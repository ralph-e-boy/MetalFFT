import Accelerate

// MARK: - YINEstimate

/// One pitch estimate from `YINDetector`.
public struct YINEstimate: Equatable, Sendable {
    /// Estimated fundamental in Hz.
    public let frequencyHz: Float
    /// Periodicity of the window, 0…1 — one minus the normalised difference at
    /// the bottom of the chosen dip, interpolated along with the lag. A clean
    /// plucked string sits above 0.9; a struck noise or a chord sits low. This
    /// is the number to gate a display on.
    public let confidence: Float
    /// Whether the absolute threshold was crossed, i.e. a genuinely periodic
    /// lag was found rather than the least-bad lag in the window.
    ///
    /// Separate from `confidence` on purpose: the two answer different
    /// questions. `confidence` says how periodic the window is, this says
    /// whether YIN believes it found *the* period at all.
    public let isPeriodic: Bool
    /// RMS of the window, before any gating.
    public let rms: Float
    /// The chosen lag in samples, interpolated. Useful for diagnostics.
    public let lag: Float
}

// MARK: - YINDetector

/// Monophonic pitch detection by the YIN algorithm of de Cheveigné and
/// Kawahara (2002).
///
/// YIN works in the **time domain**, on the samples themselves, and this is why
/// it is here alongside the spectral tools rather than built out of them. A
/// magnitude spectrum locates *partials*; which partial is the fundamental is
/// then an inference, and the inference is what goes wrong — a bin is 11.7 Hz
/// wide at a 4096-point transform on 48 kHz, which is most of a semitone down
/// where a bass string lives, and a microphone that rolls off the fundamental
/// leaves the loudest peak an octave above the note. YIN never asks the
/// question: it measures how long the waveform takes to repeat, which is the
/// pitch by definition, and resolves that period to a fraction of a sample.
///
/// The four steps are the paper's, in order:
///
/// 1. **Difference function** `d(τ) = Σ(x[j] − x[j+τ])²` — how badly the window
///    fails to match itself τ samples later.
/// 2. **Cumulative mean normalisation** — divide each `d(τ)` by the running mean
///    of everything below it. Without this, `d` falls off with τ and the search
///    drifts to long lags; with it, the first deep dip is the period.
/// 3. **Absolute threshold** — take the *first* dip below the threshold, not
///    the deepest. Taking the deepest is what produces octave-down errors,
///    since twice the period always matches too. A dip, not merely a lag under
///    the threshold: the curve has to be falling as well, or a partial above
///    `maxFrequency` gets accepted on its way back up.
/// 4. **Parabolic interpolation** around that lag, so the period is not
///    quantised to whole samples. At 48 kHz an E2 period is 582 samples and one
///    sample is 3 cents, so this step is the difference between a tuner and a
///    note-namer.
///
/// Step 1 is the expensive one — quadratic if written directly. It is computed
/// here from the autocorrelation instead, using
/// `d(τ) = p(0) + p(τ) − 2·r(τ)`, where `p(τ)` is the energy of the window
/// starting at τ (a running sum, O(1) per lag) and `r(τ)` the correlation at
/// lag τ. Each `r(τ)` is a single `vDSP_dotpr`, so the inner loop is one
/// vectorised call per lag rather than three, and no temporary buffers are
/// touched at all.
///
/// That inner loop still costs `lags × window`, and both halves of that product
/// shrink under decimation, so steps 1–3 run on a low-passed copy at a quarter
/// of the sample rate — sixteen times less work — and only step 4 returns to
/// the full rate, recomputing `d(τ)` across the handful of lags around the
/// coarse answer. Precision is set entirely by that second stage, so nothing is
/// given up for the speed: the decimated pass only has to land within a couple
/// of samples of the period, which is what it is good at. When the frequency
/// range or the window is too small for a clean decimation the factor falls to
/// 1 and the coarse pass is simply the exact one.
///
/// Allocation-free after `init`. Not thread-safe: one detector per thread, or
/// serialise calls.
public final class YINDetector {
    /// Sample rate the windows arrive at.
    public let sampleRate: Double
    /// Window length in samples. Must be at least twice the longest period to
    /// be detected — the difference function compares the first half of the
    /// window against every offset into the second.
    public let bufferSize: Int

    public let minFrequency: Float
    public let maxFrequency: Float

    /// Normalised-difference level a lag must dip below to be taken as the
    /// period. The paper suggests 0.1; 0.15 is a little more willing, which
    /// suits a plucked string whose harmonic content shifts as it decays.
    public let threshold: Float

    /// Windows quieter than this are not analysed at all. The single most
    /// effective thing in the whole class: a detector asked to find a pitch in
    /// room noise will always find one.
    public var rmsThreshold: Float

    /// Lag search bounds at the full sample rate, derived from the frequency
    /// bounds.
    private let minLag: Int
    private let maxLag: Int

    /// Rate reduction applied before steps 1–3. 1 disables the coarse stage.
    let decimation: Int
    /// Anti-alias low pass, applied by `vDSP_desamp` as it decimates. Empty
    /// when `decimation` is 1.
    private let antiAlias: [Float]

    /// The decimated window and its lag bounds.
    private let coarseCount: Int
    private let coarseMinLag: Int
    private let coarseMaxLag: Int

    private var coarseSignal: [Float]
    private var difference: [Float]
    private var normalized: [Float]
    /// The cumulative-mean divisor of step 2, kept so that step 4 can normalise
    /// the full-rate difference the same way the coarse pass was normalised.
    private var runningMean: [Float]
    /// Full-rate normalised difference over the refinement neighbourhood.
    private var local: [Float]

    /// - Parameters:
    ///   - sampleRate: Samples per second.
    ///   - bufferSize: Window length. 4096 at 48 kHz reaches down to about
    ///     23 Hz, comfortably below a bass guitar's low B.
    ///   - minFrequency: Lowest fundamental to look for. Default 55 Hz (A1),
    ///     below a guitar's low E.
    ///   - maxFrequency: Highest fundamental to look for. Default 1500 Hz.
    ///   - threshold: Absolute threshold for step 3.
    ///   - rmsThreshold: Level gate.
    public init(sampleRate: Double,
                bufferSize: Int,
                minFrequency: Float = 55,
                maxFrequency: Float = 1500,
                threshold: Float = 0.15,
                rmsThreshold: Float = 0.005) {
        precondition(bufferSize >= 64, "YIN needs a window of at least 64 samples")
        self.sampleRate = sampleRate
        self.bufferSize = bufferSize
        self.minFrequency = minFrequency
        self.maxFrequency = maxFrequency
        self.threshold = threshold
        self.rmsThreshold = rmsThreshold

        // The first half of the window is matched against offsets into the
        // second, so the longest lag examinable is half the window.
        let half = bufferSize / 2
        let longest = Int((Float(sampleRate) / max(1, minFrequency)).rounded(.up))
        let shortest = Int((Float(sampleRate) / max(1, maxFrequency)).rounded(.down))
        maxLag = Swift.min(half - 1, Swift.max(4, longest))
        minLag = Swift.max(2, Swift.min(shortest, maxLag - 1))

        let plan = YINDetector.decimationPlan(bufferSize: bufferSize,
                                              maxLag: maxLag,
                                              sampleRate: sampleRate,
                                              maxFrequency: maxFrequency)
        decimation = plan.factor
        coarseCount = plan.count
        antiAlias = plan.factor > 1
            ? YINDetector.lowPass(taps: plan.taps, cutoff: 0.4 / Float(plan.factor))
            : []

        // One decimated lag stands for `decimation` full-rate ones, and the
        // decimated window is shorter than the full one by the filter's length,
        // so the range is rounded up and then clipped to what the window holds.
        coarseMaxLag = Swift.min(plan.count / 2 - 1,
                                 Swift.max(4, (maxLag + plan.factor - 1) / plan.factor))
        coarseMinLag = Swift.max(2, Swift.min(minLag / plan.factor, coarseMaxLag - 1))

        coarseSignal = [Float](repeating: 0, count: plan.factor > 1 ? plan.count : 0)
        difference = [Float](repeating: 0, count: coarseMaxLag + 1)
        normalized = [Float](repeating: 0, count: coarseMaxLag + 1)
        runningMean = [Float](repeating: 1, count: coarseMaxLag + 1)
        local = [Float](repeating: 0, count: 4 * plan.factor + 3)
    }

    // MARK: - Decimation plan

    /// The largest usable rate reduction, with the filter length and decimated
    /// window length that go with it. Falls back to 1 — no coarse stage — when
    /// no factor leaves the advertised lag range intact.
    private static func decimationPlan(bufferSize: Int,
                                       maxLag: Int,
                                       sampleRate: Double,
                                       maxFrequency: Float) -> (factor: Int, taps: Int, count: Int) {
        // Keep the fourth harmonic of the highest fundamental above the new
        // Nyquist rate: a missing fundamental is found through its harmonics,
        // so decimating down to the fundamental alone would defeat the point.
        let headroom = Swift.min(4, Int(sampleRate / Double(8 * Swift.max(1, maxFrequency))))
        guard headroom >= 2 else { return (1, 0, bufferSize) }
        let usable = { (factor: Int) -> (factor: Int, taps: Int, count: Int)? in
            let taps = 16 * factor + 1
            let count = (bufferSize - taps) / factor + 1
            let needed = (maxLag + factor - 1) / factor + 2
            guard count >= 128, count / 2 - 1 >= needed else { return nil }
            return (factor, taps, count)
        }
        return (2 ... headroom)
            .reversed()
            .lazy
            .compactMap(usable)
            .first ?? (1, 0, bufferSize)
    }

    /// Hamming-windowed sinc at unit DC gain.
    private static func lowPass(taps: Int, cutoff: Float) -> [Float] {
        let last = Float(taps - 1)
        let response = (0 ..< taps).map { tap -> Float in
            let offset = Float(tap) - last / 2
            let sinc = offset == 0 ? 2 * cutoff : sinf(2 * .pi * cutoff * offset) / (.pi * offset)
            return sinc * (0.54 - 0.46 * cosf(2 * .pi * Float(tap) / last))
        }
        let gain = vDSP.sum(response)
        return gain > 0 ? vDSP.divide(response, gain) : response
    }

    // MARK: - Detection

    /// Estimate the pitch of one window. Returns nil when the window is below
    /// the level gate or holds no usable lag at all.
    ///
    /// `samples` must hold exactly `bufferSize` values.
    public func detect(_ samples: [Float]) -> YINEstimate? {
        samples.withUnsafeBufferPointer { detect($0) }
    }

    public func detect(_ samples: UnsafeBufferPointer<Float>) -> YINEstimate? {
        guard samples.count == bufferSize, let x = samples.baseAddress else { return nil }

        var rms: Float = 0
        vDSP_rmsqv(x, 1, &rms, vDSP_Length(bufferSize))
        guard rms > rmsThreshold else { return nil }

        withCoarseSignal(x) { coarse in
            computeDifference(coarse, count: coarseCount)
            cumulativeMeanNormalize()
        }

        guard let lag = chooseLag() else { return nil }
        guard let refined = refine(x, around: lag.index), refined > 0 else { return nil }

        let dip = lag.index < coarseMaxLag
            ? vertex(normalized[lag.index - 1], normalized[lag.index], normalized[lag.index + 1]).value
            : normalized[lag.index]
        return YINEstimate(frequencyHz: Float(sampleRate) / refined,
                           confidence: 1 - Swift.max(0, Swift.min(1, dip)),
                           isPeriodic: lag.crossedThreshold,
                           rms: rms,
                           lag: refined)
    }

    // MARK: - Stage one: the decimated signal

    /// Runs `body` over the signal steps 1–3 work on: a low-passed, decimated
    /// copy, or the window itself when the factor is 1.
    private func withCoarseSignal<R>(_ x: UnsafePointer<Float>,
                                     _ body: (UnsafePointer<Float>) -> R) -> R {
        guard decimation > 1 else { return body(x) }
        antiAlias.withUnsafeBufferPointer { filter in
            coarseSignal.withUnsafeMutableBufferPointer { out in
                vDSP_desamp(x, decimation, filter.baseAddress!, out.baseAddress!,
                            vDSP_Length(coarseCount), vDSP_Length(filter.count))
            }
        }
        return coarseSignal.withUnsafeBufferPointer { body($0.baseAddress!) }
    }

    // MARK: - Step 1: difference function

    /// `d(τ) = p(0) + p(τ) − 2·r(τ)` over the search range.
    private func computeDifference(_ x: UnsafePointer<Float>, count: Int) {
        let window = count / 2
        let length = vDSP_Length(window)

        // p(0): energy of the first half.
        var energy: Float = 0
        vDSP_svesq(x, 1, &energy, length)
        let head = energy

        difference.withUnsafeMutableBufferPointer { d in
            d[0] = 0
            var running = head
            for lag in 1 ... coarseMaxLag {
                // p(τ) from p(τ−1): the sample leaving the window at the front,
                // the one arriving at the back. O(1) instead of O(window).
                let leaving = x[lag - 1]
                let arriving = x[lag + window - 1]
                running += arriving * arriving - leaving * leaving

                var correlation: Float = 0
                vDSP_dotpr(x, 1, x + lag, 1, &correlation, length)

                d[lag] = head + running - 2 * correlation
            }
        }
    }

    // MARK: - Step 2: cumulative mean normalisation

    private func cumulativeMeanNormalize() {
        difference.withUnsafeBufferPointer { d in
            normalized.withUnsafeMutableBufferPointer { n in
                runningMean.withUnsafeMutableBufferPointer { m in
                    n[0] = 1
                    m[0] = 1
                    var running: Float = 0
                    for lag in 1 ... coarseMaxLag {
                        running += d[lag]
                        let mean = running / Float(lag)
                        m[lag] = mean > 0 ? mean : 1
                        n[lag] = mean > 0 ? d[lag] / mean : 1
                    }
                }
            }
        }
    }

    // MARK: - Step 3: absolute threshold

    /// The first *dip* below the threshold, walked down to its bottom. Falls
    /// back to the shallowest minimum in range, reported as not periodic so the
    /// caller can decide whether to trust it.
    private func chooseLag() -> (index: Int, crossedThreshold: Bool)? {
        var bestIndex = -1
        var bestValue = Float.infinity

        var lag = coarseMinLag
        while lag <= coarseMaxLag {
            // Below the threshold *and* falling. Without the second half of
            // that test a curve that is already under the threshold at the
            // bottom of the range — a strong partial above `maxFrequency` —
            // is accepted on its way back up, which is not a period at all.
            if normalized[lag] < threshold, normalized[lag] <= normalized[lag - 1] {
                // Descend to the true bottom: the threshold is crossed on the
                // way down, and the minimum a sample or two later is the
                // better estimate of the period.
                var descended = lag
                while descended + 1 <= coarseMaxLag,
                      normalized[descended + 1] < normalized[descended] {
                    descended += 1
                }
                return (descended, true)
            }
            if normalized[lag] < bestValue {
                bestValue = normalized[lag]
                bestIndex = lag
            }
            lag += 1
        }

        guard bestIndex > 0 else { return nil }
        return (bestIndex, false)
    }

    // MARK: - Step 4: full-rate refinement and interpolation

    /// The period to a fraction of a sample, at the full rate. The coarse lag
    /// is only accurate to ±`decimation` samples, so `d(τ)` is recomputed over
    /// twice that either side and the minimum interpolated.
    ///
    /// The recomputed difference is divided by the same cumulative mean the
    /// coarse pass used, read back at fractional resolution. That divisor is
    /// what makes the curve a CMNDF rather than a raw difference, and it drifts
    /// across the neighbourhood; ignoring it would tilt the parabola and move
    /// its vertex. When `decimation` is 1 the divisor is exact and this is the
    /// textbook interpolation of `d'(τ)`.
    private func refine(_ x: UnsafePointer<Float>, around coarseLag: Int) -> Float? {
        let window = bufferSize / 2
        let length = vDSP_Length(window)
        let center = coarseLag * decimation
        let span = 2 * decimation
        // Clamped to the requested lag range: the refinement must not answer
        // with a period the caller ruled out through its frequency bounds.
        let lowest = Swift.max(minLag, center - span)
        let highest = Swift.min(maxLag, center + span)
        guard lowest <= highest else { return nil }

        var head: Float = 0
        vDSP_svesq(x, 1, &head, length)
        // p(τ) for the lag below the first one examined, then the same running
        // update as step 1 across the neighbourhood.
        var running: Float = 0
        vDSP_svesq(x + lowest - 1, 1, &running, length)

        var bestIndex = -1
        var bestValue = Float.infinity
        for lag in (lowest - 1) ... (highest + 1) {
            if lag > lowest - 1 {
                let leaving = x[lag - 1]
                let arriving = x[lag + window - 1]
                running += arriving * arriving - leaving * leaving
            }
            var correlation: Float = 0
            vDSP_dotpr(x, 1, x + lag, 1, &correlation, length)

            let value = (head + running - 2 * correlation) / coarseMean(at: lag)
            local[lag - lowest + 1] = value
            if lag >= lowest, lag <= highest, value < bestValue {
                bestValue = value
                bestIndex = lag
            }
        }

        guard bestIndex > 0 else { return nil }
        let index = bestIndex - lowest + 1
        return Float(bestIndex) + vertex(local[index - 1], local[index], local[index + 1]).offset
    }

    /// The step 2 divisor at a full-rate lag, linearly interpolated between the
    /// decimated lags either side of it.
    private func coarseMean(at lag: Int) -> Float {
        let position = Float(lag) / Float(decimation)
        let index = Int(position)
        guard index >= 1 else { return runningMean[1] }
        guard index + 1 <= coarseMaxLag else { return runningMean[coarseMaxLag] }
        return runningMean[index]
            + (position - Float(index)) * (runningMean[index + 1] - runningMean[index])
    }

    /// The vertex of the parabola through three samples, as an offset from the
    /// middle one and the value there.
    private func vertex(_ left: Float, _ mid: Float, _ right: Float) -> (offset: Float, value: Float) {
        let denominator = 2 * (2 * mid - right - left)
        guard denominator != 0 else { return (0, mid) }
        let offset = (right - left) / denominator
        // A vertex further than a sample away means the three points did not
        // describe a dip; the sample itself is the better answer.
        guard abs(offset) <= 1 else { return (0, mid) }
        return (offset, mid + 0.25 * (right - left) * offset)
    }
}
