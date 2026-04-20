import Accelerate

// MARK: - PeakDetection

/// Stateless frequency-domain peak detection utilities.
public enum PeakDetection {

    // MARK: - Simple Peak

    /// Returns the index and value of the maximum in `magnitudes`, optionally restricted to `range`.
    public static func peak(
        in magnitudes: [Float],
        range: Range<Int>? = nil
    ) -> (index: Int, value: Float)? {
        guard !magnitudes.isEmpty else { return nil }
        let lo = range?.lowerBound ?? 0
        let hi = min(range?.upperBound ?? magnitudes.count, magnitudes.count)
        guard lo < hi else { return nil }

        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(Array(magnitudes[lo..<hi]), 1, &maxVal, &maxIdx, vDSP_Length(hi - lo))
        return (index: lo + Int(maxIdx), value: maxVal)
    }

    // MARK: - Fundamental Frequency (harmonic scoring)

    /// Finds the fundamental frequency bin using harmonic scoring.
    ///
    /// Scores each candidate bin by its magnitude plus 50% credit for harmonics 2–5.
    /// Low-frequency candidates (< 300 Hz) receive a 20% boost.
    ///
    /// - Parameters:
    ///   - magnitudes: Squared-magnitude spectrum (from `Spectrum.magnitudes`).
    ///   - sampleRate: Original signal sample rate.
    ///   - fftSize: Original real signal length (= 2 × `magnitudes.count`).
    ///   - minFreq: Lowest candidate frequency in Hz.
    ///   - maxFreq: Highest candidate frequency in Hz.
    ///   - magnitudeThreshold: Bins below this power are skipped. Default 0.12 (for squared magnitudes).
    /// - Returns: The winning bin index and its raw frequency, or `nil` if none qualify.
    public static func fundamentalFrequency(
        magnitudes: [Float],
        sampleRate: Double,
        fftSize: Int,
        minFreq: Double,
        maxFreq: Double,
        magnitudeThreshold: Float = 0.12
    ) -> (index: Int, rawFrequency: Double)? {
        let count = magnitudes.count
        let freqPerBin = sampleRate / Double(fftSize)
        let minIndex = Int(minFreq / freqPerBin)
        let maxIndex = min(Int(maxFreq / freqPerBin), count - 1)
        guard minIndex < maxIndex else {
            return simplePeak(magnitudes: magnitudes, sampleRate: sampleRate, fftSize: fftSize)
        }

        var bestScore: Float = 0
        var bestIndex = -1

        for i in minIndex...maxIndex where magnitudes[i] > magnitudeThreshold {
            let score = harmonicScore(magnitudes: magnitudes, candidateIndex: i,
                                      count: count, sampleRate: sampleRate, fftSize: fftSize,
                                      freqPerBin: freqPerBin)
            if score > bestScore {
                bestScore = score
                bestIndex = i
            }
        }

        guard bestIndex >= 0 else {
            return simplePeak(magnitudes: magnitudes, sampleRate: sampleRate, fftSize: fftSize)
        }
        return (index: bestIndex, rawFrequency: Double(bestIndex) * freqPerBin)
    }

    // MARK: - Parabolic Interpolation

    /// Sub-bin frequency via 3-point parabolic interpolation around `peakIndex`.
    ///
    /// - Parameters:
    ///   - magnitudes: Squared-magnitude spectrum.
    ///   - peakIndex: Bin index of the peak.
    ///   - sampleRate: Original signal sample rate.
    ///   - fftSize: Original real signal length (= 2 × `magnitudes.count`).
    /// - Returns: Interpolated frequency in Hz.
    public static func parabolicInterpolation(
        magnitudes: [Float],
        peakIndex: Int,
        sampleRate: Double,
        fftSize: Int
    ) -> Double {
        let binWidth = sampleRate / Double(fftSize)
        guard peakIndex > 0, peakIndex < magnitudes.count - 1 else {
            return Double(peakIndex) * binWidth
        }
        let left   = magnitudes[peakIndex - 1]
        let center = magnitudes[peakIndex]
        let right  = magnitudes[peakIndex + 1]
        let denom  = 2.0 * (left - 2.0 * center + right)
        guard denom != 0 else { return Double(peakIndex) * binWidth }
        let offset = Double(left - right) / Double(denom)
        return (Double(peakIndex) + offset) * binWidth
    }

    // MARK: - Internals

    private static func harmonicScore(
        magnitudes: [Float], candidateIndex: Int, count: Int,
        sampleRate: Double, fftSize: Int, freqPerBin: Double
    ) -> Float {
        var score = magnitudes[candidateIndex]
        for harmonic in 2...5 {
            let hIdx = candidateIndex * harmonic
            if hIdx < count { score += magnitudes[hIdx] * 0.5 }
        }
        let candidateFreq = Double(candidateIndex) * freqPerBin
        if candidateFreq < 300 { score *= 1.2 }
        return score
    }

    private static func simplePeak(
        magnitudes: [Float], sampleRate: Double, fftSize: Int
    ) -> (index: Int, rawFrequency: Double)? {
        guard let (idx, _) = peak(in: magnitudes) else { return nil }
        let freq = Double(idx) * sampleRate / Double(fftSize)
        return (index: idx, rawFrequency: freq)
    }
}
