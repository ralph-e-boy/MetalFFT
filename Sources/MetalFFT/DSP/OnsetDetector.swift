import Accelerate

// MARK: - OnsetDetector

/// Streaming onset/beat detector using half-wave rectified spectral flux.
/// Feed short audio buffers repeatedly; read `onsets` and `isPeak` after each call.
/// Not thread-safe.
public final class OnsetDetector {
    public let sampleRate: Double
    public let fftSize: Int
    public let hopSize: Int

    /// Timestamps in seconds where onsets were detected (cumulative across all `feed` calls).
    public private(set) var onsets: [Double] = []

    /// `true` if the most recent `feed` call ended on a detected onset frame.
    public private(set) var isPeak: Bool = false

    private let stft: STFT
    private var prevMagnitudes: [Float]
    private var fluxHistory: [Float] = []
    private var frameCount: Int = 0

    /// - Parameters:
    ///   - sampleRate: Input signal sample rate.
    ///   - fftSize: FFT size for spectral analysis. Default 1024.
    ///   - hopSize: Hop between frames. Default 256 (75% overlap at fftSize=1024).
    public init(sampleRate: Double, fftSize: Int = 1024, hopSize: Int = 256) throws {
        self.sampleRate = sampleRate
        self.fftSize    = fftSize
        self.hopSize    = hopSize
        self.stft       = try STFT(fftSize: fftSize, hopSize: hopSize, window: .hann, sampleRate: sampleRate)
        self.prevMagnitudes = [Float](repeating: 0, count: fftSize)
    }

    /// Feed the next chunk of samples. Detects onsets and appends to `onsets`.
    public func feed(_ samples: [Float]) throws {
        let frames = try stft.analyze(samples)
        var lastIsPeak = false
        for frame in frames {
            lastIsPeak = process(frame.magnitudes)
        }
        isPeak = lastIsPeak
    }

    /// Clear all accumulated state.
    public func reset() {
        prevMagnitudes = [Float](repeating: 0, count: fftSize)
        fluxHistory = []
        onsets = []
        frameCount = 0
        isPeak = false
    }

    // MARK: - Internal

    private func process(_ mags: [Float]) -> Bool {
        let flux = spectralFlux(mags)
        fluxHistory.append(flux)

        defer {
            prevMagnitudes = mags
            frameCount += 1
        }

        guard fluxHistory.count >= 8 else { return false }

        let windowSize = min(fluxHistory.count, 43)
        let recent = Array(fluxHistory.suffix(windowSize))
        var mean: Float = 0, meanSq: Float = 0
        vDSP_meanv(recent, 1, &mean, vDSP_Length(recent.count))
        vDSP_measqv(recent, 1, &meanSq, vDSP_Length(recent.count))
        let std = sqrt(max(0, meanSq - mean * mean))
        let threshold = mean + 1.5 * std

        let prevFlux = fluxHistory.count >= 2 ? fluxHistory[fluxHistory.count - 2] : 0
        guard flux > threshold, flux >= prevFlux else { return false }

        onsets.append(Double(frameCount) * Double(hopSize) / sampleRate)
        return true
    }

    private func spectralFlux(_ current: [Float]) -> Float {
        let n = min(current.count, prevMagnitudes.count)
        var flux: Float = 0
        for i in 0..<n {
            let diff = current[i] - prevMagnitudes[i]
            if diff > 0 { flux += diff }
        }
        return flux
    }
}
