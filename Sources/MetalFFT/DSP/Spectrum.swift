import Accelerate

// MARK: - Spectrum

/// Stateless spectral analysis utilities.
public enum Spectrum {

    // MARK: - RMS

    /// RMS amplitude of a real sample buffer (equivalent to `vDSP_rmsqv`).
    public static func rms(_ samples: UnsafeBufferPointer<Float>) -> Float {
        var result: Float = 0
        vDSP_rmsqv(samples.baseAddress!, 1, &result, vDSP_Length(samples.count))
        return result
    }

    /// Convenience overload for Array.
    public static func rms(_ samples: [Float]) -> Float {
        samples.withUnsafeBufferPointer { rms($0) }
    }

    // MARK: - Magnitudes (squared, matching vDSP_zvmags)

    /// Returns the squared magnitude (x²+y²) of each complex bin — matches `vDSP_zvmags` output.
    /// `count` limits the number of bins returned; defaults to `complex.count`.
    public static func magnitudes(_ complex: [SIMD2<Float>], count: Int? = nil) -> [Float] {
        let n = count ?? complex.count
        return (0..<n).map { i in
            let c = complex[i]
            return c.x * c.x + c.y * c.y
        }
    }

    // MARK: - Normalize

    /// Normalizes `magnitudes` in-place to [0, 1] by the peak value. No-op if peak is zero.
    public static func normalize(_ magnitudes: inout [Float]) {
        var peak: Float = 0
        vDSP_maxv(magnitudes, 1, &peak, vDSP_Length(magnitudes.count))
        guard peak > 0 else { return }
        var inv = 1.0 / peak
        vDSP_vsmul(magnitudes, 1, &inv, &magnitudes, 1, vDSP_Length(magnitudes.count))
    }

    // MARK: - Decibels

    /// Converts squared magnitudes to dB (10·log₁₀). `floorDB` clamps -∞ from zero-valued bins.
    public static func toDecibels(_ magnitudes: [Float], floorDB: Float = -120) -> [Float] {
        var out = [Float](repeating: 0, count: magnitudes.count)
        var b: Float = 1.0
        vDSP_vdbcon(magnitudes, 1, &b, &out, 1, vDSP_Length(magnitudes.count), 1)
        var lo = floorDB
        var hi = Float.greatestFiniteMagnitude
        vDSP_vclip(out, 1, &lo, &hi, &out, 1, vDSP_Length(out.count))
        return out
    }

    // MARK: - Phase

    /// Returns the instantaneous phase (atan2) of each complex bin, in radians (–π to π).
    public static func phase(_ complex: [SIMD2<Float>]) -> [Float] {
        complex.map { atan2($0.y, $0.x) }
    }

    // MARK: - Noise Detection

    /// Configuration for `isNoise`.
    public struct NoiseConfig {
        /// Fraction of total power in 0–200 Hz that classifies as noise. Default: 0.8.
        public var lowFreqDominanceThreshold: Float
        /// Spectral variance below this → noise. Default: 0.005.
        public var flatnessVarianceThreshold: Float
        /// Peak-to-mean ratio below this → noise. Default: 2.5.
        public var peakToMeanRatioThreshold: Float

        public init(
            lowFreqDominanceThreshold: Float = 0.8,
            flatnessVarianceThreshold: Float = 0.005,
            peakToMeanRatioThreshold: Float = 2.5
        ) {
            self.lowFreqDominanceThreshold = lowFreqDominanceThreshold
            self.flatnessVarianceThreshold = flatnessVarianceThreshold
            self.peakToMeanRatioThreshold  = peakToMeanRatioThreshold
        }

        public static let `default` = NoiseConfig()
    }

    /// Returns `true` if the power spectrum `magnitudes` looks like environmental noise.
    ///
    /// `magnitudes` is expected to be squared magnitudes as returned by `Spectrum.magnitudes(_:)`.
    /// `sampleRate` is the original signal's sample rate.
    /// `fftSize` is the length of the real signal (2 × `magnitudes.count`).
    public static func isNoise(
        _ magnitudes: [Float],
        sampleRate: Double,
        fftSize: Int,
        config: NoiseConfig = .default
    ) -> Bool {
        let count = magnitudes.count
        let freqResolution = sampleRate / Double(fftSize)
        let lowFreqEnd = min(Int(200.0 / freqResolution), count)
        let midFreqEnd = min(Int(2000.0 / freqResolution), count)
        guard lowFreqEnd < count, midFreqEnd < count else { return false }

        var lowSum: Float = 0
        vDSP_sve(magnitudes, 1, &lowSum, vDSP_Length(lowFreqEnd))
        var totalSum: Float = 0
        vDSP_sve(magnitudes, 1, &totalSum, vDSP_Length(count))
        guard totalSum > 0 else { return true }

        if lowSum / totalSum > config.lowFreqDominanceThreshold { return true }

        var mean: Float = 0, meanSq: Float = 0
        vDSP_meanv(magnitudes, 1, &mean, vDSP_Length(count))
        vDSP_measqv(magnitudes, 1, &meanSq, vDSP_Length(count))
        let variance = meanSq - mean * mean
        if variance < config.flatnessVarianceThreshold { return true }

        var peak: Float = 0
        vDSP_maxv(magnitudes, 1, &peak, vDSP_Length(count))
        if mean > 0, peak / mean < config.peakToMeanRatioThreshold { return true }

        return false
    }
}
