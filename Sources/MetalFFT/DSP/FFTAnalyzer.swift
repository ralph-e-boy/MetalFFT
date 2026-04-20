import Accelerate

// MARK: - AnalysisResult

/// All computed properties are lazy over the stored complex spectrum — call only what you need.
public struct AnalysisResult {
    public let complex: [SIMD2<Float>]
    public let sampleRate: Double
    public let fftSize: Int

    /// Squared magnitudes (x²+y²) for each bin — same as `vDSP_zvmags`.
    public var magnitudes: [Float] { Spectrum.magnitudes(complex) }

    /// Magnitudes in dB (10·log₁₀ of squared magnitudes). Floor at –120 dB.
    public var magnitudesDB: [Float] { Spectrum.toDecibels(magnitudes) }

    /// Per-bin phase in radians (–π to π).
    public var phase: [Float] { Spectrum.phase(complex) }

    /// Frequency in Hz for a given bin index.
    public func binFrequency(_ bin: Int) -> Double { Double(bin) * sampleRate / Double(fftSize) }

    /// Parabolic-interpolated dominant frequency in Hz, or `nil` if the spectrum looks like noise.
    public var dominantFreq: Double? {
        let mags = magnitudes
        guard let peak = PeakDetection.fundamentalFrequency(
            magnitudes: mags, sampleRate: sampleRate, fftSize: fftSize,
            minFreq: 20, maxFreq: sampleRate / 2
        ) else { return nil }
        return PeakDetection.parabolicInterpolation(
            magnitudes: mags, peakIndex: peak.index,
            sampleRate: sampleRate, fftSize: fftSize
        )
    }

    /// Nearest piano note for the dominant frequency, or `nil`.
    public var dominantNote: (name: String, octave: Int)? {
        guard let f = dominantFreq else { return nil }
        return Pitch.note(frequency: f)
    }

    /// `true` if the spectrum matches environmental noise heuristics.
    public var isNoise: Bool { Spectrum.isNoise(magnitudes, sampleRate: sampleRate, fftSize: fftSize) }

    /// RMS amplitude via Parseval's theorem: √(Σ|X[k]|²) / N.
    public var rms: Float {
        var total: Float = 0
        let mags = magnitudes
        vDSP_sve(mags, 1, &total, vDSP_Length(mags.count))
        return sqrt(total) / Float(fftSize)
    }
}

// MARK: - FFTAnalyzer

/// Stateful one-stop analyzer: window → pack → GPU FFT → AnalysisResult.
/// Reuses internal buffers across calls. Not thread-safe.
public final class FFTAnalyzer {
    public let size: Int
    public let sampleRate: Double

    private let fft: MetalFFT
    private let window: [Float]
    private var windowedBuf: [Float]
    private var complexBuf: [SIMD2<Float>]
    private var outputBuf: [SIMD2<Float>]

    public init(
        size: Int,
        sampleRate: Double,
        window windowType: WindowType = .hann
    ) throws {
        self.size = size
        self.sampleRate = sampleRate
        self.fft = try MetalFFT(size: size)
        self.window = windowType.coefficients(size)
        self.windowedBuf = [Float](repeating: 0, count: size)
        self.complexBuf = [SIMD2<Float>](repeating: .zero, count: size)
        self.outputBuf = [SIMD2<Float>](repeating: .zero, count: size)
    }

    /// Analyze `samples` (must have `count == size`). Returns a lazy result struct.
    public func analyze(_ samples: [Float]) throws -> AnalysisResult {
        precondition(samples.count == size)
        samples.withUnsafeBufferPointer { ptr in
            vDSP_vmul(ptr.baseAddress!, 1, window, 1, &windowedBuf, 1, vDSP_Length(size))
        }
        for i in 0..<size { complexBuf[i] = SIMD2<Float>(windowedBuf[i], 0) }
        try complexBuf.withUnsafeBufferPointer { try fft.forward(input: $0, output: &outputBuf) }
        return AnalysisResult(complex: outputBuf, sampleRate: sampleRate, fftSize: size)
    }

    /// Analyze a sub-range of `samples` starting at `offset`, without allocating a slice.
    public func analyze(_ samples: [Float], offset: Int) throws -> AnalysisResult {
        precondition(offset + size <= samples.count)
        samples.withUnsafeBufferPointer { ptr in
            vDSP_vmul(ptr.baseAddress! + offset, 1, window, 1, &windowedBuf, 1, vDSP_Length(size))
        }
        for i in 0..<size { complexBuf[i] = SIMD2<Float>(windowedBuf[i], 0) }
        try complexBuf.withUnsafeBufferPointer { try fft.forward(input: $0, output: &outputBuf) }
        return AnalysisResult(complex: outputBuf, sampleRate: sampleRate, fftSize: size)
    }
}
