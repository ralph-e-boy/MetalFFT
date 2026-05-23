import Accelerate

// MARK: - AnalysisResult

/// All computed properties are lazy over the stored complex spectrum — call only what you need.
public struct AnalysisResult {
    public let complex: [SIMD2<Float>]
    public let sampleRate: Double
    public let fftSize: Int

    /// Squared magnitudes (x²+y²) for each bin — same as `vDSP_zvmags`.
    public var magnitudes: [Float] {
        Spectrum.magnitudes(complex)
    }

    /// Magnitudes in dB (10·log₁₀ of squared magnitudes). Floor at –120 dB.
    public var magnitudesDB: [Float] {
        Spectrum.toDecibels(magnitudes)
    }

    /// Per-bin phase in radians (–π to π).
    public var phase: [Float] {
        Spectrum.phase(complex)
    }

    /// Frequency in Hz for a given bin index.
    public func binFrequency(_ bin: Int) -> Double {
        Double(bin) * sampleRate / Double(fftSize)
    }

    /// Parabolic-interpolated dominant frequency in Hz, or `nil` if the spectrum looks like noise.
    public var dominantFreq: Float? {
        let mags = magnitudes
        guard let peak = PeakDetection.fundamentalFrequency(
            magnitudes: mags, sampleRate: sampleRate, fftSize: fftSize,
            minFreq: 20, maxFreq: sampleRate / 2
        ) else { return nil }
        return Float(PeakDetection.parabolicInterpolation(
            magnitudes: mags, peakIndex: peak.index,
            sampleRate: sampleRate, fftSize: fftSize
        ))
    }

    /// Nearest piano note for the dominant frequency, or `nil`.
    public var dominantNote: (name: String, octave: Int)? {
        guard let f = dominantFreq else { return nil }
        return Pitch.note(frequency: f)
    }

    /// `true` if the spectrum matches environmental noise heuristics.
    public var isNoise: Bool {
        Spectrum.isNoise(magnitudes, sampleRate: sampleRate, fftSize: fftSize)
    }

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
    public let windowType: WindowType

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
        self.windowType = windowType
        fft = try MetalFFT(size: size)
        window = windowType.coefficients(size)
        windowedBuf = [Float](repeating: 0, count: size)
        complexBuf = [SIMD2<Float>](repeating: .zero, count: size)
        outputBuf = [SIMD2<Float>](repeating: .zero, count: size)
    }

    /// Analyze `samples` (must have `count == size`). Returns a lazy result struct.
    public func analyze(_ samples: [Float]) throws -> AnalysisResult {
        precondition(samples.count == size)
        samples.withUnsafeBufferPointer { ptr in
            vDSP_vmul(ptr.baseAddress!, 1, window, 1, &windowedBuf, 1, vDSP_Length(size))
        }
        for i in 0 ..< size {
            complexBuf[i] = SIMD2<Float>(windowedBuf[i], 0)
        }
        try complexBuf.withUnsafeBufferPointer { try fft.forward(input: $0, output: &outputBuf) }
        return AnalysisResult(complex: outputBuf, sampleRate: sampleRate, fftSize: size)
    }

    /// Analyze a sub-range of `samples` starting at `offset`, without allocating a slice.
    public func analyze(_ samples: [Float], offset: Int) throws -> AnalysisResult {
        precondition(offset + size <= samples.count)
        samples.withUnsafeBufferPointer { ptr in
            vDSP_vmul(ptr.baseAddress! + offset, 1, window, 1, &windowedBuf, 1, vDSP_Length(size))
        }
        for i in 0 ..< size {
            complexBuf[i] = SIMD2<Float>(windowedBuf[i], 0)
        }
        try complexBuf.withUnsafeBufferPointer { try fft.forward(input: $0, output: &outputBuf) }
        return AnalysisResult(complex: outputBuf, sampleRate: sampleRate, fftSize: size)
    }

    // MARK: - Zero-copy variants

    /// Zero-allocation analyze: windows the input, runs the GPU FFT, and writes
    /// squared magnitudes (x²+y²) of the first `binCount` bins into `output`.
    ///
    /// `binCount` defaults to `N/2 + 1` (rfft convention: DC through Nyquist
    /// inclusive). Callers that only need the unique-frequency half of a real-input
    /// spectrum should pass `binCount: N/2`.
    ///
    /// `output` must have capacity ≥ `binCount`. No allocations occur per call.
    public func analyze(
        samples: UnsafeBufferPointer<Float>,
        intoMagnitudes output: UnsafeMutableBufferPointer<Float>,
        binCount: Int? = nil
    ) throws {
        precondition(samples.count == size, "expected \(size) samples, got \(samples.count)")
        let count = binCount ?? (size / 2 + 1)
        precondition(count > 0 && count <= size, "binCount out of range: \(count)")
        precondition(output.count >= count, "output capacity \(output.count) < binCount \(count)")

        vDSP_vmul(samples.baseAddress!, 1, window, 1, &windowedBuf, 1, vDSP_Length(size))
        for i in 0 ..< size {
            complexBuf[i] = SIMD2<Float>(windowedBuf[i], 0)
        }
        try complexBuf.withUnsafeBufferPointer { try fft.forward(input: $0, output: &outputBuf) }

        let outPtr = output.baseAddress!
        outputBuf.withUnsafeBufferPointer { src in
            let srcPtr = src.baseAddress!
            for i in 0 ..< count {
                let c = srcPtr[i]
                outPtr[i] = c.x * c.x + c.y * c.y
            }
        }
    }

    /// Zero-allocation analyze: writes dB-scaled magnitudes (10·log₁₀(x²+y²))
    /// of the first `binCount` bins into `output`, clamped at `floorDB`.
    ///
    /// See `analyze(samples:intoMagnitudes:binCount:)` for `binCount` semantics.
    public func analyze(
        samples: UnsafeBufferPointer<Float>,
        intoMagnitudesDB output: UnsafeMutableBufferPointer<Float>,
        binCount: Int? = nil,
        floorDB: Float = -120
    ) throws {
        let count = binCount ?? (size / 2 + 1)
        try analyze(samples: samples, intoMagnitudes: output, binCount: count)
        let outPtr = output.baseAddress!
        var one: Float = 1
        vDSP_vdbcon(outPtr, 1, &one, outPtr, 1, vDSP_Length(count), 1)
        var lo = floorDB
        var hi = Float.greatestFiniteMagnitude
        vDSP_vclip(outPtr, 1, &lo, &hi, outPtr, 1, vDSP_Length(count))
    }
}
