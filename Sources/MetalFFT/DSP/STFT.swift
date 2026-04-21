import Accelerate

// MARK: - STFTFrame

public struct STFTFrame {
    public let complex: [SIMD2<Float>]
    public let magnitudes: [Float]
    public var magnitudesDB: [Float] {
        Spectrum.toDecibels(magnitudes)
    }

    public var phase: [Float] {
        Spectrum.phase(complex)
    }
}

// MARK: - STFT

/// Short-Time Fourier Transform: sliding-window FFT over a long signal.
/// Construct once and reuse — holds internal Metal and scratch buffers. Not thread-safe.
public final class STFT {
    public let fftSize: Int
    public let hopSize: Int
    public let sampleRate: Double

    private let fft: MetalFFT
    private let window: [Float]
    private var windowedBuf: [Float]
    private var complexBuf: [SIMD2<Float>]
    private var outputBuf: [SIMD2<Float>]

    /// - Parameters:
    ///   - fftSize: Must be a supported MetalFFT size (64–16384).
    ///   - hopSize: Samples advanced per frame. Overlap = `fftSize - hopSize`.
    ///              Typical choices: `fftSize/2` (50%) or `fftSize/4` (75%).
    ///   - windowType: Spectral window applied before each FFT. Default `.hann`.
    ///   - sampleRate: Input signal sample rate in Hz.
    public init(
        fftSize: Int,
        hopSize: Int,
        window windowType: WindowType = .hann,
        sampleRate: Double
    ) throws {
        self.fftSize = fftSize
        self.hopSize = hopSize
        self.sampleRate = sampleRate
        fft = try MetalFFT(size: fftSize)
        window = windowType.coefficients(fftSize)
        windowedBuf = [Float](repeating: 0, count: fftSize)
        complexBuf = [SIMD2<Float>](repeating: .zero, count: fftSize)
        outputBuf = [SIMD2<Float>](repeating: .zero, count: fftSize)
    }

    /// Number of frames that will be produced for a signal of `sampleCount` samples.
    public func frameCount(for sampleCount: Int) -> Int {
        guard sampleCount >= fftSize else { return 0 }
        return (sampleCount - fftSize) / hopSize + 1
    }

    /// Frequency in Hz for a given bin index.
    public func binFrequency(_ bin: Int) -> Double {
        Double(bin) * sampleRate / Double(fftSize)
    }

    /// Start time in seconds for a given frame index.
    public func frameTime(_ frameIndex: Int) -> Double {
        Double(frameIndex * hopSize) / sampleRate
    }

    /// Analyze `signal`, returning one `STFTFrame` per hop.
    public func analyze(_ signal: [Float]) throws -> [STFTFrame] {
        var frames = [STFTFrame]()
        frames.reserveCapacity(frameCount(for: signal.count))
        var pos = 0
        while pos + fftSize <= signal.count {
            try frames.append(frame(signal: signal, offset: pos))
            pos += hopSize
        }
        return frames
    }

    /// 2-D spectrogram as dB magnitudes: `[time][frequency]`.
    public func spectrogram(_ signal: [Float]) throws -> [[Float]] {
        try analyze(signal).map(\.magnitudesDB)
    }

    // MARK: - Internal

    private func frame(signal: [Float], offset: Int) throws -> STFTFrame {
        signal.withUnsafeBufferPointer { ptr in
            vDSP_vmul(ptr.baseAddress! + offset, 1, window, 1, &windowedBuf, 1, vDSP_Length(fftSize))
        }
        for i in 0 ..< fftSize {
            complexBuf[i] = SIMD2<Float>(windowedBuf[i], 0)
        }
        try complexBuf.withUnsafeBufferPointer { try fft.forward(input: $0, output: &outputBuf) }
        let out = outputBuf
        return STFTFrame(complex: out, magnitudes: Spectrum.magnitudes(out))
    }
}
