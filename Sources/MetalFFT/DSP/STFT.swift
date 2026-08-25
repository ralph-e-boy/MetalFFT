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

/// Complex elements of scratch per side, which sets how many frames go into one
/// GPU dispatch. See `chunkFrames`.
private let chunkElements = 262_144

/// Short-Time Fourier Transform: sliding-window FFT over a long signal.
/// Construct once and reuse — holds internal Metal and scratch buffers. Not thread-safe.
public final class STFT {
    public let fftSize: Int
    public let hopSize: Int
    public let sampleRate: Double

    private let fft: MetalFFT
    private let window: [Float]
    /// Frames per GPU dispatch — 64 at a 4096-point window. One command buffer
    /// covers the whole chunk, so this is the factor the round-trip cost is
    /// divided by, and 360 frames of 4096 measured 25 ms at 4 frames a chunk,
    /// 9.6 at 16, 6.4 at 64 and 6.1 at 128. It flattens where the transform
    /// starts to cost more than the submission; 64 is that knee, and doubling
    /// again buys 5% for twice the scratch.
    private let chunkFrames: Int
    private var inputChunk: [SIMD2<Float>]
    private var outputChunk: [SIMD2<Float>]

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
        chunkFrames = max(1, chunkElements / fftSize)
        // The imaginary halves are written once, here: the packing below fills
        // only the real slots, and nothing ever writes an imaginary one back.
        inputChunk = [SIMD2<Float>](repeating: .zero, count: chunkFrames * fftSize)
        outputChunk = [SIMD2<Float>](repeating: .zero, count: chunkFrames * fftSize)
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
    ///
    /// Frames are transformed `chunkFrames` at a time in a single GPU dispatch.
    /// Done one frame per command buffer, a 256-sample hop asks the GPU for
    /// ~190 round trips per second of audio and spends almost all of that time
    /// in submission rather than in the transform.
    public func analyze(_ signal: [Float]) throws -> [STFTFrame] {
        let total = frameCount(for: signal.count)
        guard total > 0 else { return [] }

        var frames = [STFTFrame]()
        frames.reserveCapacity(total)
        var first = 0
        while first < total {
            let count = Swift.min(chunkFrames, total - first)
            pack(signal, from: first, count: count)
            try inputChunk.withUnsafeBufferPointer { input in
                try fft.forward(batch: UnsafeBufferPointer(rebasing: input[0 ..< count * fftSize]),
                                count: count,
                                output: &outputChunk)
            }
            for index in 0 ..< count {
                let complex = Array(outputChunk[index * fftSize ..< (index + 1) * fftSize])
                frames.append(STFTFrame(complex: complex,
                                        magnitudes: Spectrum.magnitudes(complex)))
            }
            first += count
        }
        return frames
    }

    /// 2-D spectrogram as dB magnitudes: `[time][frequency]`.
    public func spectrogram(_ signal: [Float]) throws -> [[Float]] {
        try analyze(signal).map(\.magnitudesDB)
    }

    // MARK: - Internal

    /// Windows `count` frames, starting at frame `first`, straight into the real
    /// slots of the chunk buffer. Writing at stride 2 lands each sample in the
    /// `.x` of its `SIMD2<Float>` and leaves the `.y` at the zero it was
    /// initialised to, so windowing and complex packing are one `vDSP_vmul`
    /// rather than a multiply followed by a scalar interleave.
    private func pack(_ signal: [Float], from first: Int, count: Int) {
        signal.withUnsafeBufferPointer { source in
            inputChunk.withUnsafeMutableBufferPointer { destination in
                destination.withMemoryRebound(to: Float.self) { reals in
                    for index in 0 ..< count {
                        vDSP_vmul(source.baseAddress! + (first + index) * hopSize, 1,
                                  window, 1,
                                  reals.baseAddress! + index * fftSize * 2, 2,
                                  vDSP_Length(fftSize))
                    }
                }
            }
        }
    }
}
