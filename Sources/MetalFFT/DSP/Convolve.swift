import Accelerate
import Metal

// MARK: - Convolver

/// FFT-based overlap-add FIR convolution. Pre-computes the kernel spectrum at init.
/// Holds a `MetalFFT` instance — create once per (kernel, fftSize) and reuse for any signal.
public final class Convolver {
    public let blockSize: Int
    public let hopSize: Int

    private let fft: MetalFFT
    private let kernelSpectrum: [SIMD2<Float>]
    private let kernelLen: Int

    /// - Parameters:
    ///   - kernel: FIR filter coefficients.
    ///   - fftSize: Must be a supported MetalFFT size (64–16384) and strictly greater than `kernel.count`.
    public init(kernel: [Float], fftSize: Int) throws {
        precondition(fftSize > kernel.count,
                     "fftSize must be > kernel.count; got fftSize=\(fftSize), kernel.count=\(kernel.count)")
        blockSize = fftSize
        hopSize = fftSize - kernel.count + 1
        kernelLen = kernel.count
        fft = try MetalFFT(size: fftSize)

        var kernelComplex = [SIMD2<Float>](repeating: .zero, count: fftSize)
        for i in 0 ..< kernel.count {
            kernelComplex[i] = SIMD2<Float>(kernel[i], 0)
        }
        kernelSpectrum = try fft.forward(kernelComplex)
    }

    /// Convolves `signal` with the kernel using overlap-add.
    /// Output length is `signal.count + kernelLen - 1`.
    public func apply(to signal: [Float]) throws -> [Float] {
        let outputLen = signal.count + kernelLen - 1
        var output = [Float](repeating: 0, count: outputLen)
        var block = [SIMD2<Float>](repeating: .zero, count: blockSize)
        var signalSpectrum = [SIMD2<Float>](repeating: .zero, count: blockSize)
        var product = [SIMD2<Float>](repeating: .zero, count: blockSize)

        var pos = 0
        while pos < signal.count {
            let chunkLen = min(hopSize, signal.count - pos)

            // Fill block: chunk then zeros
            for i in 0 ..< chunkLen {
                block[i] = SIMD2<Float>(signal[pos + i], 0)
            }
            for i in chunkLen ..< blockSize {
                block[i] = .zero
            }

            try block.withUnsafeBufferPointer { try fft.forward(input: $0, output: &signalSpectrum) }

            // Pointwise complex multiply: S × K
            for i in 0 ..< blockSize {
                let s = signalSpectrum[i], k = kernelSpectrum[i]
                product[i] = SIMD2<Float>(s.x * k.x - s.y * k.y,
                                          s.x * k.y + s.y * k.x)
            }

            let ifftResult = try fft.inverse(product)

            // Overlap-add real parts
            for i in 0 ..< blockSize {
                let outIdx = pos + i
                if outIdx < output.count { output[outIdx] += ifftResult[i].x }
            }

            pos += hopSize
        }

        return output
    }
}

// MARK: - Convenience

public extension Convolver {
    /// Returns the smallest supported `MetalFFT` size > `kernelCount`, or `nil` if none.
    static func recommendedFFTSize(forKernelCount kernelCount: Int) -> Int? {
        let sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        return sizes.first { $0 > kernelCount }
    }
}
