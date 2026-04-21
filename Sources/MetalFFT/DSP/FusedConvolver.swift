import Metal
import Accelerate

// MARK: - FusedConvolver

/// GPU convolution that fuses FFT → multiply → IFFT into a single Metal dispatch.
///
/// Compared to `Convolver` (which issues three separate GPU dispatches per block),
/// `FusedConvolver` keeps all intermediate data in threadgroup memory and issues
/// one dispatch per block. Device memory traffic drops from 6 transfers to 2 per block.
///
/// **Constraint**: only supports `fftSize == 4096`. For other sizes or for linear
/// convolution over longer signals, use `Convolver` (overlap-add).
///
/// Create once per filter and reuse — the kernel spectrum is pre-computed at init
/// and stored in a Metal buffer.
public final class FusedConvolver {

    /// The only FFT size supported by the fused kernel.
    public static let supportedSize = 4096

    /// Arithmetic precision of the fused FFT → multiply → IFFT pipeline.
    ///
    /// All three modes use 16 KiB of threadgroup memory (half2[4096]) versus 32 KiB
    /// for `.float32`, potentially doubling threadgroup occupancy on M1/M2.
    ///
    /// - Note: FP16 modes have a ~42 dB SQNR floor. Use `.float32` when accuracy matters.
    public enum Precision {
        /// All butterfly arithmetic in FP16.
        case float16Pure
        /// FP16 threadgroup storage, FP32 butterfly compute. Recommended FP16 mode.
        case float16Storage
        /// FP16 twiddle multiply, FP32 butterfly accumulate.
        case float16Mixed
        /// Full FP32 throughout (default).
        case float32
    }

    public let kernelLen: Int
    public let hopSize: Int
    public let precision: Precision

    private let context: MetalContext
    private let fft: MetalFFT
    private let filterBuf: MTLBuffer
    private let inBuf:     MTLBuffer
    private let outBuf:    MTLBuffer
    private let pipelineName: String

    private static let byteCount = supportedSize * MemoryLayout<SIMD2<Float>>.stride

    /// - Parameters:
    ///   - kernel: Real-valued FIR filter coefficients (time domain). Must have `count < 4096`.
    ///   - fftSize: Must be `4096`.
    ///   - precision: Arithmetic precision. Default `.float32`.
    public init(kernel: [Float], fftSize: Int = 4096, precision: Precision = .float32) throws {
        precondition(fftSize == FusedConvolver.supportedSize,
                     "FusedConvolver only supports fftSize=\(FusedConvolver.supportedSize)")
        precondition(fftSize > kernel.count, "fftSize must be > kernel.count")
        self.kernelLen = kernel.count
        self.hopSize   = fftSize - kernel.count + 1
        self.precision = precision
        self.pipelineName = switch precision {
            case .float32:       "fft_fused_convolve_4096"
            case .float16Pure:   "fft_fused_convolve_fp16_pure"
            case .float16Storage:"fft_fused_convolve_fp16_storage"
            case .float16Mixed:  "fft_fused_convolve_fp16_mixed"
        }

        let ctx = try MetalContext.shared()
        self.context = ctx
        self.fft = try MetalFFT(size: fftSize)

        var kernelComplex = [SIMD2<Float>](repeating: .zero, count: fftSize)
        for i in 0..<kernel.count { kernelComplex[i] = SIMD2<Float>(kernel[i], 0) }
        let spectrum = try fft.forward(kernelComplex)

        let fb = try makeBuffer(ctx.device, length: FusedConvolver.byteCount)
        spectrum.withUnsafeBytes { src in
            fb.contents().copyMemory(from: src.baseAddress!, byteCount: FusedConvolver.byteCount)
        }
        filterBuf = fb
        inBuf  = try makeBuffer(ctx.device, length: FusedConvolver.byteCount)
        outBuf = try makeBuffer(ctx.device, length: FusedConvolver.byteCount)
    }

    /// Convolves `signal` with the kernel using overlap-add.
    /// Each block dispatches a single fused GPU kernel (FFT → multiply → IFFT).
    /// Output length is `signal.count + kernelLen - 1`.
    public func apply(to signal: [Float]) throws -> [Float] {
        let n        = FusedConvolver.supportedSize
        let pipeline = context.pipelines[pipelineName]!
        let outputLen = signal.count + kernelLen - 1
        var output    = [Float](repeating: 0, count: outputLen)

        var pos = 0
        while pos < signal.count {
            let chunkLen = min(hopSize, signal.count - pos)

            let inPtr = inBuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: n)
            for i in 0..<chunkLen { inPtr[i] = SIMD2<Float>(signal[pos + i], 0) }
            for i in chunkLen..<n { inPtr[i] = .zero }

            guard let cb  = context.queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else {
                throw FFTError.commandBufferFailed("FusedConvolver: encoder creation failed")
            }
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(inBuf,     offset: 0, index: 0)
            enc.setBuffer(outBuf,    offset: 0, index: 1)
            enc.setBuffer(filterBuf, offset: 0, index: 2)
            enc.dispatchThreadgroups(MTLSizeMake(1, 1, 1),
                                     threadsPerThreadgroup: MTLSizeMake(1024, 1, 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let err = cb.error { throw FFTError.commandBufferFailed(err.localizedDescription) }

            let outPtr = outBuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: n)
            for i in 0..<n {
                let outIdx = pos + i
                if outIdx < output.count { output[outIdx] += outPtr[i].x }
            }
            pos += hopSize
        }

        return output
    }
}
