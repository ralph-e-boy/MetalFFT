import Metal
import Foundation

// MARK: - FFTError

public enum FFTError: Error, CustomStringConvertible {
    case noMetalDevice
    case noCommandQueue
    case libraryBuildFailed(Error)
    case kernelNotFound(String)
    case commandBufferFailed(String)
    case invalidInputSize(expected: Int, got: Int)
    case batchInputSize(expected: Int, got: Int, batchIndex: Int)
    case unsupportedFFTSize(Int)
    case bufferAllocationFailed

    public var description: String {
        switch self {
        case .noMetalDevice:                     return "No Metal device available"
        case .noCommandQueue:                    return "Could not create Metal command queue"
        case .libraryBuildFailed(let e):         return "Metal library build failed: \(e)"
        case .kernelNotFound(let name):          return "Metal kernel not found: \(name)"
        case .commandBufferFailed(let msg):      return "Command buffer error: \(msg)"
        case .invalidInputSize(let e, let g):    return "Input size mismatch: expected \(e), got \(g)"
        case .batchInputSize(let e, let g, let i): return "Batch element \(i): expected \(e), got \(g)"
        case .unsupportedFFTSize(let n):         return "Unsupported FFT size: \(n). Supported: 64,128,256,512,1024,2048,4096,8192,16384"
        case .bufferAllocationFailed:            return "Metal buffer allocation failed"
        }
    }
}

// MARK: - FFTDescriptor

struct FFTDescriptor {
    enum Kind {
        case singlePass(kernel: String, threadsPerGroup: Int)
        case fourStep(
            n1: Int, n2: Int,
            pass1Kernel: String, pass1Threads: Int,
            pass2Kernel: String, pass2Threads: Int
        )
    }

    let size: Int
    let kind: Kind

    init(size: Int) throws {
        self.size = size
        switch size {
        case 64:    kind = .singlePass(kernel: "fft_64_stockham",   threadsPerGroup: 16)
        case 128:   kind = .singlePass(kernel: "fft_128_stockham",  threadsPerGroup: 32)
        case 256:   kind = .singlePass(kernel: "fft_256_stockham",  threadsPerGroup: 64)
        case 512:   kind = .singlePass(kernel: "fft_512_stockham",  threadsPerGroup: 128)
        case 1024:  kind = .singlePass(kernel: "fft_1024_stockham", threadsPerGroup: 256)
        case 2048:  kind = .singlePass(kernel: "fft_2048_stockham", threadsPerGroup: 512)
        case 4096:  kind = .singlePass(kernel: "fft_4096_stockham", threadsPerGroup: 1024)
        case 8192:  kind = .fourStep(
            n1: 64,  n2: 128,
            pass1Kernel: "fft_128_stockham", pass1Threads: 32,
            pass2Kernel: "fft_64_stockham",  pass2Threads: 16
        )
        case 16384: kind = .fourStep(
            n1: 128, n2: 128,
            pass1Kernel: "fft_128_stockham", pass1Threads: 32,
            pass2Kernel: "fft_128_stockham", pass2Threads: 32
        )
        default: throw FFTError.unsupportedFFTSize(size)
        }
    }
}

// MARK: - MetalContext

final class MetalContext {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipelines: [String: MTLComputePipelineState]

    private static let lock = NSLock()
    private static var _shared: MetalContext?

    static func shared() throws -> MetalContext {
        lock.lock(); defer { lock.unlock() }
        if let existing = _shared { return existing }
        let ctx = try MetalContext()
        _shared = ctx
        return ctx
    }

    private init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw FFTError.noMetalDevice }
        guard let q = dev.makeCommandQueue() else { throw FFTError.noCommandQueue }
        device = dev
        queue = q

        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        // Each entry: (resource name, [kernel functions it contains])
        let sources: [(resource: String, kernels: [String])] = [
            ("fft_multisize", [
                "fft_64_stockham", "fft_128_stockham",
                "fft_256_stockham", "fft_512_stockham", "fft_1024_stockham",
                "fft_2048_stockham", "fft_4096_stockham",
                "fft_twiddle_transpose", "fft_transpose",
            ]),
            ("fft_4096_batched",    ["fft_4096_batched"]),
            ("fft_fused_convolve",  ["fft_fused_convolve_4096"]),
            ("fft_cross_spectral",  ["fft_cross_spectral"]),
            ("fft_fused_convolve_fp16", [
                "fft_fused_convolve_fp16_pure",
                "fft_fused_convolve_fp16_storage",
                "fft_fused_convolve_fp16_mixed",
            ]),
        ]

        var ps: [String: MTLComputePipelineState] = [:]
        for (resource, kernels) in sources {
            guard let url = Bundle.module.url(forResource: resource, withExtension: "metal") else {
                throw FFTError.libraryBuildFailed(
                    NSError(domain: "MetalFFT", code: 1,
                            userInfo: [NSLocalizedDescriptionKey: "\(resource).metal not found in bundle"])
                )
            }
            let source: String
            do { source = try String(contentsOf: url, encoding: .utf8) }
            catch { throw FFTError.libraryBuildFailed(error) }

            let library: MTLLibrary
            do { library = try dev.makeLibrary(source: source, options: options) }
            catch { throw FFTError.libraryBuildFailed(error) }

            for name in kernels {
                guard let fn = library.makeFunction(name: name) else {
                    throw FFTError.kernelNotFound(name)
                }
                ps[name] = try dev.makeComputePipelineState(function: fn)
            }
        }
        pipelines = ps
    }
}
