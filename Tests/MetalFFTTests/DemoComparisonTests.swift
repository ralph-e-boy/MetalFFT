import XCTest
import Metal
import Accelerate
@testable import MetalFFT

/// Compares MetalFFT package output directly against the standalone demo kernels in src/.
/// Each demo kernel is loaded from source and dispatched with the same parameters the
/// demo host programs use, so divergence here means the package and the paper kernels disagree.
final class DemoComparisonTests: XCTestCase {

    // L2 threshold matches the existing package tests.
    private static let l2Threshold: Float = 1e-4
    private static let n = 4096

    // Locate src/ relative to this source file at compile time.
    private static let srcDir: URL = URL(fileURLWithPath: #file)
        .deletingLastPathComponent()   // Tests/MetalFFTTests/
        .deletingLastPathComponent()   // Tests/
        .deletingLastPathComponent()   // package root
        .appendingPathComponent("src")

    // MARK: - Demo vs Package

    func testPackageVsStockhamDemo() throws {
        let input = randomInput(n: Self.n)
        let packageOut = try packageFFT(input)
        let demoOut = try demoKernel(
            source: Self.srcDir.appendingPathComponent("fft_stockham_4096.metal"),
            function: "fft_4096_stockham",
            threads: 1024,
            input: input
        )
        let l2 = l2RelativeError(packageOut, demoOut)
        XCTAssertLessThan(l2, Self.l2Threshold,
            "Package vs Stockham demo L2=\(l2) exceeds threshold")
    }

    func testPackageVsRadix8Demo() throws {
        let input = randomInput(n: Self.n)
        let packageOut = try packageFFT(input)
        let demoOut = try demoKernel(
            source: Self.srcDir.appendingPathComponent("fft_4096_radix8.metal"),
            function: "fft_4096_mma",
            threads: 512,
            input: input
        )
        let l2 = l2RelativeError(packageOut, demoOut)
        XCTAssertLessThan(l2, Self.l2Threshold,
            "Package vs Radix-8 demo L2=\(l2) exceeds threshold")
    }

    func testPackageVsCTMMADemo() throws {
        let input = randomInput(n: Self.n)
        let packageOut = try packageFFT(input)
        let demoOut = try ctmmaKernel(
            source: Self.srcDir.appendingPathComponent("fft_4096_ct_mma.metal"),
            input: input
        )
        let l2 = l2RelativeError(packageOut, demoOut)
        XCTAssertLessThan(l2, Self.l2Threshold,
            "Package vs CT MMA demo L2=\(l2) exceeds threshold")
    }

    // All three demo kernels must agree with each other on the same input.
    func testDemoKernelsAgreeWithEachOther() throws {
        let input = randomInput(n: Self.n)
        let stockham = try demoKernel(
            source: Self.srcDir.appendingPathComponent("fft_stockham_4096.metal"),
            function: "fft_4096_stockham",
            threads: 1024,
            input: input
        )
        let radix8 = try demoKernel(
            source: Self.srcDir.appendingPathComponent("fft_4096_radix8.metal"),
            function: "fft_4096_mma",
            threads: 512,
            input: input
        )
        let ctmma = try ctmmaKernel(
            source: Self.srcDir.appendingPathComponent("fft_4096_ct_mma.metal"),
            input: input
        )
        XCTAssertLessThan(l2RelativeError(stockham, radix8), Self.l2Threshold,
            "Stockham vs Radix-8 demo kernels disagree")
        XCTAssertLessThan(l2RelativeError(stockham, ctmma), Self.l2Threshold,
            "Stockham vs CT MMA demo kernels disagree")
    }

    // Impulse response must be all-ones for every demo kernel.
    func testDemoKernelsImpulse() throws {
        var impulse = [SIMD2<Float>](repeating: .zero, count: Self.n)
        impulse[0] = SIMD2<Float>(1, 0)

        let stockham = try demoKernel(
            source: Self.srcDir.appendingPathComponent("fft_stockham_4096.metal"),
            function: "fft_4096_stockham", threads: 1024, input: impulse)
        let radix8 = try demoKernel(
            source: Self.srcDir.appendingPathComponent("fft_4096_radix8.metal"),
            function: "fft_4096_mma", threads: 512, input: impulse)
        let ctmma = try ctmmaKernel(
            source: Self.srcDir.appendingPathComponent("fft_4096_ct_mma.metal"),
            input: impulse)

        for (fn, out) in [("fft_4096_stockham", stockham), ("fft_4096_mma", radix8), ("fft_4096_ct_mma", ctmma)] {
            for bin in out {
                XCTAssertEqual(bin.x, 1.0, accuracy: 1e-3, "\(fn): impulse real part")
                XCTAssertEqual(bin.y, 0.0, accuracy: 1e-3, "\(fn): impulse imag part")
            }
        }
    }

    // MARK: - Helpers

    private func packageFFT(_ input: [SIMD2<Float>]) throws -> [SIMD2<Float>] {
        try MetalFFT(size: Self.n).forward(input)
    }

    /// Loads a Metal source file, compiles it, dispatches 1 threadgroup, returns output.
    private func demoKernel(
        source: URL,
        function: String,
        threads: Int,
        input: [SIMD2<Float>]
    ) throws -> [SIMD2<Float>] {
        let n = input.count
        let byteCount = n * MemoryLayout<SIMD2<Float>>.stride

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal device available"); return []
        }
        let metalSource = try String(contentsOf: source, encoding: .utf8)
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library  = try device.makeLibrary(source: metalSource, options: options)
        guard let fn = library.makeFunction(name: function) else {
            XCTFail("Kernel function '\(function)' not found in \(source.lastPathComponent)")
            return []
        }
        let pipeline = try device.makeComputePipelineState(function: fn)
        guard let queue = device.makeCommandQueue() else {
            XCTFail("makeCommandQueue failed"); return []
        }
        let inBuf = device.makeBuffer(bytes: input, length: byteCount, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            XCTFail("Failed to create command buffer/encoder"); return []
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inBuf,  offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.dispatchThreadgroups(MTLSizeMake(1, 1, 1),
                                 threadsPerThreadgroup: MTLSizeMake(threads, 1, 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        if let err = cb.error {
            XCTFail("GPU error in \(function): \(err)"); return []
        }
        let ptr = outBuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    /// Dispatches the CT MMA kernel, which requires four additional precomputed constant buffers
    /// beyond the standard input/output pair (DFT matrices at indices 2–4, twiddle table at 5).
    private func ctmmaKernel(source: URL, input: [SIMD2<Float>]) throws -> [SIMD2<Float>] {
        let n = input.count
        let byteCount = n * MemoryLayout<SIMD2<Float>>.stride

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal device available"); return []
        }
        let metalSource = try String(contentsOf: source, encoding: .utf8)
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: metalSource, options: options)
        guard let fn = library.makeFunction(name: "fft_4096_ct_mma") else {
            XCTFail("Kernel function 'fft_4096_ct_mma' not found"); return []
        }
        let pipeline = try device.makeComputePipelineState(function: fn)
        guard let queue = device.makeCommandQueue() else {
            XCTFail("makeCommandQueue failed"); return []
        }

        // DFT_8 matrices (row-major 8×8)
        let s: Float = 0.70710678118654752
        let f8Real: [Float] = [
             1,  1,   1,  1,   1,  1,   1,  1,
             1,  s,   0, -s,  -1, -s,   0,  s,
             1,  0,  -1,  0,   1,  0,  -1,  0,
             1, -s,   0,  s,  -1,  s,   0, -s,
             1, -1,   1, -1,   1, -1,   1, -1,
             1, -s,   0,  s,  -1,  s,   0, -s,
             1,  0,  -1,  0,   1,  0,  -1,  0,
             1,  s,   0, -s,  -1, -s,   0,  s,
        ]
        let f8Imag: [Float] = [
             0,  0,   0,  0,   0,  0,   0,  0,
             0, -s,  -1, -s,   0,  s,   1,  s,
             0, -1,   0,  1,   0, -1,   0,  1,
             0, -s,   1, -s,   0,  s,  -1,  s,
             0,  0,   0,  0,   0,  0,   0,  0,
             0,  s,  -1,  s,   0, -s,   1, -s,
             0,  1,   0, -1,   0,  1,   0, -1,
             0,  s,   1,  s,   0, -s,  -1, -s,
        ]
        let f8NegImag = f8Imag.map { -$0 }

        let floatStride = MemoryLayout<Float>.stride
        let dftRealBuf    = device.makeBuffer(bytes: f8Real,    length: 64 * floatStride, options: .storageModeShared)!
        let dftImagBuf    = device.makeBuffer(bytes: f8Imag,    length: 64 * floatStride, options: .storageModeShared)!
        let dftNegImagBuf = device.makeBuffer(bytes: f8NegImag, length: 64 * floatStride, options: .storageModeShared)!

        // Twiddle factors for 3 CT stages
        let strides    = [512, 64, 8]
        let groupSizes = [4096, 512, 64]
        var twiddleTable = [SIMD2<Float>](repeating: SIMD2<Float>(1, 0), count: 3 * n)
        for stage in 0..<3 {
            let S = strides[stage]
            let G = groupSizes[stage]
            for i in 0..<n {
                let posInGroup = i % G
                let r = posInGroup / S
                let k = posInGroup % S
                guard r > 0 && k > 0 else { continue }
                let angle = -2.0 * Float.pi * Float(r * k) / Float(G)
                twiddleTable[stage * n + i] = SIMD2<Float>(cos(angle), sin(angle))
            }
        }
        let twiddleBuf = device.makeBuffer(
            bytes: twiddleTable,
            length: 3 * n * MemoryLayout<SIMD2<Float>>.stride,
            options: .storageModeShared)!

        let inBuf  = device.makeBuffer(bytes: input, length: byteCount, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            XCTFail("Failed to create command buffer/encoder"); return []
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inBuf,         offset: 0, index: 0)
        enc.setBuffer(outBuf,        offset: 0, index: 1)
        enc.setBuffer(dftRealBuf,    offset: 0, index: 2)
        enc.setBuffer(dftImagBuf,    offset: 0, index: 3)
        enc.setBuffer(dftNegImagBuf, offset: 0, index: 4)
        enc.setBuffer(twiddleBuf,    offset: 0, index: 5)
        enc.dispatchThreadgroups(MTLSizeMake(1, 1, 1),
                                 threadsPerThreadgroup: MTLSizeMake(512, 1, 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        if let err = cb.error {
            XCTFail("GPU error in fft_4096_ct_mma: \(err)"); return []
        }
        let ptr = outBuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    private func randomInput(n: Int) -> [SIMD2<Float>] {
        (0..<n).map { _ in SIMD2<Float>(Float.random(in: -1...1), Float.random(in: -1...1)) }
    }

    private func l2RelativeError(_ a: [SIMD2<Float>], _ b: [SIMD2<Float>]) -> Float {
        var errSq: Float = 0, refSq: Float = 0
        for i in 0..<a.count {
            let d = a[i] - b[i]
            errSq += d.x * d.x + d.y * d.y
            refSq += b[i].x * b[i].x + b[i].y * b[i].y
        }
        return refSq > 0 ? sqrt(errSq / refSq) : sqrt(errSq)
    }
}
