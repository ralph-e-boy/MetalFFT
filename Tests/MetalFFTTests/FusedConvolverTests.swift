import XCTest
import Accelerate
@testable import MetalFFT

final class FusedConvolverTests: XCTestCase {

    private let fftSize = 4096

    // MARK: - FP32 correctness

    func testDeltaFilterIsIdentity() throws {
        var delta = [Float](repeating: 0, count: 64)
        delta[0] = 1.0
        let conv = try FusedConvolver(kernel: delta, fftSize: fftSize)

        let signal = randomReal(n: conv.hopSize)
        let output = try conv.apply(to: signal)

        // Output should equal input (up to output length = signal + kernelLen - 1)
        let l2 = l2RelativeError(Array(output.prefix(signal.count)), signal)
        XCTAssertLessThan(l2, 1e-4, "Delta filter L2=\(l2)")
    }

    func testMatchesConvolver() throws {
        let kernel = randomReal(n: 128)
        let signal = randomReal(n: fftSize * 2)

        let fused   = try FusedConvolver(kernel: kernel, fftSize: fftSize)
        let unfused = try Convolver(kernel: kernel, fftSize: fftSize)

        let fusedOut   = try fused.apply(to: signal)
        let unfusedOut = try unfused.apply(to: signal)

        let l2 = l2RelativeError(fusedOut, unfusedOut)
        XCTAssertLessThan(l2, 1e-4, "Fused vs Convolver L2=\(l2)")
    }

    // MARK: - FP16 modes vs FP32

    func testFP16PureMatchesFP32() throws {
        try assertFP16ModeAcceptable(.float16Pure, threshold: 2e-2)
    }

    func testFP16StorageMatchesFP32() throws {
        try assertFP16ModeAcceptable(.float16Storage, threshold: 1e-3)
    }

    func testFP16MixedMatchesFP32() throws {
        try assertFP16ModeAcceptable(.float16Mixed, threshold: 2e-2)
    }

    private func assertFP16ModeAcceptable(_ mode: FusedConvolver.Precision, threshold: Float) throws {
        let kernel = randomReal(n: 64)
        let signal = randomReal(n: fftSize + 500)

        let fp32 = try FusedConvolver(kernel: kernel, fftSize: fftSize, precision: .float32)
        let fp16 = try FusedConvolver(kernel: kernel, fftSize: fftSize, precision: mode)

        let ref = try fp32.apply(to: signal)
        let out = try fp16.apply(to: signal)

        let l2 = l2RelativeError(out, ref)
        XCTAssertLessThan(l2, threshold, "\(mode) vs float32 L2=\(l2)")
    }

    // MARK: - Helpers

    private func randomReal(n: Int) -> [Float] {
        (0..<n).map { _ in Float.random(in: -1...1) }
    }

    private func l2RelativeError(_ a: [Float], _ b: [Float]) -> Float {
        let len = min(a.count, b.count)
        var num: Float = 0, den: Float = 0
        vDSP_distancesq(a, 1, b, 1, &num, vDSP_Length(len))
        vDSP_svesq(b, 1, &den, vDSP_Length(len))
        return den > 0 ? sqrt(num / den) : sqrt(num)
    }
}
