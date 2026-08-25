import Accelerate
@testable import MetalFFT
import XCTest

/// What a 4096-point FFT actually costs by the three routes available: one GPU
/// round trip per frame, one round trip for a whole batch, and vDSP on the CPU.
/// The library's speed claims are per-dispatch; this is the per-frame number an
/// STFT at a 256-sample hop actually sees.
final class DispatchOverheadTests: XCTestCase {
    let size = 4096
    let frames = 200

    func signal() -> [SIMD2<Float>] {
        (0 ..< size).map { SIMD2<Float>(sinf(2 * .pi * 440 * Float($0) / 48000), 0) }
    }

    func time(_ label: String, _ count: Int, _ body: () throws -> Void) rethrows -> Double {
        _ = try? body()
        let start = Date()
        try body()
        let each = Date().timeIntervalSince(start) / Double(count) * 1e6
        print(String(format: "%@: %.1f us/frame", label, each))
        return each
    }

    func testDispatchOverheadDominatesPerFrameFFT() throws {
        let fft = try MetalFFT(size: size)
        let one = signal()
        let many = [[SIMD2<Float>]](repeating: one, count: frames)

        let perFrame = try time("GPU, one command buffer per frame", frames) {
            for _ in 0 ..< frames { _ = try fft.forward(one) }
        }
        let batched = try time("GPU, one command buffer for \(frames)", frames) {
            _ = try fft.forward(batch: many)
        }

        var setup: vDSP_DFT_Setup?
        setup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(size), .FORWARD)
        defer { if let setup { vDSP_DFT_DestroySetup(setup) } }
        var realIn = one.map(\.x), imagIn = one.map(\.y)
        var realOut = [Float](repeating: 0, count: size), imagOut = [Float](repeating: 0, count: size)
        let cpu = time("CPU, vDSP_DFT_zop", frames) {
            for _ in 0 ..< frames {
                vDSP_DFT_Execute(setup!, &realIn, &imagIn, &realOut, &imagOut)
            }
        }

        print(String(format: "batching wins %.1fx; CPU is %.1fx the per-frame GPU path",
                     perFrame / batched, perFrame / cpu))
        XCTAssertLessThan(batched, perFrame, "batching must beat per-frame dispatch")
    }
}
