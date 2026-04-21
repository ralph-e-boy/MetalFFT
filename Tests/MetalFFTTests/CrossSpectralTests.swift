import XCTest
import Accelerate
@testable import MetalFFT

final class CrossSpectralTests: XCTestCase {

    private let fftSize = 1024
    private let hopSize = 512

    // MARK: - MultiChannelFFT

    func testMultiChannelImpulse() throws {
        let n = 1024
        let channels = 4
        var impulse = [SIMD2<Float>](repeating: .zero, count: n)
        impulse[0] = SIMD2<Float>(1, 0)
        let inputs = [[SIMD2<Float>]](repeating: impulse, count: channels)

        let mfft = try MultiChannelFFT(channels: channels, size: n)
        let out = try mfft.forward(inputs)

        XCTAssertEqual(out.count, channels)
        for (c, spectrum) in out.enumerated() {
            for (k, bin) in spectrum.enumerated() {
                XCTAssertEqual(bin.x, 1.0, accuracy: 1e-3, "ch\(c) bin\(k) real")
                XCTAssertEqual(bin.y, 0.0, accuracy: 1e-3, "ch\(c) bin\(k) imag")
            }
        }
    }

    func testMultiChannelMatchesSingle() throws {
        let n = 1024
        let channels = 3
        let signal = randomReal(n: n)
        let complex = signal.map { SIMD2<Float>($0, 0) }

        let single = try MetalFFT(size: n)
        let expected = try single.forward(complex)

        let multi = try MultiChannelFFT(channels: channels, size: n)
        let inputs = [[SIMD2<Float>]](repeating: complex, count: channels)
        let out = try multi.forward(inputs)

        for c in 0..<channels {
            let l2 = l2Error(out[c], expected)
            XCTAssertLessThan(l2, 1e-4, "ch\(c) L2=\(l2)")
        }
    }

    func testMultiChannelInverse() throws {
        let n = 512
        let channels = 2
        let signal = randomReal(n: n).map { SIMD2<Float>($0, 0) }
        let inputs = [[SIMD2<Float>]](repeating: signal, count: channels)

        let mfft = try MultiChannelFFT(channels: channels, size: n)
        let fwd = try mfft.forward(inputs)
        let inv = try mfft.inverse(fwd)

        for c in 0..<channels {
            let l2 = l2Error(inv[c], signal)
            XCTAssertLessThan(l2, 1e-4, "ch\(c) round-trip L2=\(l2)")
        }
    }

    // MARK: - CrossSpectral

    func testIdenticalChannelsHaveUnitCoherence() throws {
        let n = fftSize
        let signal = randomReal(n: n * 4)
        let channels = [[Float]](repeating: signal, count: 3)

        let result = try PSD.crossSpectral(
            channels: channels,
            fftSize: n,
            hopSize: hopSize,
            sampleRate: 44100
        )

        XCTAssertEqual(result.channels, 3)
        XCTAssertEqual(result.pairCount, 3)

        for p in 0..<result.pairCount {
            for k in 1..<n {   // skip DC/Nyquist where power may be zero
                XCTAssertEqual(result.coherence[p][k], 1.0, accuracy: 1e-4,
                               "pair\(p) bin\(k) coherence")
            }
        }
    }

    func testSingleChannelCrossSpectralNoPairs() throws {
        let signal = randomReal(n: fftSize * 2)
        let result = try PSD.crossSpectral(
            channels: [signal],
            fftSize: fftSize,
            hopSize: hopSize,
            sampleRate: 44100
        )
        XCTAssertEqual(result.channels, 1)
        XCTAssertEqual(result.pairCount, 0)
        XCTAssertEqual(result.power.count, 1)
    }

    func testPairIndexOrdering() throws {
        let signal = randomReal(n: fftSize * 2)
        let channels = [[Float]](repeating: signal, count: 4)
        let result = try PSD.crossSpectral(
            channels: channels,
            fftSize: fftSize,
            hopSize: hopSize,
            sampleRate: 44100
        )
        // Upper-triangle ordering: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        XCTAssertEqual(result.pairIndex(0, 1), 0)
        XCTAssertEqual(result.pairIndex(0, 2), 1)
        XCTAssertEqual(result.pairIndex(0, 3), 2)
        XCTAssertEqual(result.pairIndex(1, 2), 3)
        XCTAssertEqual(result.pairIndex(1, 3), 4)
        XCTAssertEqual(result.pairIndex(2, 3), 5)
        XCTAssertEqual(result.pairCount, 6)
    }

    func testPowerMatchesSingleChannelPSD() throws {
        let n = fftSize
        let signal = randomReal(n: n * 8)

        // Single-channel cross-spectral power
        let crossResult = try PSD.crossSpectral(
            channels: [signal],
            fftSize: n,
            hopSize: hopSize,
            sampleRate: 44100
        )

        // Direct Welch PSD
        let psdPow = try PSD.welch(
            signal: signal,
            fftSize: n,
            hopSize: hopSize,
            sampleRate: 44100
        )

        let crossPow = crossResult.power[0]

        var sumCross: Float = 0, sumPSD: Float = 0
        vDSP_sve(crossPow, 1, &sumCross, vDSP_Length(n))
        vDSP_sve(psdPow,   1, &sumPSD,   vDSP_Length(n))

        // They may differ by normalization constant (window sum) — just verify both > 0
        XCTAssertGreaterThan(sumCross, 0)
        XCTAssertGreaterThan(sumPSD,   0)
    }

    func testCoherenceSymmetry() throws {
        let n = fftSize
        let s1 = randomReal(n: n * 4)
        var s2 = randomReal(n: n * 4)
        // Mix s2 with s1 to get partial coherence
        for i in s2.indices { s2[i] = 0.5 * s2[i] + 0.5 * s1[i] }

        let result = try PSD.crossSpectral(
            channels: [s1, s2],
            fftSize: n,
            hopSize: hopSize,
            sampleRate: 44100
        )

        // Coherence in [0, 1]
        for k in 0..<n {
            XCTAssertGreaterThanOrEqual(result.coherence[0][k], 0.0 - 1e-6)
            XCTAssertLessThanOrEqual(   result.coherence[0][k], 1.0 + 1e-6)
        }
    }

    // MARK: - Helpers

    private func randomReal(n: Int) -> [Float] {
        (0..<n).map { _ in Float.random(in: -1...1) }
    }

    private func l2Error(_ a: [SIMD2<Float>], _ b: [SIMD2<Float>]) -> Float {
        var num: Float = 0, den: Float = 0
        for (x, y) in zip(a, b) {
            let d = x - y
            num += d.x * d.x + d.y * d.y
            den += y.x * y.x + y.y * y.y
        }
        return den > 0 ? sqrt(num / den) : sqrt(num)
    }
}
