import Accelerate
@testable import MetalFFT
import XCTest

/// `STFT.analyze` batches its frames into one GPU dispatch. These tests hold it
/// to the answer the frame-at-a-time path gave, which is what the reference
/// inside `perFrameReference` still does.
final class STFTTests: XCTestCase {
    let sampleRate = 48000.0

    func chirp(_ count: Int) -> [Float] {
        (0 ..< count).map { n in
            let t = Float(n) / Float(sampleRate)
            return 0.4 * sinf(2 * .pi * (300 + 400 * t) * t) + 0.1 * sinf(2 * .pi * 3000 * t)
        }
    }

    /// One frame at a time, one command buffer each — the implementation the
    /// batched one replaced.
    func perFrameReference(_ signal: [Float], fftSize: Int, hopSize: Int) throws -> [[SIMD2<Float>]] {
        let fft = try MetalFFT(size: fftSize)
        let window = WindowType.hann.coefficients(fftSize)
        var windowed = [Float](repeating: 0, count: fftSize)
        var complex = [SIMD2<Float>](repeating: .zero, count: fftSize)
        var output = [SIMD2<Float>](repeating: .zero, count: fftSize)

        var frames = [[SIMD2<Float>]]()
        var position = 0
        while position + fftSize <= signal.count {
            signal.withUnsafeBufferPointer { source in
                vDSP_vmul(source.baseAddress! + position, 1, window, 1,
                          &windowed, 1, vDSP_Length(fftSize))
            }
            for i in 0 ..< fftSize { complex[i] = SIMD2<Float>(windowed[i], 0) }
            try complex.withUnsafeBufferPointer { try fft.forward(input: $0, output: &output) }
            frames.append(output)
            position += hopSize
        }
        return frames
    }

    // MARK: - Equivalence

    func testBatchedAnalysisMatchesFrameAtATime() throws {
        // 4096 spans several batch chunks; 512 and 1024 check the smaller sizes
        // and a hop that does not divide the signal evenly.
        let cases: [(fftSize: Int, hop: Int, samples: Int)] = [
            (4096, 256, 48000), (1024, 512, 20000), (512, 128, 9000), (2048, 2048, 12345),
        ]
        for spec in cases {
            let signal = chirp(spec.samples)
            let stft = try STFT(fftSize: spec.fftSize, hopSize: spec.hop, sampleRate: sampleRate)
            let frames = try stft.analyze(signal)
            let reference = try perFrameReference(signal, fftSize: spec.fftSize, hopSize: spec.hop)

            XCTAssertEqual(frames.count, reference.count, "\(spec)")
            XCTAssertEqual(frames.count, stft.frameCount(for: signal.count), "\(spec)")

            var worst: Float = 0
            for (frame, expected) in zip(frames, reference) {
                for (a, b) in zip(frame.complex, expected) {
                    worst = Swift.max(worst, Swift.max(abs(a.x - b.x), abs(a.y - b.y)))
                }
            }
            // The batched N=4096 path uses a different kernel than the single
            // one, so this is float agreement, not bit equality.
            let scale = Float(spec.fftSize)
            XCTAssertLessThan(worst, 1e-3 * scale, "\(spec): worst bin error \(worst)")
        }
    }

    func testMagnitudesSurviveTheBatching() throws {
        let stft = try STFT(fftSize: 4096, hopSize: 1024, sampleRate: sampleRate)
        let tone = (0 ..< 20000).map { 0.5 * sinf(2 * .pi * 1000 * Float($0) / Float(sampleRate)) }
        let frames = try stft.analyze(tone)
        XCTAssertGreaterThan(frames.count, 10)

        let expected = Int((1000.0 / sampleRate * 4096).rounded())
        for frame in frames {
            let peak = frame.magnitudes[0 ..< 2048].enumerated().max { $0.element < $1.element }!
            XCTAssertEqual(peak.offset, expected, "peak bin drifted to \(peak.offset)")
        }
    }

    // MARK: - Edges

    func testSignalShorterThanTheWindowGivesNoFrames() throws {
        let stft = try STFT(fftSize: 4096, hopSize: 256, sampleRate: sampleRate)
        XCTAssertEqual(try stft.analyze(chirp(4095)).count, 0)
        XCTAssertEqual(try stft.analyze([]).count, 0)
        XCTAssertEqual(try stft.analyze(chirp(4096)).count, 1)
    }

    func testAnalyzerIsReusableAcrossCalls() throws {
        let stft = try STFT(fftSize: 1024, hopSize: 256, sampleRate: sampleRate)
        let signal = chirp(8000)
        let first = try stft.analyze(signal)
        _ = try stft.analyze(chirp(30000))
        let third = try stft.analyze(signal)
        XCTAssertEqual(first.count, third.count)
        for (a, b) in zip(first, third) {
            XCTAssertEqual(a.magnitudes, b.magnitudes, "a reused STFT changed its answer")
        }
    }

    func testSpectrogramIsOneRowPerFrame() throws {
        let stft = try STFT(fftSize: 512, hopSize: 256, sampleRate: sampleRate)
        let rows = try stft.spectrogram(chirp(5000))
        XCTAssertEqual(rows.count, stft.frameCount(for: 5000))
        XCTAssertEqual(rows.first?.count, 512)
    }

    // MARK: - Cost

    func testBatchedAnalysisBeatsFrameAtATime() throws {
        let signal = chirp(48000 * 2)
        let stft = try STFT(fftSize: 4096, hopSize: 256, sampleRate: sampleRate)
        _ = try stft.analyze(signal)

        var start = Date()
        let frames = try stft.analyze(signal)
        let batched = Date().timeIntervalSince(start)

        start = Date()
        // Magnitudes too, so the only difference measured is the dispatch.
        _ = try perFrameReference(signal, fftSize: 4096, hopSize: 256).map { Spectrum.magnitudes($0) }
        let perFrame = Date().timeIntervalSince(start)

        print(String(format: "STFT %d frames: batched %.1f ms, per-frame %.1f ms (%.1fx)",
                     frames.count, batched * 1000, perFrame * 1000, perFrame / batched))
        XCTAssertLessThan(batched, perFrame)
    }
}
