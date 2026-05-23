import Accelerate
@testable import MetalFFT
import XCTest

final class MetalFFTTests: XCTestCase {
    // MARK: - FFT Accuracy

    static let allSizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    static let l2Threshold = 1e-4 as Float

    func testAllSizesImpulse() throws {
        for n in Self.allSizes {
            var impulse = [SIMD2<Float>](repeating: .zero, count: n)
            impulse[0] = SIMD2<Float>(1, 0)
            let fft = try MetalFFT(size: n)
            let out = try fft.forward(impulse)
            // DFT of impulse = all-ones
            for bin in out {
                XCTAssertEqual(bin.x, 1.0, accuracy: 1e-3, "Impulse real part at N=\(n)")
                XCTAssertEqual(bin.y, 0.0, accuracy: 1e-3, "Impulse imag part at N=\(n)")
            }
        }
    }

    func testAllSizesRandomVsVDSP() throws {
        for n in Self.allSizes {
            let signal = randomSignal(n: n)
            let fft = try MetalFFT(size: n)
            let metalOut = try fft.forward(signal)
            let vdspOut = vdspFFT(input: signal, n: n)
            let l2 = l2RelativeError(metalOut, vdspOut)
            XCTAssertLessThan(l2, Self.l2Threshold,
                              "L2 error \(l2) exceeds threshold for N=\(n)")
        }
    }

    func testSingleSinusoidPeakBin() throws {
        let n = 1024
        let freq = 100.0 // bin index 100
        let signal: [SIMD2<Float>] = (0 ..< n).map { k in
            let phase = 2.0 * Double.pi * freq * Double(k) / Double(n)
            return SIMD2<Float>(Float(cos(phase)), 0)
        }
        let fft = try MetalFFT(size: n)
        let out = try fft.forward(signal)
        let mags = Spectrum.magnitudes(out, count: n / 2)
        let peakBin = try XCTUnwrap(mags.indices.max(by: { mags[$0] < mags[$1] }))
        XCTAssertEqual(peakBin, 100, "Peak should be at bin 100")
    }

    func testBatchMatchesSingle() throws {
        let n = 512
        let a = randomSignal(n: n)
        let b = randomSignal(n: n)
        let fft = try MetalFFT(size: n)

        let batchOut = try fft.forward(batch: [a, b])
        let singleA = try fft.forward(a)
        let singleB = try fft.forward(b)

        XCTAssertLessThan(l2RelativeError(batchOut[0], singleA), Self.l2Threshold,
                          "Batch[0] diverges from single FFT")
        XCTAssertLessThan(l2RelativeError(batchOut[1], singleB), Self.l2Threshold,
                          "Batch[1] diverges from single FFT")
    }

    // MARK: - Window

    func testHannWindowEndsAtZero() {
        let w = Window.hann(1024)
        XCTAssertEqual(w.count, 1024)
        XCTAssertEqual(w[0], 0, accuracy: 1e-6)
    }

    func testWindowApply() {
        let n = 256
        let w = Window.hann(n)
        let ones = [Float](repeating: 1, count: n)
        let out = Window.apply(w, to: ones)
        for i in 0 ..< n {
            XCTAssertEqual(out[i], w[i], accuracy: 1e-6)
        }
    }

    // MARK: - Spectrum

    func testRMS() {
        let ones = [Float](repeating: 1, count: 1024)
        XCTAssertEqual(Spectrum.rms(ones), 1.0, accuracy: 1e-5)
        let zeros = [Float](repeating: 0, count: 1024)
        XCTAssertEqual(Spectrum.rms(zeros), 0.0, accuracy: 1e-5)
    }

    func testNormalize() throws {
        var mags: [Float] = [1, 2, 4, 3]
        Spectrum.normalize(&mags)
        XCTAssertEqual(try XCTUnwrap(mags.max()), 1.0, accuracy: 1e-6)
    }

    // MARK: - PeakDetection

    func testParabolicInterpolation() {
        // Symmetric parabola: peak at bin 50, no offset expected
        var mags = [Float](repeating: 0, count: 100)
        mags[49] = 0.5; mags[50] = 1.0; mags[51] = 0.5
        let freq = PeakDetection.parabolicInterpolation(magnitudes: mags, peakIndex: 50,
                                                        sampleRate: 44100, fftSize: 200)
        XCTAssertEqual(freq, 50.0 * 44100.0 / 200.0, accuracy: 0.01)
    }

    // MARK: - Pitch

    func testA4() {
        let n = Pitch.note(frequency: 440.0)
        XCTAssertEqual(n?.name, "A")
        XCTAssertEqual(n?.octave, 4)
    }

    func testC4() {
        let n = Pitch.note(frequency: 261.63)
        XCTAssertEqual(n?.name, "C")
        XCTAssertEqual(n?.octave, 4)
    }

    func testCentsDeviationInTune() throws {
        let cents = try XCTUnwrap(Pitch.centsDeviation(frequency: 440.0))
        XCTAssertEqual(cents, 0.0, accuracy: 0.01)
    }

    func testOutOfRangeReturnsNil() {
        XCTAssertNil(Pitch.note(frequency: 0))
        XCTAssertNil(Pitch.note(frequency: -1))
        XCTAssertNil(Pitch.note(frequency: 30000))
    }

    // MARK: - FrequencyTracker

    func testSmoothing() {
        let tracker = FrequencyTracker(smoothingWindow: 4)
        _ = tracker.track(440)
        _ = tracker.track(440)
        _ = tracker.track(440)
        let out = tracker.track(440)
        XCTAssertEqual(out, 440, accuracy: 0.1)
    }

    func testReset() {
        let tracker = FrequencyTracker(smoothingWindow: 4)
        _ = tracker.track(440)
        _ = tracker.track(440)
        tracker.reset()
        let out = tracker.track(880)
        XCTAssertEqual(out, 880, accuracy: 0.1, "After reset, first reading is unsmoothed")
    }

    // MARK: - Real-input Forward FFT

    func testForwardRealMatchesPackedComplex() throws {
        for n in [64, 256, 1024, 4096] {
            let realSamples = (0 ..< n).map { _ in Float.random(in: -1 ... 1) }
            let packed = realSamples.map { SIMD2<Float>($0, 0) }

            let fft = try MetalFFT(size: n)
            let viaComplex = try fft.forward(packed)
            let viaReal = try fft.forward(real: realSamples)

            let l2 = l2RelativeError(viaReal, viaComplex)
            XCTAssertLessThan(l2, Self.l2Threshold,
                              "forward(real:) diverges from packed-complex forward at N=\(n)")
        }
    }

    /// Regression: interleaving a complex `forward(input:output:)` call with a real
    /// `forward(real:output:)` call must produce a correct real-input result.
    /// (The complex path can leave non-zero imag values in the internal buffer; the
    /// real path must reset them every call, not rely on init-time zeroing.)
    func testForwardRealAfterComplexResetsImagSlots() throws {
        let n = 1024
        let fft = try MetalFFT(size: n)

        // First: a complex call with non-zero imag everywhere. Leaves imag slots dirty.
        let noisy = (0 ..< n).map { _ in
            SIMD2<Float>(Float.random(in: -1 ... 1), Float.random(in: -1 ... 1))
        }
        _ = try fft.forward(noisy)

        // Then: a real call. Must produce the same output as if no prior call ran.
        let realSamples = (0 ..< n).map { _ in Float.random(in: -1 ... 1) }
        var afterComplex = [SIMD2<Float>](repeating: .zero, count: n)
        try realSamples.withUnsafeBufferPointer {
            try fft.forward(real: $0, output: &afterComplex)
        }

        // Reference: fresh instance, same real input.
        let freshFFT = try MetalFFT(size: n)
        let reference = try freshFFT.forward(real: realSamples)

        let l2 = l2RelativeError(afterComplex, reference)
        XCTAssertLessThan(l2, Self.l2Threshold,
                          "forward(real:) is sensitive to imag state from prior complex call")
    }

    // MARK: - FFTAnalyzer zero-copy variants

    func testAnalyzeIntoMagnitudesMatchesAllocating() throws {
        let n = 2048
        let samples = (0 ..< n).map { k -> Float in
            Float(sin(2.0 * .pi * 50.0 * Double(k) / Double(n)))
        }
        let analyzer = try FFTAnalyzer(size: n, sampleRate: 44100, window: .hann)

        let resultAllocating = try analyzer.analyze(samples)
        let referenceMags = resultAllocating.magnitudes

        var zeroCopyMags = [Float](repeating: 0, count: n)
        try samples.withUnsafeBufferPointer { src in
            try zeroCopyMags.withUnsafeMutableBufferPointer { dst in
                try analyzer.analyze(samples: src, intoMagnitudes: dst, binCount: n)
            }
        }

        for i in 0 ..< n {
            XCTAssertEqual(zeroCopyMags[i], referenceMags[i], accuracy: 1e-4,
                           "intoMagnitudes bin \(i) diverges from result.magnitudes")
        }
    }

    func testAnalyzeIntoMagnitudesDBMatchesAllocating() throws {
        let n = 1024
        let samples = (0 ..< n).map { _ in Float.random(in: -0.5 ... 0.5) }
        let analyzer = try FFTAnalyzer(size: n, sampleRate: 44100, window: .blackman)

        let resultAllocating = try analyzer.analyze(samples)
        let referenceDB = resultAllocating.magnitudesDB

        var zeroCopyDB = [Float](repeating: 0, count: n)
        try samples.withUnsafeBufferPointer { src in
            try zeroCopyDB.withUnsafeMutableBufferPointer { dst in
                try analyzer.analyze(samples: src, intoMagnitudesDB: dst, binCount: n)
            }
        }

        for i in 0 ..< n {
            XCTAssertEqual(zeroCopyDB[i], referenceDB[i], accuracy: 1e-3,
                           "intoMagnitudesDB bin \(i) diverges from result.magnitudesDB")
        }
    }

    func testAnalyzeBinCountDefaultIsRfftConvention() throws {
        let n = 512
        let samples = [Float](repeating: 0.1, count: n)
        let analyzer = try FFTAnalyzer(size: n, sampleRate: 44100, window: .hann)

        // Default should fill N/2 + 1 entries.
        var mags = [Float](repeating: -999, count: n)
        try samples.withUnsafeBufferPointer { src in
            try mags.withUnsafeMutableBufferPointer { dst in
                try analyzer.analyze(samples: src, intoMagnitudes: dst)
            }
        }
        for i in 0 ... (n / 2) {
            XCTAssertNotEqual(mags[i], -999, "bin \(i) should have been written")
        }
        XCTAssertEqual(mags[n / 2 + 1], -999, "default binCount must stop at N/2+1")
    }

    // MARK: - Helpers

    private func randomSignal(n: Int) -> [SIMD2<Float>] {
        (0 ..< n).map { _ in
            SIMD2<Float>(Float.random(in: -1 ... 1), Float.random(in: -1 ... 1))
        }
    }

    private func vdspFFT(input: [SIMD2<Float>], n: Int) -> [SIMD2<Float>] {
        var log2n = 0; var v = n; while v > 1 {
            v >>= 1; log2n += 1
        }
        guard let setup = vDSP_create_fftsetup(vDSP_Length(log2n), FFTRadix(kFFTRadix2)) else {
            fatalError("vDSP_create_fftsetup failed")
        }
        defer { vDSP_destroy_fftsetup(setup) }

        var realIn = input.map(\.x), imagIn = input.map(\.y)
        var realOut = [Float](repeating: 0, count: n), imagOut = [Float](repeating: 0, count: n)
        realIn.withUnsafeMutableBufferPointer { rIn in
            imagIn.withUnsafeMutableBufferPointer { iIn in
                realOut.withUnsafeMutableBufferPointer { rOut in
                    imagOut.withUnsafeMutableBufferPointer { iOut in
                        var splitIn = DSPSplitComplex(realp: rIn.baseAddress!, imagp: iIn.baseAddress!)
                        var splitOut = DSPSplitComplex(realp: rOut.baseAddress!, imagp: iOut.baseAddress!)
                        vDSP_fft_zop(setup, &splitIn, 1, &splitOut, 1,
                                     vDSP_Length(log2n), FFTDirection(kFFTDirection_Forward))
                    }
                }
            }
        }
        return (0 ..< n).map { SIMD2<Float>(realOut[$0], imagOut[$0]) }
    }

    private func l2RelativeError(_ a: [SIMD2<Float>], _ b: [SIMD2<Float>]) -> Float {
        var errSq: Float = 0, refSq: Float = 0
        for i in 0 ..< a.count {
            let d = a[i] - b[i]
            errSq += d.x * d.x + d.y * d.y
            refSq += b[i].x * b[i].x + b[i].y * b[i].y
        }
        return refSq > 0 ? sqrt(errSq / refSq) : sqrt(errSq)
    }
}
