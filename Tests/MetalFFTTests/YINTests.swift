import Accelerate
@testable import MetalFFT
import XCTest

/// Pitch detection is judged on the cases that break spectral peak-picking:
/// low notes a bin cannot resolve, a missing fundamental, and the octave
/// errors that come out of both.
final class YINTests: XCTestCase {
    let sampleRate = 48000.0
    let bufferSize = 4096

    // MARK: - Signal generators

    /// A harmonic tone. `partials` are linear amplitudes starting at the
    /// fundamental, so a leading zero is a note whose fundamental the
    /// microphone never captured.
    func tone(_ f0: Float,
              partials: [Float] = [1.0, 0.5, 0.33, 0.25],
              noise: Float = 0,
              phase: Float = 0.3) -> [Float] {
        var out = [Float](repeating: 0, count: bufferSize)
        var seed: UInt32 = 22222
        for n in 0 ..< bufferSize {
            let t = Float(n) / Float(sampleRate)
            var v: Float = 0
            for (index, amplitude) in partials.enumerated() where amplitude > 0 {
                let harmonic = Float(index + 1)
                v += amplitude * sinf(2 * .pi * f0 * harmonic * t + phase * harmonic)
            }
            if noise > 0 {
                seed = seed &* 1_664_525 &+ 1_013_904_223
                v += noise * (Float(seed >> 8) / Float(1 << 24) - 0.5) * 2
            }
            out[n] = v * 0.25
        }
        return out
    }

    func detector(min: Float = 55, max: Float = 1500) -> YINDetector {
        YINDetector(sampleRate: sampleRate,
                    bufferSize: bufferSize,
                    minFrequency: min,
                    maxFrequency: max)
    }

    func cents(_ measured: Float, _ reference: Float) -> Float {
        1200 * log2f(measured / reference)
    }

    // MARK: - Accuracy

    func testGuitarOpenStrings() throws {
        // Standard tuning. E2 is the case a 4096-point FFT cannot resolve:
        // its neighbours sit well inside one 11.7 Hz bin.
        let strings: [(String, Float)] = [
            ("E2", 82.41), ("A2", 110.00), ("D3", 146.83),
            ("G3", 196.00), ("B3", 246.94), ("E4", 329.63),
        ]
        let yin = detector()
        for (name, frequency) in strings {
            let estimate = try XCTUnwrap(yin.detect(tone(frequency)), name)
            XCTAssertEqual(cents(estimate.frequencyHz, frequency), 0, accuracy: 5,
                           "\(name): got \(estimate.frequencyHz) Hz, want \(frequency) Hz")
            XCTAssertTrue(estimate.isPeriodic, "\(name) should read as periodic")
            XCTAssertGreaterThan(estimate.confidence, 0.8, name)
        }
    }

    /// Sub-cent accuracy is what separates a tuner from a note-namer, and it
    /// comes entirely from the parabolic interpolation.
    func testResolvesDetuningFinerThanABin() throws {
        let yin = detector()
        for offset in [-30, -12, -5, 5, 12, 30] as [Float] {
            let target = 110 * powf(2, offset / 1200)
            let estimate = try XCTUnwrap(yin.detect(tone(target)))
            XCTAssertEqual(cents(estimate.frequencyHz, 110), offset, accuracy: 3,
                           "\(offset) cents off A2")
        }
    }

    /// The failure that sent the old spectral detector an octave up: a voice or
    /// a pickup that loses the fundamental entirely. The period of the waveform
    /// is unchanged, so YIN is unmoved.
    func testMissingFundamental() throws {
        let yin = detector()
        let estimate = try XCTUnwrap(yin.detect(tone(146.83, partials: [0, 1.0, 0.6, 0.4])))
        XCTAssertEqual(cents(estimate.frequencyHz, 146.83), 0, accuracy: 8)
    }

    /// And the opposite error: a bright tone whose second harmonic is the
    /// loudest thing present must not read an octave up.
    func testBrightToneIsNotReadAnOctaveUp() throws {
        let yin = detector()
        let estimate = try XCTUnwrap(yin.detect(tone(196, partials: [0.15, 1.0, 0.7, 0.45, 0.3])))
        XCTAssertEqual(cents(estimate.frequencyHz, 196), 0, accuracy: 10,
                       "got \(estimate.frequencyHz) Hz — an octave up would be ~392")
    }

    /// A weak tone under a lot of hiss should still be found, and should say it
    /// is less sure about it.
    func testNoiseLowersConfidence() throws {
        let yin = detector()
        let clean = try XCTUnwrap(yin.detect(tone(220)))
        let noisy = try XCTUnwrap(yin.detect(tone(220, noise: 0.6)))
        XCTAssertEqual(cents(noisy.frequencyHz, 220), 0, accuracy: 15)
        XCTAssertLessThan(noisy.confidence, clean.confidence)
    }

    /// Every semitone across the advertised range, not just the six the guitar
    /// happens to use. Steps 1–3 run on a quarter-rate copy of the window, and
    /// this is what says that costs nothing: the refinement is at the full rate,
    /// so the answer is as sharp as it ever was, everywhere.
    func testAccuracyAcrossTheWholeRange() throws {
        let yin = detector()
        var frequency = Float(58)
        while frequency < 1500 {
            let estimate = try XCTUnwrap(yin.detect(tone(frequency)), "\(frequency) Hz")
            XCTAssertEqual(cents(estimate.frequencyHz, frequency), 0, accuracy: 2,
                           "\(frequency) Hz: got \(estimate.frequencyHz) Hz")
            XCTAssertTrue(estimate.isPeriodic, "\(frequency) Hz should read as periodic")
            frequency *= powf(2, 1.0 / 12)
        }
    }

    /// The coarse stage is an optimisation, so it has to agree with the search
    /// that does not use it. A detector whose range is too wide to decimate
    /// safely turns the stage off, which gives the exact search to compare to.
    func testDecimatedSearchAgreesWithTheExactOne() throws {
        let coarse = detector()
        let exact = detector(min: 55, max: 6000)
        XCTAssertEqual(coarse.decimation, 4, "the default range should decimate")
        XCTAssertEqual(exact.decimation, 1, "a range up to 6 kHz cannot be decimated")

        for frequency in [82.41, 146.83, 329.63, 440, 987.77] as [Float] {
            let a = try XCTUnwrap(coarse.detect(tone(frequency)))
            let b = try XCTUnwrap(exact.detect(tone(frequency)))
            XCTAssertEqual(cents(a.frequencyHz, b.frequencyHz), 0, accuracy: 1,
                           "\(frequency) Hz: decimated \(a.frequencyHz), exact \(b.frequencyHz)")
        }
    }

    /// Hiss costs precision but must not cost the note.
    func testAccuracyUnderNoise() throws {
        let yin = detector()
        var frequency = Float(58)
        while frequency < 1500 {
            let estimate = try XCTUnwrap(yin.detect(tone(frequency, noise: 0.4)), "\(frequency) Hz")
            XCTAssertEqual(cents(estimate.frequencyHz, frequency), 0, accuracy: 20,
                           "\(frequency) Hz: got \(estimate.frequencyHz) Hz")
            frequency *= powf(2, 1.0 / 12)
        }
    }

    /// A period shorter than `minLag` is not the caller's to be given. Before
    /// step 3 required a falling curve, a tone above `maxFrequency` could be
    /// accepted on the shoulder of its dip and reported as itself — outside the
    /// range that was asked for. The subharmonic is the honest answer.
    func testPitchAboveTheRangeIsNotReportedAnyway() throws {
        let yin = detector(min: 55, max: 285)
        let estimate = try XCTUnwrap(yin.detect(tone(300)))
        XCTAssertLessThanOrEqual(estimate.frequencyHz, 285,
                                 "reported \(estimate.frequencyHz) Hz above its own maximum")
        XCTAssertEqual(cents(estimate.frequencyHz, 150), 0, accuracy: 5)
    }

    /// The decimation factor falls out of the sample rate, the window and the
    /// frequency range, so every combination has to land somewhere valid — 44.1
    /// kHz gives a factor of three, a low rate or an odd window length gives
    /// whatever fits, and none of them may lose the note.
    func testConfigurationMatrix() throws {
        let cases: [(rate: Double, size: Int, top: Float)] = [
            (44100, 4096, 1500), (44100, 2048, 1500), (48000, 8192, 1500),
            (48000, 1000, 1500), (16000, 1024, 1500), (8000, 1024, 800),
            (96000, 4096, 2000), (48000, 4096, 400),
        ]
        for spec in cases {
            let yin = YINDetector(sampleRate: spec.rate, bufferSize: spec.size, maxFrequency: spec.top)
            var samples = [Float](repeating: 0, count: spec.size)
            for n in 0 ..< spec.size {
                let t = Float(n) / Float(spec.rate)
                samples[n] = 0.25 * (sinf(2 * .pi * 220 * t) + 0.5 * sinf(4 * .pi * 220 * t))
            }
            let estimate = try XCTUnwrap(yin.detect(samples), "\(spec)")
            XCTAssertEqual(cents(estimate.frequencyHz, 220), 0, accuracy: 5,
                           "\(spec) decimation \(yin.decimation): got \(estimate.frequencyHz) Hz")
        }
    }

    /// A window too short to decimate must still work, on the exact path.
    func testShortWindowFallsBackToTheExactSearch() {
        let yin = YINDetector(sampleRate: sampleRate, bufferSize: 64)
        XCTAssertEqual(yin.decimation, 1)
        XCTAssertNotNil(yin.detect(Array(tone(1000)[0 ..< 64])))
    }

    // MARK: - Refusing to answer

    func testSilenceIsRefused() {
        let yin = detector()
        XCTAssertNil(yin.detect([Float](repeating: 0, count: bufferSize)))
    }

    func testLevelGateRefusesQuietNoise() {
        let yin = detector()
        var quiet = tone(220, partials: [0], noise: 0.002)
        vDSP.multiply(0.3, quiet, result: &quiet)
        XCTAssertNil(yin.detect(quiet), "room noise must not produce a pitch")
    }

    /// Broadband noise is loud enough to analyse but has no period. It must
    /// come back as not periodic, which is the signal a display gates on.
    func testNoiseIsNotPeriodic() throws {
        let yin = detector()
        let estimate = yin.detect(tone(220, partials: [0], noise: 1.0))
        if let estimate {
            XCTAssertFalse(estimate.isPeriodic, "noise reported a period")
            XCTAssertLessThan(estimate.confidence, 0.6)
        }
    }

    func testWrongWindowLengthIsRefused() {
        let yin = detector()
        XCTAssertNil(yin.detect([Float](repeating: 0.5, count: bufferSize / 2)))
    }

    // MARK: - Cost

    /// The window arrives about twelve times a second; the detector has to be
    /// far inside that budget or it belongs on the GPU instead.
    func testDetectionIsFasterThanTheWindowItConsumes() throws {
        let yin = detector()
        let samples = tone(146.83)
        let start = Date()
        let rounds = 100
        for _ in 0 ..< rounds { _ = yin.detect(samples) }
        let each = Date().timeIntervalSince(start) / Double(rounds)
        let windowDuration = Double(bufferSize) / sampleRate
        XCTAssertLessThan(each, windowDuration * 0.5,
                          "detect() took \(each * 1000) ms against a \(windowDuration * 1000) ms window")
    }
}
