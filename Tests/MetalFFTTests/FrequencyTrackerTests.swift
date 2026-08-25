import Accelerate
@testable import MetalFFT
import XCTest

/// The tracker exists for one job: to stop a single bad frame from reaching the
/// display. These tests are about that frame.
final class FrequencyTrackerTests: XCTestCase {
    // MARK: - Mean, unchanged

    func testMeanIsTheRunningAverage() {
        let tracker = FrequencyTracker(smoothingWindow: 4)
        XCTAssertEqual(tracker.track(100), 100, accuracy: 1e-4)
        XCTAssertEqual(tracker.track(200), 150, accuracy: 1e-4)
        XCTAssertEqual(tracker.track(300), 200, accuracy: 1e-4)
    }

    func testMeanIsStillTheDefault() {
        let tracker = FrequencyTracker(smoothingWindow: 3)
        _ = tracker.track(100)
        XCTAssertEqual(tracker.track(400), 250, accuracy: 1e-4)
    }

    // MARK: - Median

    /// The whole reason for the mode: an octave error passes through the mean
    /// and is stopped dead by the median.
    func testMedianRejectsAnIsolatedOctaveError() {
        let steady: [Float] = [220, 220.4, 219.6, 220.2, 220.1]
        let mean = FrequencyTracker(smoothingWindow: 5, smoothing: .mean)
        let median = FrequencyTracker(smoothingWindow: 5, smoothing: .median)
        for frequency in steady {
            _ = mean.track(frequency)
            _ = median.track(frequency)
        }

        let jump: Float = 440
        XCTAssertEqual(median.track(jump), 220.2, accuracy: 0.5,
                       "the median must not move for one frame")
        XCTAssertGreaterThan(mean.track(jump), 250, "the mean is expected to move")
    }

    /// And the frame after it is clean again, with no tail.
    func testMedianLeavesNoTailAfterTheOutlier() {
        let tracker = FrequencyTracker(smoothingWindow: 5, smoothing: .median)
        for frequency in [220, 220, 220, 220, 440] as [Float] { _ = tracker.track(frequency) }
        XCTAssertEqual(tracker.track(220), 220, accuracy: 1e-4)
    }

    /// A median is not a freeze: a real slide has to arrive, late but whole.
    func testMedianFollowsARealSlide() {
        let tracker = FrequencyTracker(smoothingWindow: 5, smoothing: .median)
        var last: Float = 0
        for step in 0 ..< 12 { last = tracker.track(220 + Float(step) * 10) }
        XCTAssertEqual(last, 220 + 90, accuracy: 1e-4, "three frames of lag at a window of five")
    }

    func testMedianOfAPartialWindow() {
        let tracker = FrequencyTracker(smoothingWindow: 5, smoothing: .median)
        XCTAssertEqual(tracker.track(300), 300, accuracy: 1e-4)
        XCTAssertEqual(tracker.track(100), 200, accuracy: 1e-4, "two frames average their middles")
        XCTAssertEqual(tracker.track(200), 200, accuracy: 1e-4)
    }

    // MARK: - Shared behaviour

    func testUnvoicedFramesPassThroughUntouched() {
        let tracker = FrequencyTracker(smoothingWindow: 5, smoothing: .median)
        _ = tracker.track(220)
        XCTAssertEqual(tracker.track(0), 0)
        XCTAssertEqual(tracker.track(-1), -1)
        XCTAssertEqual(tracker.track(220), 220, accuracy: 1e-4, "and do not enter the window")
    }

    func testResetForgetsTheNote() {
        let tracker = FrequencyTracker(smoothingWindow: 5, smoothing: .median)
        for _ in 0 ..< 5 { _ = tracker.track(220) }
        tracker.reset()
        XCTAssertEqual(tracker.track(880), 880, accuracy: 1e-4)
    }

    func testWindowOfOneIsAPassThrough() {
        for smoothing in [FrequencyTracker.Smoothing.mean, .median] {
            let tracker = FrequencyTracker(smoothingWindow: 1, smoothing: smoothing)
            XCTAssertEqual(tracker.track(123), 123, accuracy: 1e-4, "\(smoothing)")
            XCTAssertEqual(tracker.track(456), 456, accuracy: 1e-4, "\(smoothing)")
        }
    }

    // MARK: - Against the detector it exists for

    /// End to end: a run of windows where one is a burst of noise. The median
    /// tracker is what keeps that frame off the tuner.
    func testMedianSurvivesOneBadWindowFromYIN() throws {
        let sampleRate = 48000.0
        let size = 4096
        let yin = YINDetector(sampleRate: sampleRate, bufferSize: size)
        let tracker = FrequencyTracker(smoothingWindow: 5, smoothing: .median)

        func window(_ f0: Float, noisy: Bool) -> [Float] {
            var seed: UInt32 = 7777
            return (0 ..< size).map { n in
                let t = Float(n) / Float(sampleRate)
                if noisy {
                    seed = seed &* 1_664_525 &+ 1_013_904_223
                    return 0.5 * (Float(seed >> 8) / Float(1 << 24) - 0.5)
                }
                return 0.25 * (sinf(2 * .pi * f0 * t) + 0.5 * sinf(4 * .pi * f0 * t))
            }
        }

        var smoothed: Float = 0
        for frame in 0 ..< 9 {
            let estimate = yin.detect(window(196, noisy: frame == 5))
            smoothed = tracker.track(estimate?.frequencyHz ?? 0)
        }
        XCTAssertEqual(1200 * log2f(smoothed / 196), 0, accuracy: 5,
                       "one noise burst pulled the tracked pitch to \(smoothed) Hz")
    }
}
