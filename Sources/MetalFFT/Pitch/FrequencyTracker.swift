import Foundation

// MARK: - FrequencyTracker

/// Stateful frequency smoother with harmonic-jump correction.
///
/// Uses a fixed-size ring buffer to compute a running mean, and corrects
/// octave/suboctave jumps against the last stable frequency.
///
/// Not thread-safe: serialize all calls to `track` and `reset`.
public final class FrequencyTracker {

    // MARK: - Properties

    /// Number of frames used for the running mean.
    public let smoothingWindow: Int

    // MARK: - Private State

    private var buffer: [Double]
    private var lastStable: Double = 0.0

    // MARK: - Init

    public init(smoothingWindow: Int = 5) {
        precondition(smoothingWindow > 0)
        self.smoothingWindow = smoothingWindow
        self.buffer = []
        self.buffer.reserveCapacity(smoothingWindow)
    }

    // MARK: - Public API

    /// Feeds `rawFrequency` through harmonic-jump correction and a ring-buffer mean.
    ///
    /// - Parameter rawFrequency: Raw interpolated frequency in Hz. Pass 0 or negative to indicate silence.
    /// - Returns: The smoothed frequency. Returns `rawFrequency` unchanged when it is ≤ 0.
    @discardableResult
    public func track(_ rawFrequency: Double) -> Double {
        guard rawFrequency > 0 else { return rawFrequency }
        let corrected = correctHarmonicJump(rawFrequency)
        buffer.append(corrected)
        if buffer.count > smoothingWindow { buffer.removeFirst() }
        let smoothed = buffer.reduce(0.0, +) / Double(buffer.count)
        if abs(smoothed - lastStable) < 50 { lastStable = smoothed }
        return smoothed
    }

    /// Clears the smoothing buffer and the stable frequency reference.
    public func reset() {
        buffer.removeAll(keepingCapacity: true)
        lastStable = 0.0
    }

    // MARK: - Private

    private func correctHarmonicJump(_ frequency: Double) -> Double {
        guard lastStable > 0 else { return frequency }
        let ratio = frequency / lastStable
        let harmonicRatios: [(target: Double, divisor: Double)] = [
            (2.0, 2.0), (3.0, 3.0), (4.0, 4.0),
            (0.5, 0.5), (1.0 / 3.0, 1.0 / 3.0), (0.25, 0.25),
        ]
        for (target, divisor) in harmonicRatios {
            if abs(ratio - target) < 0.1 { return frequency / divisor }
        }
        return frequency
    }
}
