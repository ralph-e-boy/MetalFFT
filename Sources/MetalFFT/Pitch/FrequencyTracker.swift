import Accelerate

/// Stateful frequency smoother using a fixed-capacity ring buffer.
/// Construct once and reuse. Not thread-safe.
public final class FrequencyTracker {
    public let smoothingWindow: Int

    private var ring: [Float]
    private var head: Int = 0
    private var filled: Int = 0

    /// - Parameter smoothingWindow: Number of frames to average. Clamped to [1, 64].
    public init(smoothingWindow: Int = 5) {
        self.smoothingWindow = max(1, min(smoothingWindow, 64))
        ring = [Float](repeating: 0, count: self.smoothingWindow)
    }

    /// Pushes `frequency` into the ring buffer and returns the smoothed mean.
    /// Returns `frequency` unchanged if it is ≤ 0.
    public func track(_ frequency: Float) -> Float {
        guard frequency > 0 else { return frequency }
        ring[head] = frequency
        head = (head + 1) % smoothingWindow
        filled = min(filled + 1, smoothingWindow)
        var mean: Float = 0
        vDSP_meanv(ring, 1, &mean, vDSP_Length(filled))
        return mean
    }

    /// Clears the smoothing buffer.
    public func reset() {
        vDSP_vclr(&ring, 1, vDSP_Length(smoothingWindow))
        head = 0
        filled = 0
    }
}
