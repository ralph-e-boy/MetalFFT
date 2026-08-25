import Accelerate

/// Stateful frequency smoother using a fixed-capacity ring buffer.
/// Construct once and reuse. Not thread-safe.
public final class FrequencyTracker {
    /// How the frames in the window are combined.
    public enum Smoothing: Sendable {
        /// Running mean. Steady on a stable note, but every frame contributes:
        /// one wrong frame moves the answer and goes on moving it until it
        /// falls out of the window. An octave error in a window of five drags
        /// the mean up by a fifth of an octave for five frames — a wrong
        /// reading turned into five slightly wrong ones.
        case mean
        /// Running median. The frame is outvoted rather than averaged in, so
        /// an isolated octave jump — the characteristic YIN failure, and the
        /// one thing steps 1–4 cannot rule out from a single window — leaves no
        /// trace at all. A real slide still tracks it, delayed by half the
        /// window.
        ///
        /// This is what pitch wants. It is not the default only because the
        /// mean was here first.
        case median
    }

    public let smoothingWindow: Int
    public let smoothing: Smoothing

    private var ring: [Float]
    private var sorted: [Float]
    private var head: Int = 0
    private var filled: Int = 0

    /// - Parameters:
    ///   - smoothingWindow: Number of frames to combine. Clamped to [1, 64].
    ///     Prefer an odd count under `.median`: an even one averages the two
    ///     middle frames, which lets a single outlier back in when half the
    ///     window is outliers.
    ///   - smoothing: How to combine them. See `Smoothing`.
    public init(smoothingWindow: Int = 5, smoothing: Smoothing = .mean) {
        self.smoothingWindow = max(1, min(smoothingWindow, 64))
        self.smoothing = smoothing
        ring = [Float](repeating: 0, count: self.smoothingWindow)
        sorted = [Float](repeating: 0, count: self.smoothingWindow)
    }

    /// Pushes `frequency` into the ring buffer and returns the smoothed value.
    /// Returns `frequency` unchanged if it is ≤ 0.
    public func track(_ frequency: Float) -> Float {
        guard frequency > 0 else { return frequency }
        ring[head] = frequency
        head = (head + 1) % smoothingWindow
        filled = min(filled + 1, smoothingWindow)

        switch smoothing {
        case .mean:
            var mean: Float = 0
            vDSP_meanv(ring, 1, &mean, vDSP_Length(filled))
            return mean
        case .median:
            return median()
        }
    }

    /// Clears the smoothing buffer.
    public func reset() {
        vDSP_vclr(&ring, 1, vDSP_Length(smoothingWindow))
        head = 0
        filled = 0
    }

    // MARK: - Median

    /// The ring is written from index 0 up until it first wraps, so the frames
    /// held are always `ring[0 ..< filled]` whatever the head is doing.
    private func median() -> Float {
        sorted.withUnsafeMutableBufferPointer { out in
            ring.withUnsafeBufferPointer { source in
                out.baseAddress!.update(from: source.baseAddress!, count: filled)
            }
            vDSP_vsort(out.baseAddress!, vDSP_Length(filled), 1)
        }
        let middle = filled / 2
        return filled % 2 == 1 ? sorted[middle] : 0.5 * (sorted[middle - 1] + sorted[middle])
    }
}
