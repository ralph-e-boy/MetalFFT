import Accelerate

// MARK: - Window

/// Windowing functions and application helpers.
public enum Window {

    /// Returns a Hann window of `size` samples.
    public static func hann(_ size: Int) -> [Float] {
        var w = [Float](repeating: 0, count: size)
        vDSP_hann_window(&w, vDSP_Length(size), Int32(vDSP_HANN_NORM))
        return w
    }

    /// Returns a Hamming window of `size` samples.
    public static func hamming(_ size: Int) -> [Float] {
        var w = [Float](repeating: 0, count: size)
        vDSP_hamm_window(&w, vDSP_Length(size), 0)
        return w
    }

    /// Element-wise multiply `window` by `input`, writing into `output`.
    /// Caller must ensure `window.count == input.count == output.count`.
    public static func apply(
        _ window: [Float],
        input: UnsafeBufferPointer<Float>,
        output: inout [Float]
    ) {
        precondition(window.count == input.count && window.count == output.count)
        vDSP_vmul(input.baseAddress!, 1, window, 1, &output, 1, vDSP_Length(window.count))
    }

    /// Convenience: apply window to `samples` returning a new windowed array.
    public static func apply(_ window: [Float], to samples: [Float]) -> [Float] {
        precondition(window.count == samples.count)
        var out = [Float](repeating: 0, count: window.count)
        vDSP_vmul(samples, 1, window, 1, &out, 1, vDSP_Length(window.count))
        return out
    }
}
