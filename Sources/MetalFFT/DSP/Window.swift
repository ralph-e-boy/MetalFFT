import Accelerate

// MARK: - WindowType

/// Selects a windowing function for use with `FFTAnalyzer`, `STFT`, and other components.
public enum WindowType {
    case hann
    case hamming
    case blackman
    case flatTop
    case kaiser(beta: Double)
    case rectangular

    public func coefficients(_ size: Int) -> [Float] {
        switch self {
        case .hann:              return Window.hann(size)
        case .hamming:           return Window.hamming(size)
        case .blackman:          return Window.blackman(size)
        case .flatTop:           return Window.flatTop(size)
        case .kaiser(let beta):  return Window.kaiser(size, beta: beta)
        case .rectangular:       return [Float](repeating: 1, count: size)
        }
    }
}

// MARK: - Window

/// Windowing functions and application helpers.
public enum Window {

    // MARK: - Window Generators

    public static func hann(_ size: Int) -> [Float] {
        var w = [Float](repeating: 0, count: size)
        vDSP_hann_window(&w, vDSP_Length(size), Int32(vDSP_HANN_NORM))
        return w
    }

    public static func hamming(_ size: Int) -> [Float] {
        var w = [Float](repeating: 0, count: size)
        vDSP_hamm_window(&w, vDSP_Length(size), 0)
        return w
    }

    /// Blackman window — lower sidelobes than Hann, good general-purpose choice.
    public static func blackman(_ size: Int) -> [Float] {
        var w = [Float](repeating: 0, count: size)
        vDSP_blkman_window(&w, vDSP_Length(size), 0)
        return w
    }

    /// Flat-top window — maximally flat passband for accurate amplitude measurement.
    public static func flatTop(_ size: Int) -> [Float] {
        let a0: Float = 0.21557895
        let a1: Float = 0.41663158
        let a2: Float = 0.277263158
        let a3: Float = 0.083578947
        let a4: Float = 0.006947368
        let n = Float(size)
        return (0..<size).map { k in
            let x = Float(k) / n
            return a0
                - a1 * cos(2 * .pi * x)
                + a2 * cos(4 * .pi * x)
                - a3 * cos(6 * .pi * x)
                + a4 * cos(8 * .pi * x)
        }
    }

    /// Kaiser window — tunable sidelobe attenuation via `beta` (common values: 5–10).
    public static func kaiser(_ size: Int, beta: Double = 6.0) -> [Float] {
        let halfN = Double(size - 1) / 2.0
        let i0Beta = besselI0(beta)
        return (0..<size).map { k in
            let x = (Double(k) - halfN) / halfN
            return Float(besselI0(beta * sqrt(max(0, 1 - x * x))) / i0Beta)
        }
    }

    // MARK: - Application

    /// Element-wise multiply `window` by `input`, writing into `output`.
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

    // MARK: - Internal

    private static func besselI0(_ x: Double) -> Double {
        var result = 1.0
        var term = 1.0
        let halfX = x / 2.0
        for k in 1...30 {
            term *= (halfX * halfX) / Double(k * k)
            result += term
            if term < 1e-12 * result { break }
        }
        return result
    }
}
