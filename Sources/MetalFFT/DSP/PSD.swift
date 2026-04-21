import Accelerate

// MARK: - PSD

/// Power spectral density and coherence estimation.
public enum PSD {
    /// Power spectral density via Welch's method (averaged overlapping periodograms).
    ///
    /// Returns PSD in linear units (squared magnitude per Hz). Take `Spectrum.toDecibels`
    /// of the result to convert to dB/Hz.
    ///
    /// - Parameters:
    ///   - signal: Real-valued input signal.
    ///   - fftSize: Must be a supported MetalFFT size (64–16384).
    ///   - hopSize: Hop between analysis frames (controls overlap).
    ///   - sampleRate: Signal sample rate.
    ///   - windowType: Spectral window. Default `.hann`.
    public static func welch(
        signal: [Float],
        fftSize: Int,
        hopSize: Int,
        sampleRate: Double,
        window windowType: WindowType = .hann
    ) throws -> [Float] {
        let w = windowType.coefficients(fftSize)

        var windowSumSq: Float = 0
        vDSP_svesq(w, 1, &windowSumSq, vDSP_Length(fftSize))

        let fft = try MetalFFT(size: fftSize)
        var accumulator = [Float](repeating: 0, count: fftSize)
        var windowed = [Float](repeating: 0, count: fftSize)
        var complex = [SIMD2<Float>](repeating: .zero, count: fftSize)
        var spectrum = [SIMD2<Float>](repeating: .zero, count: fftSize)
        var frameCount = 0

        var pos = 0
        while pos + fftSize <= signal.count {
            signal.withUnsafeBufferPointer { ptr in
                vDSP_vmul(ptr.baseAddress! + pos, 1, w, 1, &windowed, 1, vDSP_Length(fftSize))
            }
            for i in 0 ..< fftSize {
                complex[i] = SIMD2<Float>(windowed[i], 0)
            }
            try complex.withUnsafeBufferPointer { try fft.forward(input: $0, output: &spectrum) }

            let mags = Spectrum.magnitudes(spectrum)
            vDSP_vadd(accumulator, 1, mags, 1, &accumulator, 1, vDSP_Length(fftSize))
            frameCount += 1
            pos += hopSize
        }

        guard frameCount > 0 else { return accumulator }

        // Normalize: divide by (frames × windowSumSq × sampleRate)
        var invScale = 1.0 / (Float(frameCount) * windowSumSq * Float(sampleRate))
        vDSP_vsmul(accumulator, 1, &invScale, &accumulator, 1, vDSP_Length(fftSize))
        return accumulator
    }

    /// Magnitude-squared coherence between two signals: C(f) = |S_xy(f)|² / (S_xx(f)·S_yy(f)).
    ///
    /// Output is in [0, 1] per bin: 1 = fully linearly coherent at that frequency.
    /// Useful for measuring coupling, common excitation, or signal similarity.
    ///
    /// - Parameters:
    ///   - a: First real-valued signal.
    ///   - b: Second real-valued signal (same length as `a`).
    ///   - fftSize: Must be a supported MetalFFT size (64–16384).
    ///   - hopSize: Hop between analysis frames.
    ///   - sampleRate: Signal sample rate.
    ///   - windowType: Spectral window. Default `.hann`.
    public static func coherence(
        a: [Float],
        b: [Float],
        fftSize: Int,
        hopSize: Int,
        sampleRate: Double,
        window windowType: WindowType = .hann
    ) throws -> [Float] {
        let w = windowType.coefficients(fftSize)
        let fft = try MetalFFT(size: fftSize)

        var sxx = [Float](repeating: 0, count: fftSize)
        var syy = [Float](repeating: 0, count: fftSize)
        var sxyR = [Float](repeating: 0, count: fftSize)
        var sxyI = [Float](repeating: 0, count: fftSize)

        var wA = [Float](repeating: 0, count: fftSize)
        var wB = [Float](repeating: 0, count: fftSize)
        var cA = [SIMD2<Float>](repeating: .zero, count: fftSize)
        var cB = [SIMD2<Float>](repeating: .zero, count: fftSize)
        var oA = [SIMD2<Float>](repeating: .zero, count: fftSize)
        var oB = [SIMD2<Float>](repeating: .zero, count: fftSize)

        let limit = min(a.count, b.count)
        var pos = 0
        var frameCount = 0

        while pos + fftSize <= limit {
            a.withUnsafeBufferPointer { ptr in
                vDSP_vmul(ptr.baseAddress! + pos, 1, w, 1, &wA, 1, vDSP_Length(fftSize))
            }
            b.withUnsafeBufferPointer { ptr in
                vDSP_vmul(ptr.baseAddress! + pos, 1, w, 1, &wB, 1, vDSP_Length(fftSize))
            }
            for i in 0 ..< fftSize {
                cA[i] = SIMD2<Float>(wA[i], 0); cB[i] = SIMD2<Float>(wB[i], 0)
            }
            try cA.withUnsafeBufferPointer { try fft.forward(input: $0, output: &oA) }
            try cB.withUnsafeBufferPointer { try fft.forward(input: $0, output: &oB) }

            // Accumulate auto- and cross-spectra
            let magsA = Spectrum.magnitudes(oA)
            let magsB = Spectrum.magnitudes(oB)
            vDSP_vadd(sxx, 1, magsA, 1, &sxx, 1, vDSP_Length(fftSize))
            vDSP_vadd(syy, 1, magsB, 1, &syy, 1, vDSP_Length(fftSize))
            for i in 0 ..< fftSize {
                sxyR[i] += oA[i].x * oB[i].x + oA[i].y * oB[i].y // Re(A·conj(B))
                sxyI[i] += oA[i].y * oB[i].x - oA[i].x * oB[i].y // Im(A·conj(B))
            }

            frameCount += 1
            pos += hopSize
        }

        guard frameCount > 0 else { return [Float](repeating: 0, count: fftSize) }

        return (0 ..< fftSize).map { i in
            let denom = sxx[i] * syy[i]
            guard denom > 0 else { return 0 }
            return (sxyR[i] * sxyR[i] + sxyI[i] * sxyI[i]) / denom
        }
    }
}
