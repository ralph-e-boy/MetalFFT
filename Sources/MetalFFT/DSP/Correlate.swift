import Metal

// MARK: - Correlator

/// GPU-accelerated cross-correlation and autocorrelation via FFT.
/// Holds a `MetalFFT` instance — create once per `fftSize` and reuse.
public final class Correlator {
    public let fftSize: Int

    private let fft: MetalFFT
    private var bufA: [SIMD2<Float>]
    private var bufB: [SIMD2<Float>]
    private var outA: [SIMD2<Float>]
    private var outB: [SIMD2<Float>]

    public init(fftSize: Int) throws {
        self.fftSize = fftSize
        fft = try MetalFFT(size: fftSize)
        bufA = [SIMD2<Float>](repeating: .zero, count: fftSize)
        bufB = [SIMD2<Float>](repeating: .zero, count: fftSize)
        outA = [SIMD2<Float>](repeating: .zero, count: fftSize)
        outB = [SIMD2<Float>](repeating: .zero, count: fftSize)
    }

    /// Circular autocorrelation of `signal` (zero-padded to `fftSize`).
    /// Peak at lag 0 equals signal energy; peaks at lag m indicate periodicity at m samples.
    public func auto(_ signal: [Float]) throws -> [Float] {
        try cross(signal, signal)
    }

    /// Circular cross-correlation R_ab[m] = Σ a[n] · b[n+m] (zero-padded to `fftSize`).
    /// Via FFT: IFFT(FFT(a) · conj(FFT(b))).
    public func cross(_ a: [Float], _ b: [Float]) throws -> [Float] {
        pack(a, into: &bufA)
        pack(b, into: &bufB)
        try bufA.withUnsafeBufferPointer { try fft.forward(input: $0, output: &outA) }
        try bufB.withUnsafeBufferPointer { try fft.forward(input: $0, output: &outB) }

        var product = [SIMD2<Float>](repeating: .zero, count: fftSize)
        for i in 0 ..< fftSize {
            let ai = outA[i], bi = outB[i]
            // A · conj(B) = (ar+j·ai)(br-j·bi)
            product[i] = SIMD2<Float>(ai.x * bi.x + ai.y * bi.y,
                                      ai.y * bi.x - ai.x * bi.y)
        }

        let result = try fft.inverse(product)
        return result.map(\.x)
    }

    // MARK: - Internal

    private func pack(_ samples: [Float], into buf: inout [SIMD2<Float>]) {
        let n = min(samples.count, fftSize)
        for i in 0 ..< n {
            buf[i] = SIMD2<Float>(samples[i], 0)
        }
        for i in n ..< fftSize {
            buf[i] = .zero
        }
    }
}
