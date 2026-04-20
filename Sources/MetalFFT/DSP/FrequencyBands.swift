import Accelerate

// MARK: - BandEnergy

/// Per-band summed energy from a magnitude spectrum.
public struct BandEnergy {
    public let sub:   Float   // 20–60 Hz   — rumble, kick fundamental
    public let bass:  Float   // 60–250 Hz  — bass guitar, kick body
    public let low:   Float   // 250–500 Hz — lower mids
    public let mid:   Float   // 500–2 kHz  — vocals, guitars
    public let upper: Float   // 2–4 kHz    — presence, attack
    public let air:   Float   // 4–20 kHz   — cymbals, sibilance, sheen

    public var all: [(name: String, energy: Float)] {
        [("sub", sub), ("bass", bass), ("low", low), ("mid", mid), ("upper", upper), ("air", air)]
    }

    public var total: Float { sub + bass + low + mid + upper + air }
}

// MARK: - FrequencyBands

/// Splits a magnitude spectrum into named perceptual bands.
/// Construct once per (sampleRate, fftSize) pair and reuse.
public struct FrequencyBands {
    public let sampleRate: Double
    public let fftSize: Int

    private let freqPerBin: Double

    public init(sampleRate: Double, fftSize: Int) {
        self.sampleRate = sampleRate
        self.fftSize = fftSize
        self.freqPerBin = sampleRate / Double(fftSize)
    }

    /// Summed energy per band from squared magnitudes (as returned by `Spectrum.magnitudes`).
    public func analyze(_ magnitudes: [Float]) -> BandEnergy {
        BandEnergy(
            sub:   sum(magnitudes, lo: 20,   hi: 60),
            bass:  sum(magnitudes, lo: 60,   hi: 250),
            low:   sum(magnitudes, lo: 250,  hi: 500),
            mid:   sum(magnitudes, lo: 500,  hi: 2000),
            upper: sum(magnitudes, lo: 2000, hi: 4000),
            air:   sum(magnitudes, lo: 4000, hi: 20000)
        )
    }

    /// Same as `analyze` but each band is divided by total energy → values in [0, 1].
    public func analyzeNormalized(_ magnitudes: [Float]) -> BandEnergy {
        let e = analyze(magnitudes)
        let total = e.total
        guard total > 0 else { return e }
        let inv = 1.0 / total
        return BandEnergy(
            sub:   e.sub   * inv,
            bass:  e.bass  * inv,
            low:   e.low   * inv,
            mid:   e.mid   * inv,
            upper: e.upper * inv,
            air:   e.air   * inv
        )
    }

    // MARK: - Internal

    private func sum(_ magnitudes: [Float], lo: Double, hi: Double) -> Float {
        let a = max(0, Int(lo / freqPerBin))
        let b = min(magnitudes.count - 1, Int(hi / freqPerBin))
        guard a <= b else { return 0 }
        var s: Float = 0
        magnitudes.withUnsafeBufferPointer { ptr in
            vDSP_sve(ptr.baseAddress! + a, 1, &s, vDSP_Length(b - a + 1))
        }
        return s
    }
}
