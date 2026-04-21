import Accelerate

// MARK: - OctaveBands

/// Standard 1/3-octave band analysis (31 bands, ISO 266 center frequencies).
public enum OctaveBands {
    /// ISO 266 nominal 1/3-octave center frequencies from 16 Hz to 16 kHz.
    public static let centerFrequencies: [Double] = [
        16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
        315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
        4000, 5000, 6300, 8000, 10000, 12500, 16000
    ]

    /// Returns summed energy for each 1/3-octave band.
    /// Input `magnitudes` is squared (as returned by `Spectrum.magnitudes`).
    public static func analyze(
        magnitudes: [Float],
        sampleRate: Double,
        fftSize: Int
    ) -> [(center: Double, energy: Float)] {
        let freqPerBin = sampleRate / Double(fftSize)
        let halfBand = pow(2.0, 1.0 / 6.0) // ±1/6 octave around each center
        return magnitudes.withUnsafeBufferPointer { ptr in
            centerFrequencies.map { center in
                let lo = Int(max(0.0, (center / halfBand) / freqPerBin))
                let hi = Int(min(Double(magnitudes.count - 1), (center * halfBand) / freqPerBin))
                guard lo <= hi else { return (center: center, energy: 0) }
                var s: Float = 0
                vDSP_sve(ptr.baseAddress! + lo, 1, &s, vDSP_Length(hi - lo + 1))
                return (center: center, energy: s)
            }
        }
    }

    /// Returns only the energy values (same order as `centerFrequencies`).
    public static func energies(
        magnitudes: [Float],
        sampleRate: Double,
        fftSize: Int
    ) -> [Float] {
        analyze(magnitudes: magnitudes, sampleRate: sampleRate, fftSize: fftSize).map(\.energy)
    }
}
