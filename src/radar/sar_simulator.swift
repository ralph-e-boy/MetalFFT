// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

import Foundation

// ============================================================================
// SAR Point-Target Simulator
//
// Generates synthetic SAR raw data (complex baseband I/Q) for known point
// targets using a flat-earth broadside-looking geometry.
//
// Signal model:
//   Chirp: s(t) = exp(j * pi * Kr * t^2)
//   Range delay: tau = 2*R(eta)/c  where R(eta) is instantaneous slant range
//   Azimuth modulation: quadratic phase from changing range during aperture
//   Range cell migration: target range changes across azimuth pulses
// ============================================================================

struct SARParameters {
    // Range parameters
    let nRange: Int
    let nAzimuth: Int
    let bandwidth: Double = 100e6 // Chirp bandwidth (Hz)
    let pulseDuration: Double = 10e-6 // Pulse duration (s)
    let prf: Double = 1000.0 // Pulse repetition frequency (Hz)
    let samplingRate: Double // Range sampling rate (Hz)

    // Platform parameters
    let velocity: Double = 100.0 // Platform velocity (m/s)
    let carrierFreq: Double = 10e9 // Carrier frequency (Hz) — X-band
    let wavelength: Double // Carrier wavelength (m)

    // Derived
    let chirpRate: Double // Kr = B / Tp
    let rangeResolution: Double // delta_r = c / (2*B)
    let azimuthResolution: Double // delta_a = wavelength / (2 * beamwidthAngle) — approximate
    let c: Double = 299_792_458.0 // Speed of light (m/s)

    /// Scene geometry
    let centerRange: Double = 20000.0 // Scene center range (m)

    // Sampling
    let rangePixelSpacing: Double // c / (2 * fs)
    let azimuthPixelSpacing: Double // v / PRF

    init(nRange: Int = 4096, nAzimuth: Int = 4096) {
        self.nRange = nRange
        self.nAzimuth = nAzimuth
        wavelength = c / carrierFreq
        chirpRate = bandwidth / pulseDuration
        rangeResolution = c / (2.0 * bandwidth)
        samplingRate = bandwidth * 1.2 // 20% oversampling
        rangePixelSpacing = c / (2.0 * samplingRate)
        azimuthPixelSpacing = velocity / prf

        // Azimuth resolution depends on antenna length; for simplicity use
        // the synthetic aperture resolution: delta_a ≈ D/2 where D is antenna length
        // We'll use wavelength * R / (2 * L_sa) where L_sa is synthetic aperture length
        // For this simulation, just compute from parameters
        azimuthResolution = velocity / (2.0 * bandwidth) * (c / carrierFreq) * 1000
        // More precisely, azimuth resolution = v / (2 * Doppler_bandwidth)
        // We'll compute this properly in the metrics
    }
}

struct PointTarget {
    let rangePosition: Double // Slant range to target (m)
    let azimuthPosition: Double // Along-track position of target (m)
    let amplitude: Double // Reflectivity amplitude

    // Grid indices (computed after simulation)
    var expectedRangeBin: Int = 0
    var expectedAzimuthBin: Int = 0
}

class SARSimulator {
    let params: SARParameters
    var targets: [PointTarget]

    // Output: nAzimuth rows x nRange columns, row-major
    // rawData[azIdx * nRange + rgIdx] = complex sample
    var rawData: [SIMD2<Float>]

    init(params: SARParameters, targets: [PointTarget]) {
        self.params = params
        self.targets = targets
        rawData = [SIMD2<Float>](repeating: .zero, count: params.nAzimuth * params.nRange)
    }

    func simulate() {
        let p = params
        let Nr = p.nRange
        let Na = p.nAzimuth

        // Time axes
        let rangeTimeStart = p.centerRange * 2.0 / p.c - Double(Nr / 2) / p.samplingRate
        let azimuthTimeStart = -Double(Na / 2) / p.prf

        // Platform moves along azimuth (x-axis), targets are at (x_t, 0, 0) in ground plane
        // Platform at (v*eta, 0, h) but we use slant range directly for flat-earth

        print("SAR Simulator: Generating raw data for \(targets.count) point targets")
        print("  Range samples: \(Nr), Azimuth samples: \(Na)")
        print("  Center range: \(p.centerRange) m")
        print("  Chirp rate: \(String(format: "%.2e", p.chirpRate)) Hz/s")
        print("  Wavelength: \(String(format: "%.4f", p.wavelength)) m")
        print()

        for (tIdx, target) in targets.enumerated() {
            let R0 = target.rangePosition
            let x_t = target.azimuthPosition
            let amp = target.amplitude

            // Compute expected bin positions
            let expectedRgBin = Int((2.0 * R0 / p.c - rangeTimeStart) * p.samplingRate)
            let expectedAzBin = Int((x_t / p.velocity - azimuthTimeStart) * p.prf)
            targets[tIdx].expectedRangeBin = expectedRgBin
            targets[tIdx].expectedAzimuthBin = expectedAzBin

            print("  Target \(tIdx): range=\(R0) m, azimuth=\(x_t) m")
            print("    Expected bins: range=\(expectedRgBin), azimuth=\(expectedAzBin)")

            // For each azimuth pulse
            for azIdx in 0 ..< Na {
                let eta = azimuthTimeStart + Double(azIdx) / p.prf // Slow time
                let platformX = p.velocity * eta // Platform position

                // Instantaneous slant range (flat earth, broadside)
                let dx = platformX - x_t
                let R_eta = sqrt(R0 * R0 + dx * dx)

                // Range delay
                let tau0 = 2.0 * R_eta / p.c

                // For each range sample, compute received signal
                for rgIdx in 0 ..< Nr {
                    let tau = rangeTimeStart + Double(rgIdx) / p.samplingRate // Fast time
                    let dt = tau - tau0 // Time relative to echo arrival

                    // Check if within pulse duration
                    if abs(dt) <= p.pulseDuration / 2.0 {
                        // Chirp signal (baseband): exp(j * pi * Kr * dt^2)
                        // Phase from two-way path: exp(-j * 4*pi*R_eta/lambda)
                        let chirpPhase = Double.pi * p.chirpRate * dt * dt
                        let rangePhase = -4.0 * Double.pi * R_eta / p.wavelength
                        let totalPhase = chirpPhase + rangePhase

                        let real = Float(amp * cos(totalPhase))
                        let imag = Float(amp * sin(totalPhase))

                        rawData[azIdx * Nr + rgIdx].x += real
                        rawData[azIdx * Nr + rgIdx].y += imag
                    }
                }
            }
        }

        // Add noise
        addNoise(snrDB: 20.0)

        print()
        print("  Raw data generated: \(Na) x \(Nr) complex samples")
    }

    private func addNoise(snrDB: Double) {
        // Compute signal power
        var signalPower: Double = 0
        var signalCount = 0
        for i in 0 ..< rawData.count {
            let mag2 = Double(rawData[i].x * rawData[i].x + rawData[i].y * rawData[i].y)
            if mag2 > 0 {
                signalPower += mag2
                signalCount += 1
            }
        }
        guard signalCount > 0 else { return }
        signalPower /= Double(signalCount)

        let noisePower = signalPower / pow(10.0, snrDB / 10.0)
        let noiseStd = Float(sqrt(noisePower / 2.0)) // Per component (real, imag)

        // Box-Muller for Gaussian noise
        for i in stride(from: 0, to: rawData.count - 1, by: 2) {
            let u1 = max(Float.random(in: 0 ..< 1), 1e-10)
            let u2 = Float.random(in: 0 ..< 1)
            let r = noiseStd * sqrt(-2.0 * log(u1))
            let theta = 2.0 * Float.pi * u2
            let n1 = r * cos(theta)
            let n2 = r * sin(theta)
            rawData[i].x += n1
            rawData[i].y += n2
            if i + 1 < rawData.count {
                let u3 = max(Float.random(in: 0 ..< 1), 1e-10)
                let u4 = Float.random(in: 0 ..< 1)
                let r2 = noiseStd * sqrt(-2.0 * log(u3))
                let theta2 = 2.0 * Float.pi * u4
                rawData[i + 1].x += r2 * cos(theta2)
                rawData[i + 1].y += r2 * sin(theta2)
            }
        }
    }

    /// Generate the range chirp reference signal (for matched filtering)
    func generateChirpReference() -> [SIMD2<Float>] {
        let Nr = params.nRange
        var chirpRef = [SIMD2<Float>](repeating: .zero, count: Nr)

        let nSamples = Int(params.pulseDuration * params.samplingRate)
        let halfSamples = nSamples / 2

        for i in 0 ..< nSamples {
            let t = Double(i - halfSamples) / params.samplingRate
            let phase = Double.pi * params.chirpRate * t * t
            let idx = i < halfSamples ? (Nr - halfSamples + i) : (i - halfSamples)
            if idx < Nr {
                chirpRef[idx] = SIMD2<Float>(Float(cos(phase)), Float(sin(phase)))
            }
        }

        return chirpRef
    }

    /// Generate default point targets for testing
    static func defaultTargets(params: SARParameters) -> [PointTarget] {
        let R0 = params.centerRange
        let dr = params.rangeResolution * 50 // Spacing in range
        let da = params.azimuthPixelSpacing * 50 // Spacing in azimuth

        return [
            // Center target
            PointTarget(rangePosition: R0, azimuthPosition: 0, amplitude: 1.0),
            // Offset in range only
            PointTarget(rangePosition: R0 + dr, azimuthPosition: 0, amplitude: 1.0),
            // Offset in azimuth only
            PointTarget(rangePosition: R0, azimuthPosition: da, amplitude: 1.0),
            // Offset in both
            PointTarget(rangePosition: R0 - dr, azimuthPosition: -da, amplitude: 1.0),
            // Far offset
            PointTarget(rangePosition: R0 + 2 * dr, azimuthPosition: -2 * da, amplitude: 0.8)
        ]
    }
}
