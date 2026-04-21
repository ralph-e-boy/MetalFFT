// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

import Foundation

// ============================================================================
// SAR Image Quality Metrics
//
// Measures point-target response quality in a focused SAR image:
//   - PSLR (Peak Sidelobe Ratio): highest sidelobe / mainlobe peak, in dB
//   - ISLR (Integrated Sidelobe Ratio): sidelobe energy / mainlobe energy, in dB
//   - SNR: signal-to-noise ratio
//   - 3 dB resolution: mainlobe -3dB width in range and azimuth
// ============================================================================

struct TargetMetrics {
    let targetIndex: Int
    let expectedRangeBin: Int
    let expectedAzimuthBin: Int
    let measuredRangeBin: Int
    let measuredAzimuthBin: Int
    let peakMagnitude: Float
    let pslrRange: Float // dB
    let pslrAzimuth: Float // dB
    let islrRange: Float // dB
    let islrAzimuth: Float // dB
    let snr: Float // dB
    let resolution3dBRange: Float // bins
    let resolution3dBAzimuth: Float // bins
}

class RadarMetrics {
    let nRange: Int
    let nAzimuth: Int
    let image: [SIMD2<Float>] // Complex focused image (nAzimuth x nRange)

    /// Magnitude-squared image
    let magImage: [Float]

    init(image: [SIMD2<Float>], nRange: Int, nAzimuth: Int) {
        self.image = image
        self.nRange = nRange
        self.nAzimuth = nAzimuth

        // Compute magnitude-squared
        var mag = [Float](repeating: 0, count: image.count)
        for i in 0 ..< image.count {
            mag[i] = image[i].x * image[i].x + image[i].y * image[i].y
        }
        magImage = mag
    }

    /// Measure metrics for a target near the expected position
    func measureTarget(index: Int, expectedRange: Int, expectedAzimuth: Int,
                       searchRadius: Int = 20) -> TargetMetrics {
        // Find peak near expected position
        let (peakRg, peakAz, peakVal) = findPeak(
            nearRange: expectedRange, nearAzimuth: expectedAzimuth, radius: searchRadius
        )

        // Extract range and azimuth cuts through the peak
        let rangeCut = extractRangeCut(azimuthBin: peakAz)
        let azimuthCut = extractAzimuthCut(rangeBin: peakRg)

        // Measure metrics on each cut
        let (pslrR, islrR, res3dbR) = measureCutMetrics(cut: rangeCut, peakBin: peakRg)
        let (pslrA, islrA, res3dbA) = measureCutMetrics(cut: azimuthCut, peakBin: peakAz)

        // SNR: peak power / mean noise power
        let snr = measureSNR(peakRange: peakRg, peakAzimuth: peakAz, peakValue: peakVal)

        return TargetMetrics(
            targetIndex: index,
            expectedRangeBin: expectedRange,
            expectedAzimuthBin: expectedAzimuth,
            measuredRangeBin: peakRg,
            measuredAzimuthBin: peakAz,
            peakMagnitude: peakVal,
            pslrRange: pslrR,
            pslrAzimuth: pslrA,
            islrRange: islrR,
            islrAzimuth: islrA,
            snr: snr,
            resolution3dBRange: res3dbR,
            resolution3dBAzimuth: res3dbA
        )
    }

    /// Find the magnitude peak near a given position
    private func findPeak(nearRange: Int, nearAzimuth: Int, radius: Int)
        -> (rangeBin: Int, azimuthBin: Int, magnitude: Float) {
        var bestRg = nearRange
        var bestAz = nearAzimuth
        var bestVal: Float = -1

        let rgStart = max(0, nearRange - radius)
        let rgEnd = min(nRange - 1, nearRange + radius)
        let azStart = max(0, nearAzimuth - radius)
        let azEnd = min(nAzimuth - 1, nearAzimuth + radius)

        for az in azStart ... azEnd {
            for rg in rgStart ... rgEnd {
                let val = magImage[az * nRange + rg]
                if val > bestVal {
                    bestVal = val
                    bestRg = rg
                    bestAz = az
                }
            }
        }

        return (bestRg, bestAz, bestVal)
    }

    /// Extract a range cut (row) at a given azimuth bin
    private func extractRangeCut(azimuthBin: Int) -> [Float] {
        var cut = [Float](repeating: 0, count: nRange)
        for rg in 0 ..< nRange {
            cut[rg] = magImage[azimuthBin * nRange + rg]
        }
        return cut
    }

    /// Extract an azimuth cut (column) at a given range bin
    private func extractAzimuthCut(rangeBin: Int) -> [Float] {
        var cut = [Float](repeating: 0, count: nAzimuth)
        for az in 0 ..< nAzimuth {
            cut[az] = magImage[az * nRange + rangeBin]
        }
        return cut
    }

    /// Measure PSLR, ISLR, and 3dB resolution on a 1D magnitude-squared cut
    private func measureCutMetrics(cut: [Float], peakBin: Int)
        -> (pslr: Float, islr: Float, resolution3dB: Float) {
        let n = cut.count
        let peakVal = cut[peakBin]

        guard peakVal > 0 else {
            return (0, 0, Float(n))
        }

        let peakDB = 10 * log10(peakVal)

        // Find mainlobe extent: region around peak where magnitude stays within
        // a threshold (we use first nulls or -20 dB width for mainlobe definition)
        let mainlobeThreshDB: Float = -20.0 // Define mainlobe as region above -20dB from peak

        // Search left from peak
        var mainlobeLeft = peakBin
        for i in stride(from: peakBin - 1, through: 0, by: -1) {
            let valDB = cut[i] > 0 ? 10 * log10(cut[i]) - peakDB : -100
            if valDB < mainlobeThreshDB {
                mainlobeLeft = i + 1
                break
            }
            if i == 0 { mainlobeLeft = 0 }
        }

        // Search right from peak
        var mainlobeRight = peakBin
        for i in (peakBin + 1) ..< n {
            let valDB = cut[i] > 0 ? 10 * log10(cut[i]) - peakDB : -100
            if valDB < mainlobeThreshDB {
                mainlobeRight = i - 1
                break
            }
            if i == n - 1 { mainlobeRight = n - 1 }
        }

        // Mainlobe energy
        var mainlobeEnergy: Float = 0
        for i in mainlobeLeft ... mainlobeRight {
            mainlobeEnergy += cut[i]
        }

        // Sidelobe: everything outside mainlobe within a reasonable window
        let windowHalf = min(n / 4, 200)
        let windowLeft = max(0, peakBin - windowHalf)
        let windowRight = min(n - 1, peakBin + windowHalf)

        var sidelobeEnergy: Float = 0
        var maxSidelobe: Float = 0

        for i in windowLeft ... windowRight {
            if i < mainlobeLeft || i > mainlobeRight {
                sidelobeEnergy += cut[i]
                if cut[i] > maxSidelobe {
                    maxSidelobe = cut[i]
                }
            }
        }

        // PSLR: peak sidelobe / mainlobe peak (in power, convert to dB)
        let pslr: Float = maxSidelobe > 0 ? 10 * log10(maxSidelobe / peakVal) : -60

        // ISLR: integrated sidelobe / integrated mainlobe (in power, dB)
        let islr: Float = (sidelobeEnergy > 0 && mainlobeEnergy > 0)
            ? 10 * log10(sidelobeEnergy / mainlobeEnergy) : -60

        // 3 dB resolution: width at -3 dB from peak
        let thresh3dB = peakVal * 0.5 // -3 dB in power

        // Find -3dB crossings by linear interpolation
        var left3dB = Float(peakBin)
        for i in stride(from: peakBin, through: max(0, peakBin - windowHalf), by: -1) {
            if cut[i] < thresh3dB {
                // Interpolate
                if i + 1 < n, cut[i + 1] > thresh3dB {
                    let frac = (thresh3dB - cut[i]) / (cut[i + 1] - cut[i])
                    left3dB = Float(i) + frac
                } else {
                    left3dB = Float(i)
                }
                break
            }
        }

        var right3dB = Float(peakBin)
        for i in peakBin ..< min(n, peakBin + windowHalf) {
            if cut[i] < thresh3dB {
                if i > 0, cut[i - 1] > thresh3dB {
                    let frac = (thresh3dB - cut[i]) / (cut[i - 1] - cut[i])
                    right3dB = Float(i) - frac
                } else {
                    right3dB = Float(i)
                }
                break
            }
        }

        let resolution3dB = right3dB - left3dB

        return (pslr, islr, max(resolution3dB, 1.0))
    }

    // SNR: peak power / mean noise floor (estimated from image corners)
    private func measureSNR(peakRange: Int, peakAzimuth: Int, peakValue: Float) -> Float {
        // Sample noise from corners of the image (away from targets)
        let cornerSize = 64
        var noiseSum: Float = 0
        var noiseCount = 0

        // Top-left corner
        for az in 0 ..< cornerSize {
            for rg in 0 ..< cornerSize {
                noiseSum += magImage[az * nRange + rg]
                noiseCount += 1
            }
        }
        // Bottom-right corner
        for az in (nAzimuth - cornerSize) ..< nAzimuth {
            for rg in (nRange - cornerSize) ..< nRange {
                noiseSum += magImage[az * nRange + rg]
                noiseCount += 1
            }
        }

        guard noiseCount > 0 else { return 0 }
        let noiseMean = noiseSum / Float(noiseCount)

        guard noiseMean > 0 else { return 60 }
        return 10 * log10(peakValue / noiseMean)
    }

    /// Print formatted report
    static func printReport(metrics: [TargetMetrics], params: SARParameters) {
        print()
        print(String(repeating: "=", count: 80))
        print("SAR Image Quality Report")
        print(String(repeating: "=", count: 80))
        print()
        print("Target  Exp(rg,az)    Meas(rg,az)   PSLR_r(dB) PSLR_a(dB) ISLR_r  ISLR_a   SNR")
        print(String(repeating: "-", count: 80))

        for m in metrics {
            let line = "  #\(m.targetIndex)"
                + "    (\(m.expectedRangeBin),\(m.expectedAzimuthBin))"
                + "   (\(m.measuredRangeBin),\(m.measuredAzimuthBin))"
                + "     \(String(format: "%6.1f", m.pslrRange))"
                + "      \(String(format: "%6.1f", m.pslrAzimuth))"
                + "    \(String(format: "%6.1f", m.islrRange))"
                + "  \(String(format: "%6.1f", m.islrAzimuth))"
                + "  \(String(format: "%5.1f", m.snr))"
            print(line)
        }

        print()
        print("Resolution (3 dB width):")
        for m in metrics {
            let rangeResM = Float(params.rangePixelSpacing) * m.resolution3dBRange
            let azResM = Float(params.azimuthPixelSpacing) * m.resolution3dBAzimuth
            print("  Target #\(m.targetIndex): Range = \(String(format: "%.2f", m.resolution3dBRange)) bins (\(String(format: "%.2f", rangeResM)) m), Azimuth = \(String(format: "%.2f", m.resolution3dBAzimuth)) bins (\(String(format: "%.2f", azResM)) m)")
        }

        print()
        print("Expected PSLR for uniform weighting: -13.3 dB (sinc sidelobes)")
        print("Expected ISLR for uniform weighting: ~-10 dB")
        print(String(repeating: "=", count: 80))
    }
}
