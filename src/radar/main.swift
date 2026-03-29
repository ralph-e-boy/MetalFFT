// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

import Foundation

// ============================================================================
// SAR Simulation + Range Doppler Algorithm Pipeline
//
// End-to-end: simulate point targets -> RDA processing -> quality metrics
//
// Usage:
//   SARRadar [size]              — Run unfused baseline (default size=4096)
//   SARRadar [size] --fused      — Run both fused and unfused, compare results
//   SARRadar [size] --precision  — Run FP16 precision comparison (all modes)
// ============================================================================

// Flush output after every print for debugging
setbuf(stdout, nil)

print(String(repeating: "=", count: 80))
print("SAR Point-Target Simulation & Range Doppler Algorithm Processing")
print(String(repeating: "=", count: 80))
print()

// --- Configuration ---
var fftSize: Int = 4096
var runFused = false
var runPrecision = false

for arg in CommandLine.arguments.dropFirst() {
    if arg == "--fused" {
        runFused = true
    } else if arg == "--precision" {
        runPrecision = true
    } else if let sz = Int(arg) {
        fftSize = sz
    }
}

let params = SARParameters(nRange: fftSize, nAzimuth: fftSize)
print("SAR Parameters:")
print("  Range samples:    \(params.nRange)")
print("  Azimuth samples:  \(params.nAzimuth)")
print("  Bandwidth:        \(params.bandwidth / 1e6) MHz")
print("  Pulse duration:   \(params.pulseDuration * 1e6) us")
print("  PRF:              \(params.prf) Hz")
print("  Platform velocity: \(params.velocity) m/s")
print("  Carrier frequency: \(params.carrierFreq / 1e9) GHz")
print("  Wavelength:       \(String(format: "%.4f", params.wavelength)) m")
print("  Range resolution: \(String(format: "%.2f", params.rangeResolution)) m")
print("  Range pixel:      \(String(format: "%.2f", params.rangePixelSpacing)) m")
print("  Azimuth pixel:    \(String(format: "%.4f", params.azimuthPixelSpacing)) m")
print("  Center range:     \(params.centerRange) m")
if runFused {
    print("  Mode:             FUSED vs UNFUSED comparison")
}
if runPrecision {
    print("  Mode:             FP16 PRECISION COMPARISON")
}
print()

// --- Generate point targets ---
var targets = SARSimulator.defaultTargets(params: params)
let simulator = SARSimulator(params: params, targets: targets)

let simStart = CFAbsoluteTimeGetCurrent()
simulator.simulate()
let simTime = CFAbsoluteTimeGetCurrent() - simStart
targets = simulator.targets  // Update with computed bin positions
print(String(format: "  Simulation time: %.2f seconds", simTime))
print()

// --- Generate chirp reference ---
let chirpRef = simulator.generateChirpReference()

// --- RDA Processing ---
print(String(repeating: "-", count: 80))
print("Range Doppler Algorithm Processing")
print(String(repeating: "-", count: 80))

do {
    // Always run unfused baseline
    let pipeline = try RDAPipeline(params: params)

    let rdaStart = CFAbsoluteTimeGetCurrent()
    let focusedImage = pipeline.process(rawData: simulator.rawData, chirpRef: chirpRef)
    let rdaTime = CFAbsoluteTimeGetCurrent() - rdaStart
    print(String(format: "\n  Unfused RDA processing time: %.2f seconds", rdaTime))

    // --- Quality Metrics (unfused) ---
    print()
    print(String(repeating: "-", count: 80))
    print("Measuring Image Quality Metrics (Unfused)")
    print(String(repeating: "-", count: 80))

    let metrics = RadarMetrics(image: focusedImage, nRange: params.nRange, nAzimuth: params.nAzimuth)

    var targetMetrics: [TargetMetrics] = []
    for (i, target) in targets.enumerated() {
        print("  Measuring target \(i) at expected (\(target.expectedRangeBin), \(target.expectedAzimuthBin))...")
        let m = metrics.measureTarget(
            index: i,
            expectedRange: target.expectedRangeBin,
            expectedAzimuth: target.expectedAzimuthBin,
            searchRadius: 30
        )
        print("    Peak at (\(m.measuredRangeBin), \(m.measuredAzimuthBin)), SNR=\(String(format: "%.1f", m.snr)) dB")
        targetMetrics.append(m)
    }

    RadarMetrics.printReport(metrics: targetMetrics, params: params)

    // --- Fused pipeline ---
    if runFused {
        print()
        print(String(repeating: "=", count: 80))
        print("Running FUSED Pipeline")
        print(String(repeating: "=", count: 80))

        let fusedPipeline = try RDAFusedPipeline(params: params)

        let fusedStart = CFAbsoluteTimeGetCurrent()
        let fusedImage = fusedPipeline.process(rawData: simulator.rawData, chirpRef: chirpRef)
        let fusedTime = CFAbsoluteTimeGetCurrent() - fusedStart
        print(String(format: "\n  Fused RDA processing time: %.2f seconds", fusedTime))

        fusedPipeline.printTimingBreakdown()

        // --- Quality Metrics (fused) ---
        print()
        print(String(repeating: "-", count: 80))
        print("Measuring Image Quality Metrics (Fused)")
        print(String(repeating: "-", count: 80))

        let fusedMetrics = RadarMetrics(image: fusedImage, nRange: params.nRange, nAzimuth: params.nAzimuth)

        var fusedTargetMetrics: [TargetMetrics] = []
        for (i, target) in targets.enumerated() {
            print("  Measuring target \(i) at expected (\(target.expectedRangeBin), \(target.expectedAzimuthBin))...")
            let m = fusedMetrics.measureTarget(
                index: i,
                expectedRange: target.expectedRangeBin,
                expectedAzimuth: target.expectedAzimuthBin,
                searchRadius: 30
            )
            print("    Peak at (\(m.measuredRangeBin), \(m.measuredAzimuthBin)), SNR=\(String(format: "%.1f", m.snr)) dB")
            fusedTargetMetrics.append(m)
        }

        RadarMetrics.printReport(metrics: fusedTargetMetrics, params: params)

        // --- Validation: compare fused vs unfused output ---
        print()
        print(String(repeating: "=", count: 80))
        print("Validation: Fused vs Unfused Comparison")
        print(String(repeating: "=", count: 80))

        // L2 relative error
        var diffSqSum: Double = 0
        var refSqSum: Double = 0
        for i in 0..<focusedImage.count {
            let dx = Double(fusedImage[i].x - focusedImage[i].x)
            let dy = Double(fusedImage[i].y - focusedImage[i].y)
            diffSqSum += dx * dx + dy * dy
            refSqSum += Double(focusedImage[i].x * focusedImage[i].x + focusedImage[i].y * focusedImage[i].y)
        }
        let l2RelError = refSqSum > 0 ? sqrt(diffSqSum / refSqSum) : 0
        print(String(format: "  L2 relative error: %.2e", l2RelError))

        let errorThreshold = 1e-5
        if l2RelError < errorThreshold {
            print("  PASS: Fused output matches unfused (error < \(errorThreshold))")
        } else if l2RelError < 1e-3 {
            print("  ACCEPTABLE: Error above threshold but within float32 tolerance")
        } else {
            print("  WARNING: Significant difference between fused and unfused output")
        }

        // Per-target comparison
        print()
        print("  Per-target SNR comparison:")
        print("  Target    Unfused SNR    Fused SNR    Delta")
        print("  " + String(repeating: "-", count: 50))
        for i in 0..<targetMetrics.count {
            let snrU = targetMetrics[i].snr
            let snrF = fusedTargetMetrics[i].snr
            let delta = snrF - snrU
            print(String(format: "    #%d       %6.1f dB     %6.1f dB    %+.1f dB",
                         i, snrU, snrF, delta))
        }

        // --- Summary ---
        print()
        print(String(repeating: "=", count: 80))
        print("Pipeline Comparison Summary")
        print(String(repeating: "=", count: 80))
        print(String(format: "  Data size:        %d x %d = %.1f M complex samples",
                     params.nAzimuth, params.nRange,
                     Double(params.nAzimuth * params.nRange) / 1e6))
        print(String(format: "  Simulation:       %.2f s", simTime))
        print(String(format: "  Unfused RDA:      %.2f s", rdaTime))
        print(String(format: "  Fused RDA:        %.2f s", fusedTime))
        let speedup = rdaTime / fusedTime
        print(String(format: "  Speedup:          %.2fx", speedup))
        print(String(format: "  L2 relative error: %.2e", l2RelError))
        print()

        let allFocusedFused = fusedTargetMetrics.allSatisfy { $0.snr > 10 }
        if allFocusedFused {
            print("  STATUS: All targets detected with SNR > 10 dB (fused pipeline)")
        } else {
            print("  STATUS: Some targets may not be well-focused (fused pipeline)")
        }
        print(String(repeating: "=", count: 80))

    } else {
        // --- Summary (unfused only) ---
        print()
        print(String(repeating: "=", count: 80))
        print("Pipeline Summary")
        print(String(repeating: "=", count: 80))
        print(String(format: "  Data size:        %d x %d = %.1f M complex samples",
                     params.nAzimuth, params.nRange,
                     Double(params.nAzimuth * params.nRange) / 1e6))
        print(String(format: "  Simulation:       %.2f s", simTime))
        print(String(format: "  RDA processing:   %.2f s", rdaTime))
        print(String(format: "  Total:            %.2f s", simTime + rdaTime))
        print()

        let allFocused = targetMetrics.allSatisfy { $0.snr > 10 }
        if allFocused {
            print("  STATUS: All targets detected with SNR > 10 dB")
        } else {
            print("  STATUS: Some targets may not be well-focused (SNR < 10 dB)")
        }
        print(String(repeating: "=", count: 80))
    }

    // --- FP16 Precision Comparison ---
    if runPrecision {
        print()
        print(String(repeating: "=", count: 80))
        print("FP16 Precision Comparison: Mixed-Precision SAR FFT Analysis")
        print(String(repeating: "=", count: 80))

        let fp16Pipeline = try RDAFusedFP16Pipeline(params: params)

        let precisionResults = fp16Pipeline.runComparison(
            rawData: simulator.rawData,
            chirpRef: chirpRef,
            targets: targets
        )

        RDAFusedFP16Pipeline.printComparisonReport(results: precisionResults, params: params)
    }

} catch {
    print("ERROR: \(error)")
}
