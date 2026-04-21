// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

import Accelerate
import Foundation

// ============================================================================
// vDSP FFT Baseline Benchmark
//
// Measures vDSP_fft_zop for sizes 256..16384, batch sizes 1..1024.
// Reports wall-clock time and GFLOPS (5 * N * log2(N) FLOPs per complex FFT).
// ============================================================================

struct VDSPBenchResult {
    let fftSize: Int
    let batchSize: Int
    let totalTimeUs: Double
    let timePerFFTUs: Double
    let gflops: Double
}

func benchmarkVDSP(fftSize: Int, batchSize: Int, warmup: Int = 5, repeats: Int = 20) -> VDSPBenchResult {
    let log2n = vDSP_Length(Int(log2(Double(fftSize))))
    guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
        fatalError("vDSP_create_fftsetup failed for size \(fftSize)")
    }
    defer { vDSP_destroy_fftsetup(fftSetup) }

    let totalCount = fftSize * batchSize
    let realIn = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
    let imagIn = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
    let realOut = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
    let imagOut = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
    defer {
        realIn.deallocate()
        imagIn.deallocate()
        realOut.deallocate()
        imagOut.deallocate()
    }

    for i in 0 ..< totalCount {
        realIn[i] = Float.random(in: -1 ... 1)
        imagIn[i] = Float.random(in: -1 ... 1)
    }

    /// Helper to run one batch
    func runBatch() {
        for b in 0 ..< batchSize {
            let offset = b * fftSize
            var splitIn = DSPSplitComplex(realp: realIn + offset, imagp: imagIn + offset)
            var splitOut = DSPSplitComplex(realp: realOut + offset, imagp: imagOut + offset)
            vDSP_fft_zop(fftSetup, &splitIn, 1, &splitOut, 1, log2n, FFTDirection(kFFTDirection_Forward))
        }
    }

    // Warmup
    for _ in 0 ..< warmup {
        runBatch()
    }

    // Get timebase info once
    var info = mach_timebase_info_data_t()
    mach_timebase_info(&info)
    let ticksToUs = Double(info.numer) / Double(info.denom) / 1000.0

    // Timed runs
    var times = [Double](repeating: 0, count: repeats)
    for r in 0 ..< repeats {
        let start = mach_absolute_time()
        runBatch()
        let end = mach_absolute_time()
        times[r] = Double(end - start) * ticksToUs
    }

    times.sort()
    let medianUs = times[times.count / 2]

    let flopsPerFFT = 5.0 * Double(fftSize) * log2(Double(fftSize))
    let totalFlops = flopsPerFFT * Double(batchSize)
    let gflops = totalFlops / (medianUs * 1e-6) / 1e9

    return VDSPBenchResult(
        fftSize: fftSize,
        batchSize: batchSize,
        totalTimeUs: medianUs,
        timePerFFTUs: medianUs / Double(batchSize),
        gflops: gflops
    )
}

// ============================================================================
// Main
// ============================================================================

func runVDSPBaseline() {
    print("=" * 72)
    print("vDSP FFT Baseline Benchmark (vDSP_fft_zop)")
    print("=" * 72)
    print()

    let fftSizes = [256, 512, 1024, 2048, 4096, 8192, 16384]
    let batchSizes = [1, 16, 64, 256, 1024]

    // Header
    print("  N       ", terminator: "")
    for bs in batchSizes {
        print(String(format: "  | Batch=%-4d           ", bs), terminator: "")
    }
    print()
    print("          ", terminator: "")
    for _ in batchSizes {
        print("  |   us/FFT   GFLOPS  ", terminator: "")
    }
    print()
    print("  " + String(repeating: "-", count: 8 + batchSizes.count * 24))

    for n in fftSizes {
        print(String(format: "  %-8d", n), terminator: "")
        for bs in batchSizes {
            let result = benchmarkVDSP(fftSize: n, batchSize: bs)
            print(String(format: "  | %8.2f %8.2f  ", result.timePerFFTUs, result.gflops), terminator: "")
        }
        print()
    }

    print()

    // Detailed report for N=4096 (our target)
    print("-- Detailed: N=4096 --")
    print()
    print("  Batch     Total (us)   Per FFT (us)       GFLOPS")
    print("  " + String(repeating: "-", count: 52))
    for bs in batchSizes {
        let r = benchmarkVDSP(fftSize: 4096, batchSize: bs)
        print(String(format: "  %-8d  %12.2f  %12.2f  %12.2f", r.batchSize, r.totalTimeUs, r.timePerFFTUs, r.gflops))
    }
    print()

    print("=" * 72)
    print("Notes:")
    print("  FLOP count: 5 * N * log2(N) per complex FFT (standard estimate)")
    print("  Times are median of 20 trials after 5 warmup iterations")
    print("  vDSP_fft_zop: out-of-place, forward FFT, split complex format")
    print("=" * 72)
}
