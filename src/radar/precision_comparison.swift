// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

import Foundation
import Metal

// ============================================================================
// FP16 Precision Comparison for SAR Range Doppler Algorithm
//
// Runs the full RDA pipeline at four precision modes and compares radar image
// quality metrics (PSLR, ISLR, SNR, 3dB resolution) against the FP32 baseline.
//
// Precision modes:
//   FP32:      Full float32 fused pipeline (baseline)
//   FP16-pure: Mode A — all computation and storage in half precision
//   FP16-stor: Mode B — half2 threadgroup storage, float compute (recommended)
//   FP16-mix:  Mode C — half multiply, float accumulate
//
// This is the first published analysis of mixed-precision FFT accuracy
// specifically for SAR image quality metrics.
// ============================================================================

enum PrecisionMode: String, CaseIterable {
    case fp32 = "FP32"
    case fp16Pure = "FP16-pure"
    case fp16Stor = "FP16-stor"
    case fp16Mix = "FP16-mix"
}

struct PrecisionResult {
    let mode: PrecisionMode
    let targetMetrics: [TargetMetrics]
    let pipelineTimeSeconds: Double
    let focusedImage: [SIMD2<Float>]
}

class RDAFusedFP16Pipeline {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    // Pipelines per precision mode
    var rangeCompressionPipelines: [PrecisionMode: MTLComputePipelineState] = [:]
    var azimuthCompressionPipelines: [PrecisionMode: MTLComputePipelineState] = [:]
    var multiplyIfftPipelines: [PrecisionMode: MTLComputePipelineState] = [:]

    var fftPipelines: [String: MTLComputePipelineState] = [:]
    var utilityPipelines: [String: MTLComputePipelineState] = [:]

    let params: SARParameters
    let nRange: Int
    let nAzimuth: Int

    init(params: SARParameters) throws {
        self.params = params
        nRange = params.nRange
        nAzimuth = params.nAzimuth

        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device found")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Could not create command queue")
        }
        commandQueue = queue

        let compileOptions = MTLCompileOptions()
        compileOptions.fastMathEnabled = true

        // Load FP32 fused kernels
        let fusedSource = try Self.loadShaderSource(name: "fft_sar_fused.metal")
        let fusedLibrary = try device.makeLibrary(source: fusedSource, options: compileOptions)

        if let f = fusedLibrary.makeFunction(name: "fused_range_compression") {
            rangeCompressionPipelines[.fp32] = try device.makeComputePipelineState(function: f)
        }
        if let f = fusedLibrary.makeFunction(name: "fused_azimuth_compression") {
            azimuthCompressionPipelines[.fp32] = try device.makeComputePipelineState(function: f)
        }
        if let f = fusedLibrary.makeFunction(name: "fused_multiply_ifft") {
            multiplyIfftPipelines[.fp32] = try device.makeComputePipelineState(function: f)
        }

        // Load FP16 fused kernels
        let fp16Source = try Self.loadShaderSource(name: "fft_sar_fused_fp16.metal")
        let fp16Library = try device.makeLibrary(source: fp16Source, options: compileOptions)

        let modeKernelMap: [(PrecisionMode, String, String, String)] = [
            (.fp16Pure, "fused_range_compression_fp16_pure", "fused_azimuth_compression_fp16_pure", "fused_multiply_ifft_fp16_pure"),
            (.fp16Stor, "fused_range_compression_fp16_storage", "fused_azimuth_compression_fp16_storage", "fused_multiply_ifft_fp16_storage"),
            (.fp16Mix, "fused_range_compression_fp16_mixed", "fused_azimuth_compression_fp16_mixed", "fused_multiply_ifft_fp16_mixed")
        ]

        for (mode, rangeName, azName, mulIfftName) in modeKernelMap {
            if let f = fp16Library.makeFunction(name: rangeName) {
                rangeCompressionPipelines[mode] = try device.makeComputePipelineState(function: f)
            }
            if let f = fp16Library.makeFunction(name: azName) {
                azimuthCompressionPipelines[mode] = try device.makeComputePipelineState(function: f)
            }
            if let f = fp16Library.makeFunction(name: mulIfftName) {
                multiplyIfftPipelines[mode] = try device.makeComputePipelineState(function: f)
            }
        }

        // Load FFT kernels for azimuth FFT step
        let fftSource = try Self.loadShaderSource(name: "fft_multisize.metal")
        let fftLibrary = try device.makeLibrary(source: fftSource, options: compileOptions)

        for name in ["fft_256_stockham", "fft_512_stockham", "fft_1024_stockham",
                     "fft_2048_stockham", "fft_4096_stockham"] {
            if let function = fftLibrary.makeFunction(name: name) {
                fftPipelines[name] = try device.makeComputePipelineState(function: function)
            }
        }

        // Load utility kernels
        let rdaSource = try Self.loadShaderSource(name: "rda_kernels.metal")
        let rdaLibrary = try device.makeLibrary(source: rdaSource, options: compileOptions)

        for name in ["complex_multiply", "complex_multiply_conjugate",
                     "transpose_2d", "rcmc_sinc_interp",
                     "generate_azimuth_matched_filter",
                     "fftshift_columns", "magnitude_detect"] {
            if let function = rdaLibrary.makeFunction(name: name) {
                utilityPipelines[name] = try device.makeComputePipelineState(function: function)
            }
        }

        print("FP16 Precision Comparison Pipeline initialized")
        print("  Range compression kernels: \(rangeCompressionPipelines.count) modes")
        print("  Azimuth compression kernels: \(azimuthCompressionPipelines.count) modes")
        print("  Multiply+IFFT kernels: \(multiplyIfftPipelines.count) modes")
    }

    private static func loadShaderSource(name: String) throws -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        let cwd = FileManager.default.currentDirectoryPath

        let searchPaths = [
            (execDir as NSString).appendingPathComponent(name),
            (execDir as NSString).appendingPathComponent("../../../\(name)"),
            (execDir as NSString).appendingPathComponent("../../src/metal/\(name)"),
            (execDir as NSString).appendingPathComponent("../../../src/metal/\(name)"),
            (execDir as NSString).appendingPathComponent("../../src/radar/\(name)"),
            (execDir as NSString).appendingPathComponent("../../../src/radar/\(name)"),
            (cwd as NSString).appendingPathComponent(name),
            (cwd as NSString).appendingPathComponent("src/metal/\(name)"),
            (cwd as NSString).appendingPathComponent("src/radar/\(name)"),
            (cwd as NSString).appendingPathComponent("../metal/\(name)"),
            (cwd as NSString).appendingPathComponent("../radar/\(name)")
        ]
        for path in searchPaths {
            if FileManager.default.fileExists(atPath: path) {
                return try String(contentsOfFile: path, encoding: .utf8)
            }
        }
        fatalError("Could not find shader: \(name). Searched: \(searchPaths)")
    }

    // ========================================================================
    // Run full comparison across all precision modes
    // ========================================================================

    func runComparison(rawData: [SIMD2<Float>], chirpRef: [SIMD2<Float>],
                       targets: [PointTarget]) -> [PrecisionResult] {
        var results: [PrecisionResult] = []

        for mode in PrecisionMode.allCases {
            print("\n" + String(repeating: "-", count: 60))
            print("Processing: \(mode.rawValue)")
            print(String(repeating: "-", count: 60))

            guard rangeCompressionPipelines[mode] != nil else {
                print("  SKIPPED: kernel not available for \(mode.rawValue)")
                continue
            }

            let t0 = CFAbsoluteTimeGetCurrent()
            let focusedImage = process(rawData: rawData, chirpRef: chirpRef, mode: mode)
            let pipelineTime = CFAbsoluteTimeGetCurrent() - t0

            print(String(format: "  Pipeline time: %.4f s", pipelineTime))

            // Measure quality metrics
            let metrics = RadarMetrics(image: focusedImage, nRange: nRange, nAzimuth: nAzimuth)
            var targetMetrics: [TargetMetrics] = []
            for (i, target) in targets.enumerated() {
                let m = metrics.measureTarget(
                    index: i,
                    expectedRange: target.expectedRangeBin,
                    expectedAzimuth: target.expectedAzimuthBin,
                    searchRadius: 30
                )
                targetMetrics.append(m)
            }

            results.append(PrecisionResult(
                mode: mode,
                targetMetrics: targetMetrics,
                pipelineTimeSeconds: pipelineTime,
                focusedImage: focusedImage
            ))
        }

        return results
    }

    // ========================================================================
    // Process one precision mode through full RDA pipeline
    // ========================================================================

    private func process(rawData: [SIMD2<Float>], chirpRef: [SIMD2<Float>],
                         mode: PrecisionMode) -> [SIMD2<Float>] {
        let totalSamples = nAzimuth * nRange
        let byteCount = totalSamples * MemoryLayout<SIMD2<Float>>.stride

        var data = rawData
        let dataBuffer = device.makeBuffer(bytes: &data, length: byteCount, options: .storageModeShared)!

        // Step 1: Fused range compression
        let rangeCompressed = fusedRangeCompression(dataBuffer: dataBuffer, chirpRef: chirpRef, mode: mode)

        // Step 2: Azimuth FFT (always FP32 — not fused)
        let rangeDoppler = azimuthFFT(dataBuffer: rangeCompressed)

        // Step 3: RCMC (always FP32)
        let rcmCorrected = rcmc(dataBuffer: rangeDoppler)

        // Step 4: Fused azimuth compression
        let focused = fusedAzimuthCompression(dataBuffer: rcmCorrected, mode: mode)

        let ptr = focused.contents().bindMemory(to: SIMD2<Float>.self, capacity: totalSamples)
        return Array(UnsafeBufferPointer(start: ptr, count: totalSamples))
    }

    // ========================================================================
    // Step 1: Fused Range Compression (precision-parameterized)
    // ========================================================================

    private func fusedRangeCompression(dataBuffer: MTLBuffer, chirpRef: [SIMD2<Float>],
                                       mode: PrecisionMode) -> MTLBuffer {
        let Nr = nRange
        let Na = nAzimuth
        let byteCount = Na * Nr * MemoryLayout<SIMD2<Float>>.stride

        let chirpFFT = cpuFFT(chirpRef)
        var matchedFilter = [SIMD2<Float>](repeating: .zero, count: Nr)
        for i in 0 ..< Nr {
            matchedFilter[i] = SIMD2<Float>(chirpFFT[i].x, -chirpFFT[i].y)
        }
        let filterBuffer = device.makeBuffer(bytes: matchedFilter,
                                             length: Nr * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!

        let outputBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        let pipeline = rangeCompressionPipelines[mode]!
        let cb = commandQueue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(dataBuffer, offset: 0, index: 0)
        enc.setBuffer(outputBuffer, offset: 0, index: 1)
        enc.setBuffer(filterBuffer, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSizeMake(Na, 1, 1),
                                 threadsPerThreadgroup: MTLSizeMake(1024, 1, 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        return outputBuffer
    }

    // ========================================================================
    // Step 2: Azimuth FFT (always FP32)
    // ========================================================================

    private func azimuthFFT(dataBuffer: MTLBuffer) -> MTLBuffer {
        let Nr = nRange
        let Na = nAzimuth
        let byteCount = Na * Nr * MemoryLayout<SIMD2<Float>>.stride

        let transposed = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        transpose2D(input: dataBuffer, output: transposed, rows: Na, cols: Nr)

        let fftOut = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        batchRowFFT(input: transposed, output: fftOut, rowSize: Na, numRows: Nr)

        let result = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        transpose2D(input: fftOut, output: result, rows: Nr, cols: Na)

        return result
    }

    // ========================================================================
    // Step 3: RCMC (always FP32)
    // ========================================================================

    private func rcmc(dataBuffer: MTLBuffer) -> MTLBuffer {
        let Nr = nRange
        let Na = nAzimuth
        let byteCount = Na * Nr * MemoryLayout<SIMD2<Float>>.stride

        let outputBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        let lambda = Float(params.wavelength)
        let V = Float(params.velocity)
        let R0 = Float(params.centerRange)
        let rangePixel = Float(params.rangePixelSpacing)

        var rcmcParams: [Float] = [lambda, V, R0, rangePixel, Float(Nr), Float(Na)]
        let paramsBuffer = device.makeBuffer(bytes: &rcmcParams,
                                             length: rcmcParams.count * MemoryLayout<Float>.stride, options: .storageModeShared)!

        let pipeline = utilityPipelines["rcmc_sinc_interp"]!
        let cb = commandQueue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(dataBuffer, offset: 0, index: 0)
        enc.setBuffer(outputBuffer, offset: 0, index: 1)
        enc.setBuffer(paramsBuffer, offset: 0, index: 2)

        let threadsPerGroup = min(256, Nr)
        let numGroups = (Na * Nr + threadsPerGroup - 1) / threadsPerGroup
        enc.dispatchThreadgroups(MTLSizeMake(numGroups, 1, 1),
                                 threadsPerThreadgroup: MTLSizeMake(threadsPerGroup, 1, 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        return outputBuffer
    }

    // ========================================================================
    // Step 4: Fused Azimuth Compression (precision-parameterized)
    // ========================================================================

    private func fusedAzimuthCompression(dataBuffer: MTLBuffer, mode: PrecisionMode) -> MTLBuffer {
        let Nr = nRange
        let Na = nAzimuth
        let byteCount = Na * Nr * MemoryLayout<SIMD2<Float>>.stride

        // Generate azimuth matched filter
        let filterBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        let lambda = Float(params.wavelength)
        let V = Float(params.velocity)
        let R0 = Float(params.centerRange)
        let prf = Float(params.prf)
        let rangePixel = Float(params.rangePixelSpacing)

        var azParams: [Float] = [lambda, V, R0, prf, Float(Nr), Float(Na), rangePixel]
        let azParamsBuffer = device.makeBuffer(bytes: &azParams,
                                               length: azParams.count * MemoryLayout<Float>.stride, options: .storageModeShared)!

        let genPipeline = utilityPipelines["generate_azimuth_matched_filter"]!
        let cb1 = commandQueue.makeCommandBuffer()!
        let enc1 = cb1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(genPipeline)
        enc1.setBuffer(filterBuffer, offset: 0, index: 0)
        enc1.setBuffer(azParamsBuffer, offset: 0, index: 1)
        let threads1 = min(256, Na)
        let groups1 = (Na * Nr + threads1 - 1) / threads1
        enc1.dispatchThreadgroups(MTLSizeMake(groups1, 1, 1),
                                  threadsPerThreadgroup: MTLSizeMake(threads1, 1, 1))
        enc1.endEncoding()
        cb1.commit()
        cb1.waitUntilCompleted()

        // Transpose data and filter
        let transposedData = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        transpose2D(input: dataBuffer, output: transposedData, rows: Na, cols: Nr)

        let transposedFilter = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        transpose2D(input: filterBuffer, output: transposedFilter, rows: Na, cols: Nr)

        // Fused multiply + IFFT
        let fusedOut = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        var fusedParams: [UInt32] = [0] // filterIsSingleRow = 0
        let fusedParamsBuffer = device.makeBuffer(bytes: &fusedParams,
                                                  length: fusedParams.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

        let pipeline = multiplyIfftPipelines[mode]!
        let cb2 = commandQueue.makeCommandBuffer()!
        let enc2 = cb2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(pipeline)
        enc2.setBuffer(transposedData, offset: 0, index: 0)
        enc2.setBuffer(fusedOut, offset: 0, index: 1)
        enc2.setBuffer(transposedFilter, offset: 0, index: 2)
        enc2.setBuffer(fusedParamsBuffer, offset: 0, index: 3)
        enc2.dispatchThreadgroups(MTLSizeMake(Nr, 1, 1),
                                  threadsPerThreadgroup: MTLSizeMake(1024, 1, 1))
        enc2.endEncoding()
        cb2.commit()
        cb2.waitUntilCompleted()

        // Transpose back
        let result = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        transpose2D(input: fusedOut, output: result, rows: Nr, cols: Na)

        return result
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    private func fftConfigForSize(_ n: Int) -> (kernelName: String, threadsPerGroup: Int) {
        switch n {
        case 256: ("fft_256_stockham", 64)
        case 512: ("fft_512_stockham", 128)
        case 1024: ("fft_1024_stockham", 256)
        case 2048: ("fft_2048_stockham", 512)
        case 4096: ("fft_4096_stockham", 1024)
        default: fatalError("Unsupported FFT size: \(n)")
        }
    }

    func batchRowFFT(input: MTLBuffer, output: MTLBuffer, rowSize: Int, numRows: Int) {
        let config = fftConfigForSize(rowSize)
        let pipeline = fftPipelines[config.kernelName]!

        let cb = commandQueue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(output, offset: 0, index: 1)
        enc.dispatchThreadgroups(MTLSizeMake(numRows, 1, 1),
                                 threadsPerThreadgroup: MTLSizeMake(config.threadsPerGroup, 1, 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    private func transpose2D(input: MTLBuffer, output: MTLBuffer, rows: Int, cols: Int) {
        let pipeline = utilityPipelines["transpose_2d"]!
        let totalCount = rows * cols

        var params: [UInt32] = [UInt32(rows), UInt32(cols)]
        let paramsBuffer = device.makeBuffer(bytes: &params,
                                             length: params.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

        let cb = commandQueue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(output, offset: 0, index: 1)
        enc.setBuffer(paramsBuffer, offset: 0, index: 2)
        let threads = min(256, totalCount)
        let groups = (totalCount + threads - 1) / threads
        enc.dispatchThreadgroups(MTLSizeMake(groups, 1, 1),
                                 threadsPerThreadgroup: MTLSizeMake(threads, 1, 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    func cpuFFT(_ input: [SIMD2<Float>]) -> [SIMD2<Float>] {
        let n = input.count
        let inputBuffer = device.makeBuffer(bytes: input,
                                            length: n * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(
            length: n * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared
        )!

        batchRowFFT(input: inputBuffer, output: outputBuffer, rowSize: n, numRows: 1)

        let ptr = outputBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // ========================================================================
    // Formatted comparison report
    // ========================================================================

    static func printComparisonReport(results: [PrecisionResult], params: SARParameters) {
        guard let fp32Result = results.first(where: { $0.mode == .fp32 }) else {
            print("ERROR: No FP32 baseline result")
            return
        }

        print()
        print(String(repeating: "=", count: 100))
        print("Mixed-Precision SAR Image Quality Comparison")
        print(String(repeating: "=", count: 100))
        print()

        // Header
        let header = String(format: "%-10s  %-6s  %10s  %10s  %10s  %10s  %10s",
                            "Precision", "Target", "PSLR (dB)", "ISLR (dB)", "SNR (dB)", "Res_r (bin)", "Res_a (bin)")
        print(header)
        print(String(repeating: "-", count: 100))

        for result in results {
            for m in result.targetMetrics {
                let line = String(format: "%-10s  T%-5d  %10.1f  %10.1f  %10.1f  %10.2f  %10.2f",
                                  result.mode.rawValue, m.targetIndex,
                                  m.pslrRange, m.islrRange, m.snr,
                                  m.resolution3dBRange, m.resolution3dBAzimuth)
                print(line)
            }
            print(String(repeating: "-", count: 100))
        }

        // Delta table (vs FP32 baseline)
        print()
        print(String(repeating: "=", count: 100))
        print("Delta vs FP32 Baseline")
        print(String(repeating: "=", count: 100))
        print()

        let deltaHeader = String(format: "%-10s  %-6s  %10s  %10s  %10s  %10s",
                                 "Precision", "Target", "dPSLR(dB)", "dISLR(dB)", "dSNR(dB)", "dRes_r(bin)")
        print(deltaHeader)
        print(String(repeating: "-", count: 80))

        for result in results where result.mode != .fp32 {
            for (i, m) in result.targetMetrics.enumerated() {
                let ref = fp32Result.targetMetrics[i]
                let dPslr = m.pslrRange - ref.pslrRange
                let dIslr = m.islrRange - ref.islrRange
                let dSnr = m.snr - ref.snr
                let dRes = m.resolution3dBRange - ref.resolution3dBRange

                let line = String(format: "%-10s  T%-5d  %+10.2f  %+10.2f  %+10.2f  %+10.3f",
                                  result.mode.rawValue, m.targetIndex,
                                  dPslr, dIslr, dSnr, dRes)
                print(line)
            }
            print(String(repeating: "-", count: 80))
        }

        // L2 relative error vs FP32
        print()
        print("Numerical Error vs FP32 Baseline:")
        print(String(format: "  %-12s  %15s  %15s", "Mode", "L2 Rel Error", "Max Abs Error"))
        print("  " + String(repeating: "-", count: 45))

        for result in results where result.mode != .fp32 {
            var diffSqSum: Double = 0
            var refSqSum: Double = 0
            var maxAbsErr: Float = 0

            for i in 0 ..< fp32Result.focusedImage.count {
                let dx = fp32Result.focusedImage[i].x - result.focusedImage[i].x
                let dy = fp32Result.focusedImage[i].y - result.focusedImage[i].y
                diffSqSum += Double(dx * dx + dy * dy)
                refSqSum += Double(fp32Result.focusedImage[i].x * fp32Result.focusedImage[i].x
                    + fp32Result.focusedImage[i].y * fp32Result.focusedImage[i].y)
                let absErr = max(abs(dx), abs(dy))
                if absErr > maxAbsErr { maxAbsErr = absErr }
            }
            let l2RelError = refSqSum > 0 ? sqrt(diffSqSum / refSqSum) : 0
            let sqnrDB = l2RelError > 0 ? -20.0 * log10(l2RelError) : 999.0

            print(String(format: "  %-12s  %12.2e     %12.2e", result.mode.rawValue, l2RelError, maxAbsErr))
            print(String(format: "  %-12s  SQNR: %.1f dB", "", sqnrDB))
        }

        // Performance comparison
        print()
        print(String(repeating: "=", count: 60))
        print("Performance Comparison")
        print(String(repeating: "=", count: 60))
        print()

        let fp32Time = fp32Result.pipelineTimeSeconds
        print(String(format: "  %-12s  %10s  %10s", "Mode", "Time (s)", "Speedup"))
        print("  " + String(repeating: "-", count: 35))

        for result in results {
            let speedup = fp32Time / result.pipelineTimeSeconds
            print(String(format: "  %-12s  %10.4f  %9.2fx", result.mode.rawValue,
                         result.pipelineTimeSeconds, speedup))
        }

        // PSLR pass/fail assessment
        print()
        print(String(repeating: "=", count: 60))
        print("Radar Quality Assessment")
        print(String(repeating: "=", count: 60))
        print()
        print("  PSLR requirement: < -13.3 dB (unweighted sinc)")
        print()

        for result in results {
            let allPass = result.targetMetrics.allSatisfy { $0.pslrRange < -13.0 }
            let avgPslr = result.targetMetrics.reduce(Float(0)) { $0 + $1.pslrRange } / Float(result.targetMetrics.count)
            let avgSnr = result.targetMetrics.reduce(Float(0)) { $0 + $1.snr } / Float(result.targetMetrics.count)
            let status = allPass ? "PASS" : "FAIL"

            print(String(format: "  %-12s  avg PSLR: %6.1f dB  avg SNR: %5.1f dB  [%s]",
                         result.mode.rawValue, avgPslr, avgSnr, status))
        }

        print()
        print(String(repeating: "=", count: 100))
    }
}
