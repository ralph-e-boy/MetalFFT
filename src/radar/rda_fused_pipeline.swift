// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

import Foundation
import Metal

// ============================================================================
// Range Doppler Algorithm (RDA) — Fused Pipeline
//
// Key optimization: FFT + matched filter multiply + IFFT are fused into a
// single Metal dispatch per compression step, keeping data in threadgroup
// memory between operations. This eliminates 4 of 6 device memory transfers
// per line compared to the unfused baseline.
//
// Steps:
//   1. Fused range compression: FFT(range) * H_r(f) then IFFT — single dispatch
//   2. Azimuth FFT: column-wise via transpose-FFT-transpose
//   3. RCMC: Range cell migration correction
//   4. Fused azimuth compression: FFT(az) * H_a(f) then IFFT — single dispatch
// ============================================================================

class RDAFusedPipeline {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var fusedPipelines: [String: MTLComputePipelineState] = [:]
    var fftPipelines: [String: MTLComputePipelineState] = [:]
    var utilityPipelines: [String: MTLComputePipelineState] = [:]

    let params: SARParameters
    let nRange: Int
    let nAzimuth: Int

    /// Timing breakdown
    var stepTimes: [(String, Double)] = []

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

        // Load fused kernels
        let fusedSource = try RDAFusedPipeline.loadShaderSource(name: "fft_sar_fused.metal")
        let fusedLibrary = try device.makeLibrary(source: fusedSource, options: compileOptions)

        for name in ["fused_range_compression", "fused_azimuth_compression", "fused_multiply_ifft"] {
            if let function = fusedLibrary.makeFunction(name: name) {
                fusedPipelines[name] = try device.makeComputePipelineState(function: function)
            }
        }

        // Load FFT kernels (still needed for azimuth FFT step in RCMC path)
        let fftSource = try RDAFusedPipeline.loadShaderSource(name: "fft_multisize.metal")
        let fftLibrary = try device.makeLibrary(source: fftSource, options: compileOptions)

        let fftKernelNames = [
            "fft_256_stockham", "fft_512_stockham", "fft_1024_stockham",
            "fft_2048_stockham", "fft_4096_stockham",
            "fft_64_stockham", "fft_128_stockham"
        ]
        for name in fftKernelNames {
            if let function = fftLibrary.makeFunction(name: name) {
                fftPipelines[name] = try device.makeComputePipelineState(function: function)
            }
        }

        // Load RDA utility kernels
        let rdaSource = try RDAFusedPipeline.loadShaderSource(name: "rda_kernels.metal")
        let rdaLibrary = try device.makeLibrary(source: rdaSource, options: compileOptions)

        let rdaKernelNames = [
            "complex_multiply", "complex_multiply_conjugate",
            "transpose_2d", "rcmc_sinc_interp",
            "generate_azimuth_matched_filter",
            "fftshift_columns", "magnitude_detect"
        ]
        for name in rdaKernelNames {
            if let function = rdaLibrary.makeFunction(name: name) {
                utilityPipelines[name] = try device.makeComputePipelineState(function: function)
            }
        }

        print("RDA Fused Pipeline: Metal device = \(device.name)")
        print("  Fused kernels: \(fusedPipelines.count), FFT kernels: \(fftPipelines.count), Utility kernels: \(utilityPipelines.count)")
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
    // Main RDA Processing (Fused)
    // ========================================================================

    func process(rawData: [SIMD2<Float>], chirpRef: [SIMD2<Float>]) -> [SIMD2<Float>] {
        assert(rawData.count == nAzimuth * nRange)
        assert(chirpRef.count == nRange)

        let totalSamples = nAzimuth * nRange
        let byteCount = totalSamples * MemoryLayout<SIMD2<Float>>.stride

        stepTimes = []
        print("\nRDA Fused Pipeline: Processing \(nAzimuth) x \(nRange) data")

        var data = rawData
        let dataBuffer = device.makeBuffer(bytes: &data, length: byteCount, options: .storageModeShared)!

        // Step 1: Fused range compression (single dispatch)
        print("  Step 1: Fused range compression (FFT+multiply+IFFT)...")
        var t0 = CFAbsoluteTimeGetCurrent()
        let rangeCompressed = fusedRangeCompression(dataBuffer: dataBuffer, chirpRef: chirpRef)
        var t1 = CFAbsoluteTimeGetCurrent()
        stepTimes.append(("Range compression (fused)", t1 - t0))

        // Step 2: Azimuth FFT (column-wise via transpose-FFT-transpose)
        print("  Step 2: Azimuth FFT...")
        t0 = CFAbsoluteTimeGetCurrent()
        let rangeDoppler = azimuthFFT(dataBuffer: rangeCompressed)
        t1 = CFAbsoluteTimeGetCurrent()
        stepTimes.append(("Azimuth FFT", t1 - t0))

        // Step 3: RCMC
        print("  Step 3: Range Cell Migration Correction...")
        t0 = CFAbsoluteTimeGetCurrent()
        let rcmCorrected = rcmc(dataBuffer: rangeDoppler)
        t1 = CFAbsoluteTimeGetCurrent()
        stepTimes.append(("RCMC", t1 - t0))

        // Step 4: Fused azimuth compression (single dispatch)
        print("  Step 4: Fused azimuth compression (FFT+multiply+IFFT)...")
        t0 = CFAbsoluteTimeGetCurrent()
        let focused = fusedAzimuthCompression(dataBuffer: rcmCorrected)
        t1 = CFAbsoluteTimeGetCurrent()
        stepTimes.append(("Azimuth compression (fused)", t1 - t0))

        // Read back result
        let ptr = focused.contents().bindMemory(to: SIMD2<Float>.self, capacity: totalSamples)
        let result = Array(UnsafeBufferPointer(start: ptr, count: totalSamples))

        print("  RDA fused processing complete.")
        return result
    }

    // ========================================================================
    // Step 1: Fused Range Compression
    // ========================================================================

    private func fusedRangeCompression(dataBuffer: MTLBuffer, chirpRef: [SIMD2<Float>]) -> MTLBuffer {
        let Nr = nRange
        let Na = nAzimuth
        let byteCount = Na * Nr * MemoryLayout<SIMD2<Float>>.stride

        // FFT the chirp reference once (on GPU via existing kernel)
        let chirpFFT = cpuFFT(chirpRef)

        // Matched filter: H_r(f) = conj(FFT(chirp_ref))
        var matchedFilter = [SIMD2<Float>](repeating: .zero, count: Nr)
        for i in 0 ..< Nr {
            matchedFilter[i] = SIMD2<Float>(chirpFFT[i].x, -chirpFFT[i].y)
        }
        let filterBuffer = device.makeBuffer(bytes: matchedFilter,
                                             length: Nr * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!

        let outputBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        // Single fused dispatch: all range lines processed in parallel
        let pipeline = fusedPipelines["fused_range_compression"]!
        let cb = commandQueue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(dataBuffer, offset: 0, index: 0)
        enc.setBuffer(outputBuffer, offset: 0, index: 1)
        enc.setBuffer(filterBuffer, offset: 0, index: 2)
        // Each threadgroup processes one range line
        enc.dispatchThreadgroups(MTLSizeMake(Na, 1, 1),
                                 threadsPerThreadgroup: MTLSizeMake(1024, 1, 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        return outputBuffer
    }

    // ========================================================================
    // Step 2: Azimuth FFT (column-wise)
    // ========================================================================

    private func azimuthFFT(dataBuffer: MTLBuffer) -> MTLBuffer {
        let Nr = nRange
        let Na = nAzimuth
        let byteCount = Na * Nr * MemoryLayout<SIMD2<Float>>.stride

        // Transpose: (Na x Nr) -> (Nr x Na) so columns become rows
        let transposed = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        transpose2D(input: dataBuffer, output: transposed, rows: Na, cols: Nr)

        // Batch FFT on rows (now azimuth dimension)
        let fftOut = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        batchRowFFT(input: transposed, output: fftOut, rowSize: Na, numRows: Nr)

        // Transpose back: (Nr x Na) -> (Na x Nr)
        let result = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        transpose2D(input: fftOut, output: result, rows: Nr, cols: Na)

        return result
    }

    // ========================================================================
    // Step 3: RCMC
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
    // Step 4: Fused Azimuth Compression
    //
    // In the range-Doppler domain, we need to:
    //   1. Transpose to get azimuth lines as rows
    //   2. For each azimuth line: FFT -> multiply by azimuth filter -> IFFT
    //   3. Transpose back
    //
    // The fused kernel handles step 2 in a single dispatch.
    // The azimuth filter varies per range bin, so we generate a 2D filter
    // and transpose it along with the data.
    // ========================================================================

    private func fusedAzimuthCompression(dataBuffer: MTLBuffer) -> MTLBuffer {
        let Nr = nRange
        let Na = nAzimuth
        let byteCount = Na * Nr * MemoryLayout<SIMD2<Float>>.stride

        // Generate azimuth matched filter (2D: Na x Nr) in range-Doppler domain
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

        // Data is in range-Doppler domain (Na x Nr), already azimuth-FFTed.
        // We need to: multiply by filter, then IFFT along azimuth (columns).
        // To IFFT columns: transpose -> IFFT rows -> transpose back.
        // Fuse: transpose -> multiply+IFFT rows -> transpose back.

        // Transpose data: (Na x Nr) -> (Nr x Na)
        let transposedData = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        transpose2D(input: dataBuffer, output: transposedData, rows: Na, cols: Nr)

        // Transpose filter: (Na x Nr) -> (Nr x Na)
        let transposedFilter = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        transpose2D(input: filterBuffer, output: transposedFilter, rows: Na, cols: Nr)

        // Fused multiply + IFFT: each threadgroup handles one row (one range bin's
        // azimuth line). Data is already in frequency domain, so no forward FFT needed.
        let fusedOut = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        var fusedParams: [UInt32] = [0] // filterIsSingleRow = 0 (filter is per-row)
        let fusedParamsBuffer = device.makeBuffer(bytes: &fusedParams,
                                                  length: fusedParams.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

        let pipeline = fusedPipelines["fused_multiply_ifft"]!
        let cb2 = commandQueue.makeCommandBuffer()!
        let enc2 = cb2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(pipeline)
        enc2.setBuffer(transposedData, offset: 0, index: 0)
        enc2.setBuffer(fusedOut, offset: 0, index: 1)
        enc2.setBuffer(transposedFilter, offset: 0, index: 2)
        enc2.setBuffer(fusedParamsBuffer, offset: 0, index: 3)
        // Nr rows, each is an azimuth line of length Na
        enc2.dispatchThreadgroups(MTLSizeMake(Nr, 1, 1),
                                  threadsPerThreadgroup: MTLSizeMake(1024, 1, 1))
        enc2.endEncoding()
        cb2.commit()
        cb2.waitUntilCompleted()

        // Transpose back: (Nr x Na) -> (Na x Nr)
        let result = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        transpose2D(input: fusedOut, output: result, rows: Nr, cols: Na)

        return result
    }

    // ========================================================================
    // FFT Helpers (reused from unfused pipeline for azimuth FFT step)
    // ========================================================================

    private struct FFTConfig {
        let kernelName: String
        let threadsPerGroup: Int
    }

    private func fftConfigForSize(_ n: Int) -> FFTConfig {
        switch n {
        case 256: FFTConfig(kernelName: "fft_256_stockham", threadsPerGroup: 64)
        case 512: FFTConfig(kernelName: "fft_512_stockham", threadsPerGroup: 128)
        case 1024: FFTConfig(kernelName: "fft_1024_stockham", threadsPerGroup: 256)
        case 2048: FFTConfig(kernelName: "fft_2048_stockham", threadsPerGroup: 512)
        case 4096: FFTConfig(kernelName: "fft_4096_stockham", threadsPerGroup: 1024)
        default: fatalError("Unsupported FFT size for single-pass: \(n)")
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

    // ========================================================================
    // Utility GPU operations
    // ========================================================================

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

    /// Print step timing breakdown
    func printTimingBreakdown() {
        print("\n  Fused pipeline step breakdown:")
        for (name, time) in stepTimes {
            let padded = name.padding(toLength: 35, withPad: " ", startingAt: 0)
            print(String(format: "    %@ %.4f s", padded, time))
        }
    }
}
