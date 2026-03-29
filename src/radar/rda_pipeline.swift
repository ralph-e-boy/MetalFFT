// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

import Metal
import Foundation

// ============================================================================
// Range Doppler Algorithm (RDA) — Unfused Baseline
//
// Uses existing Metal FFT kernels as separate dispatches.
// Data layout: nAzimuth rows x nRange columns, row-major.
//
// Steps:
//   1. Range compression: FFT(range) * H_r(f) then IFFT(range) for each azimuth line
//   2. Azimuth FFT: FFT along azimuth (column-wise) -> range-Doppler domain
//   3. RCMC: Range cell migration correction via interpolation
//   4. Azimuth compression: Multiply by azimuth matched filter
//   5. Azimuth IFFT: IFFT along azimuth -> focused image
// ============================================================================

class RDAPipeline {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var fftPipelines: [String: MTLComputePipelineState] = [:]
    var utilityPipelines: [String: MTLComputePipelineState] = [:]

    let params: SARParameters
    let nRange: Int
    let nAzimuth: Int

    init(params: SARParameters) throws {
        self.params = params
        self.nRange = params.nRange
        self.nAzimuth = params.nAzimuth

        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device found")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Could not create command queue")
        }
        self.commandQueue = queue

        // Load FFT kernels from fft_multisize.metal
        let fftSource = try RDAPipeline.loadShaderSource(name: "fft_multisize.metal")
        let fftOptions = MTLCompileOptions()
        fftOptions.fastMathEnabled = true
        let fftLibrary = try device.makeLibrary(source: fftSource, options: fftOptions)

        let fftKernelNames = [
            "fft_256_stockham", "fft_512_stockham", "fft_1024_stockham",
            "fft_2048_stockham", "fft_4096_stockham",
            "fft_64_stockham", "fft_128_stockham",
            "fft_twiddle_transpose", "fft_transpose",
        ]
        for name in fftKernelNames {
            if let function = fftLibrary.makeFunction(name: name) {
                fftPipelines[name] = try device.makeComputePipelineState(function: function)
            }
        }

        // Load RDA utility kernels
        let rdaSource = try RDAPipeline.loadShaderSource(name: "rda_kernels.metal")
        let rdaLibrary = try device.makeLibrary(source: rdaSource, options: fftOptions)

        let rdaKernelNames = [
            "complex_multiply", "complex_multiply_conjugate",
            "transpose_2d", "rcmc_sinc_interp",
            "generate_azimuth_matched_filter",
            "fftshift_columns", "magnitude_detect",
        ]
        for name in rdaKernelNames {
            if let function = rdaLibrary.makeFunction(name: name) {
                utilityPipelines[name] = try device.makeComputePipelineState(function: function)
            }
        }

        print("RDA Pipeline: Metal device = \(device.name)")
        print("  FFT kernels: \(fftPipelines.count), Utility kernels: \(utilityPipelines.count)")
    }

    private static func loadShaderSource(name: String) throws -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        let cwd = FileManager.default.currentDirectoryPath

        let searchPaths = [
            // Relative to executable
            (execDir as NSString).appendingPathComponent(name),
            (execDir as NSString).appendingPathComponent("../../../\(name)"),
            (execDir as NSString).appendingPathComponent("../../src/metal/\(name)"),
            (execDir as NSString).appendingPathComponent("../../../src/metal/\(name)"),
            (execDir as NSString).appendingPathComponent("../../src/radar/\(name)"),
            (execDir as NSString).appendingPathComponent("../../../src/radar/\(name)"),
            // Relative to current working directory
            (cwd as NSString).appendingPathComponent(name),
            (cwd as NSString).appendingPathComponent("src/metal/\(name)"),
            (cwd as NSString).appendingPathComponent("src/radar/\(name)"),
            (cwd as NSString).appendingPathComponent("../metal/\(name)"),
            (cwd as NSString).appendingPathComponent("../radar/\(name)"),
        ]
        for path in searchPaths {
            if FileManager.default.fileExists(atPath: path) {
                return try String(contentsOfFile: path, encoding: .utf8)
            }
        }
        fatalError("Could not find shader: \(name). Searched: \(searchPaths)")
    }

    // ========================================================================
    // Main RDA Processing
    // ========================================================================

    func process(rawData: [SIMD2<Float>], chirpRef: [SIMD2<Float>]) -> [SIMD2<Float>] {
        assert(rawData.count == nAzimuth * nRange)
        assert(chirpRef.count == nRange)

        let totalSamples = nAzimuth * nRange
        let byteCount = totalSamples * MemoryLayout<SIMD2<Float>>.stride

        print("\nRDA Pipeline: Processing \(nAzimuth) x \(nRange) data")

        // Upload raw data to GPU
        var data = rawData
        let dataBuffer = device.makeBuffer(bytes: &data, length: byteCount, options: .storageModeShared)!

        // Step 1: Range compression
        print("  Step 1: Range compression...")
        let rangeCompressed = rangeCompression(dataBuffer: dataBuffer, chirpRef: chirpRef)

        // Step 2: Azimuth FFT (column-wise via transpose-FFT-transpose)
        print("  Step 2: Azimuth FFT...")
        let rangeDoppler = azimuthFFT(dataBuffer: rangeCompressed)

        // Step 3: RCMC
        print("  Step 3: Range Cell Migration Correction...")
        let rcmCorrected = rcmc(dataBuffer: rangeDoppler)

        // Step 4: Azimuth compression
        print("  Step 4: Azimuth compression...")
        let azCompressed = azimuthCompression(dataBuffer: rcmCorrected)

        // Step 5: Azimuth IFFT
        print("  Step 5: Azimuth IFFT...")
        let focused = azimuthIFFT(dataBuffer: azCompressed)

        // Read back result
        let ptr = focused.contents().bindMemory(to: SIMD2<Float>.self, capacity: totalSamples)
        let result = Array(UnsafeBufferPointer(start: ptr, count: totalSamples))

        print("  RDA processing complete.")
        return result
    }

    // ========================================================================
    // Step 1: Range Compression
    // ========================================================================

    private func rangeCompression(dataBuffer: MTLBuffer, chirpRef: [SIMD2<Float>]) -> MTLBuffer {
        let Nr = nRange
        let Na = nAzimuth
        let byteCount = Na * Nr * MemoryLayout<SIMD2<Float>>.stride

        // FFT the chirp reference once
        let chirpFFT = cpuFFT(chirpRef)

        // Create matched filter: H_r(f) = conj(FFT(chirp_ref))
        var matchedFilter = [SIMD2<Float>](repeating: .zero, count: Nr)
        for i in 0..<Nr {
            matchedFilter[i] = SIMD2<Float>(chirpFFT[i].x, -chirpFFT[i].y)
        }
        let filterBuffer = device.makeBuffer(bytes: matchedFilter,
            length: Nr * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!

        // Buffers for FFT -> multiply -> IFFT
        let fftOutBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        let mulOutBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        let ifftOutBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        // Batch FFT all range lines (each azimuth line is one FFT)
        batchRowFFT(input: dataBuffer, output: fftOutBuffer, rowSize: Nr, numRows: Na)

        // Multiply each row by matched filter
        complexMultiply(a: fftOutBuffer, b: filterBuffer, output: mulOutBuffer,
                       rowSize: Nr, numRows: Na, bIsSingleRow: true)

        // Batch IFFT all range lines
        batchRowIFFT(input: mulOutBuffer, output: ifftOutBuffer, rowSize: Nr, numRows: Na)

        return ifftOutBuffer
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
    // Step 3: Range Cell Migration Correction (RCMC)
    // ========================================================================

    private func rcmc(dataBuffer: MTLBuffer) -> MTLBuffer {
        let Nr = nRange
        let Na = nAzimuth
        let byteCount = Na * Nr * MemoryLayout<SIMD2<Float>>.stride

        let outputBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        // RCMC parameters
        let lambda = Float(params.wavelength)
        let V = Float(params.velocity)
        let R0 = Float(params.centerRange)
        let rangePixel = Float(params.rangePixelSpacing)

        // Pack parameters into a buffer
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
    // Step 4: Azimuth Compression
    // ========================================================================

    private func azimuthCompression(dataBuffer: MTLBuffer) -> MTLBuffer {
        let Nr = nRange
        let Na = nAzimuth
        let byteCount = Na * Nr * MemoryLayout<SIMD2<Float>>.stride

        // Generate azimuth matched filter for each range bin
        // In range-Doppler domain, the azimuth matched filter is:
        //   H_a(f_a, R0) = exp(j * 4*pi*R0/lambda * sqrt(1 - (lambda*f_a/(2*V))^2))
        let filterBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        let lambda = Float(params.wavelength)
        let V = Float(params.velocity)
        let R0 = Float(params.centerRange)
        let prf = Float(params.prf)
        let rangePixel = Float(params.rangePixelSpacing)

        var azParams: [Float] = [lambda, V, R0, prf, Float(Nr), Float(Na), rangePixel]
        let azParamsBuffer = device.makeBuffer(bytes: &azParams,
            length: azParams.count * MemoryLayout<Float>.stride, options: .storageModeShared)!

        // Generate the filter on GPU
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

        // Multiply data by conjugate of azimuth filter
        let outputBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        complexMultiply(a: dataBuffer, b: filterBuffer, output: outputBuffer,
                       rowSize: Nr, numRows: Na, bIsSingleRow: false)

        return outputBuffer
    }

    // ========================================================================
    // Step 5: Azimuth IFFT (column-wise)
    // ========================================================================

    private func azimuthIFFT(dataBuffer: MTLBuffer) -> MTLBuffer {
        let Nr = nRange
        let Na = nAzimuth
        let byteCount = Na * Nr * MemoryLayout<SIMD2<Float>>.stride

        // Transpose: (Na x Nr) -> (Nr x Na)
        let transposed = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        transpose2D(input: dataBuffer, output: transposed, rows: Na, cols: Nr)

        // Batch IFFT on rows
        let ifftOut = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        batchRowIFFT(input: transposed, output: ifftOut, rowSize: Na, numRows: Nr)

        // Transpose back: (Nr x Na) -> (Na x Nr)
        let result = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        transpose2D(input: ifftOut, output: result, rows: Nr, cols: Na)

        return result
    }

    // ========================================================================
    // FFT/IFFT Helpers using Metal kernels
    // ========================================================================

    private struct FFTConfig {
        let kernelName: String
        let threadsPerGroup: Int
    }

    private func fftConfigForSize(_ n: Int) -> FFTConfig {
        switch n {
        case 256:  return FFTConfig(kernelName: "fft_256_stockham",  threadsPerGroup: 64)
        case 512:  return FFTConfig(kernelName: "fft_512_stockham",  threadsPerGroup: 128)
        case 1024: return FFTConfig(kernelName: "fft_1024_stockham", threadsPerGroup: 256)
        case 2048: return FFTConfig(kernelName: "fft_2048_stockham", threadsPerGroup: 512)
        case 4096: return FFTConfig(kernelName: "fft_4096_stockham", threadsPerGroup: 1024)
        default:   fatalError("Unsupported FFT size for single-pass: \(n)")
        }
    }

    // Batch row-wise FFT: each row of length rowSize gets an FFT
    func batchRowFFT(input: MTLBuffer, output: MTLBuffer, rowSize: Int, numRows: Int) {
        let config = fftConfigForSize(rowSize)
        let pipeline = fftPipelines[config.kernelName]!

        let cb = commandQueue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(output, offset: 0, index: 1)
        // Each threadgroup processes one row (one FFT)
        enc.dispatchThreadgroups(MTLSizeMake(numRows, 1, 1),
                                 threadsPerThreadgroup: MTLSizeMake(config.threadsPerGroup, 1, 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    // Batch row-wise IFFT: conjugate -> FFT -> conjugate -> scale by 1/N
    func batchRowIFFT(input: MTLBuffer, output: MTLBuffer, rowSize: Int, numRows: Int) {
        let totalCount = rowSize * numRows
        let byteCount = totalCount * MemoryLayout<SIMD2<Float>>.stride

        // IFFT = (1/N) * conj(FFT(conj(x)))
        // Step 1: Conjugate input
        let conjInput = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        conjugateBuffer(input: input, output: conjInput, count: totalCount)

        // Step 2: FFT
        let fftOut = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        batchRowFFT(input: conjInput, output: fftOut, rowSize: rowSize, numRows: numRows)

        // Step 3: Conjugate and scale by 1/N
        conjugateAndScale(input: fftOut, output: output, count: totalCount, scale: 1.0 / Float(rowSize))
    }

    // ========================================================================
    // Utility GPU operations
    // ========================================================================

    private func complexMultiply(a: MTLBuffer, b: MTLBuffer, output: MTLBuffer,
                                  rowSize: Int, numRows: Int, bIsSingleRow: Bool) {
        let pipeline = utilityPipelines["complex_multiply"]!
        let totalCount = rowSize * numRows

        var params: [UInt32] = [UInt32(rowSize), UInt32(numRows), bIsSingleRow ? 1 : 0]
        let paramsBuffer = device.makeBuffer(bytes: &params,
            length: params.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

        let cb = commandQueue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        enc.setBuffer(paramsBuffer, offset: 0, index: 3)
        let threads = min(256, totalCount)
        let groups = (totalCount + threads - 1) / threads
        enc.dispatchThreadgroups(MTLSizeMake(groups, 1, 1),
                                 threadsPerThreadgroup: MTLSizeMake(threads, 1, 1))
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

    // CPU-side conjugate (simple utility for IFFT)
    private func conjugateBuffer(input: MTLBuffer, output: MTLBuffer, count: Int) {
        let src = input.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
        let dst = output.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
        for i in 0..<count {
            dst[i] = SIMD2<Float>(src[i].x, -src[i].y)
        }
    }

    private func conjugateAndScale(input: MTLBuffer, output: MTLBuffer, count: Int, scale: Float) {
        let src = input.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
        let dst = output.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
        for i in 0..<count {
            dst[i] = SIMD2<Float>(src[i].x * scale, -src[i].y * scale)
        }
    }

    // CPU FFT for small reference signals (chirp reference)
    func cpuFFT(_ input: [SIMD2<Float>]) -> [SIMD2<Float>] {
        let n = input.count
        // Use GPU FFT via Metal kernel
        let inputBuffer = device.makeBuffer(bytes: input,
            length: n * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(
            length: n * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!

        batchRowFFT(input: inputBuffer, output: outputBuffer, rowSize: n, numRows: 1)

        let ptr = outputBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }
}
