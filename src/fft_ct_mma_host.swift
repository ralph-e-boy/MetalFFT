// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

import Metal
import Accelerate
import Foundation

// ============================================================================
// FFT 4096 — In-place Cooley-Tukey DIF with simdgroup_matrix MMA
// Host code with vDSP validation and benchmarking
// ============================================================================

struct FFTHostCTMMA {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipeline: MTLComputePipelineState
    let dftRealBuffer: MTLBuffer
    let dftImagBuffer: MTLBuffer
    let dftNegImagBuffer: MTLBuffer
    let twiddleBuffer: MTLBuffer

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device found")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Could not create command queue")
        }
        self.commandQueue = queue

        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        let searchPaths = [
            (execDir as NSString).appendingPathComponent("fft_4096_ct_mma.metal"),
            (execDir as NSString).appendingPathComponent("FFTCTMMAHost_FFTCTMMAHost.bundle/fft_4096_ct_mma.metal"),
            (execDir as NSString).appendingPathComponent("../Sources/fft_4096_ct_mma.metal"),
            (execDir as NSString).appendingPathComponent("../../Sources/fft_4096_ct_mma.metal"),
            (execDir as NSString).appendingPathComponent("../../src/metal/fft_4096_ct_mma.metal"),
            (execDir as NSString).appendingPathComponent("../../../fft_4096_ct_mma.metal"),
        ]
        var foundPath: String? = nil
        for path in searchPaths {
            if FileManager.default.fileExists(atPath: path) {
                foundPath = path
                break
            }
        }
        guard let metalPath = foundPath else {
            print("ERROR: Could not find fft_4096_ct_mma.metal")
            print("Searched: \(searchPaths)")
            fatalError("CT MMA Metal shader not found")
        }
        let metalSource = try String(contentsOfFile: metalPath, encoding: .utf8)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: metalSource, options: options)
        guard let function = library.makeFunction(name: "fft_4096_ct_mma") else {
            fatalError("Could not find kernel function 'fft_4096_ct_mma'")
        }
        self.pipeline = try device.makeComputePipelineState(function: function)

        // Precompute DFT_8 real and imaginary matrices (row-major 8x8)
        let sqrt2_2: Float = 0.70710678118654752
        let f8Real: [Float] = [
             1.0,  1.0,       1.0,  1.0,       1.0,  1.0,       1.0,  1.0,
             1.0,  sqrt2_2,   0.0, -sqrt2_2,  -1.0, -sqrt2_2,   0.0,  sqrt2_2,
             1.0,  0.0,      -1.0,  0.0,       1.0,  0.0,      -1.0,  0.0,
             1.0, -sqrt2_2,   0.0,  sqrt2_2,  -1.0,  sqrt2_2,   0.0, -sqrt2_2,
             1.0, -1.0,       1.0, -1.0,       1.0, -1.0,       1.0, -1.0,
             1.0, -sqrt2_2,   0.0,  sqrt2_2,  -1.0,  sqrt2_2,   0.0, -sqrt2_2,
             1.0,  0.0,      -1.0,  0.0,       1.0,  0.0,      -1.0,  0.0,
             1.0,  sqrt2_2,   0.0, -sqrt2_2,  -1.0, -sqrt2_2,   0.0,  sqrt2_2,
        ]
        let f8Imag: [Float] = [
             0.0,  0.0,       0.0,  0.0,       0.0,  0.0,       0.0,  0.0,
             0.0, -sqrt2_2,  -1.0, -sqrt2_2,   0.0,  sqrt2_2,   1.0,  sqrt2_2,
             0.0, -1.0,       0.0,  1.0,       0.0, -1.0,       0.0,  1.0,
             0.0, -sqrt2_2,   1.0, -sqrt2_2,   0.0,  sqrt2_2,  -1.0,  sqrt2_2,
             0.0,  0.0,       0.0,  0.0,       0.0,  0.0,       0.0,  0.0,
             0.0,  sqrt2_2,  -1.0,  sqrt2_2,   0.0, -sqrt2_2,   1.0, -sqrt2_2,
             0.0,  1.0,       0.0, -1.0,       0.0,  1.0,       0.0, -1.0,
             0.0,  sqrt2_2,   1.0,  sqrt2_2,   0.0, -sqrt2_2,  -1.0, -sqrt2_2,
        ]

        self.dftRealBuffer = device.makeBuffer(
            bytes: f8Real,
            length: 64 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        self.dftImagBuffer = device.makeBuffer(
            bytes: f8Imag,
            length: 64 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        let f8NegImag = f8Imag.map { -$0 }
        self.dftNegImagBuffer = device.makeBuffer(
            bytes: f8NegImag,
            length: 64 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        // Precompute twiddle factors for 3 stages
        let n = 4096
        let strides: [Int] = [512, 64, 8]
        let groupSizes: [Int] = [4096, 512, 64]
        var twiddleTable = [SIMD2<Float>](repeating: SIMD2<Float>(1.0, 0.0), count: 3 * n)
        for stage in 0..<3 {
            let S = strides[stage]
            let G = groupSizes[stage]
            for i in 0..<n {
                let posInGroup = i % G
                let r = posInGroup / S
                let k = posInGroup % S
                if r > 0 && k > 0 {
                    let angle = -2.0 * Float.pi * Float(r * k) / Float(G)
                    twiddleTable[stage * n + i] = SIMD2<Float>(cos(angle), sin(angle))
                }
            }
        }
        self.twiddleBuffer = device.makeBuffer(
            bytes: twiddleTable,
            length: 3 * n * MemoryLayout<SIMD2<Float>>.stride,
            options: .storageModeShared
        )!

        print("CT MMA Kernel:")
        print("  Device:                  \(device.name)")
        print("  Max threads/threadgroup: \(pipeline.maxTotalThreadsPerThreadgroup)")
        print("  Thread execution width:  \(pipeline.threadExecutionWidth)")
        if pipeline.maxTotalThreadsPerThreadgroup < 512 {
            print("  WARNING: Kernel needs 512 threads, pipeline supports \(pipeline.maxTotalThreadsPerThreadgroup).")
        }
        print()
    }

    func runFFT(input: [SIMD2<Float>], batchSize: Int = 1) -> [SIMD2<Float>] {
        let n = 4096
        assert(input.count == n * batchSize)

        let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<SIMD2<Float>>.stride,
            options: .storageModeShared
        )!
        let outputBuffer = device.makeBuffer(
            length: input.count * MemoryLayout<SIMD2<Float>>.stride,
            options: .storageModeShared
        )!

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBuffer(dftRealBuffer, offset: 0, index: 2)
        encoder.setBuffer(dftImagBuffer, offset: 0, index: 3)
        encoder.setBuffer(dftNegImagBuffer, offset: 0, index: 4)
        encoder.setBuffer(twiddleBuffer, offset: 0, index: 5)

        let threadsPerThreadgroup = MTLSizeMake(512, 1, 1)
        let threadgroups = MTLSizeMake(batchSize, 1, 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPtr = outputBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: input.count)
        return Array(UnsafeBufferPointer(start: outputPtr, count: input.count))
    }

    func timeFFT(input: [SIMD2<Float>], batchSize: Int = 1, warmup: Int = 10, repeats: Int = 50) -> Double {
        let n = 4096
        assert(input.count == n * batchSize)

        let inputBuffer = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<SIMD2<Float>>.stride,
            options: .storageModeShared
        )!
        let outputBuffer = device.makeBuffer(
            length: input.count * MemoryLayout<SIMD2<Float>>.stride,
            options: .storageModeShared
        )!

        let threadsPerThreadgroup = MTLSizeMake(512, 1, 1)
        let threadgroups = MTLSizeMake(batchSize, 1, 1)

        for _ in 0..<warmup {
            let cb = commandQueue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(inputBuffer, offset: 0, index: 0)
            enc.setBuffer(outputBuffer, offset: 0, index: 1)
            enc.setBuffer(dftRealBuffer, offset: 0, index: 2)
            enc.setBuffer(dftImagBuffer, offset: 0, index: 3)
            enc.setBuffer(dftNegImagBuffer, offset: 0, index: 4)
            enc.setBuffer(twiddleBuffer, offset: 0, index: 5)
            enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }

        var times: [Double] = []
        for _ in 0..<repeats {
            let cb = commandQueue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(inputBuffer, offset: 0, index: 0)
            enc.setBuffer(outputBuffer, offset: 0, index: 1)
            enc.setBuffer(dftRealBuffer, offset: 0, index: 2)
            enc.setBuffer(dftImagBuffer, offset: 0, index: 3)
            enc.setBuffer(dftNegImagBuffer, offset: 0, index: 4)
            enc.setBuffer(twiddleBuffer, offset: 0, index: 5)
            enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            times.append((cb.gpuEndTime - cb.gpuStartTime) * 1e6)
        }

        times.sort()
        return times[times.count / 2]
    }
}

// ============================================================================
// CT Scalar Kernel Host (for debugging)
// ============================================================================

struct FFTHostCTScalar {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipeline: MTLComputePipelineState
    let dftRealBuffer: MTLBuffer
    let dftImagBuffer: MTLBuffer

    init(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary, dftReal: MTLBuffer, dftImag: MTLBuffer) throws {
        self.device = device
        self.commandQueue = queue
        self.dftRealBuffer = dftReal
        self.dftImagBuffer = dftImag
        guard let function = library.makeFunction(name: "fft_4096_ct_scalar") else {
            fatalError("Could not find kernel function 'fft_4096_ct_scalar'")
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    func runFFT(input: [SIMD2<Float>], batchSize: Int = 1) -> [SIMD2<Float>] {
        let n = 4096
        assert(input.count == n * batchSize)

        let inputBuffer = device.makeBuffer(bytes: input, length: input.count * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(length: input.count * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBuffer(dftRealBuffer, offset: 0, index: 2)
        encoder.setBuffer(dftImagBuffer, offset: 0, index: 3)

        encoder.dispatchThreadgroups(MTLSizeMake(batchSize, 1, 1), threadsPerThreadgroup: MTLSizeMake(512, 1, 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPtr = outputBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: input.count)
        return Array(UnsafeBufferPointer(start: outputPtr, count: input.count))
    }
}

// ============================================================================
// vDSP Reference FFT
// ============================================================================

func vdspFFT4096(input: [SIMD2<Float>]) -> [SIMD2<Float>] {
    let n = 4096
    let log2n = vDSP_Length(12)
    assert(input.count == n)

    guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
        fatalError("vDSP_create_fftsetup failed")
    }
    defer { vDSP_destroy_fftsetup(fftSetup) }

    var realIn = [Float](repeating: 0, count: n)
    var imagIn = [Float](repeating: 0, count: n)
    for i in 0..<n {
        realIn[i] = input[i].x
        imagIn[i] = input[i].y
    }

    var realOut = [Float](repeating: 0, count: n)
    var imagOut = [Float](repeating: 0, count: n)

    realIn.withUnsafeMutableBufferPointer { rIn in
        imagIn.withUnsafeMutableBufferPointer { iIn in
            realOut.withUnsafeMutableBufferPointer { rOut in
                imagOut.withUnsafeMutableBufferPointer { iOut in
                    var splitIn = DSPSplitComplex(realp: rIn.baseAddress!, imagp: iIn.baseAddress!)
                    var splitOut = DSPSplitComplex(realp: rOut.baseAddress!, imagp: iOut.baseAddress!)
                    vDSP_fft_zop(fftSetup, &splitIn, 1, &splitOut, 1, log2n, FFTDirection(kFFTDirection_Forward))
                }
            }
        }
    }

    var result = [SIMD2<Float>](repeating: .zero, count: n)
    for i in 0..<n {
        result[i] = SIMD2<Float>(realOut[i], imagOut[i])
    }
    return result
}

// ============================================================================
// Test Signal Generation
// ============================================================================

func generateTestSignal(n: Int) -> [SIMD2<Float>] {
    var signal = [SIMD2<Float>](repeating: .zero, count: n)
    let invN = 2.0 * Float.pi / Float(n)
    for k in 0..<n {
        let t = Float(k) * invN
        let real = cos(100.0 * t) + 0.5 * cos(200.0 * t) + 0.3 * sin(500.0 * t)
        signal[k] = SIMD2<Float>(real, 0.0)
    }
    return signal
}

func generateRandomSignal(n: Int) -> [SIMD2<Float>] {
    var signal = [SIMD2<Float>](repeating: .zero, count: n)
    for k in 0..<n {
        signal[k] = SIMD2<Float>(Float.random(in: -1...1), Float.random(in: -1...1))
    }
    return signal
}

// ============================================================================
// Validation
// ============================================================================

func validate(metal: [SIMD2<Float>], reference: [SIMD2<Float>], label: String) -> Bool {
    let n = metal.count
    assert(n == reference.count)

    var maxAbsError: Float = 0
    var maxRelError: Float = 0
    var worstIdx = 0
    var l2ErrorSq: Float = 0
    var l2RefSq: Float = 0

    for i in 0..<n {
        let diff = metal[i] - reference[i]
        let absErr = sqrt(diff.x * diff.x + diff.y * diff.y)
        let refMag = sqrt(reference[i].x * reference[i].x + reference[i].y * reference[i].y)

        if absErr > maxAbsError {
            maxAbsError = absErr
            worstIdx = i
        }

        let relErr = refMag > 1e-3 ? absErr / refMag : 0
        if relErr > maxRelError {
            maxRelError = relErr
        }

        l2ErrorSq += diff.x * diff.x + diff.y * diff.y
        l2RefSq += reference[i].x * reference[i].x + reference[i].y * reference[i].y
    }

    let l2RelError = l2RefSq > 0 ? sqrt(l2ErrorSq / l2RefSq) : sqrt(l2ErrorSq)
    let pass = l2RelError < 1e-5

    print("  [\(label)]")
    print("    Max absolute error: \(String(format: "%.6e", maxAbsError)) (at bin \(worstIdx))")
    print("    Max relative error: \(String(format: "%.6e", maxRelError))")
    print("    L2 relative error:  \(String(format: "%.6e", l2RelError))")
    print("    Result: \(pass ? "PASS" : "FAIL")")
    if !pass {
        let refVal = reference[worstIdx]
        let metalVal = metal[worstIdx]
        print("    Worst bin \(worstIdx): Metal=(\(metalVal.x), \(metalVal.y))  vDSP=(\(refVal.x), \(refVal.y))")
    }
    print()
    return pass
}

// ============================================================================
// Main
// ============================================================================

@main
struct FFTCTMMAHostMain {
    static func main() throws {
        let n = 4096

        let host = try FFTHostCTMMA()

        // === First: test scalar CT DIF to validate the algorithm ===
        print(String(repeating: "=", count: 72))
        print("CT DIF SCALAR -- Quick validation")
        print(String(repeating: "=", count: 72))

        // Find and compile the shader
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        let shaderSearchPaths = [
            (execDir as NSString).appendingPathComponent("fft_4096_ct_mma.metal"),
            (execDir as NSString).appendingPathComponent("../../src/metal/fft_4096_ct_mma.metal"),
            (execDir as NSString).appendingPathComponent("../../../fft_4096_ct_mma.metal"),
        ]
        var shaderPath: String? = nil
        for p in shaderSearchPaths {
            if FileManager.default.fileExists(atPath: p) { shaderPath = p; break }
        }
        guard let foundShaderPath = shaderPath else {
            print("WARNING: Could not find shader for scalar kernel test, skipping")
            let scalarHost: FFTHostCTScalar? = nil
            _ = scalarHost  // suppress unused warning
            // skip scalar tests
            let _ = 0 // placeholder
            print()
            // jump to MMA tests below
            var allPassed = true
            // (rest handled below)
            fatalError("shader not found for scalar test")
        }
        let shaderSource = try String(contentsOfFile: foundShaderPath, encoding: .utf8)
        let compileOpts = MTLCompileOptions()
        compileOpts.fastMathEnabled = true
        let scalarLib = try host.device.makeLibrary(source: shaderSource, options: compileOpts)
        let scalarHost = try FFTHostCTScalar(
            device: host.device, queue: host.commandQueue,
            library: scalarLib,
            dftReal: host.dftRealBuffer, dftImag: host.dftImagBuffer
        )

        var impulse = [SIMD2<Float>](repeating: .zero, count: n)
        impulse[0] = SIMD2<Float>(1.0, 0.0)
        let refImpulse = vdspFFT4096(input: impulse)
        let _ = validate(metal: scalarHost.runFFT(input: impulse), reference: refImpulse, label: "scalar-impulse")

        let dc = [SIMD2<Float>](repeating: SIMD2<Float>(1.0, 0.0), count: n)
        let refDC = vdspFFT4096(input: dc)
        let _ = validate(metal: scalarHost.runFFT(input: dc), reference: refDC, label: "scalar-dc")

        let randSignal = generateRandomSignal(n: n)
        let refRand = vdspFFT4096(input: randSignal)
        let _ = validate(metal: scalarHost.runFFT(input: randSignal), reference: refRand, label: "scalar-random")

        // Also test split-layout scalar
        print()
        print("Split-layout scalar:")
        guard let splitScalarFn = scalarLib.makeFunction(name: "fft_4096_ct_split_scalar") else {
            fatalError("could not find fft_4096_ct_split_scalar")
        }
        let splitScalarPipeline = try host.device.makeComputePipelineState(function: splitScalarFn)
        // Quick inline test
        do {
            let inBuf = host.device.makeBuffer(bytes: randSignal, length: n * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
            let outBuf = host.device.makeBuffer(length: n * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
            let cb = host.commandQueue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(splitScalarPipeline)
            enc.setBuffer(inBuf, offset: 0, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            enc.setBuffer(host.dftRealBuffer, offset: 0, index: 2)
            enc.setBuffer(host.dftImagBuffer, offset: 0, index: 3)
            enc.dispatchThreadgroups(MTLSizeMake(1,1,1), threadsPerThreadgroup: MTLSizeMake(512,1,1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            let ptr = outBuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: n)
            let res = Array(UnsafeBufferPointer(start: ptr, count: n))
            let _ = validate(metal: res, reference: refRand, label: "split-scalar-random")
        }

        print()

        // === MMA kernel validation ===
        var allPassed = true

        print(String(repeating: "=", count: 72))
        print("FFT 4096 CT DIF MMA -- Validation")
        print(String(repeating: "=", count: 72))
        print()

        print("-- Test 1: Sinusoidal signal --")
        let sinSignal = generateTestSignal(n: n)
        allPassed = validate(metal: host.runFFT(input: sinSignal), reference: vdspFFT4096(input: sinSignal), label: "sinusoids") && allPassed

        print("-- Test 2: Random complex signal --")
        allPassed = validate(metal: host.runFFT(input: randSignal), reference: refRand, label: "random") && allPassed

        print("-- Test 3: Impulse --")
        allPassed = validate(metal: host.runFFT(input: impulse), reference: refImpulse, label: "impulse") && allPassed

        print("-- Test 4: DC --")
        allPassed = validate(metal: host.runFFT(input: dc), reference: refDC, label: "dc") && allPassed

        print("-- Test 5: Batch of 16 --")
        let batchSize = 16
        var batchInput = [SIMD2<Float>](repeating: .zero, count: n * batchSize)
        for i in 0..<(n * batchSize) {
            batchInput[i] = SIMD2<Float>(Float.random(in: -1...1), Float.random(in: -1...1))
        }
        let metalBatch = host.runFFT(input: batchInput, batchSize: batchSize)
        for b in 0..<batchSize {
            let slice = Array(batchInput[b * n..<(b + 1) * n])
            let ref = vdspFFT4096(input: slice)
            let metalSlice = Array(metalBatch[b * n..<(b + 1) * n])
            let ok = validate(metal: metalSlice, reference: ref, label: "batch[\(b)]")
            allPassed = allPassed && ok
        }

        // --- Performance ---
        print(String(repeating: "=", count: 72))
        print("Performance Benchmarking")
        print(String(repeating: "=", count: 72))
        print()

        for bs in [1, 16, 64, 256] {
            var bi = [SIMD2<Float>](repeating: .zero, count: n * bs)
            for i in 0..<bi.count { bi[i] = SIMD2<Float>(Float.random(in: -1...1), Float.random(in: -1...1)) }
            let us = host.timeFFT(input: bi, batchSize: bs)
            let totalFFTs = Double(bs)
            let flopsPerFFT = 5.0 * Double(n) * log2(Double(n))
            let gflops = totalFFTs * flopsPerFFT / (us * 1e-6) / 1e9
            let usPerFFT = us / totalFFTs
            print(String(format: "  Batch %4d:  %8.1f us total  %8.2f us/FFT  %8.2f GFLOPS", bs, us, usPerFFT, gflops))
        }
        print()

        print(String(repeating: "=", count: 72))
        print("FINAL: \(allPassed ? "ALL PASSED" : "SOME FAILED")")
        print(String(repeating: "=", count: 72))
    }
}
