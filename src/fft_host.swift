// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

import Accelerate
import Foundation
import Metal

// ============================================================================
// FFT 4096 — Metal Host + vDSP Validation
// Supports both Stockham (radix-4) and MMA (simdgroup_matrix radix-8) kernels
// ============================================================================

// ============================================================================
// Stockham Kernel Host
// ============================================================================

struct FFTHost {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipeline: MTLComputePipelineState

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device found")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Could not create command queue")
        }
        commandQueue = queue

        // Load Metal shader source from file adjacent to executable or source tree
        let metalSource: String
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        let searchPaths = [
            (execDir as NSString).appendingPathComponent("fft_stockham_4096.metal"),
            (execDir as NSString).appendingPathComponent("FFTHost_FFTHost.bundle/fft_stockham_4096.metal"),
            (execDir as NSString).appendingPathComponent("../Sources/fft_stockham_4096.metal"),
            (execDir as NSString).appendingPathComponent("../../Sources/fft_stockham_4096.metal"),
            (execDir as NSString).appendingPathComponent("../../src/metal/fft_stockham_4096.metal")
        ]
        var foundPath: String? = nil
        for path in searchPaths {
            if FileManager.default.fileExists(atPath: path) {
                foundPath = path
                break
            }
        }
        guard let metalPath = foundPath else {
            print("ERROR: Could not find fft_stockham_4096.metal")
            print("Searched: \(searchPaths)")
            fatalError("Metal shader not found")
        }
        metalSource = try String(contentsOfFile: metalPath, encoding: .utf8)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: metalSource, options: options)
        guard let function = library.makeFunction(name: "fft_4096_stockham") else {
            fatalError("Could not find kernel function 'fft_4096_stockham'")
        }
        pipeline = try device.makeComputePipelineState(function: function)

        print("Device:                  \(device.name)")
        print("Max threads/threadgroup: \(pipeline.maxTotalThreadsPerThreadgroup)")
        print("Thread execution width:  \(pipeline.threadExecutionWidth)")
        if pipeline.maxTotalThreadsPerThreadgroup < 1024 {
            print("WARNING: Kernel needs 1024 threads, pipeline supports \(pipeline.maxTotalThreadsPerThreadgroup).")
            print("         Register pressure may be too high. Results may be incorrect.")
        }
        print()
    }

    /// Run the Metal FFT kernel on the given interleaved complex input
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

        let threadsPerThreadgroup = MTLSizeMake(1024, 1, 1)
        let threadgroups = MTLSizeMake(batchSize, 1, 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPtr = outputBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: input.count)
        return Array(UnsafeBufferPointer(start: outputPtr, count: input.count))
    }

    /// Time the Metal FFT kernel (median of multiple runs)
    func timeFFT(input: [SIMD2<Float>], batchSize: Int = 1, warmup: Int = 3, repeats: Int = 10) -> Double {
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

        let threadsPerThreadgroup = MTLSizeMake(1024, 1, 1)
        let threadgroups = MTLSizeMake(batchSize, 1, 1)

        // Warmup
        for _ in 0 ..< warmup {
            let cb = commandQueue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(inputBuffer, offset: 0, index: 0)
            enc.setBuffer(outputBuffer, offset: 0, index: 1)
            enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }

        // Timed runs
        var times: [Double] = []
        for _ in 0 ..< repeats {
            let cb = commandQueue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(inputBuffer, offset: 0, index: 0)
            enc.setBuffer(outputBuffer, offset: 0, index: 1)
            enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            times.append((cb.gpuEndTime - cb.gpuStartTime) * 1e6) // microseconds
        }

        times.sort()
        return times[times.count / 2]
    }
}

// ============================================================================
// MMA (simdgroup_matrix) Kernel Host
// ============================================================================

struct FFTHostMMA {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipeline: MTLComputePipelineState
    let dftRealBuffer: MTLBuffer
    let dftImagBuffer: MTLBuffer

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device found")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Could not create command queue")
        }
        commandQueue = queue

        // Load Metal shader source
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        let searchPaths = [
            (execDir as NSString).appendingPathComponent("fft_4096_mma.metal"),
            (execDir as NSString).appendingPathComponent("FFTHost_FFTHost.bundle/fft_4096_mma.metal"),
            (execDir as NSString).appendingPathComponent("../Sources/fft_4096_mma.metal"),
            (execDir as NSString).appendingPathComponent("../../Sources/fft_4096_mma.metal"),
            (execDir as NSString).appendingPathComponent("../../src/metal/fft_4096_mma.metal"),
            (execDir as NSString).appendingPathComponent("../../../fft_4096_mma.metal")
        ]
        var foundPath: String? = nil
        for path in searchPaths {
            if FileManager.default.fileExists(atPath: path) {
                foundPath = path
                break
            }
        }
        guard let metalPath = foundPath else {
            print("ERROR: Could not find fft_4096_mma.metal")
            print("Searched: \(searchPaths)")
            fatalError("MMA Metal shader not found")
        }
        let metalSource = try String(contentsOfFile: metalPath, encoding: .utf8)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: metalSource, options: options)
        guard let function = library.makeFunction(name: "fft_4096_mma") else {
            fatalError("Could not find kernel function 'fft_4096_mma'")
        }
        pipeline = try device.makeComputePipelineState(function: function)

        // Precompute DFT_8 real and imaginary matrices (row-major 8x8)
        let sqrt2_2: Float = 0.70710678118654752
        let f8Real: [Float] = [
            // Row 0
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            // Row 1
            1.0, sqrt2_2, 0.0, -sqrt2_2, -1.0, -sqrt2_2, 0.0, sqrt2_2,
            // Row 2
            1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0,
            // Row 3
            1.0, -sqrt2_2, 0.0, sqrt2_2, -1.0, sqrt2_2, 0.0, -sqrt2_2,
            // Row 4
            1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
            // Row 5
            1.0, -sqrt2_2, 0.0, sqrt2_2, -1.0, sqrt2_2, 0.0, -sqrt2_2,
            // Row 6
            1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0,
            // Row 7
            1.0, sqrt2_2, 0.0, -sqrt2_2, -1.0, -sqrt2_2, 0.0, sqrt2_2
        ]
        let f8Imag: [Float] = [
            // Row 0
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // Row 1
            0.0, -sqrt2_2, -1.0, -sqrt2_2, 0.0, sqrt2_2, 1.0, sqrt2_2,
            // Row 2
            0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0,
            // Row 3
            0.0, -sqrt2_2, 1.0, -sqrt2_2, 0.0, sqrt2_2, -1.0, sqrt2_2,
            // Row 4
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // Row 5
            0.0, sqrt2_2, -1.0, sqrt2_2, 0.0, -sqrt2_2, 1.0, -sqrt2_2,
            // Row 6
            0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0,
            // Row 7
            0.0, sqrt2_2, 1.0, sqrt2_2, 0.0, -sqrt2_2, -1.0, -sqrt2_2
        ]

        dftRealBuffer = device.makeBuffer(
            bytes: f8Real,
            length: 64 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        dftImagBuffer = device.makeBuffer(
            bytes: f8Imag,
            length: 64 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!

        print("MMA Kernel:")
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

        let threadsPerThreadgroup = MTLSizeMake(512, 1, 1)
        let threadgroups = MTLSizeMake(batchSize, 1, 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPtr = outputBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: input.count)
        return Array(UnsafeBufferPointer(start: outputPtr, count: input.count))
    }

    func timeFFT(input: [SIMD2<Float>], batchSize: Int = 1, warmup: Int = 3, repeats: Int = 10) -> Double {
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

        for _ in 0 ..< warmup {
            let cb = commandQueue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(inputBuffer, offset: 0, index: 0)
            enc.setBuffer(outputBuffer, offset: 0, index: 1)
            enc.setBuffer(dftRealBuffer, offset: 0, index: 2)
            enc.setBuffer(dftImagBuffer, offset: 0, index: 3)
            enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }

        var times: [Double] = []
        for _ in 0 ..< repeats {
            let cb = commandQueue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(inputBuffer, offset: 0, index: 0)
            enc.setBuffer(outputBuffer, offset: 0, index: 1)
            enc.setBuffer(dftRealBuffer, offset: 0, index: 2)
            enc.setBuffer(dftImagBuffer, offset: 0, index: 3)
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

    // Convert interleaved to split complex
    var realIn = [Float](repeating: 0, count: n)
    var imagIn = [Float](repeating: 0, count: n)
    for i in 0 ..< n {
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

    // Convert split complex back to interleaved
    var result = [SIMD2<Float>](repeating: .zero, count: n)
    for i in 0 ..< n {
        result[i] = SIMD2<Float>(realOut[i], imagOut[i])
    }
    return result
}

// ============================================================================
// Test Signal Generation
// ============================================================================

func generateTestSignal(n: Int) -> [SIMD2<Float>] {
    // Sum of sinusoids at known frequencies for easy verification:
    //   x[k] = cos(2pi * 100 * k / N) + 0.5 * cos(2pi * 200 * k / N)
    //         + 0.3 * sin(2pi * 500 * k / N)
    var signal = [SIMD2<Float>](repeating: .zero, count: n)
    let invN = 2.0 * Float.pi / Float(n)
    for k in 0 ..< n {
        let t = Float(k) * invN
        let real = cos(100.0 * t) + 0.5 * cos(200.0 * t) + 0.3 * sin(500.0 * t)
        signal[k] = SIMD2<Float>(real, 0.0)
    }
    return signal
}

func generateRandomSignal(n: Int) -> [SIMD2<Float>] {
    var signal = [SIMD2<Float>](repeating: .zero, count: n)
    for k in 0 ..< n {
        signal[k] = SIMD2<Float>(Float.random(in: -1 ... 1), Float.random(in: -1 ... 1))
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

    for i in 0 ..< n {
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
    // Float32 FFT with N=4096: ~12 stages, expect ~sqrt(12)*eps ~= 1e-6 L2 error.
    // Max relative error on small bins can be large; use L2 as primary pass criterion.
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

// ============================================================================
// Generic validation runner (works with either kernel)
// ============================================================================

func runValidationSuite(
    label: String,
    runFFT: ([SIMD2<Float>], Int) -> [SIMD2<Float>],
    timeFFT: ([SIMD2<Float>], Int) -> Double
) -> Bool {
    let n = 4096
    var allPassed = true

    print("=" * 72)
    print("\(label) — Validation")
    print("=" * 72)
    print()

    // --- Test 1: Known sinusoidal signal ---
    print("-- Test 1: Sinusoidal signal (real-valued input) --")
    let sinSignal = generateTestSignal(n: n)
    let metalResult1 = runFFT(sinSignal, 1)
    let vdspResult1 = vdspFFT4096(input: sinSignal)
    allPassed = validate(metal: metalResult1, reference: vdspResult1, label: "sinusoids") && allPassed

    // --- Test 2: Random complex signal ---
    print("-- Test 2: Random complex signal --")
    let randSignal = generateRandomSignal(n: n)
    let metalResult2 = runFFT(randSignal, 1)
    let vdspResult2 = vdspFFT4096(input: randSignal)
    allPassed = validate(metal: metalResult2, reference: vdspResult2, label: "random") && allPassed

    // --- Test 3: Impulse (delta function) ---
    print("-- Test 3: Impulse (delta at k=0) --")
    var impulse = [SIMD2<Float>](repeating: .zero, count: n)
    impulse[0] = SIMD2<Float>(1.0, 0.0)
    let metalResult3 = runFFT(impulse, 1)
    let vdspResult3 = vdspFFT4096(input: impulse)
    allPassed = validate(metal: metalResult3, reference: vdspResult3, label: "impulse") && allPassed

    // --- Test 4: DC signal (all ones) ---
    print("-- Test 4: DC signal (all ones) --")
    let dc = [SIMD2<Float>](repeating: SIMD2<Float>(1.0, 0.0), count: n)
    let metalResult4 = runFFT(dc, 1)
    let vdspResult4 = vdspFFT4096(input: dc)
    allPassed = validate(metal: metalResult4, reference: vdspResult4, label: "dc") && allPassed

    // --- Test 5: Batch of 16 random signals ---
    print("-- Test 5: Batch of 16 random signals --")
    let batchSize = 16
    var batchInput = [SIMD2<Float>](repeating: .zero, count: n * batchSize)
    for i in 0 ..< (n * batchSize) {
        batchInput[i] = SIMD2<Float>(Float.random(in: -1 ... 1), Float.random(in: -1 ... 1))
    }
    let metalBatch = runFFT(batchInput, batchSize)
    var batchPassed = true
    for b in 0 ..< batchSize {
        let slice = Array(batchInput[b * n ..< (b + 1) * n])
        let ref = vdspFFT4096(input: slice)
        let metalSlice = Array(metalBatch[b * n ..< (b + 1) * n])
        let ok = validate(metal: metalSlice, reference: ref, label: "batch[\(b)]")
        batchPassed = batchPassed && ok
    }
    allPassed = allPassed && batchPassed

    // --- Performance ---
    print("=" * 72)
    print("\(label) — Performance Benchmarking")
    print("=" * 72)
    print()

    for bs in [1, 16, 64, 256] {
        var bi = [SIMD2<Float>](repeating: .zero, count: n * bs)
        for i in 0 ..< bi.count {
            bi[i] = SIMD2<Float>(Float.random(in: -1 ... 1), Float.random(in: -1 ... 1))
        }
        let us = timeFFT(bi, bs)
        let totalFFTs = Double(bs)
        let flopsPerFFT = 5.0 * Double(n) * log2(Double(n))
        let gflops = totalFFTs * flopsPerFFT / (us * 1e-6) / 1e9
        let usPerFFT = us / totalFFTs
        print(String(format: "  Batch %4d:  %8.1f us total  %8.2f us/FFT  %8.2f GFLOPS",
                     bs, us, usPerFFT, gflops))
    }
    print()

    return allPassed
}

// ============================================================================
// Main
// ============================================================================

@main
struct FFTHostMain {
    static func main() throws {
        var allPassed = true

        // --- Stockham Kernel ---
        let stockham = try FFTHost()
        let stockhamPassed = runValidationSuite(
            label: "FFT 4096 Stockham (radix-4)",
            runFFT: { input, batch in stockham.runFFT(input: input, batchSize: batch) },
            timeFFT: { input, batch in stockham.timeFFT(input: input, batchSize: batch) }
        )
        allPassed = allPassed && stockhamPassed

        // --- MMA Kernel ---
        let mma = try FFTHostMMA()
        let mmaPassed = runValidationSuite(
            label: "FFT 4096 MMA (simdgroup_matrix radix-8)",
            runFFT: { input, batch in mma.runFFT(input: input, batchSize: batch) },
            timeFFT: { input, batch in mma.timeFFT(input: input, batchSize: batch) }
        )
        allPassed = allPassed && mmaPassed

        // --- Final Summary ---
        print("=" * 72)
        print("FINAL SUMMARY")
        print("=" * 72)
        print("  Stockham: \(stockhamPassed ? "ALL PASSED" : "SOME FAILED")")
        print("  MMA:      \(mmaPassed ? "ALL PASSED" : "SOME FAILED")")
        print()
        if allPassed {
            print("ALL TESTS PASSED")
        } else {
            print("SOME TESTS FAILED")
        }
        print("=" * 72)
    }
}

// MARK: - Utilities

extension String {
    static func * (lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}
