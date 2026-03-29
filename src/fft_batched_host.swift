// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

import Metal
import Accelerate
import Foundation

// ============================================================================
// Batched FFT 4096 — Metal Host with simdgroup_matrix MMA Kernel
//
// Validates batched FFT against vDSP and benchmarks at multiple batch sizes.
// Each threadgroup processes one N=4096 FFT; batching via dispatch.
// ============================================================================

// MARK: - vDSP Reference

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

// MARK: - Batched MMA Kernel Host

struct BatchedFFTHost {
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
        self.commandQueue = queue

        // Find the Metal shader
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        let searchPaths = [
            (execDir as NSString).appendingPathComponent("fft_4096_batched.metal"),
            (execDir as NSString).appendingPathComponent("../Sources/fft_4096_batched.metal"),
            (execDir as NSString).appendingPathComponent("../../Sources/fft_4096_batched.metal"),
            (execDir as NSString).appendingPathComponent("../../src/metal/fft_4096_batched.metal"),
            (execDir as NSString).appendingPathComponent("../../../fft_4096_batched.metal"),
        ]
        var foundPath: String? = nil
        for path in searchPaths {
            if FileManager.default.fileExists(atPath: path) {
                foundPath = path
                break
            }
        }
        guard let metalPath = foundPath else {
            print("ERROR: Could not find fft_4096_batched.metal")
            print("Searched: \(searchPaths)")
            fatalError("Metal shader not found")
        }
        let metalSource = try String(contentsOfFile: metalPath, encoding: .utf8)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        if #available(macOS 14.0, *) {
            options.languageVersion = .version3_1
        }
        print("  Compiling Metal shader...")
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: metalSource, options: options)
            print("  Metal library compiled successfully")
        } catch {
            print("Metal compilation error: \(error)")
            throw error
        }
        guard let function = library.makeFunction(name: "fft_4096_batched") else {
            fatalError("Could not find kernel function 'fft_4096_batched'")
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
        print("  Pipeline state created successfully")

        // Precompute DFT_8 matrices
        let sqrt2_2: Float = 0.70710678118654752
        let f8Real: [Float] = [
             1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
             1.0,  sqrt2_2,  0.0, -sqrt2_2, -1.0, -sqrt2_2,  0.0,  sqrt2_2,
             1.0,  0.0, -1.0,  0.0,  1.0,  0.0, -1.0,  0.0,
             1.0, -sqrt2_2,  0.0,  sqrt2_2, -1.0,  sqrt2_2,  0.0, -sqrt2_2,
             1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,
             1.0, -sqrt2_2,  0.0,  sqrt2_2, -1.0,  sqrt2_2,  0.0, -sqrt2_2,
             1.0,  0.0, -1.0,  0.0,  1.0,  0.0, -1.0,  0.0,
             1.0,  sqrt2_2,  0.0, -sqrt2_2, -1.0, -sqrt2_2,  0.0,  sqrt2_2,
        ]
        let f8Imag: [Float] = [
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0, -sqrt2_2, -1.0, -sqrt2_2,  0.0,  sqrt2_2,  1.0,  sqrt2_2,
             0.0, -1.0,  0.0,  1.0,  0.0, -1.0,  0.0,  1.0,
             0.0, -sqrt2_2,  1.0, -sqrt2_2,  0.0,  sqrt2_2, -1.0,  sqrt2_2,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  sqrt2_2, -1.0,  sqrt2_2,  0.0, -sqrt2_2,  1.0, -sqrt2_2,
             0.0,  1.0,  0.0, -1.0,  0.0,  1.0,  0.0, -1.0,
             0.0,  sqrt2_2,  1.0,  sqrt2_2,  0.0, -sqrt2_2, -1.0, -sqrt2_2,
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

        print("Batched FFT Kernel (scalar radix-8):")
        print("  Device:                  \(device.name)")
        print("  Max threads/threadgroup: \(pipeline.maxTotalThreadsPerThreadgroup)")
        print("  Thread execution width:  \(pipeline.threadExecutionWidth)")
        if pipeline.maxTotalThreadsPerThreadgroup < 512 {
            print("  WARNING: Kernel needs 512 threads, device supports \(pipeline.maxTotalThreadsPerThreadgroup).")
        }
        print()
    }

    func runFFT(input: [SIMD2<Float>], batchSize: Int) -> [SIMD2<Float>] {
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

    func timeFFT(input: [SIMD2<Float>], batchSize: Int, warmup: Int = 5, repeats: Int = 20) -> Double {
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

        // Warmup
        for _ in 0..<warmup {
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

        // Timed runs
        var times: [Double] = []
        for _ in 0..<repeats {
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

// MARK: - Validation

func validate(metal: [SIMD2<Float>], reference: [SIMD2<Float>], label: String) -> (Bool, Double) {
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

    print("    [\(label)]")
    print("      Max abs error: \(String(format: "%.6e", maxAbsError)) (bin \(worstIdx))")
    print("      Max rel error: \(String(format: "%.6e", maxRelError))")
    print("      L2 rel error:  \(String(format: "%.6e", l2RelError))")
    print("      Result: \(pass ? "PASS" : "FAIL")")
    if !pass {
        let refVal = reference[worstIdx]
        let metalVal = metal[worstIdx]
        print("      Worst bin \(worstIdx): Metal=(\(metalVal.x), \(metalVal.y))  vDSP=(\(refVal.x), \(refVal.y))")
    }
    return (pass, Double(l2RelError))
}

// MARK: - Main

@main
struct BatchedFFTMain {
    static func main() throws {
        print("Starting BatchedFFTMain...")
        let n = 4096
        let host = try BatchedFFTHost()

        var allPassed = true

        // ================================================================
        // Validation: batch=8 random signals, each validated independently
        // ================================================================
        print(String(repeating: "=", count: 72))
        print("VALIDATION: Batch=8 random complex signals")
        print(String(repeating: "=", count: 72))
        print()

        let valBatch = 8
        var valInput = [SIMD2<Float>](repeating: .zero, count: n * valBatch)
        for i in 0..<valInput.count {
            valInput[i] = SIMD2<Float>(Float.random(in: -1...1), Float.random(in: -1...1))
        }

        let metalResult = host.runFFT(input: valInput, batchSize: valBatch)

        for b in 0..<valBatch {
            let slice = Array(valInput[b * n..<(b + 1) * n])
            let ref = vdspFFT4096(input: slice)
            let metalSlice = Array(metalResult[b * n..<(b + 1) * n])
            let (pass, _) = validate(metal: metalSlice, reference: ref, label: "batch[\(b)]")
            allPassed = allPassed && pass
        }
        print()

        // Additional validation: impulse and DC
        print(String(repeating: "=", count: 72))
        print("VALIDATION: Special signals (impulse, DC)")
        print(String(repeating: "=", count: 72))
        print()

        // Impulse
        var impulse = [SIMD2<Float>](repeating: .zero, count: n)
        impulse[0] = SIMD2<Float>(1.0, 0.0)
        let metalImpulse = host.runFFT(input: impulse, batchSize: 1)
        let refImpulse = vdspFFT4096(input: impulse)
        let (passI, _) = validate(metal: metalImpulse, reference: refImpulse, label: "impulse")
        allPassed = allPassed && passI

        // DC
        let dc = [SIMD2<Float>](repeating: SIMD2<Float>(1.0, 0.0), count: n)
        let metalDC = host.runFFT(input: dc, batchSize: 1)
        let refDC = vdspFFT4096(input: dc)
        let (passDC, _) = validate(metal: metalDC, reference: refDC, label: "dc")
        allPassed = allPassed && passDC

        // Sinusoidal
        var sinSignal = [SIMD2<Float>](repeating: .zero, count: n)
        let invN = 2.0 * Float.pi / Float(n)
        for k in 0..<n {
            let t = Float(k) * invN
            sinSignal[k] = SIMD2<Float>(cos(100.0 * t) + 0.5 * cos(200.0 * t), 0.0)
        }
        let metalSin = host.runFFT(input: sinSignal, batchSize: 1)
        let refSin = vdspFFT4096(input: sinSignal)
        let (passS, _) = validate(metal: metalSin, reference: refSin, label: "sinusoids")
        allPassed = allPassed && passS

        print()

        // ================================================================
        // Benchmarking: batch=8, 16, 64, 256
        // ================================================================
        print(String(repeating: "=", count: 72))
        print("PERFORMANCE BENCHMARKING")
        print(String(repeating: "=", count: 72))
        print()
        print("  Batch       Total (us)     us/FFT     GFLOPS")
        print("  -----       ----------     ------     ------")

        for batchSize in [8, 16, 64, 256] {
            var batchInput = [SIMD2<Float>](repeating: .zero, count: n * batchSize)
            for i in 0..<batchInput.count {
                batchInput[i] = SIMD2<Float>(Float.random(in: -1...1), Float.random(in: -1...1))
            }

            let totalUs = host.timeFFT(input: batchInput, batchSize: batchSize)
            let usPerFFT = totalUs / Double(batchSize)
            let flopsPerFFT = 5.0 * Double(n) * log2(Double(n))
            let gflops = Double(batchSize) * flopsPerFFT / (totalUs * 1e-6) / 1e9

            print(String(format: "  Batch %4d  %10.1f  %10.2f  %10.2f",
                         batchSize, totalUs, usPerFFT, gflops))
        }
        print()

        // ================================================================
        // Summary
        // ================================================================
        print(String(repeating: "=", count: 72))
        print("FINAL RESULT: \(allPassed ? "ALL TESTS PASSED" : "SOME TESTS FAILED")")
        print(String(repeating: "=", count: 72))
    }
}
