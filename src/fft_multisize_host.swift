// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

import Metal
import Accelerate
import Foundation

// ============================================================================
// Multi-size FFT Host — Metal Kernel Validation & Benchmarking
// Supports N = 256, 512, 1024, 2048, 4096, 8192, 16384
// ============================================================================

struct FFTSize {
    let n: Int
    let kernelName: String
    let threadsPerGroup: Int
    let log2n: Int

    init(_ n: Int, kernel: String, threads: Int) {
        self.n = n
        self.kernelName = kernel
        self.threadsPerGroup = threads
        var v = n; var l = 0; while v > 1 { v >>= 1; l += 1 }
        self.log2n = l
    }
}

let singlePassSizes: [FFTSize] = [
    FFTSize(256,  kernel: "fft_256_stockham",  threads: 64),
    FFTSize(512,  kernel: "fft_512_stockham",  threads: 128),
    FFTSize(1024, kernel: "fft_1024_stockham", threads: 256),
    FFTSize(2048, kernel: "fft_2048_stockham", threads: 512),
    FFTSize(4096, kernel: "fft_4096_stockham", threads: 1024),
]

struct FourStepConfig {
    let n: Int
    let n1: Int
    let n2: Int
    let log2n: Int
    let pass1Kernel: String
    let pass1Threads: Int
    let pass2Kernel: String
    let pass2Threads: Int
}

let fourStepSizes: [FourStepConfig] = [
    // N=8192 = N1*N2 = 64*128
    // Step 1: N1=64 row-FFTs of size N2=128, Step 3: N2=128 row-FFTs of size N1=64
    FourStepConfig(n: 8192, n1: 64, n2: 128, log2n: 13,
                   pass1Kernel: "fft_128_stockham", pass1Threads: 32,
                   pass2Kernel: "fft_64_stockham", pass2Threads: 16),
    // N=16384 = N1*N2 = 128*128
    // Step 1: 128 row-FFTs of size 128, Step 3: 128 row-FFTs of size 128
    FourStepConfig(n: 16384, n1: 128, n2: 128, log2n: 14,
                   pass1Kernel: "fft_128_stockham", pass1Threads: 32,
                   pass2Kernel: "fft_128_stockham", pass2Threads: 32),
]

struct FFTMultiHost {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var pipelines: [String: MTLComputePipelineState] = [:]

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device found")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Could not create command queue")
        }
        self.commandQueue = queue

        // Load Metal shader source
        let metalSource: String
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        let searchPaths = [
            (execDir as NSString).appendingPathComponent("fft_multisize.metal"),
            (execDir as NSString).appendingPathComponent("../../../fft_multisize.metal"),
            (execDir as NSString).appendingPathComponent("../../src/metal/fft_multisize.metal"),
            (execDir as NSString).appendingPathComponent("../../../src/metal/fft_multisize.metal"),
        ]
        var foundPath: String? = nil
        for path in searchPaths {
            if FileManager.default.fileExists(atPath: path) {
                foundPath = path
                break
            }
        }
        guard let metalPath = foundPath else {
            print("ERROR: Could not find fft_multisize.metal")
            print("Searched: \(searchPaths)")
            fatalError("Metal shader not found")
        }
        metalSource = try String(contentsOfFile: metalPath, encoding: .utf8)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: metalSource, options: options)

        // Build pipelines for all kernels
        let kernelNames = [
            "fft_256_stockham", "fft_512_stockham", "fft_1024_stockham",
            "fft_2048_stockham", "fft_4096_stockham",
            "fft_64_stockham", "fft_128_stockham",
            "fft_twiddle_transpose", "fft_transpose",
        ]
        for name in kernelNames {
            guard let function = library.makeFunction(name: name) else {
                fatalError("Could not find kernel function '\(name)'")
            }
            pipelines[name] = try device.makeComputePipelineState(function: function)
        }

        print("Device: \(device.name)")
        print("Compiled \(pipelines.count) kernel variants")
        print()
    }

    // Single-pass FFT for N <= 4096
    func runSinglePassFFT(input: [SIMD2<Float>], size: FFTSize, batchSize: Int = 1) -> [SIMD2<Float>] {
        assert(input.count == size.n * batchSize)
        let pipeline = pipelines[size.kernelName]!

        let inputBuffer = device.makeBuffer(bytes: input,
            length: input.count * MemoryLayout<SIMD2<Float>>.stride,
            options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(
            length: input.count * MemoryLayout<SIMD2<Float>>.stride,
            options: .storageModeShared)!

        let cb = commandQueue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inputBuffer, offset: 0, index: 0)
        enc.setBuffer(outputBuffer, offset: 0, index: 1)
        enc.dispatchThreadgroups(MTLSizeMake(batchSize, 1, 1),
                                 threadsPerThreadgroup: MTLSizeMake(size.threadsPerGroup, 1, 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let ptr = outputBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: input.count)
        return Array(UnsafeBufferPointer(start: ptr, count: input.count))
    }

    // Four-step FFT for N = N1 * N2 (8192, 16384)
    //
    // View input as N2 rows x N1 cols: x[n2*N1 + n1]
    // Step 0: Transpose N2 x N1 -> N1 x N2 (so rows become columns)
    // Step 1: N1 row-FFTs of size N2
    // Step 2: Twiddle W_N^{row*col} + transpose N1 x N2 -> N2 x N1
    // Step 3: N2 row-FFTs of size N1
    // Step 4: Transpose N2 x N1 -> N1 x N2 (standard output order)
    //
    // Result: output[N2*k1+k2] = X[N2*k1+k2] = DFT(x)[k]
    func runFourStepFFT(input: [SIMD2<Float>], config: FourStepConfig, batchSize: Int = 1) -> [SIMD2<Float>] {
        let n = config.n
        assert(input.count == n * batchSize)

        let transposePipeline = pipelines["fft_transpose"]!
        let pass1Pipeline = pipelines[config.pass1Kernel]!    // size N2 FFT
        let twiddlePipeline = pipelines["fft_twiddle_transpose"]!
        let pass2Pipeline = pipelines[config.pass2Kernel]!    // size N1 FFT

        let byteCount = input.count * MemoryLayout<SIMD2<Float>>.stride
        let inputBuffer = device.makeBuffer(bytes: input, length: byteCount, options: .storageModeShared)!
        let tempA = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        let tempB = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        let tempC = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        let tempD = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        var n1Val = UInt32(config.n1)
        var n2Val = UInt32(config.n2)
        let n1Buf = device.makeBuffer(bytes: &n1Val, length: 4, options: .storageModeShared)!
        let n2Buf = device.makeBuffer(bytes: &n2Val, length: 4, options: .storageModeShared)!

        let elemThreads = min(256, n)
        let elemTGs = (n + elemThreads - 1) / elemThreads

        for b in 0..<batchSize {
            let batchOffset = b * n * MemoryLayout<SIMD2<Float>>.stride

            let cb = commandQueue.makeCommandBuffer()!

            // Step 0: Transpose input from N2 x N1 -> N1 x N2
            let enc0 = cb.makeComputeCommandEncoder()!
            enc0.setComputePipelineState(transposePipeline)
            enc0.setBuffer(inputBuffer, offset: batchOffset, index: 0)
            enc0.setBuffer(tempA, offset: batchOffset, index: 1)
            enc0.setBuffer(n2Buf, offset: 0, index: 2)  // ROWS = N2
            enc0.setBuffer(n1Buf, offset: 0, index: 3)  // COLS = N1
            enc0.dispatchThreadgroups(MTLSizeMake(elemTGs, 1, 1),
                                      threadsPerThreadgroup: MTLSizeMake(elemThreads, 1, 1))
            enc0.endEncoding()

            // Step 1: N1 row-FFTs of size N2 (input is N1 x N2, N1 threadgroups)
            let enc1 = cb.makeComputeCommandEncoder()!
            enc1.setComputePipelineState(pass1Pipeline)
            enc1.setBuffer(tempA, offset: batchOffset, index: 0)
            enc1.setBuffer(tempB, offset: batchOffset, index: 1)
            enc1.dispatchThreadgroups(MTLSizeMake(config.n1, 1, 1),
                                      threadsPerThreadgroup: MTLSizeMake(config.pass1Threads, 1, 1))
            enc1.endEncoding()

            // Step 2: Twiddle W_N^{row*col} + transpose N1 x N2 -> N2 x N1
            let enc2 = cb.makeComputeCommandEncoder()!
            enc2.setComputePipelineState(twiddlePipeline)
            enc2.setBuffer(tempB, offset: batchOffset, index: 0)
            enc2.setBuffer(tempC, offset: batchOffset, index: 1)
            enc2.setBuffer(n1Buf, offset: 0, index: 2)  // N1 (rows of input)
            enc2.setBuffer(n2Buf, offset: 0, index: 3)  // N2 (cols of input)
            enc2.dispatchThreadgroups(MTLSizeMake(elemTGs, 1, 1),
                                      threadsPerThreadgroup: MTLSizeMake(elemThreads, 1, 1))
            enc2.endEncoding()

            // Step 3: N2 row-FFTs of size N1 (input is N2 x N1, N2 threadgroups)
            let enc3 = cb.makeComputeCommandEncoder()!
            enc3.setComputePipelineState(pass2Pipeline)
            enc3.setBuffer(tempC, offset: batchOffset, index: 0)
            enc3.setBuffer(tempD, offset: batchOffset, index: 1)
            enc3.dispatchThreadgroups(MTLSizeMake(config.n2, 1, 1),
                                      threadsPerThreadgroup: MTLSizeMake(config.pass2Threads, 1, 1))
            enc3.endEncoding()

            // Step 4: Transpose N2 x N1 -> N1 x N2 (standard output order)
            let enc4 = cb.makeComputeCommandEncoder()!
            enc4.setComputePipelineState(transposePipeline)
            enc4.setBuffer(tempD, offset: batchOffset, index: 0)
            enc4.setBuffer(outputBuffer, offset: batchOffset, index: 1)
            enc4.setBuffer(n2Buf, offset: 0, index: 2)  // ROWS = N2
            enc4.setBuffer(n1Buf, offset: 0, index: 3)  // COLS = N1
            enc4.dispatchThreadgroups(MTLSizeMake(elemTGs, 1, 1),
                                      threadsPerThreadgroup: MTLSizeMake(elemThreads, 1, 1))
            enc4.endEncoding()

            cb.commit()
            cb.waitUntilCompleted()
        }

        let ptr = outputBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: input.count)
        return Array(UnsafeBufferPointer(start: ptr, count: input.count))
    }

    // Unified dispatch
    func runFFT(input: [SIMD2<Float>], n: Int, batchSize: Int = 1) -> [SIMD2<Float>] {
        if let size = singlePassSizes.first(where: { $0.n == n }) {
            return runSinglePassFFT(input: input, size: size, batchSize: batchSize)
        }
        if let config = fourStepSizes.first(where: { $0.n == n }) {
            return runFourStepFFT(input: input, config: config, batchSize: batchSize)
        }
        fatalError("Unsupported FFT size: \(n)")
    }

    // Timing for single-pass sizes
    func timeSinglePassFFT(size: FFTSize, batchSize: Int, warmup: Int = 3, repeats: Int = 10) -> Double {
        let n = size.n
        let pipeline = pipelines[size.kernelName]!
        var input = [SIMD2<Float>](repeating: .zero, count: n * batchSize)
        for i in 0..<input.count { input[i] = SIMD2<Float>(Float.random(in: -1...1), Float.random(in: -1...1)) }

        let inputBuffer = device.makeBuffer(bytes: input,
            length: input.count * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(
            length: input.count * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!

        let tg = MTLSizeMake(batchSize, 1, 1)
        let tpt = MTLSizeMake(size.threadsPerGroup, 1, 1)

        for _ in 0..<warmup {
            let cb = commandQueue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(inputBuffer, offset: 0, index: 0)
            enc.setBuffer(outputBuffer, offset: 0, index: 1)
            enc.dispatchThreadgroups(tg, threadsPerThreadgroup: tpt)
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
        }

        var times: [Double] = []
        for _ in 0..<repeats {
            let cb = commandQueue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(inputBuffer, offset: 0, index: 0)
            enc.setBuffer(outputBuffer, offset: 0, index: 1)
            enc.dispatchThreadgroups(tg, threadsPerThreadgroup: tpt)
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
            times.append((cb.gpuEndTime - cb.gpuStartTime) * 1e6)
        }
        times.sort()
        return times[times.count / 2]
    }

    // Timing for four-step sizes (5 GPU dispatches: transpose, FFT, twiddle+transpose, FFT, transpose)
    func timeFourStepFFT(config: FourStepConfig, batchSize: Int, warmup: Int = 3, repeats: Int = 10) -> Double {
        let n = config.n
        var input = [SIMD2<Float>](repeating: .zero, count: n * batchSize)
        for i in 0..<input.count { input[i] = SIMD2<Float>(Float.random(in: -1...1), Float.random(in: -1...1)) }

        let byteCount = input.count * MemoryLayout<SIMD2<Float>>.stride
        let inputBuffer = device.makeBuffer(bytes: input, length: byteCount, options: .storageModeShared)!
        let tempA = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        let tempB = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        let tempC = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        let tempD = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        var n1Val = UInt32(config.n1)
        var n2Val = UInt32(config.n2)
        let n1Buf = device.makeBuffer(bytes: &n1Val, length: 4, options: .storageModeShared)!
        let n2Buf = device.makeBuffer(bytes: &n2Val, length: 4, options: .storageModeShared)!

        let transposePipeline = pipelines["fft_transpose"]!
        let pass1Pipeline = pipelines[config.pass1Kernel]!
        let twiddlePipeline = pipelines["fft_twiddle_transpose"]!
        let pass2Pipeline = pipelines[config.pass2Kernel]!

        let elemThreads = min(256, n)
        let elemTGs = (n + elemThreads - 1) / elemThreads

        func encodeFiveStep(cb: MTLCommandBuffer) {
            // Step 0: Transpose N2 x N1 -> N1 x N2
            let enc0 = cb.makeComputeCommandEncoder()!
            enc0.setComputePipelineState(transposePipeline)
            enc0.setBuffer(inputBuffer, offset: 0, index: 0)
            enc0.setBuffer(tempA, offset: 0, index: 1)
            enc0.setBuffer(n2Buf, offset: 0, index: 2)
            enc0.setBuffer(n1Buf, offset: 0, index: 3)
            enc0.dispatchThreadgroups(MTLSizeMake(elemTGs * batchSize, 1, 1),
                                      threadsPerThreadgroup: MTLSizeMake(elemThreads, 1, 1))
            enc0.endEncoding()

            // Step 1: N1 row-FFTs of size N2
            let enc1 = cb.makeComputeCommandEncoder()!
            enc1.setComputePipelineState(pass1Pipeline)
            enc1.setBuffer(tempA, offset: 0, index: 0)
            enc1.setBuffer(tempB, offset: 0, index: 1)
            enc1.dispatchThreadgroups(MTLSizeMake(config.n1 * batchSize, 1, 1),
                                      threadsPerThreadgroup: MTLSizeMake(config.pass1Threads, 1, 1))
            enc1.endEncoding()

            // Step 2: Twiddle + transpose N1 x N2 -> N2 x N1
            let enc2 = cb.makeComputeCommandEncoder()!
            enc2.setComputePipelineState(twiddlePipeline)
            enc2.setBuffer(tempB, offset: 0, index: 0)
            enc2.setBuffer(tempC, offset: 0, index: 1)
            enc2.setBuffer(n1Buf, offset: 0, index: 2)
            enc2.setBuffer(n2Buf, offset: 0, index: 3)
            enc2.dispatchThreadgroups(MTLSizeMake(elemTGs * batchSize, 1, 1),
                                      threadsPerThreadgroup: MTLSizeMake(elemThreads, 1, 1))
            enc2.endEncoding()

            // Step 3: N2 row-FFTs of size N1
            let enc3 = cb.makeComputeCommandEncoder()!
            enc3.setComputePipelineState(pass2Pipeline)
            enc3.setBuffer(tempC, offset: 0, index: 0)
            enc3.setBuffer(tempD, offset: 0, index: 1)
            enc3.dispatchThreadgroups(MTLSizeMake(config.n2 * batchSize, 1, 1),
                                      threadsPerThreadgroup: MTLSizeMake(config.pass2Threads, 1, 1))
            enc3.endEncoding()

            // Step 4: Transpose N2 x N1 -> N1 x N2
            let enc4 = cb.makeComputeCommandEncoder()!
            enc4.setComputePipelineState(transposePipeline)
            enc4.setBuffer(tempD, offset: 0, index: 0)
            enc4.setBuffer(outputBuffer, offset: 0, index: 1)
            enc4.setBuffer(n2Buf, offset: 0, index: 2)
            enc4.setBuffer(n1Buf, offset: 0, index: 3)
            enc4.dispatchThreadgroups(MTLSizeMake(elemTGs * batchSize, 1, 1),
                                      threadsPerThreadgroup: MTLSizeMake(elemThreads, 1, 1))
            enc4.endEncoding()
        }

        for _ in 0..<warmup {
            let cb = commandQueue.makeCommandBuffer()!
            encodeFiveStep(cb: cb)
            cb.commit(); cb.waitUntilCompleted()
        }

        var times: [Double] = []
        for _ in 0..<repeats {
            let cb = commandQueue.makeCommandBuffer()!
            encodeFiveStep(cb: cb)
            cb.commit(); cb.waitUntilCompleted()
            times.append((cb.gpuEndTime - cb.gpuStartTime) * 1e6)
        }
        times.sort()
        return times[times.count / 2]
    }

    func timeFFT(n: Int, batchSize: Int, warmup: Int = 3, repeats: Int = 10) -> Double {
        if let size = singlePassSizes.first(where: { $0.n == n }) {
            return timeSinglePassFFT(size: size, batchSize: batchSize, warmup: warmup, repeats: repeats)
        }
        if let config = fourStepSizes.first(where: { $0.n == n }) {
            return timeFourStepFFT(config: config, batchSize: batchSize, warmup: warmup, repeats: repeats)
        }
        fatalError("Unsupported FFT size: \(n)")
    }
}

// ============================================================================
// vDSP Reference FFT (generic size)
// ============================================================================

func vdspFFT(input: [SIMD2<Float>], n: Int) -> [SIMD2<Float>] {
    var log2n_val = 0
    var v = n; while v > 1 { v >>= 1; log2n_val += 1 }
    let log2n = vDSP_Length(log2n_val)

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
// Validation
// ============================================================================

func generateRandomSignal(n: Int) -> [SIMD2<Float>] {
    var signal = [SIMD2<Float>](repeating: .zero, count: n)
    for k in 0..<n {
        signal[k] = SIMD2<Float>(Float.random(in: -1...1), Float.random(in: -1...1))
    }
    return signal
}

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
        if relErr > maxRelError { maxRelError = relErr }

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
struct FFTMultiMain {
    static func main() throws {
        print("=" * 72)
        print("Multi-size FFT Stockham — Metal Kernel Validation & Benchmarking")
        print("=" * 72)
        print()

        let host = try FFTMultiHost()
        var allPassed = true

        let allSizes = [256, 512, 1024, 2048, 4096, 8192, 16384]

        // --- Validation for each size ---
        for n in allSizes {
            print("-" * 72)
            print("Validating N = \(n)")
            print("-" * 72)

            // Random complex signal
            let signal = generateRandomSignal(n: n)
            let metalResult = host.runFFT(input: signal, n: n)
            let vdspResult = vdspFFT(input: signal, n: n)
            let ok = validate(metal: metalResult, reference: vdspResult, label: "N=\(n) random")
            allPassed = allPassed && ok

            // Impulse test
            var impulse = [SIMD2<Float>](repeating: .zero, count: n)
            impulse[0] = SIMD2<Float>(1.0, 0.0)
            let metalImpulse = host.runFFT(input: impulse, n: n)
            let vdspImpulse = vdspFFT(input: impulse, n: n)
            let ok2 = validate(metal: metalImpulse, reference: vdspImpulse, label: "N=\(n) impulse")
            allPassed = allPassed && ok2
        }

        // --- Performance ---
        print()
        print("=" * 72)
        print("Performance Benchmarking")
        print("=" * 72)
        print()

        for n in allSizes {
            print("N = \(n):")
            for batchSize in [1, 64] {
                let us = host.timeFFT(n: n, batchSize: batchSize)
                let totalFFTs = Double(batchSize)
                let flopsPerFFT = 5.0 * Double(n) * log2(Double(n))
                let gflops = totalFFTs * flopsPerFFT / (us * 1e-6) / 1e9
                let usPerFFT = us / totalFFTs
                print(String(format: "  Batch %4d:  %8.1f us total  %8.2f us/FFT  %8.2f GFLOPS",
                             batchSize, us, usPerFFT, gflops))
            }
            print()
        }

        // --- Summary ---
        print("=" * 72)
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
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}
