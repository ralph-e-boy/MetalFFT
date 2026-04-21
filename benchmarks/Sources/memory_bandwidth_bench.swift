// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

import Foundation
import Metal

// MARK: - GPU Benchmark Harness

struct BenchmarkResult {
    let name: String
    let threadsPerThreadgroup: Int
    let iterations: Int
    let elapsedMs: Double
    let throughputGBs: Double?
    let opsPerSecond: Double?
    let notes: String
}

class GPUBenchmarkHarness {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw BenchError.noDevice
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw BenchError.noQueue
        }
        commandQueue = queue

        // Load Metal shaders from .metal file adjacent to executable or source dir
        let metalSource: String
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        let searchPaths = [
            (execDir as NSString).appendingPathComponent("memory_bandwidth.metal"),
            (execDir as NSString).appendingPathComponent("FFTMicroBenchmarks_FFTMicroBenchmarks.bundle/memory_bandwidth.metal"),
            (execDir as NSString).appendingPathComponent("../Sources/memory_bandwidth.metal"),
            (execDir as NSString).appendingPathComponent("../../Sources/memory_bandwidth.metal")
        ]
        var foundPath: String? = nil
        for path in searchPaths {
            if FileManager.default.fileExists(atPath: path) {
                foundPath = path
                break
            }
        }
        guard let metalPath = foundPath else {
            print("ERROR: Could not find memory_bandwidth.metal")
            print("Searched: \(searchPaths)")
            throw BenchError.noFunction("memory_bandwidth.metal not found")
        }
        metalSource = try String(contentsOfFile: metalPath, encoding: .utf8)

        let options = MTLCompileOptions()
        do {
            library = try device.makeLibrary(source: metalSource, options: options)
        } catch {
            print("ERROR: Metal shader compilation failed:")
            print(error)
            throw error
        }

        printDeviceInfo()
    }

    func printDeviceInfo() {
        print("=" * 72)
        print("FFT Microbenchmark Harness — Apple Silicon GPU")
        print("=" * 72)
        print("Device:                    \(device.name)")
        print("Max threadgroup memory:    \(device.maxThreadgroupMemoryLength) bytes (\(device.maxThreadgroupMemoryLength / 1024) KiB)")
        print("Max threads/threadgroup:   \(device.maxThreadsPerThreadgroup)")
        print("Unified memory:            \(device.hasUnifiedMemory)")
        print("Recommended working set:   \(device.recommendedMaxWorkingSetSize / 1024 / 1024) MiB")
        print("=" * 72)
        print()
    }

    /// Pad a string to the given width with trailing spaces.
    func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }

    func makePipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw BenchError.noFunction(name)
        }
        return try device.makeComputePipelineState(function: function)
    }

    /// Run a compute kernel and return wall-clock GPU execution time in milliseconds.
    func timeKernel(
        pipeline: MTLComputePipelineState,
        threadgroups: MTLSize,
        threadsPerThreadgroup: MTLSize,
        threadgroupMemoryLength: Int = 0,
        buffers: [(MTLBuffer, Int)] = [],
        warmup: Int = 2,
        repeats: Int = 5
    ) throws -> Double {
        func encodeAndRun() throws -> Double {
            guard let cb = commandQueue.makeCommandBuffer() else {
                fputs("  FATAL: makeCommandBuffer returned nil\n", stderr)
                return 0
            }
            guard let encoder = cb.makeComputeCommandEncoder() else {
                fputs("  FATAL: makeComputeCommandEncoder returned nil\n", stderr)
                return 0
            }
            encoder.setComputePipelineState(pipeline)
            for (buf, idx) in buffers {
                encoder.setBuffer(buf, offset: 0, index: idx)
            }
            if threadgroupMemoryLength > 0 {
                encoder.setThreadgroupMemoryLength(threadgroupMemoryLength, index: 0)
            }
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if cb.status == .error {
                fputs("GPU error: \(cb.error?.localizedDescription ?? "unknown")\n", stderr)
            }
            return (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
        }

        // Warmup runs
        for _ in 0 ..< warmup {
            _ = try encodeAndRun()
        }

        // Timed runs
        var times: [Double] = []
        for _ in 0 ..< repeats {
            let ms = try encodeAndRun()
            times.append(ms)
        }

        // Return median
        times.sort()
        return times[times.count / 2]
    }

    func makeBuffer(_ value: UInt32) -> MTLBuffer {
        var val = value
        return device.makeBuffer(bytes: &val, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    }

    func makeOutputBuffer(count: Int) -> MTLBuffer {
        device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared)!
    }
}

// MARK: - Benchmark Suites

extension GPUBenchmarkHarness {
    /// ========================================================================
    /// Benchmark 1: Threadgroup Memory Bandwidth
    /// ========================================================================
    func benchThreadgroupMemory() throws {
        print("── Benchmark 1: Threadgroup Memory Bandwidth ──────────────────────────")
        print()

        let iterations: UInt32 = 1000
        let threadsPerTG = 256
        let numThreadgroups = 64
        let elementsPerThread = 32
        let bytesPerIteration = threadsPerTG * elementsPerThread * MemoryLayout<Float>.size * 2 // read + write
        let totalBytes = Double(bytesPerIteration) * Double(iterations) * Double(numThreadgroups)

        let iterBuf = makeBuffer(iterations)
        let outputBuf = makeOutputBuffer(count: numThreadgroups)
        let tgMemSize = threadsPerTG * elementsPerThread * MemoryLayout<Float>.size

        let patterns = [
            ("tgmem_sequential_rw", "Sequential"),
            ("tgmem_strided_rw", "Strided (stride=N)"),
            ("tgmem_conflict_rw", "Bank-conflict (stride=32)")
        ]

        for (kernelName, label) in patterns {
            let pipeline = try makePipeline(kernelName)
            let ms = try timeKernel(
                pipeline: pipeline,
                threadgroups: MTLSizeMake(numThreadgroups, 1, 1),
                threadsPerThreadgroup: MTLSizeMake(threadsPerTG, 1, 1),
                threadgroupMemoryLength: tgMemSize,
                buffers: [(outputBuf, 0), (iterBuf, 1)]
            )
            let gbps = totalBytes / (ms / 1000.0) / 1e9
            print("  \(pad(label, 28)) \(String(format: "%8.3f", ms)) ms  \(String(format: "%8.1f", gbps)) GB/s")
        }
        print()
    }

    /// ========================================================================
    /// Benchmark 2: SIMD Shuffle Throughput
    /// ========================================================================
    func benchSimdShuffle() throws {
        print("── Benchmark 2: SIMD Shuffle Throughput ────────────────────────────────")
        print()

        let iterations: UInt32 = 10000
        let threadsPerTG = 256
        let numThreadgroups = 64
        let shufflesPerIter = 10 // 5 shuffle + 5 shuffle_xor per iteration
        let totalShuffles = Double(shufflesPerIter) * Double(iterations) * Double(threadsPerTG) * Double(numThreadgroups)

        let iterBuf = makeBuffer(iterations)

        // Float (32-bit) shuffle
        let outputBuf = makeOutputBuffer(count: numThreadgroups)
        let pipeline = try makePipeline("simd_shuffle_throughput")
        let ms = try timeKernel(
            pipeline: pipeline,
            threadgroups: MTLSizeMake(numThreadgroups, 1, 1),
            threadsPerThreadgroup: MTLSizeMake(threadsPerTG, 1, 1),
            buffers: [(outputBuf, 0), (iterBuf, 1)]
        )
        let gops = totalShuffles / (ms / 1000.0) / 1e9
        let bytesPerShuffle = 4.0 // float
        let gbps = gops * bytesPerShuffle
        print("  \(pad("float simd_shuffle", 28)) \(String(format: "%8.3f", ms)) ms  \(String(format: "%8.1f", gops)) Gshuffles/s  \(String(format: "%8.1f", gbps)) GB/s")

        // Float2 (complex, 64-bit) shuffle
        let outputBuf2 = device.makeBuffer(length: numThreadgroups * MemoryLayout<Float>.size * 2, options: .storageModeShared)!
        let pipeline2 = try makePipeline("simd_shuffle_complex_throughput")
        let ms2 = try timeKernel(
            pipeline: pipeline2,
            threadgroups: MTLSizeMake(numThreadgroups, 1, 1),
            threadsPerThreadgroup: MTLSizeMake(threadsPerTG, 1, 1),
            buffers: [(outputBuf2, 0), (iterBuf, 1)]
        )
        let gops2 = totalShuffles / (ms2 / 1000.0) / 1e9
        let gbps2 = gops2 * 8.0 // float2 = 8 bytes
        print("  \(pad("float2 simd_shuffle", 28)) \(String(format: "%8.3f", ms2)) ms  \(String(format: "%8.1f", gops2)) Gshuffles/s  \(String(format: "%8.1f", gbps2)) GB/s")
        print()
    }

    /// ========================================================================
    /// Benchmark 3: Register ↔ Threadgroup Copy
    /// ========================================================================
    func benchRegToTgmem() throws {
        print("── Benchmark 3: Register ↔ Threadgroup Memory Copy ─────────────────────")
        print()

        let iterations: UInt32 = 2000
        let numThreadgroups = 64

        let iterBuf = makeBuffer(iterations)
        let pipeline = try makePipeline("reg_to_tgmem_copy")

        for threadsPerTG in [64, 128, 256, 512] {
            let floatsPerThread = 16
            // Each iteration: write 16 floats + barrier + read 16 floats + barrier
            let bytesPerIter = threadsPerTG * floatsPerThread * MemoryLayout<Float>.size * 2
            let totalBytes = Double(bytesPerIter) * Double(iterations) * Double(numThreadgroups)
            let tgMemSize = threadsPerTG * floatsPerThread * MemoryLayout<Float>.size

            // Skip if threadgroup memory would exceed 32 KiB
            if tgMemSize > 32768 {
                print("  \(pad("\(threadsPerTG) threads", 28)) SKIPPED (tgmem \(tgMemSize) > 32768)")
                continue
            }

            let outputBuf = makeOutputBuffer(count: numThreadgroups)
            let ms = try timeKernel(
                pipeline: pipeline,
                threadgroups: MTLSizeMake(numThreadgroups, 1, 1),
                threadsPerThreadgroup: MTLSizeMake(threadsPerTG, 1, 1),
                threadgroupMemoryLength: tgMemSize,
                buffers: [(outputBuf, 0), (iterBuf, 1)]
            )
            let gbps = totalBytes / (ms / 1000.0) / 1e9
            print("  \(pad("\(threadsPerTG) threads x 16 floats", 28)) \(String(format: "%8.3f", ms)) ms  \(String(format: "%8.1f", gbps)) GB/s")
        }
        print()
    }

    /// ========================================================================
    /// Benchmark 4: Occupancy vs Register Pressure
    /// ========================================================================
    func benchOccupancy() throws {
        print("── Benchmark 4: Occupancy vs Register Pressure ─────────────────────────")
        print()

        let iterations: UInt32 = 500
        let threadsPerTG = 256
        let numThreadgroups = 64
        let tgMemSize = 4096 // 4 KiB — small to not constrain occupancy

        let iterBuf = makeBuffer(iterations)

        let kernels: [(String, String, Int)] = [
            ("occupancy_low_regs", "~8 GPRs  (1 float)", 1),
            ("occupancy_med_regs", "~32 GPRs (16 floats)", 16),
            ("occupancy_high_regs", "~64 GPRs (32 floats)", 32),
            ("occupancy_vhigh_regs", "~128 GPRs (64 floats)", 64)
        ]

        for (kernelName, label, _) in kernels {
            let pipeline = try makePipeline(kernelName)
            let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
            let execWidth = pipeline.threadExecutionWidth

            let outputBuf = makeOutputBuffer(count: numThreadgroups)
            let ms = try timeKernel(
                pipeline: pipeline,
                threadgroups: MTLSizeMake(numThreadgroups, 1, 1),
                threadsPerThreadgroup: MTLSizeMake(min(threadsPerTG, maxThreads), 1, 1),
                threadgroupMemoryLength: tgMemSize,
                buffers: [(outputBuf, 0), (iterBuf, 1)]
            )
            print("  \(pad(label, 28)) \(String(format: "%8.3f", ms)) ms  maxThreads=\(String(format: "%4d", maxThreads))  execWidth=\(execWidth)")
        }
        print()
        print("  Note: maxTotalThreadsPerThreadgroup reflects occupancy — lower values")
        print("  indicate the compiler allocated more registers, reducing concurrent threads.")
        print()
    }

    /// ========================================================================
    /// Benchmark 5: Optimal Thread Count Sweep
    /// ========================================================================
    func benchThreadCountSweep() throws {
        print("── Benchmark 5: Optimal Thread Count Sweep ─────────────────────────────")
        print()

        let iterations: UInt32 = 200
        let totalElements: UInt32 = 8192
        let numThreadgroups = 64
        let tgMemSize = Int(totalElements) * MemoryLayout<Float>.size

        let iterBuf = makeBuffer(iterations)
        let pipeline = try makePipeline("thread_count_sweep")

        for threadsPerTG in [32, 64, 128, 256, 512, 1024] {
            let elementsPerThread: UInt32 = totalElements / UInt32(threadsPerTG)
            let eptBuf = makeBuffer(elementsPerThread)
            let outputBuf = makeOutputBuffer(count: numThreadgroups)

            let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
            let actualThreads = min(threadsPerTG, maxThreads)

            let ms = try timeKernel(
                pipeline: pipeline,
                threadgroups: MTLSizeMake(numThreadgroups, 1, 1),
                threadsPerThreadgroup: MTLSizeMake(actualThreads, 1, 1),
                threadgroupMemoryLength: tgMemSize,
                buffers: [(outputBuf, 0), (iterBuf, 1), (eptBuf, 2)]
            )

            // Throughput: total butterfly-like ops
            let opsPerIter = Double(totalElements) * Double(numThreadgroups) * 4.0 // ~4 FLOPs per element
            let totalOps = opsPerIter * Double(iterations)
            let gflops = totalOps / (ms / 1000.0) / 1e9

            print("  \(String(format: "%4d", actualThreads)) threads x \(String(format: "%3d", elementsPerThread)) elements  \(String(format: "%8.3f", ms)) ms  \(String(format: "%8.2f", gflops)) GFLOPS (butterfly-like)")
        }
        print()
    }
}

// MARK: - Error Types

enum BenchError: Error, CustomStringConvertible {
    case noDevice
    case noQueue
    case noFunction(String)

    var description: String {
        switch self {
        case .noDevice: "No Metal device found"
        case .noQueue: "Could not create command queue"
        case let .noFunction(name): "Metal function '\(name)' not found"
        }
    }
}

// MARK: - Main Entry Point

func benchmarkMain() throws {
    let args = CommandLine.arguments
    let runAll = args.count < 2
    let needsGPU = runAll
        || args.contains("1") || args.contains("tgmem")
        || args.contains("2") || args.contains("shuffle")
        || args.contains("3") || args.contains("regcopy")
        || args.contains("4") || args.contains("occupancy")
        || args.contains("5") || args.contains("threadcount")

    var harness: GPUBenchmarkHarness? = nil
    if needsGPU {
        harness = try GPUBenchmarkHarness()
    }

    if runAll || args.contains("1") || args.contains("tgmem") {
        try harness!.benchThreadgroupMemory()
    }
    if runAll || args.contains("2") || args.contains("shuffle") {
        try harness!.benchSimdShuffle()
    }
    if runAll || args.contains("3") || args.contains("regcopy") {
        try harness!.benchRegToTgmem()
    }
    if runAll || args.contains("4") || args.contains("occupancy") {
        try harness!.benchOccupancy()
    }
    if runAll || args.contains("5") || args.contains("threadcount") {
        try harness!.benchThreadCountSweep()
    }
    if runAll || args.contains("6") || args.contains("vdsp") {
        runVDSPBaseline()
    }

    print("=" * 72)
    print("All benchmarks complete.")
    print()
    print("To run individual benchmarks:")
    print("  swift run FFTMicroBenchmarks 1          # Threadgroup memory bandwidth")
    print("  swift run FFTMicroBenchmarks 2          # SIMD shuffle throughput")
    print("  swift run FFTMicroBenchmarks 3          # Register-to-threadgroup copy")
    print("  swift run FFTMicroBenchmarks 4          # Occupancy vs register pressure")
    print("  swift run FFTMicroBenchmarks 5          # Optimal thread count sweep")
    print("  swift run FFTMicroBenchmarks 6          # vDSP FFT baseline")
    print("  swift run FFTMicroBenchmarks vdsp       # Same as 6")
    print("  swift run FFTMicroBenchmarks tgmem      # Same as 1")
    print("  swift run FFTMicroBenchmarks shuffle     # Same as 2")
    print("=" * 72)
}
