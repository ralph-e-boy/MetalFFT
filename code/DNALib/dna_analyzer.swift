import Foundation
import Metal

// ============================================================================
// DNA Spectral Analysis — Metal Host
//
// Orchestrates the GPU pipeline:
// 1. Read DNA from FASTA or generate synthetic sequences
// 2. Dispatch 4-channel FFT kernel
// 3. Dispatch cross-spectral analysis
// 4. Compute spectrogram via sliding-window STFT
// 5. Output results as TSV files
// ============================================================================

public struct DNASpectralAnalyzer {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let fftPipeline: MTLComputePipelineState
    public let crossSpectralPipeline: MTLComputePipelineState
    public let period3Pipeline: MTLComputePipelineState
    public let spectrogramPipeline: MTLComputePipelineState
    public let spectrogram4chPipeline: MTLComputePipelineState

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device found")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Could not create command queue")
        }
        commandQueue = queue

        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
        let searchDirs = [
            execDir,
            "\(execDir)/../src/dna",
            "\(execDir)/../../src/dna",
            "\(execDir)/../../../src/dna",
            "\(execDir)/../../../../src/dna",
            "\(execDir)/../../../..",
            FileManager.default.currentDirectoryPath
        ]

        func loadShader(_ filename: String) throws -> String {
            for dir in searchDirs {
                let path = "\(dir)/\(filename)"
                if FileManager.default.fileExists(atPath: path) {
                    return try String(contentsOfFile: path, encoding: .utf8)
                }
            }
            print("ERROR: Could not find \(filename)")
            print("Searched directories:")
            for dir in searchDirs {
                print("  \(dir)")
            }
            fatalError("Metal shader not found: \(filename)")
        }

        let fftSource = try loadShader("dna_spectral.metal")
        let crossSource = try loadShader("dna_cross_spectral.metal")
        let spectrogramSource = try loadShader("dna_spectrogram.metal")

        let fftLib = try device.makeLibrary(source: fftSource, options: options)
        let crossLib = try device.makeLibrary(source: crossSource, options: options)
        let specLib = try device.makeLibrary(source: spectrogramSource, options: options)

        func makePipeline(_ lib: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
            guard let fn = lib.makeFunction(name: name) else {
                fatalError("Kernel function '\(name)' not found")
            }
            return try device.makeComputePipelineState(function: fn)
        }

        fftPipeline = try makePipeline(fftLib, "dna_fft_1024")
        crossSpectralPipeline = try makePipeline(crossLib, "dna_cross_spectral")
        period3Pipeline = try makePipeline(crossLib, "dna_period3_detect")
        spectrogramPipeline = try makePipeline(specLib, "dna_spectrogram_1024")
        spectrogram4chPipeline = try makePipeline(specLib, "dna_spectrogram_4ch_1024")

        print("DNA Spectral Analyzer initialized on: \(device.name)")
        print("  FFT pipeline max threads: \(fftPipeline.maxTotalThreadsPerThreadgroup)")
        print()
    }

    public func runFFT(dna: [UInt8]) -> [SIMD2<Float>] {
        let N = 1024
        let numWindows = dna.count / N
        assert(dna.count % N == 0, "DNA length must be multiple of \(N)")

        let dnaBuffer = device.makeBuffer(bytes: dna, length: dna.count, options: .storageModeShared)!
        let spectraBuffer = device.makeBuffer(length: numWindows * 4 * N * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
        var params: [UInt32] = [0]
        let paramsBuffer = device.makeBuffer(bytes: &params, length: params.count * 4, options: .storageModeShared)!

        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(fftPipeline)
        encoder.setBuffer(dnaBuffer, offset: 0, index: 0)
        encoder.setBuffer(spectraBuffer, offset: 0, index: 1)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
        encoder.dispatchThreadgroups(
            MTLSize(width: numWindows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = spectraBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: numWindows * 4 * N)
        return Array(UnsafeBufferPointer(start: ptr, count: numWindows * 4 * N))
    }

    public struct CrossSpectralResult {
        public let power: [Float]
        public let totalPower: [Float]
        public let crossSpectra: [SIMD2<Float>]
        public let coherence: [Float]

        public init(power: [Float], totalPower: [Float], crossSpectra: [SIMD2<Float>], coherence: [Float]) {
            self.power = power
            self.totalPower = totalPower
            self.crossSpectra = crossSpectra
            self.coherence = coherence
        }
    }

    public func runCrossSpectral(spectra: [SIMD2<Float>], N: Int) -> CrossSpectralResult {
        let spectraBuffer = device.makeBuffer(bytes: spectra, length: spectra.count * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
        let powerBuffer = device.makeBuffer(length: 4 * N * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let totalPowerBuffer = device.makeBuffer(length: N * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let crossBuffer = device.makeBuffer(length: 6 * N * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
        let coherenceBuffer = device.makeBuffer(length: 6 * N * MemoryLayout<Float>.stride, options: .storageModeShared)!
        var params: [UInt32] = [UInt32(N)]
        let paramsBuffer = device.makeBuffer(bytes: &params, length: params.count * 4, options: .storageModeShared)!

        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(crossSpectralPipeline)
        encoder.setBuffer(spectraBuffer, offset: 0, index: 0)
        encoder.setBuffer(powerBuffer, offset: 0, index: 1)
        encoder.setBuffer(totalPowerBuffer, offset: 0, index: 2)
        encoder.setBuffer(crossBuffer, offset: 0, index: 3)
        encoder.setBuffer(coherenceBuffer, offset: 0, index: 4)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 5)

        let numGroups = (N + 255) / 256
        encoder.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let powerPtr = powerBuffer.contents().bindMemory(to: Float.self, capacity: 4 * N)
        let totalPtr = totalPowerBuffer.contents().bindMemory(to: Float.self, capacity: N)
        let crossPtr = crossBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: 6 * N)
        let cohPtr = coherenceBuffer.contents().bindMemory(to: Float.self, capacity: 6 * N)

        return CrossSpectralResult(
            power: Array(UnsafeBufferPointer(start: powerPtr, count: 4 * N)),
            totalPower: Array(UnsafeBufferPointer(start: totalPtr, count: N)),
            crossSpectra: Array(UnsafeBufferPointer(start: crossPtr, count: 6 * N)),
            coherence: Array(UnsafeBufferPointer(start: cohPtr, count: 6 * N))
        )
    }

    public struct SpectrogramResult {
        public let data: [Float]
        public let numWindows: Int
        public let numFreqs: Int
    }

    public func runSpectrogram(dna: [UInt8], windowSize: Int = 1024, hopSize: Int = 256) -> SpectrogramResult {
        let N = windowSize
        assert(N == 1024, "Only N=1024 spectrogram supported")
        let numFreqs = N / 2 + 1
        let numWindows = (dna.count - N) / hopSize + 1

        let dnaBuffer = device.makeBuffer(bytes: dna, length: dna.count, options: .storageModeShared)!
        let outBuffer = device.makeBuffer(length: numWindows * numFreqs * MemoryLayout<Float>.stride, options: .storageModeShared)!
        var params: [UInt32] = [UInt32(dna.count), UInt32(hopSize), UInt32(numWindows)]
        let paramsBuffer = device.makeBuffer(bytes: &params, length: params.count * 4, options: .storageModeShared)!

        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(spectrogramPipeline)
        encoder.setBuffer(dnaBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
        encoder.dispatchThreadgroups(
            MTLSize(width: numWindows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = outBuffer.contents().bindMemory(to: Float.self, capacity: numWindows * numFreqs)
        return SpectrogramResult(
            data: Array(UnsafeBufferPointer(start: ptr, count: numWindows * numFreqs)),
            numWindows: numWindows,
            numFreqs: numFreqs
        )
    }

    public func writePowerSpectrumTSV(result: CrossSpectralResult, N: Int, path: String) throws {
        var lines = ["frequency\tP_A\tP_T\tP_G\tP_C\tP_total"]
        for k in 0 ..< (N / 2 + 1) {
            let pa = result.power[4 * k]
            let pt = result.power[4 * k + 1]
            let pg = result.power[4 * k + 2]
            let pc = result.power[4 * k + 3]
            lines.append("\(k)\t\(pa)\t\(pt)\t\(pg)\t\(pc)\t\(result.totalPower[k])")
        }
        try lines.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
    }

    public func writeCrossSpectrumTSV(result: CrossSpectralResult, N: Int, path: String) throws {
        var lines = ["frequency\tcoh_AT\tcoh_AG\tcoh_AC\tcoh_TG\tcoh_TC\tcoh_GC"]
        for k in 0 ..< (N / 2 + 1) {
            let c = (0 ..< 6).map { String(result.coherence[6 * k + $0]) }
            lines.append("\(k)\t\(c.joined(separator: "\t"))")
        }
        try lines.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
    }

    public func writeSpectrogramTSV(result: SpectrogramResult, hopSize: Int, path: String) throws {
        var lines = ["position\tfrequency\tpower"]
        for w in 0 ..< result.numWindows {
            let pos = w * hopSize
            for f in 0 ..< result.numFreqs {
                let power = result.data[w * result.numFreqs + f]
                if power > 1e-6 {
                    lines.append("\(pos)\t\(f)\t\(power)")
                }
            }
        }
        try lines.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
    }

    public func writePeriod3TSV(spectrogram: SpectrogramResult, hopSize: Int, path: String) throws {
        let k3 = 1024 / 3
        var lines = ["position\tperiod3_power\tnormalized_score"]
        for w in 0 ..< spectrogram.numWindows {
            let pos = w * hopSize
            let p3 = spectrogram.data[w * spectrogram.numFreqs + k3]
            var sumPower: Float = 0
            for f in 1 ..< (spectrogram.numFreqs - 1) {
                sumPower += spectrogram.data[w * spectrogram.numFreqs + f]
            }
            let meanPower = sumPower / Float(spectrogram.numFreqs - 2)
            let score = meanPower > 1e-30 ? p3 / meanPower : 0
            lines.append("\(pos)\t\(p3)\t\(score)")
        }
        try lines.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
    }
}

public extension String {
    static func * (lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}
