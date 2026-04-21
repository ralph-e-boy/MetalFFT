import Metal
import Accelerate

// MARK: - CrossSpectralResult

/// Result of a multi-channel cross-spectral analysis.
///
/// Pairs are ordered as the upper triangle of the channel × channel matrix:
/// (0,1), (0,2), …, (0,C-1), (1,2), …, (C-2, C-1).
/// Use `pairIndex(_:_:)` to look up by channel indices.
public struct CrossSpectralResult {
    public let channels: Int
    public let fftSize: Int

    /// Power spectrum per channel. `power[c][k]` = |X_c(k)|².
    public let power: [[Float]]

    /// Complex cross-spectra for each pair. `crossSpectra[pair][k]` = X_i(k) · conj(X_j(k)).
    public let crossSpectra: [[SIMD2<Float>]]

    /// Magnitude-squared coherence per pair per bin. Values in [0, 1].
    public let coherence: [[Float]]

    /// Number of channel pairs (upper triangle count = C*(C-1)/2).
    public var pairCount: Int { channels * (channels - 1) / 2 }

    /// Returns the flat pair index for channels `i` and `j` (i < j).
    public func pairIndex(_ i: Int, _ j: Int) -> Int {
        precondition(i < j && j < channels)
        var idx = 0
        for a in 0..<i { idx += channels - 1 - a }
        return idx + (j - i - 1)
    }
}

// MARK: - PSD cross-spectral extension

extension PSD {

    /// Multi-channel cross-spectral density matrix via Welch's method.
    ///
    /// Slides a window over each channel, FFTs each window, computes cross-spectra
    /// on the GPU, and averages over windows.
    ///
    /// - Parameters:
    ///   - channels: Time-domain signals, all the same length. Maximum 16 channels.
    ///   - fftSize: Analysis window length. Must be a supported `MetalFFT` size.
    ///   - hopSize: Window advance per frame.
    ///   - sampleRate: Sample rate in Hz (used only for bin-frequency labelling).
    ///   - window: Spectral window applied before each FFT.
    /// - Returns: Averaged power, cross-spectra, and coherence per bin.
    public static func crossSpectral(
        channels signals: [[Float]],
        fftSize: Int,
        hopSize: Int,
        sampleRate: Double,
        window windowType: WindowType = .hann
    ) throws -> CrossSpectralResult {
        precondition(!signals.isEmpty && signals.count <= 16,
                     "crossSpectral requires 1–16 channels")
        precondition(signals.allSatisfy { $0.count == signals[0].count },
                     "all channels must have the same length")

        let C      = signals.count
        let N      = fftSize
        let pairs  = C * (C - 1) / 2

        let ctx    = try MetalContext.shared()
        let fft    = try MultiChannelFFT(channels: C, size: N)
        let win    = windowType.coefficients(N)
        let winBuf = [Float](repeating: 0, count: N)

        // Accumulators: average cross-spectra and power over frames
        var avgPower  = [[Float]](repeating: [Float](repeating: 0, count: N), count: C)
        var avgCross  = [[SIMD2<Float>]](repeating: [SIMD2<Float>](repeating: .zero, count: N), count: pairs)
        var avgCoh    = [[Float]](repeating: [Float](repeating: 0, count: N), count: pairs)
        var frameCount = 0

        // Allocate Metal buffers for the cross-spectral kernel
        let specBuf   = try makeBuffer(ctx.device, length: C * N * MemoryLayout<SIMD2<Float>>.stride)
        let powBuf    = try makeBuffer(ctx.device, length: C * N * MemoryLayout<Float>.stride)
        let crossBuf  = try makeBuffer(ctx.device, length: max(1, pairs) * N * MemoryLayout<SIMD2<Float>>.stride)
        let cohBuf    = try makeBuffer(ctx.device, length: max(1, pairs) * N * MemoryLayout<Float>.stride)
        var paramData = [UInt32(N), UInt32(C)]
        let paramBuf  = try makeBuffer(ctx.device, length: 2 * MemoryLayout<UInt32>.stride)
        paramBuf.contents().copyMemory(from: &paramData, byteCount: 2 * MemoryLayout<UInt32>.stride)

        let pipeline  = ctx.pipelines["fft_cross_spectral"]!
        let tgSize    = MTLSizeMake(256, 1, 1)
        let tgCount   = MTLSizeMake((N + 255) / 256, 1, 1)

        var pos = 0
        let signalLen = signals[0].count
        while pos + N <= signalLen {
            // Window and FFT all channels
            var windowedChannels = [[SIMD2<Float>]](repeating: [SIMD2<Float>](repeating: .zero, count: N), count: C)
            for c in 0..<C {
                signals[c].withUnsafeBufferPointer { ptr in
                    var windowed = winBuf
                    vDSP_vmul(ptr.baseAddress! + pos, 1, win, 1, &windowed, 1, vDSP_Length(N))
                    for i in 0..<N { windowedChannels[c][i] = SIMD2<Float>(windowed[i], 0) }
                }
            }
            let spectra = try fft.forward(windowedChannels)

            // Pack spectra into flat Metal buffer: [ch0 | ch1 | ... | ch(C-1)]
            let specPtr = specBuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: C * N)
            for c in 0..<C {
                spectra[c].withUnsafeBufferPointer { src in
                    (specPtr + c * N).update(from: src.baseAddress!, count: N)
                }
            }

            // Dispatch cross-spectral kernel
            guard let cb  = ctx.queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else {
                throw FFTError.commandBufferFailed("crossSpectral: encoder creation failed")
            }
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(specBuf,  offset: 0, index: 0)
            enc.setBuffer(powBuf,   offset: 0, index: 1)
            enc.setBuffer(crossBuf, offset: 0, index: 2)
            enc.setBuffer(cohBuf,   offset: 0, index: 3)
            enc.setBuffer(paramBuf, offset: 0, index: 4)
            enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
            enc.endEncoding()
            try commitAndWait(cb)

            // Accumulate into averages
            let powPtr   = powBuf.contents().bindMemory(to: Float.self, capacity: C * N)
            let crossPtr = crossBuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: pairs * N)
            let cohPtr   = cohBuf.contents().bindMemory(to: Float.self, capacity: pairs * N)

            for c in 0..<C {
                vDSP_vadd(avgPower[c], 1, powPtr + c * N, 1, &avgPower[c], 1, vDSP_Length(N))
            }
            for p in 0..<pairs {
                let cs = UnsafeBufferPointer(start: crossPtr + p * N, count: N)
                for k in 0..<N { avgCross[p][k].x += cs[k].x; avgCross[p][k].y += cs[k].y }
                vDSP_vadd(avgCoh[p], 1, cohPtr + p * N, 1, &avgCoh[p], 1, vDSP_Length(N))
            }

            frameCount += 1
            pos += hopSize
        }

        guard frameCount > 0 else {
            return CrossSpectralResult(channels: C, fftSize: N,
                                       power: avgPower, crossSpectra: avgCross, coherence: avgCoh)
        }

        // Divide accumulators by frame count
        var invFrames = Float(1) / Float(frameCount)
        for c in 0..<C {
            vDSP_vsmul(avgPower[c], 1, &invFrames, &avgPower[c], 1, vDSP_Length(N))
        }
        for p in 0..<pairs {
            for k in 0..<N { avgCross[p][k].x *= invFrames; avgCross[p][k].y *= invFrames }
            vDSP_vsmul(avgCoh[p], 1, &invFrames, &avgCoh[p], 1, vDSP_Length(N))
        }

        return CrossSpectralResult(channels: C, fftSize: N,
                                   power: avgPower, crossSpectra: avgCross, coherence: avgCoh)
    }
}
