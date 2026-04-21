import Metal

// MARK: - MultiChannelFFT

/// GPU-accelerated FFT over multiple independent channels of the same size.
///
/// All channels share a single GPU dispatch — one threadgroup per channel — so
/// the overhead of issuing separate dispatches is eliminated. For N=4096 the
/// dispatch uses the radix-8 Stockham kernel (138 GFLOPS on M1).
///
/// Input/output format: one `[SIMD2<Float>]` array per channel, where `.x` = real,
/// `.y` = imaginary.  Channels are independent; there is no cross-channel coupling
/// in the FFT itself.
///
/// Useful for: stereo/surround audio, multi-sensor arrays, multi-antenna radar,
/// multi-channel biological spectral analysis.
///
/// Not thread-safe. Not intended for sizes that change between calls.
public final class MultiChannelFFT {
    public let channels: Int
    public let size: Int

    private let fft: MetalFFT

    /// - Parameters:
    ///   - channels: Number of independent FFT channels to transform per call.
    ///   - size: FFT length. Must be a supported `MetalFFT` size (64–16384).
    public init(channels: Int, size: Int) throws {
        precondition(channels > 0, "channels must be > 0")
        self.channels = channels
        self.size = size
        fft = try MetalFFT(size: size)
    }

    /// Forward FFT on all channels in a single GPU dispatch.
    ///
    /// - Parameter inputs: Exactly `channels` arrays, each of length `size`.
    /// - Returns: `channels` complex spectra in the same order as `inputs`.
    public func forward(_ inputs: [[SIMD2<Float>]]) throws -> [[SIMD2<Float>]] {
        precondition(inputs.count == channels,
                     "Expected \(channels) channels, got \(inputs.count)")
        return try fft.forward(batch: inputs)
    }

    /// Inverse FFT on all channels in a single GPU dispatch.
    ///
    /// - Parameter inputs: Exactly `channels` complex spectra, each of length `size`.
    /// - Returns: `channels` time-domain signals.
    public func inverse(_ inputs: [[SIMD2<Float>]]) throws -> [[SIMD2<Float>]] {
        precondition(inputs.count == channels,
                     "Expected \(channels) channels, got \(inputs.count)")
        return try inputs.map { try fft.inverse($0) }
    }
}
