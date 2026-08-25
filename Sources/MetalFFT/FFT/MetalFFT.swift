import Foundation
import Metal

// MARK: - MetalFFT

/// Metal-accelerated complex FFT. Sizes: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384.
///
/// Input/output format: interleaved SIMD2<Float> where .x = real, .y = imaginary.
/// Not thread-safe: serialize all calls on a single writer.
public final class MetalFFT {
    // MARK: - Public

    public let size: Int

    // MARK: - Private

    private let context: MetalContext
    private let descriptor: FFTDescriptor
    private let byteCount: Int

    private let inputBuf: MTLBuffer
    private let outputBuf: MTLBuffer
    private let fourStepState: FourStepState

    // Scratch for the batch path, grown on demand and kept. It starts aliased
    // to the single-transform buffers, so an instance that never batches costs
    // nothing extra, and one that batches every call allocates once.
    private var batchCapacity: Int
    private var batchIn: MTLBuffer
    private var batchOut: MTLBuffer
    private var batchState: FourStepState

    // MARK: - Init

    public init(size: Int) throws {
        // Real-input paths reinterpret inputBuf as alternating (real, imag) Float pairs,
        // so SIMD2<Float> must be tightly packed at stride = 2 * sizeof(Float).
        assert(MemoryLayout<SIMD2<Float>>.stride == 2 * MemoryLayout<Float>.stride,
               "SIMD2<Float> layout assumption broken — real-input FFT path is unsafe")
        let ctx = try MetalContext.shared()
        let desc = try FFTDescriptor(size: size)
        context = ctx
        descriptor = desc
        self.size = size
        byteCount = size * MemoryLayout<SIMD2<Float>>.stride

        inputBuf = try makeBuffer(ctx.device, length: byteCount)
        outputBuf = try makeBuffer(ctx.device, length: byteCount)

        if case let .fourStep(n1, n2, _, _, _, _) = desc.kind {
            var n1v = UInt32(n1), n2v = UInt32(n2)
            fourStepState = try .fourStep(
                tempA: makeBuffer(ctx.device, length: byteCount),
                tempB: makeBuffer(ctx.device, length: byteCount),
                tempC: makeBuffer(ctx.device, length: byteCount),
                tempD: makeBuffer(ctx.device, length: byteCount),
                n1Buf: makeBuffer(ctx.device, uint32: &n1v),
                n2Buf: makeBuffer(ctx.device, uint32: &n2v)
            )
        } else {
            fourStepState = .singlePass
        }

        batchCapacity = 1
        batchIn = inputBuf
        batchOut = outputBuf
        batchState = fourStepState
    }

    /// Grows the batch scratch to hold `count` transforms, keeping whatever was
    /// already large enough. Called before every batch dispatch.
    private func reserveBatch(_ count: Int) throws {
        guard count > batchCapacity else { return }
        let total = byteCount * count
        batchIn = try makeBuffer(context.device, length: total)
        batchOut = try makeBuffer(context.device, length: total)
        if case let .fourStep(_, _, _, _, n1Buf, n2Buf) = fourStepState {
            batchState = try .fourStep(
                tempA: makeBuffer(context.device, length: total),
                tempB: makeBuffer(context.device, length: total),
                tempC: makeBuffer(context.device, length: total),
                tempD: makeBuffer(context.device, length: total),
                n1Buf: n1Buf,
                n2Buf: n2Buf
            )
        }
        batchCapacity = count
    }

    // MARK: - Inverse FFT

    /// Inverse FFT via the conjugate trick: IFFT(X) = conj(FFT(conj(X))) / N.
    public func inverse(_ input: [SIMD2<Float>]) throws -> [SIMD2<Float>] {
        guard input.count == size else {
            throw FFTError.invalidInputSize(expected: size, got: input.count)
        }
        let conjInput = input.map { SIMD2<Float>($0.x, -$0.y) }
        var out = [SIMD2<Float>](repeating: .zero, count: size)
        try conjInput.withUnsafeBufferPointer { try forward(input: $0, output: &out) }
        let invN = Float(1) / Float(size)
        return out.map { SIMD2<Float>($0.x * invN, -$0.y * invN) }
    }

    // MARK: - Forward FFT

    public func forward(_ input: [SIMD2<Float>]) throws -> [SIMD2<Float>] {
        guard input.count == size else {
            throw FFTError.invalidInputSize(expected: size, got: input.count)
        }
        var out = [SIMD2<Float>](repeating: .zero, count: size)
        input.withUnsafeBufferPointer { inp in
            inputBuf.contents().copyMemory(from: inp.baseAddress!, byteCount: byteCount)
        }
        try dispatchSingle(from: inputBuf, to: outputBuf)
        out.withUnsafeMutableBufferPointer { buf in
            buf.baseAddress!.update(
                from: outputBuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: size),
                count: size
            )
        }
        return out
    }

    /// Zero-copy variant: copies input from caller-managed buffer pointer.
    public func forward(
        input: UnsafeBufferPointer<SIMD2<Float>>,
        output: inout [SIMD2<Float>]
    ) throws {
        guard input.count == size else {
            throw FFTError.invalidInputSize(expected: size, got: input.count)
        }
        if output.count != size { output = [SIMD2<Float>](repeating: .zero, count: size) }
        inputBuf.contents().copyMemory(from: input.baseAddress!, byteCount: byteCount)
        try dispatchSingle(from: inputBuf, to: outputBuf)
        output.withUnsafeMutableBufferPointer { buf in
            buf.baseAddress!.update(
                from: outputBuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: size),
                count: size
            )
        }
    }

    // MARK: - Real-input Forward FFT

    /// Forward FFT of a real-valued signal. The samples are packed as complex with
    /// imag = 0 internally; the returned spectrum is the full N-point complex output
    /// (Hermitian-symmetric for real input — bins above N/2 mirror bins below).
    ///
    /// Allocates the output array; use `forward(real:output:)` for the zero-allocation variant.
    public func forward(real input: [Float]) throws -> [SIMD2<Float>] {
        guard input.count == size else {
            throw FFTError.invalidInputSize(expected: size, got: input.count)
        }
        var out = [SIMD2<Float>](repeating: .zero, count: size)
        try input.withUnsafeBufferPointer { try forward(real: $0, output: &out) }
        return out
    }

    /// Zero-copy real-input forward FFT. Writes both real and imag slots of the
    /// internal input buffer on every call (imag := 0), so interleaving with the
    /// complex `forward(input:output:)` path is safe.
    public func forward(
        real input: UnsafeBufferPointer<Float>,
        output: inout [SIMD2<Float>]
    ) throws {
        guard input.count == size else {
            throw FFTError.invalidInputSize(expected: size, got: input.count)
        }
        if output.count != size { output = [SIMD2<Float>](repeating: .zero, count: size) }

        // Pack real → (real, 0) directly into inputBuf. Setting the full SIMD2<Float>
        // each iteration guarantees the imag slot is zeroed even if a prior complex
        // call left stale values there.
        let dst = inputBuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: size)
        let src = input.baseAddress!
        for i in 0 ..< size {
            dst[i] = SIMD2<Float>(src[i], 0)
        }

        try dispatchSingle(from: inputBuf, to: outputBuf)
        output.withUnsafeMutableBufferPointer { buf in
            buf.baseAddress!.update(
                from: outputBuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: size),
                count: size
            )
        }
    }

    /// Batch FFT: all `input` elements must have `count == size`.
    /// Single-pass sizes use one GPU dispatch; four-step uses one command buffer per element.
    public func forward(batch input: [[SIMD2<Float>]]) throws -> [[SIMD2<Float>]] {
        guard !input.isEmpty else { return [] }
        for (i, el) in input.enumerated() {
            guard el.count == size else {
                throw FFTError.batchInputSize(expected: size, got: el.count, batchIndex: i)
            }
        }
        var flat = [SIMD2<Float>](repeating: .zero, count: size * input.count)
        for (i, el) in input.enumerated() {
            flat.replaceSubrange(i * size ..< (i + 1) * size, with: el)
        }
        var out = [SIMD2<Float>](repeating: .zero, count: size * input.count)
        try flat.withUnsafeBufferPointer {
            try forward(batch: $0, count: input.count, output: &out)
        }
        return (0 ..< input.count).map { Array(out[$0 * size ..< ($0 + 1) * size]) }
    }

    /// Zero-copy batch FFT. `input` holds `count` transforms of `size` elements
    /// laid end to end, and `output` is filled the same way, resized if needed.
    ///
    /// This is the shape to use when the transforms are already contiguous —
    /// the frames of an STFT, the channels of a multichannel capture. One
    /// dispatch covers the whole batch at single-pass sizes, which is the
    /// difference between paying the command-buffer round trip once and paying
    /// it `count` times. The scratch buffers are kept and reused, so a caller
    /// that batches the same count repeatedly allocates on the first call only.
    public func forward(
        batch input: UnsafeBufferPointer<SIMD2<Float>>,
        count: Int,
        output: inout [SIMD2<Float>]
    ) throws {
        guard count > 0 else { return }
        let total = size * count
        guard input.count == total else {
            throw FFTError.invalidInputSize(expected: total, got: input.count)
        }
        // Grown, never shrunk: a caller whose last chunk is short keeps the
        // buffer the full chunks needed rather than reallocating on the next one.
        if output.count < total { output = [SIMD2<Float>](repeating: .zero, count: total) }

        try reserveBatch(count)
        batchIn.contents().copyMemory(from: input.baseAddress!, byteCount: byteCount * count)
        try dispatchBatch(from: batchIn, to: batchOut, batchSize: count)
        output.withUnsafeMutableBufferPointer { buf in
            buf.baseAddress!.update(
                from: batchOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: total),
                count: total
            )
        }
    }

    // MARK: - Dispatch

    private func dispatchSingle(from inBuf: MTLBuffer, to outBuf: MTLBuffer) throws {
        switch descriptor.kind {
        case let .singlePass(kernelName, threads):
            guard let cb = context.queue.makeCommandBuffer() else {
                throw FFTError.commandBufferFailed("makeCommandBuffer returned nil")
            }
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw FFTError.commandBufferFailed("makeComputeCommandEncoder returned nil")
            }
            enc.setComputePipelineState(context.pipelines[kernelName]!)
            enc.setBuffer(inBuf, offset: 0, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            enc.dispatchThreadgroups(MTLSizeMake(1, 1, 1),
                                     threadsPerThreadgroup: MTLSizeMake(threads, 1, 1))
            enc.endEncoding()
            try commitAndWait(cb)

        case .fourStep:
            try dispatchFourStep(from: inBuf, to: outBuf, batchOffset: 0, temps: fourStepState)
        }
    }

    private func dispatchBatch(from inBuf: MTLBuffer, to outBuf: MTLBuffer, batchSize: Int) throws {
        switch descriptor.kind {
        case let .singlePass(kernelName, threads):
            // For N=4096 batch, use the dedicated radix-8 Stockham kernel (138 GFLOPS vs 113 GFLOPS).
            let (kName, kThreads): (String, Int) = size == 4096
                ? ("fft_4096_batched", 512)
                : (kernelName, threads)
            guard let cb = context.queue.makeCommandBuffer() else {
                throw FFTError.commandBufferFailed("makeCommandBuffer returned nil")
            }
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw FFTError.commandBufferFailed("makeComputeCommandEncoder returned nil")
            }
            enc.setComputePipelineState(context.pipelines[kName]!)
            enc.setBuffer(inBuf, offset: 0, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            enc.dispatchThreadgroups(MTLSizeMake(batchSize, 1, 1),
                                     threadsPerThreadgroup: MTLSizeMake(kThreads, 1, 1))
            enc.endEncoding()
            try commitAndWait(cb)

        case .fourStep:
            // Four-step still costs one command buffer per element: its five
            // passes write through shared temporaries, so the elements cannot
            // share a buffer without serialising on them anyway.
            for bIdx in 0 ..< batchSize {
                try dispatchFourStep(from: inBuf, to: outBuf,
                                     batchOffset: bIdx * byteCount,
                                     temps: batchState)
            }
        }
    }

    private func dispatchFourStep(
        from inBuf: MTLBuffer, to outBuf: MTLBuffer,
        batchOffset: Int, temps: FourStepState
    ) throws {
        guard case let .fourStep(n1, n2, pass1Kernel, pass1Threads, pass2Kernel, pass2Threads) = descriptor.kind,
              case let .fourStep(tempA, tempB, tempC, tempD, n1Buf, n2Buf) = temps
        else { fatalError("unreachable") }

        let transposePL = context.pipelines["fft_transpose"]!
        let twiddlePL = context.pipelines["fft_twiddle_transpose"]!
        let pass1PL = context.pipelines[pass1Kernel]!
        let pass2PL = context.pipelines[pass2Kernel]!
        let elemThreads = min(256, size)
        let elemTGs = (size + elemThreads - 1) / elemThreads

        guard let cb = context.queue.makeCommandBuffer() else {
            throw FFTError.commandBufferFailed("makeCommandBuffer returned nil")
        }

        // Step 0: transpose input N2×N1 → N1×N2
        let enc0 = cb.makeComputeCommandEncoder()!
        enc0.setComputePipelineState(transposePL)
        enc0.setBuffer(inBuf, offset: batchOffset, index: 0)
        enc0.setBuffer(tempA, offset: batchOffset, index: 1)
        enc0.setBuffer(n2Buf, offset: 0, index: 2)
        enc0.setBuffer(n1Buf, offset: 0, index: 3)
        enc0.dispatchThreadgroups(MTLSizeMake(elemTGs, 1, 1),
                                  threadsPerThreadgroup: MTLSizeMake(elemThreads, 1, 1))
        enc0.endEncoding()

        // Step 1: N1 row-FFTs of size N2
        let enc1 = cb.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(pass1PL)
        enc1.setBuffer(tempA, offset: batchOffset, index: 0)
        enc1.setBuffer(tempB, offset: batchOffset, index: 1)
        enc1.dispatchThreadgroups(MTLSizeMake(n1, 1, 1),
                                  threadsPerThreadgroup: MTLSizeMake(pass1Threads, 1, 1))
        enc1.endEncoding()

        // Step 2: twiddle W_N^{row*col} + transpose N1×N2 → N2×N1
        let enc2 = cb.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(twiddlePL)
        enc2.setBuffer(tempB, offset: batchOffset, index: 0)
        enc2.setBuffer(tempC, offset: batchOffset, index: 1)
        enc2.setBuffer(n1Buf, offset: 0, index: 2)
        enc2.setBuffer(n2Buf, offset: 0, index: 3)
        enc2.dispatchThreadgroups(MTLSizeMake(elemTGs, 1, 1),
                                  threadsPerThreadgroup: MTLSizeMake(elemThreads, 1, 1))
        enc2.endEncoding()

        // Step 3: N2 row-FFTs of size N1
        let enc3 = cb.makeComputeCommandEncoder()!
        enc3.setComputePipelineState(pass2PL)
        enc3.setBuffer(tempC, offset: batchOffset, index: 0)
        enc3.setBuffer(tempD, offset: batchOffset, index: 1)
        enc3.dispatchThreadgroups(MTLSizeMake(n2, 1, 1),
                                  threadsPerThreadgroup: MTLSizeMake(pass2Threads, 1, 1))
        enc3.endEncoding()

        // Step 4: transpose N2×N1 → N1×N2 (canonical output order)
        let enc4 = cb.makeComputeCommandEncoder()!
        enc4.setComputePipelineState(transposePL)
        enc4.setBuffer(tempD, offset: batchOffset, index: 0)
        enc4.setBuffer(outBuf, offset: batchOffset, index: 1)
        enc4.setBuffer(n2Buf, offset: 0, index: 2)
        enc4.setBuffer(n1Buf, offset: 0, index: 3)
        enc4.dispatchThreadgroups(MTLSizeMake(elemTGs, 1, 1),
                                  threadsPerThreadgroup: MTLSizeMake(elemThreads, 1, 1))
        enc4.endEncoding()

        try commitAndWait(cb)
    }
}

// MARK: - FourStepState

private enum FourStepState {
    case singlePass
    case fourStep(
        tempA: MTLBuffer, tempB: MTLBuffer,
        tempC: MTLBuffer, tempD: MTLBuffer,
        n1Buf: MTLBuffer, n2Buf: MTLBuffer
    )
}
