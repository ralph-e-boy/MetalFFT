import Metal
import Foundation

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

    // MARK: - Init

    public init(size: Int) throws {
        let ctx = try MetalContext.shared()
        let desc = try FFTDescriptor(size: size)
        self.context = ctx
        self.descriptor = desc
        self.size = size
        self.byteCount = size * MemoryLayout<SIMD2<Float>>.stride

        inputBuf  = try makeBuffer(ctx.device, length: byteCount)
        outputBuf = try makeBuffer(ctx.device, length: byteCount)

        if case .fourStep(let n1, let n2, _, _, _, _) = desc.kind {
            var n1v = UInt32(n1), n2v = UInt32(n2)
            fourStepState = .fourStep(
                tempA: try makeBuffer(ctx.device, length: byteCount),
                tempB: try makeBuffer(ctx.device, length: byteCount),
                tempC: try makeBuffer(ctx.device, length: byteCount),
                tempD: try makeBuffer(ctx.device, length: byteCount),
                n1Buf: try makeBuffer(ctx.device, uint32: &n1v),
                n2Buf: try makeBuffer(ctx.device, uint32: &n2v)
            )
        } else {
            fourStepState = .singlePass
        }
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

    /// Batch FFT: all `input` elements must have `count == size`.
    /// Single-pass sizes use one GPU dispatch; four-step uses one command buffer per element.
    public func forward(batch input: [[SIMD2<Float>]]) throws -> [[SIMD2<Float>]] {
        guard !input.isEmpty else { return [] }
        for (i, el) in input.enumerated() {
            guard el.count == size else {
                throw FFTError.batchInputSize(expected: size, got: el.count, batchIndex: i)
            }
        }
        let batchSize = input.count
        let totalBytes = byteCount * batchSize
        let flatIn  = try makeBuffer(context.device, length: totalBytes)
        let flatOut = try makeBuffer(context.device, length: totalBytes)

        for (i, el) in input.enumerated() {
            el.withUnsafeBytes { src in
                (flatIn.contents() + i * byteCount).copyMemory(from: src.baseAddress!, byteCount: byteCount)
            }
        }

        try dispatchBatch(from: flatIn, to: flatOut, batchSize: batchSize)

        return (0..<batchSize).map { i in
            let ptr = (flatOut.contents() + i * byteCount)
                .bindMemory(to: SIMD2<Float>.self, capacity: size)
            return Array(UnsafeBufferPointer(start: ptr, count: size))
        }
    }

    // MARK: - Dispatch

    private func dispatchSingle(from inBuf: MTLBuffer, to outBuf: MTLBuffer) throws {
        switch descriptor.kind {
        case .singlePass(let kernelName, let threads):
            guard let cb = context.queue.makeCommandBuffer() else {
                throw FFTError.commandBufferFailed("makeCommandBuffer returned nil")
            }
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw FFTError.commandBufferFailed("makeComputeCommandEncoder returned nil")
            }
            enc.setComputePipelineState(context.pipelines[kernelName]!)
            enc.setBuffer(inBuf,  offset: 0, index: 0)
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
        case .singlePass(let kernelName, let threads):
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
            enc.setBuffer(inBuf,  offset: 0, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            enc.dispatchThreadgroups(MTLSizeMake(batchSize, 1, 1),
                                     threadsPerThreadgroup: MTLSizeMake(kThreads, 1, 1))
            enc.endEncoding()
            try commitAndWait(cb)

        case .fourStep:
            let batchTemps: FourStepState
            if batchSize == 1 {
                batchTemps = fourStepState
            } else {
                let total = byteCount * batchSize
                guard case .fourStep(_, _, _, _, let n1Buf, let n2Buf) = fourStepState else {
                    fatalError("unreachable")
                }
                batchTemps = try .fourStep(
                    tempA: makeBuffer(context.device, length: total),
                    tempB: makeBuffer(context.device, length: total),
                    tempC: makeBuffer(context.device, length: total),
                    tempD: makeBuffer(context.device, length: total),
                    n1Buf: n1Buf,
                    n2Buf: n2Buf
                )
            }
            for bIdx in 0..<batchSize {
                try dispatchFourStep(from: inBuf, to: outBuf,
                                     batchOffset: bIdx * byteCount,
                                     temps: batchTemps)
            }
        }
    }

    private func dispatchFourStep(
        from inBuf: MTLBuffer, to outBuf: MTLBuffer,
        batchOffset: Int, temps: FourStepState
    ) throws {
        guard case .fourStep(let n1, let n2, let pass1Kernel, let pass1Threads, let pass2Kernel, let pass2Threads) = descriptor.kind,
              case .fourStep(let tempA, let tempB, let tempC, let tempD, let n1Buf, let n2Buf) = temps
        else { fatalError("unreachable") }

        let transposePL  = context.pipelines["fft_transpose"]!
        let twiddlePL    = context.pipelines["fft_twiddle_transpose"]!
        let pass1PL      = context.pipelines[pass1Kernel]!
        let pass2PL      = context.pipelines[pass2Kernel]!
        let elemThreads  = min(256, size)
        let elemTGs      = (size + elemThreads - 1) / elemThreads

        guard let cb = context.queue.makeCommandBuffer() else {
            throw FFTError.commandBufferFailed("makeCommandBuffer returned nil")
        }

        // Step 0: transpose input N2×N1 → N1×N2
        let enc0 = cb.makeComputeCommandEncoder()!
        enc0.setComputePipelineState(transposePL)
        enc0.setBuffer(inBuf,  offset: batchOffset, index: 0)
        enc0.setBuffer(tempA,  offset: batchOffset, index: 1)
        enc0.setBuffer(n2Buf,  offset: 0,           index: 2)
        enc0.setBuffer(n1Buf,  offset: 0,           index: 3)
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
        enc2.setBuffer(n1Buf, offset: 0,           index: 2)
        enc2.setBuffer(n2Buf, offset: 0,           index: 3)
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
        enc4.setBuffer(tempD,  offset: batchOffset, index: 0)
        enc4.setBuffer(outBuf, offset: batchOffset, index: 1)
        enc4.setBuffer(n2Buf,  offset: 0,           index: 2)
        enc4.setBuffer(n1Buf,  offset: 0,           index: 3)
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
