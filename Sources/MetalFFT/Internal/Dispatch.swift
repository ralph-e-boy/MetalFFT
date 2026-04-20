import Metal

@inline(__always)
func commitAndWait(_ commandBuffer: MTLCommandBuffer) throws {
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error {
        throw FFTError.commandBufferFailed(error.localizedDescription)
    }
}

func makeBuffer(_ device: MTLDevice, length: Int) throws -> MTLBuffer {
    guard let buf = device.makeBuffer(length: length, options: .storageModeShared) else {
        throw FFTError.bufferAllocationFailed
    }
    return buf
}

func makeBuffer(_ device: MTLDevice, uint32 value: inout UInt32) throws -> MTLBuffer {
    guard let buf = device.makeBuffer(bytes: &value, length: 4, options: .storageModeShared) else {
        throw FFTError.bufferAllocationFailed
    }
    return buf
}
