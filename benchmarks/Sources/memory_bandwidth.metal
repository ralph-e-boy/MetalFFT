// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Benchmark 1: Threadgroup Memory Bandwidth
// Measures read/write throughput with different access patterns.
// ============================================================================

// Sequential access: each thread reads/writes contiguous elements
kernel void tgmem_sequential_rw(
    device float *output [[buffer(0)]],
    constant uint &iterations [[buffer(1)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    const uint ELEMENTS_PER_THREAD = 32;
    uint base = tid * ELEMENTS_PER_THREAD;

    // Write phase: fill threadgroup memory
    for (uint i = 0; i < ELEMENTS_PER_THREAD; i++) {
        shared[base + i] = float(tid + i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Read-modify-write loop (iterated for timing stability)
    float accum = 0.0;
    for (uint iter = 0; iter < iterations; iter++) {
        for (uint i = 0; i < ELEMENTS_PER_THREAD; i++) {
            float val = shared[base + i];
            accum += val;
            shared[base + i] = val + 1.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Prevent dead-code elimination
    if (tid == 0) {
        output[tgid] = accum;
    }
}

// Strided access: threads access with stride = threadgroup_size
// This tests bank conflict behavior
kernel void tgmem_strided_rw(
    device float *output [[buffer(0)]],
    constant uint &iterations [[buffer(1)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    const uint ELEMENTS_PER_THREAD = 32;

    // Strided write: thread k writes positions k, k+N, k+2N, ...
    for (uint i = 0; i < ELEMENTS_PER_THREAD; i++) {
        shared[tid + i * threads_per_tg] = float(tid + i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accum = 0.0;
    for (uint iter = 0; iter < iterations; iter++) {
        for (uint i = 0; i < ELEMENTS_PER_THREAD; i++) {
            float val = shared[tid + i * threads_per_tg];
            accum += val;
            shared[tid + i * threads_per_tg] = val + 1.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[tgid] = accum;
    }
}

// Bank-conflict-inducing pattern: stride of 32 floats (128 bytes)
// On Apple GPU, this should maximize bank conflicts
kernel void tgmem_conflict_rw(
    device float *output [[buffer(0)]],
    constant uint &iterations [[buffer(1)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    const uint STRIDE = 32; // Same as SIMD width — likely conflicts
    const uint ELEMENTS_PER_THREAD = 8;

    for (uint i = 0; i < ELEMENTS_PER_THREAD; i++) {
        shared[(tid * STRIDE + i) % 8192] = float(tid + i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accum = 0.0;
    for (uint iter = 0; iter < iterations; iter++) {
        for (uint i = 0; i < ELEMENTS_PER_THREAD; i++) {
            float val = shared[(tid * STRIDE + i) % 8192];
            accum += val;
            shared[(tid * STRIDE + i) % 8192] = val + 1.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[tgid] = accum;
    }
}

// ============================================================================
// Benchmark 2: SIMD Shuffle Throughput
// Measures simd_shuffle bandwidth within a SIMD group.
// ============================================================================

kernel void simd_shuffle_throughput(
    device float *output [[buffer(0)]],
    constant uint &iterations [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    float val = float(lane);

    // Sustained shuffle chain: each iteration shuffles to next lane
    for (uint iter = 0; iter < iterations; iter++) {
        // Shuffle through all 32 lanes in the SIMD group
        val = simd_shuffle(val, (lane + 1) % 32);
        val = simd_shuffle(val, (lane + 2) % 32);
        val = simd_shuffle(val, (lane + 4) % 32);
        val = simd_shuffle(val, (lane + 8) % 32);
        val = simd_shuffle(val, (lane + 16) % 32);

        // Also measure shuffle_xor (butterfly pattern — FFT-relevant)
        val = simd_shuffle_xor(val, 1);
        val = simd_shuffle_xor(val, 2);
        val = simd_shuffle_xor(val, 4);
        val = simd_shuffle_xor(val, 8);
        val = simd_shuffle_xor(val, 16);
    }

    if (tid == 0) {
        output[tgid] = val;
    }
}

// Float2 (complex) shuffle — measures throughput for complex FFT data
kernel void simd_shuffle_complex_throughput(
    device float2 *output [[buffer(0)]],
    constant uint &iterations [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    float2 val = float2(float(lane), float(lane + 32));

    for (uint iter = 0; iter < iterations; iter++) {
        val = simd_shuffle(val, (lane + 1) % 32);
        val = simd_shuffle(val, (lane + 2) % 32);
        val = simd_shuffle(val, (lane + 4) % 32);
        val = simd_shuffle(val, (lane + 8) % 32);
        val = simd_shuffle(val, (lane + 16) % 32);

        val = simd_shuffle_xor(val, 1);
        val = simd_shuffle_xor(val, 2);
        val = simd_shuffle_xor(val, 4);
        val = simd_shuffle_xor(val, 8);
        val = simd_shuffle_xor(val, 16);
    }

    if (tid == 0) {
        output[tgid] = val;
    }
}

// ============================================================================
// Benchmark 3: Register-to-Threadgroup Copy Throughput
// Measures transfer rate between private registers and shared memory.
// ============================================================================

kernel void reg_to_tgmem_copy(
    device float *output [[buffer(0)]],
    constant uint &iterations [[buffer(1)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    // Each thread has 16 private float values (64 bytes) in registers
    float r0  = float(tid);
    float r1  = float(tid + 1);
    float r2  = float(tid + 2);
    float r3  = float(tid + 3);
    float r4  = float(tid + 4);
    float r5  = float(tid + 5);
    float r6  = float(tid + 6);
    float r7  = float(tid + 7);
    float r8  = float(tid + 8);
    float r9  = float(tid + 9);
    float r10 = float(tid + 10);
    float r11 = float(tid + 11);
    float r12 = float(tid + 12);
    float r13 = float(tid + 13);
    float r14 = float(tid + 14);
    float r15 = float(tid + 15);

    float accum = 0.0;
    uint base = tid * 16;

    for (uint iter = 0; iter < iterations; iter++) {
        // Register → Threadgroup (write)
        shared[base + 0]  = r0;
        shared[base + 1]  = r1;
        shared[base + 2]  = r2;
        shared[base + 3]  = r3;
        shared[base + 4]  = r4;
        shared[base + 5]  = r5;
        shared[base + 6]  = r6;
        shared[base + 7]  = r7;
        shared[base + 8]  = r8;
        shared[base + 9]  = r9;
        shared[base + 10] = r10;
        shared[base + 11] = r11;
        shared[base + 12] = r12;
        shared[base + 13] = r13;
        shared[base + 14] = r14;
        shared[base + 15] = r15;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Threadgroup → Register (read from different thread's data)
        uint src = ((tid + 1) % threads_per_tg) * 16;
        r0  = shared[src + 0];
        r1  = shared[src + 1];
        r2  = shared[src + 2];
        r3  = shared[src + 3];
        r4  = shared[src + 4];
        r5  = shared[src + 5];
        r6  = shared[src + 6];
        r7  = shared[src + 7];
        r8  = shared[src + 8];
        r9  = shared[src + 9];
        r10 = shared[src + 10];
        r11 = shared[src + 11];
        r12 = shared[src + 12];
        r13 = shared[src + 13];
        r14 = shared[src + 14];
        r15 = shared[src + 15];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        accum += r0 + r8;
    }

    if (tid == 0) {
        output[tgid] = accum;
    }
}

// ============================================================================
// Benchmark 4: Occupancy vs Register Pressure
// Kernels using different register counts to measure occupancy impact.
// The compiler will allocate registers based on live variables.
// ============================================================================

// Minimal registers (~8 GPRs)
kernel void occupancy_low_regs(
    device float *output [[buffer(0)]],
    constant uint &iterations [[buffer(1)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    float a = float(tid);
    for (uint i = 0; i < iterations; i++) {
        shared[tid] = a;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        a = shared[(tid + 1) % threads_per_tg] + 1.0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[tgid] = a;
}

// Medium registers (~32 GPRs): 16 live float values
kernel void occupancy_med_regs(
    device float *output [[buffer(0)]],
    constant uint &iterations [[buffer(1)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    float v[16];
    for (uint j = 0; j < 16; j++) v[j] = float(tid + j);

    for (uint i = 0; i < iterations; i++) {
        // Use all values to keep them live in registers
        float sum = 0.0;
        for (uint j = 0; j < 16; j++) sum += v[j];
        shared[tid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float neighbor = shared[(tid + 1) % threads_per_tg];
        for (uint j = 0; j < 16; j++) v[j] += neighbor * 0.01;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float final_sum = 0.0;
    for (uint j = 0; j < 16; j++) final_sum += v[j];
    if (tid == 0) output[tgid] = final_sum;
}

// High registers (~64 GPRs): 32 live float values
kernel void occupancy_high_regs(
    device float *output [[buffer(0)]],
    constant uint &iterations [[buffer(1)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    float v[32];
    for (uint j = 0; j < 32; j++) v[j] = float(tid + j);

    for (uint i = 0; i < iterations; i++) {
        float sum = 0.0;
        for (uint j = 0; j < 32; j++) sum += v[j];
        shared[tid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float neighbor = shared[(tid + 1) % threads_per_tg];
        for (uint j = 0; j < 32; j++) v[j] += neighbor * 0.01;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float final_sum = 0.0;
    for (uint j = 0; j < 32; j++) final_sum += v[j];
    if (tid == 0) output[tgid] = final_sum;
}

// Very high registers (~128 GPRs): 64 live float values
kernel void occupancy_vhigh_regs(
    device float *output [[buffer(0)]],
    constant uint &iterations [[buffer(1)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    float v[64];
    for (uint j = 0; j < 64; j++) v[j] = float(tid + j);

    for (uint i = 0; i < iterations; i++) {
        float sum = 0.0;
        for (uint j = 0; j < 64; j++) sum += v[j];
        shared[tid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float neighbor = shared[(tid + 1) % threads_per_tg];
        for (uint j = 0; j < 64; j++) v[j] += neighbor * 0.01;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float final_sum = 0.0;
    for (uint j = 0; j < 64; j++) final_sum += v[j];
    if (tid == 0) output[tgid] = final_sum;
}

// ============================================================================
// Benchmark 5: Optimal Thread Count
// Same workload (threadgroup memory read-modify-write) at different thread counts.
// Host-side dispatches with varying threadsPerThreadgroup.
// ============================================================================

kernel void thread_count_sweep(
    device float *output [[buffer(0)]],
    constant uint &iterations [[buffer(1)]],
    constant uint &elements_per_thread [[buffer(2)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    // Total work is constant: threads × elements_per_thread = 8192
    // Each thread does more work at lower thread counts
    float accum = 0.0;

    // Initialize threadgroup memory
    for (uint i = 0; i < elements_per_thread; i++) {
        uint idx = tid * elements_per_thread + i;
        if (idx < 8192) {
            shared[idx] = float(idx);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint iter = 0; iter < iterations; iter++) {
        // Butterfly-like access pattern: read pair, compute, write back
        for (uint i = 0; i < elements_per_thread; i++) {
            uint idx = tid * elements_per_thread + i;
            if (idx < 8192) {
                uint partner = idx ^ (1 << (iter % 13)); // butterfly partner
                if (partner < 8192) {
                    float a = shared[idx];
                    float b = shared[partner];
                    accum += a + b;
                    shared[idx] = a + b;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[tgid] = accum;
    }
}
