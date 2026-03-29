// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Fused FFT -> Matched Filter Multiply -> IFFT kernels for SAR compression
//
// Each threadgroup processes one line (range or azimuth) entirely in
// threadgroup memory. Device memory traffic is reduced from 6 transfers
// (3 reads + 3 writes in unfused) to 2 (1 read + 1 write).
//
// Architecture: radix-4 Stockham, 6 passes for N=4096, 1024 threads.
// Forward FFT: standard radix-4 with twiddle = -2*pi/N.
// Inverse FFT: IFFT(x) = (1/N) * conj(FFT(conj(x))).
//   Conjugation is done in-place in threadgroup memory (just negate .y).
//   This reuses the forward FFT butterfly unchanged, ensuring correctness.
// ============================================================================

constant uint N_FFT = 4096;
constant float TWO_PI_OVER_N = -2.0f * M_PI_F / float(N_FFT);
constant float INV_SCALE = 1.0f / float(N_FFT);

// --- Shared helpers ---

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline void radix4(thread float2& x0, thread float2& x1,
                   thread float2& x2, thread float2& x3) {
    float2 t0 = x0 + x2;
    float2 t1 = x1 + x3;
    float2 t2 = x0 - x2;
    float2 t3 = x1 - x3;
    float2 t3r = float2(t3.y, -t3.x);
    x0 = t0 + t1;
    x1 = t2 + t3r;
    x2 = t0 - t1;
    x3 = t2 - t3r;
}

inline void apply_twiddle3(thread float2& x1, thread float2& x2, thread float2& x3,
                           float2 w1) {
    float2 w2 = cmul(w1, w1);
    float2 w3 = cmul(w2, w1);
    x1 = cmul(x1, w1);
    x2 = cmul(x2, w2);
    x3 = cmul(x3, w3);
}

// ============================================================================
// Stockham FFT passes (forward only — IFFT uses conj-FFT-conj)
// ============================================================================

// Pass 0: stride=1, no twiddles — reads from device memory into threadgroup buf
inline void stockham_pass0_load(
    device const float2* src, uint base, uint tid,
    threadgroup float2* buf
) {
    float2 x0 = src[base + tid];
    float2 x1 = src[base + tid + 1024];
    float2 x2 = src[base + tid + 2048];
    float2 x3 = src[base + tid + 3072];
    radix4(x0, x1, x2, x3);
    uint wr = tid << 2;
    buf[wr]     = x0;
    buf[wr + 1] = x1;
    buf[wr + 2] = x2;
    buf[wr + 3] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 0: reads from threadgroup buf (used for IFFT after conjugation)
inline void stockham_pass0_tg(threadgroup float2* buf, uint tid) {
    float2 x0 = buf[tid];
    float2 x1 = buf[tid + 1024];
    float2 x2 = buf[tid + 2048];
    float2 x3 = buf[tid + 3072];
    radix4(x0, x1, x2, x3);
    uint wr = tid << 2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]     = x0;
    buf[wr + 1] = x1;
    buf[wr + 2] = x2;
    buf[wr + 3] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 1: stride=4, tw_scale=256
inline void stockham_pass1(threadgroup float2* buf, uint tid) {
    uint pos = tid & 3u;
    uint grp = tid >> 2;
    float2 x0 = buf[tid];
    float2 x1 = buf[tid + 1024];
    float2 x2 = buf[tid + 2048];
    float2 x3 = buf[tid + 3072];
    {
        float a1 = TWO_PI_OVER_N * float(pos << 8);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3(x1, x2, x3, float2(c1, s1));
    }
    radix4(x0, x1, x2, x3);
    uint wr = grp * 16u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]      = x0;
    buf[wr + 4]  = x1;
    buf[wr + 8]  = x2;
    buf[wr + 12] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 2: stride=16, tw_scale=64
inline void stockham_pass2(threadgroup float2* buf, uint tid) {
    uint pos = tid & 15u;
    uint grp = tid >> 4;
    float2 x0 = buf[tid];
    float2 x1 = buf[tid + 1024];
    float2 x2 = buf[tid + 2048];
    float2 x3 = buf[tid + 3072];
    {
        float a1 = TWO_PI_OVER_N * float(pos << 6);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3(x1, x2, x3, float2(c1, s1));
    }
    radix4(x0, x1, x2, x3);
    uint wr = grp * 64u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]      = x0;
    buf[wr + 16] = x1;
    buf[wr + 32] = x2;
    buf[wr + 48] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 3: stride=64, tw_scale=16
inline void stockham_pass3(threadgroup float2* buf, uint tid) {
    uint pos = tid & 63u;
    uint grp = tid >> 6;
    float2 x0 = buf[tid];
    float2 x1 = buf[tid + 1024];
    float2 x2 = buf[tid + 2048];
    float2 x3 = buf[tid + 3072];
    {
        float a1 = TWO_PI_OVER_N * float(pos << 4);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3(x1, x2, x3, float2(c1, s1));
    }
    radix4(x0, x1, x2, x3);
    uint wr = grp * 256u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]       = x0;
    buf[wr + 64]  = x1;
    buf[wr + 128] = x2;
    buf[wr + 192] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 4: stride=256, tw_scale=4
inline void stockham_pass4(threadgroup float2* buf, uint tid) {
    uint pos = tid & 255u;
    uint grp = tid >> 8;
    float2 x0 = buf[tid];
    float2 x1 = buf[tid + 1024];
    float2 x2 = buf[tid + 2048];
    float2 x3 = buf[tid + 3072];
    float a1 = TWO_PI_OVER_N * float(pos << 2);
    float s1, c1; s1 = sincos(a1, c1);
    apply_twiddle3(x1, x2, x3, float2(c1, s1));
    radix4(x0, x1, x2, x3);
    uint wr = grp * 1024u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]        = x0;
    buf[wr + 256]  = x1;
    buf[wr + 512]  = x2;
    buf[wr + 768]  = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 5 (final): writes result back to threadgroup buf
inline void stockham_pass5_to_tg(threadgroup float2* buf, uint tid) {
    float2 x0 = buf[tid];
    float2 x1 = buf[tid + 1024];
    float2 x2 = buf[tid + 2048];
    float2 x3 = buf[tid + 3072];
    float a1 = TWO_PI_OVER_N * float(tid);
    float s1, c1; s1 = sincos(a1, c1);
    apply_twiddle3(x1, x2, x3, float2(c1, s1));
    radix4(x0, x1, x2, x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[tid]        = x0;
    buf[tid + 1024] = x1;
    buf[tid + 2048] = x2;
    buf[tid + 3072] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 5 variant: writes to device memory with conjugate + scale (for IFFT output)
// Combines the final FFT pass, conjugation, and 1/N scaling into one step.
inline void stockham_pass5_conj_scale_store(
    threadgroup float2* buf, uint tid,
    device float2* dst, uint base
) {
    float2 x0 = buf[tid];
    float2 x1 = buf[tid + 1024];
    float2 x2 = buf[tid + 2048];
    float2 x3 = buf[tid + 3072];
    float a1 = TWO_PI_OVER_N * float(tid);
    float s1, c1; s1 = sincos(a1, c1);
    apply_twiddle3(x1, x2, x3, float2(c1, s1));
    radix4(x0, x1, x2, x3);
    // Conjugate + scale: output = conj(FFT_result) / N
    dst[base + tid]        = float2(x0.x, -x0.y) * INV_SCALE;
    dst[base + tid + 1024] = float2(x1.x, -x1.y) * INV_SCALE;
    dst[base + tid + 2048] = float2(x2.x, -x2.y) * INV_SCALE;
    dst[base + tid + 3072] = float2(x3.x, -x3.y) * INV_SCALE;
}

// Conjugate 4 elements per thread in threadgroup memory
inline void conjugate_tg(threadgroup float2* buf, uint tid) {
    buf[tid]        = float2(buf[tid].x,        -buf[tid].y);
    buf[tid + 1024] = float2(buf[tid + 1024].x, -buf[tid + 1024].y);
    buf[tid + 2048] = float2(buf[tid + 2048].x, -buf[tid + 2048].y);
    buf[tid + 3072] = float2(buf[tid + 3072].x, -buf[tid + 3072].y);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ============================================================================
// Fused Range Compression Kernel
//
// Single dispatch: load range line -> FFT -> multiply by matched filter
//                  -> IFFT -> store compressed line
//
// IFFT = (1/N) * conj(FFT(conj(x)))
//   Step 1: conjugate in threadgroup memory
//   Step 2: forward FFT (same passes as forward)
//   Step 3: conjugate + scale when writing to device memory
//
// Each threadgroup = one range line. 1024 threads, 4 elements each.
// ============================================================================

kernel void fused_range_compression(
    device const float2* input   [[buffer(0)]],  // Raw range data (nAz x nRg)
    device float2*       output  [[buffer(1)]],  // Compressed range data
    device const float2* filter  [[buffer(2)]],  // Matched filter in freq domain (nRg)
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_id  [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FFT;
    threadgroup float2 buf[N_FFT];

    // ===================== FORWARD FFT =====================
    stockham_pass0_load(input, base, tid, buf);
    stockham_pass1(buf, tid);
    stockham_pass2(buf, tid);
    stockham_pass3(buf, tid);
    stockham_pass4(buf, tid);
    stockham_pass5_to_tg(buf, tid);

    // ===================== MATCHED FILTER MULTIPLY =====================
    {
        float2 f0 = filter[tid];
        float2 f1 = filter[tid + 1024];
        float2 f2 = filter[tid + 2048];
        float2 f3 = filter[tid + 3072];

        buf[tid]        = cmul(buf[tid],        f0);
        buf[tid + 1024] = cmul(buf[tid + 1024], f1);
        buf[tid + 2048] = cmul(buf[tid + 2048], f2);
        buf[tid + 3072] = cmul(buf[tid + 3072], f3);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ===================== INVERSE FFT =====================
    // IFFT(x) = (1/N) * conj(FFT(conj(x)))
    // Step 1: Conjugate in threadgroup memory
    conjugate_tg(buf, tid);

    // Step 2: Forward FFT (passes 0-4 in threadgroup, pass 5 writes to device)
    stockham_pass0_tg(buf, tid);
    stockham_pass1(buf, tid);
    stockham_pass2(buf, tid);
    stockham_pass3(buf, tid);
    stockham_pass4(buf, tid);

    // Step 3: Final pass + conjugate + scale, write to device
    stockham_pass5_conj_scale_store(buf, tid, output, base);
}

// ============================================================================
// Fused Azimuth Compression Kernel (FFT -> multiply -> IFFT)
//
// Same structure as range compression but for azimuth (column) processing.
// Applied after data has been transposed so columns become rows.
// ============================================================================

kernel void fused_azimuth_compression(
    device const float2* input     [[buffer(0)]],
    device float2*       output    [[buffer(1)]],
    device const float2* az_filter [[buffer(2)]],
    device const uint*   params    [[buffer(3)]],  // [filterIsSingleRow]
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_id  [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FFT;
    const uint filterSingleRow = params[0];
    const uint filter_base = filterSingleRow ? 0 : (tg_id * N_FFT);

    threadgroup float2 buf[N_FFT];

    // ===================== FORWARD FFT =====================
    stockham_pass0_load(input, base, tid, buf);
    stockham_pass1(buf, tid);
    stockham_pass2(buf, tid);
    stockham_pass3(buf, tid);
    stockham_pass4(buf, tid);
    stockham_pass5_to_tg(buf, tid);

    // ===================== MATCHED FILTER MULTIPLY =====================
    {
        float2 f0 = az_filter[filter_base + tid];
        float2 f1 = az_filter[filter_base + tid + 1024];
        float2 f2 = az_filter[filter_base + tid + 2048];
        float2 f3 = az_filter[filter_base + tid + 3072];

        buf[tid]        = cmul(buf[tid],        f0);
        buf[tid + 1024] = cmul(buf[tid + 1024], f1);
        buf[tid + 2048] = cmul(buf[tid + 2048], f2);
        buf[tid + 3072] = cmul(buf[tid + 3072], f3);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ===================== INVERSE FFT =====================
    conjugate_tg(buf, tid);

    stockham_pass0_tg(buf, tid);
    stockham_pass1(buf, tid);
    stockham_pass2(buf, tid);
    stockham_pass3(buf, tid);
    stockham_pass4(buf, tid);

    stockham_pass5_conj_scale_store(buf, tid, output, base);
}

// ============================================================================
// Fused Multiply + IFFT Kernel
//
// For azimuth processing when data is already in frequency domain
// (after azimuth FFT in step 2). Fuses steps 4+5 of the RDA pipeline:
//   - Load frequency-domain data into threadgroup memory
//   - Multiply by matched filter
//   - IFFT via conj -> FFT -> conj+scale
//   - Store to device memory
//
// Eliminates 2 device memory transfers vs separate multiply + IFFT dispatches.
// ============================================================================

kernel void fused_multiply_ifft(
    device const float2* input     [[buffer(0)]],  // Freq-domain data (transposed rows)
    device float2*       output    [[buffer(1)]],  // Time-domain output
    device const float2* filter    [[buffer(2)]],  // Matched filter (per-row or single)
    device const uint*   params    [[buffer(3)]],  // [filterIsSingleRow]
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_id  [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FFT;
    const uint filterSingleRow = params[0];
    const uint filter_base = filterSingleRow ? 0 : (tg_id * N_FFT);

    threadgroup float2 buf[N_FFT];

    // ===================== LOAD + MULTIPLY =====================
    // Load data and filter, multiply, store to threadgroup memory
    {
        float2 d0 = input[base + tid];
        float2 d1 = input[base + tid + 1024];
        float2 d2 = input[base + tid + 2048];
        float2 d3 = input[base + tid + 3072];

        float2 f0 = filter[filter_base + tid];
        float2 f1 = filter[filter_base + tid + 1024];
        float2 f2 = filter[filter_base + tid + 2048];
        float2 f3 = filter[filter_base + tid + 3072];

        buf[tid]        = cmul(d0, f0);
        buf[tid + 1024] = cmul(d1, f1);
        buf[tid + 2048] = cmul(d2, f2);
        buf[tid + 3072] = cmul(d3, f3);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ===================== INVERSE FFT =====================
    // IFFT(x) = (1/N) * conj(FFT(conj(x)))
    conjugate_tg(buf, tid);

    stockham_pass0_tg(buf, tid);
    stockham_pass1(buf, tid);
    stockham_pass2(buf, tid);
    stockham_pass3(buf, tid);
    stockham_pass4(buf, tid);

    stockham_pass5_conj_scale_store(buf, tid, output, base);
}
