// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// FP16 Fused FFT -> Matched Filter Multiply -> IFFT kernels for SAR
//
// Three precision modes for mixed-precision radar signal processing:
//
// Mode A: Pure FP16 — all data, twiddles, and computation in half precision
//   - Threadgroup buffer: half2[4096] = 16 KiB (half of FP32's 32 KiB)
//   - Expected: 2x throughput, ~42 dB SQNR floor
//   - Risk: overflow for large signal values, insufficient PSLR for radar
//
// Mode B: FP16 storage + FP32 compute (recommended)
//   - Data stored as half2 in threadgroup memory (16 KiB)
//   - Each thread converts to float2 for butterfly computation (free conversion)
//   - Twiddle factors computed in FP32
//   - Expected: ~1.5x throughput (reduced memory traffic), near-FP32 accuracy
//
// Mode C: FP16 multiply + FP32 accumulate
//   - Twiddle multiplication in FP16, butterfly additions in FP32
//   - Store intermediate in half2
//   - Expected: best accuracy/throughput tradeoff
//
// Architecture: radix-4 Stockham, 6 passes for N=4096, 1024 threads.
// Forward FFT: standard radix-4 with twiddle = -2*pi/N.
// Inverse FFT: IFFT(x) = (1/N) * conj(FFT(conj(x))).
// ============================================================================

constant uint N_FFT = 4096;
constant float TWO_PI_OVER_N_F = -2.0f * M_PI_F / float(N_FFT);
constant half  TWO_PI_OVER_N_H = half(-2.0f * M_PI_F / float(N_FFT));
constant float INV_SCALE_F = 1.0f / float(N_FFT);
constant half  INV_SCALE_H = half(1.0f / float(N_FFT));

// ============================================================================
// MODE A: Pure FP16
// ============================================================================

inline half2 cmul_h(half2 a, half2 b) {
    return half2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline void radix4_h(thread half2& x0, thread half2& x1,
                     thread half2& x2, thread half2& x3) {
    half2 t0 = x0 + x2;
    half2 t1 = x1 + x3;
    half2 t2 = x0 - x2;
    half2 t3 = x1 - x3;
    half2 t3r = half2(t3.y, -t3.x);
    x0 = t0 + t1;
    x1 = t2 + t3r;
    x2 = t0 - t1;
    x3 = t2 - t3r;
}

inline void apply_twiddle3_h(thread half2& x1, thread half2& x2, thread half2& x3,
                             half2 w1) {
    half2 w2 = cmul_h(w1, w1);
    half2 w3 = cmul_h(w2, w1);
    x1 = cmul_h(x1, w1);
    x2 = cmul_h(x2, w2);
    x3 = cmul_h(x3, w3);
}

// --- Mode A: Stockham passes (pure FP16) ---

// Pass 0: load from device (float2) into threadgroup (half2)
inline void stockham_pass0_load_h(
    device const float2* src, uint base, uint tid,
    threadgroup half2* buf
) {
    half2 x0 = half2(src[base + tid]);
    half2 x1 = half2(src[base + tid + 1024]);
    half2 x2 = half2(src[base + tid + 2048]);
    half2 x3 = half2(src[base + tid + 3072]);
    radix4_h(x0, x1, x2, x3);
    uint wr = tid << 2;
    buf[wr]     = x0;
    buf[wr + 1] = x1;
    buf[wr + 2] = x2;
    buf[wr + 3] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 0: from threadgroup (for IFFT after conjugation)
inline void stockham_pass0_tg_h(threadgroup half2* buf, uint tid) {
    half2 x0 = buf[tid];
    half2 x1 = buf[tid + 1024];
    half2 x2 = buf[tid + 2048];
    half2 x3 = buf[tid + 3072];
    radix4_h(x0, x1, x2, x3);
    uint wr = tid << 2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]     = x0;
    buf[wr + 1] = x1;
    buf[wr + 2] = x2;
    buf[wr + 3] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 1: stride=4, tw_scale=256
inline void stockham_pass1_h(threadgroup half2* buf, uint tid) {
    uint pos = tid & 3u;
    uint grp = tid >> 2;
    half2 x0 = buf[tid];
    half2 x1 = buf[tid + 1024];
    half2 x2 = buf[tid + 2048];
    half2 x3 = buf[tid + 3072];
    {
        // sincos only works with float; compute in float, convert result
        float a1 = TWO_PI_OVER_N_F * float(pos << 8);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3_h(x1, x2, x3, half2(c1, s1));
    }
    radix4_h(x0, x1, x2, x3);
    uint wr = grp * 16u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]      = x0;
    buf[wr + 4]  = x1;
    buf[wr + 8]  = x2;
    buf[wr + 12] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 2: stride=16, tw_scale=64
inline void stockham_pass2_h(threadgroup half2* buf, uint tid) {
    uint pos = tid & 15u;
    uint grp = tid >> 4;
    half2 x0 = buf[tid];
    half2 x1 = buf[tid + 1024];
    half2 x2 = buf[tid + 2048];
    half2 x3 = buf[tid + 3072];
    {
        float a1 = TWO_PI_OVER_N_F * float(pos << 6);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3_h(x1, x2, x3, half2(c1, s1));
    }
    radix4_h(x0, x1, x2, x3);
    uint wr = grp * 64u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]      = x0;
    buf[wr + 16] = x1;
    buf[wr + 32] = x2;
    buf[wr + 48] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 3: stride=64, tw_scale=16
inline void stockham_pass3_h(threadgroup half2* buf, uint tid) {
    uint pos = tid & 63u;
    uint grp = tid >> 6;
    half2 x0 = buf[tid];
    half2 x1 = buf[tid + 1024];
    half2 x2 = buf[tid + 2048];
    half2 x3 = buf[tid + 3072];
    {
        float a1 = TWO_PI_OVER_N_F * float(pos << 4);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3_h(x1, x2, x3, half2(c1, s1));
    }
    radix4_h(x0, x1, x2, x3);
    uint wr = grp * 256u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]       = x0;
    buf[wr + 64]  = x1;
    buf[wr + 128] = x2;
    buf[wr + 192] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 4: stride=256, tw_scale=4
inline void stockham_pass4_h(threadgroup half2* buf, uint tid) {
    uint pos = tid & 255u;
    uint grp = tid >> 8;
    half2 x0 = buf[tid];
    half2 x1 = buf[tid + 1024];
    half2 x2 = buf[tid + 2048];
    half2 x3 = buf[tid + 3072];
    float a1 = TWO_PI_OVER_N_F * float(pos << 2);
    float s1, c1; s1 = sincos(a1, c1);
    apply_twiddle3_h(x1, x2, x3, half2(c1, s1));
    radix4_h(x0, x1, x2, x3);
    uint wr = grp * 1024u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]        = x0;
    buf[wr + 256]  = x1;
    buf[wr + 512]  = x2;
    buf[wr + 768]  = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 5 (final): writes to threadgroup
inline void stockham_pass5_to_tg_h(threadgroup half2* buf, uint tid) {
    half2 x0 = buf[tid];
    half2 x1 = buf[tid + 1024];
    half2 x2 = buf[tid + 2048];
    half2 x3 = buf[tid + 3072];
    float a1 = TWO_PI_OVER_N_F * float(tid);
    float s1, c1; s1 = sincos(a1, c1);
    apply_twiddle3_h(x1, x2, x3, half2(c1, s1));
    radix4_h(x0, x1, x2, x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[tid]        = x0;
    buf[tid + 1024] = x1;
    buf[tid + 2048] = x2;
    buf[tid + 3072] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 5: writes to device with conjugate + scale (IFFT output)
inline void stockham_pass5_conj_scale_store_h(
    threadgroup half2* buf, uint tid,
    device float2* dst, uint base
) {
    half2 x0 = buf[tid];
    half2 x1 = buf[tid + 1024];
    half2 x2 = buf[tid + 2048];
    half2 x3 = buf[tid + 3072];
    float a1 = TWO_PI_OVER_N_F * float(tid);
    float s1, c1; s1 = sincos(a1, c1);
    apply_twiddle3_h(x1, x2, x3, half2(c1, s1));
    radix4_h(x0, x1, x2, x3);
    // Conjugate + scale, convert back to float2 for output
    dst[base + tid]        = float2(half2(x0.x, -x0.y) * INV_SCALE_H);
    dst[base + tid + 1024] = float2(half2(x1.x, -x1.y) * INV_SCALE_H);
    dst[base + tid + 2048] = float2(half2(x2.x, -x2.y) * INV_SCALE_H);
    dst[base + tid + 3072] = float2(half2(x3.x, -x3.y) * INV_SCALE_H);
}

// Conjugate in threadgroup (half2)
inline void conjugate_tg_h(threadgroup half2* buf, uint tid) {
    buf[tid]        = half2(buf[tid].x,        -buf[tid].y);
    buf[tid + 1024] = half2(buf[tid + 1024].x, -buf[tid + 1024].y);
    buf[tid + 2048] = half2(buf[tid + 2048].x, -buf[tid + 2048].y);
    buf[tid + 3072] = half2(buf[tid + 3072].x, -buf[tid + 3072].y);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ============================================================================
// MODE A KERNEL: Pure FP16 Fused Range Compression
// ============================================================================

kernel void fused_range_compression_fp16_pure(
    device const float2* input   [[buffer(0)]],
    device float2*       output  [[buffer(1)]],
    device const float2* filter  [[buffer(2)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_id  [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FFT;
    threadgroup half2 buf[N_FFT];  // 16 KiB vs 32 KiB for float2

    // Forward FFT (pure FP16)
    stockham_pass0_load_h(input, base, tid, buf);
    stockham_pass1_h(buf, tid);
    stockham_pass2_h(buf, tid);
    stockham_pass3_h(buf, tid);
    stockham_pass4_h(buf, tid);
    stockham_pass5_to_tg_h(buf, tid);

    // Matched filter multiply (FP16)
    {
        half2 f0 = half2(filter[tid]);
        half2 f1 = half2(filter[tid + 1024]);
        half2 f2 = half2(filter[tid + 2048]);
        half2 f3 = half2(filter[tid + 3072]);

        buf[tid]        = cmul_h(buf[tid],        f0);
        buf[tid + 1024] = cmul_h(buf[tid + 1024], f1);
        buf[tid + 2048] = cmul_h(buf[tid + 2048], f2);
        buf[tid + 3072] = cmul_h(buf[tid + 3072], f3);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Inverse FFT
    conjugate_tg_h(buf, tid);
    stockham_pass0_tg_h(buf, tid);
    stockham_pass1_h(buf, tid);
    stockham_pass2_h(buf, tid);
    stockham_pass3_h(buf, tid);
    stockham_pass4_h(buf, tid);
    stockham_pass5_conj_scale_store_h(buf, tid, output, base);
}


// ============================================================================
// MODE B: FP16 Storage + FP32 Compute
//
// Data stored as half2 in threadgroup memory (16 KiB).
// Each thread loads half2, converts to float2 (free), does butterfly in FP32,
// converts back to half2 (free) for storage.
// Twiddle factors always in FP32.
// ============================================================================

inline float2 cmul_f(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline void radix4_f(thread float2& x0, thread float2& x1,
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

inline void apply_twiddle3_f(thread float2& x1, thread float2& x2, thread float2& x3,
                             float2 w1) {
    float2 w2 = cmul_f(w1, w1);
    float2 w3 = cmul_f(w2, w1);
    x1 = cmul_f(x1, w1);
    x2 = cmul_f(x2, w2);
    x3 = cmul_f(x3, w3);
}

// --- Mode B: Stockham passes (FP16 storage, FP32 compute) ---

// Pass 0: load from device (float2), compute in FP32, store as half2
inline void stockham_pass0_load_b(
    device const float2* src, uint base, uint tid,
    threadgroup half2* buf
) {
    float2 x0 = src[base + tid];
    float2 x1 = src[base + tid + 1024];
    float2 x2 = src[base + tid + 2048];
    float2 x3 = src[base + tid + 3072];
    radix4_f(x0, x1, x2, x3);
    uint wr = tid << 2;
    buf[wr]     = half2(x0);
    buf[wr + 1] = half2(x1);
    buf[wr + 2] = half2(x2);
    buf[wr + 3] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 0: from threadgroup (for IFFT)
inline void stockham_pass0_tg_b(threadgroup half2* buf, uint tid) {
    float2 x0 = float2(buf[tid]);
    float2 x1 = float2(buf[tid + 1024]);
    float2 x2 = float2(buf[tid + 2048]);
    float2 x3 = float2(buf[tid + 3072]);
    radix4_f(x0, x1, x2, x3);
    uint wr = tid << 2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]     = half2(x0);
    buf[wr + 1] = half2(x1);
    buf[wr + 2] = half2(x2);
    buf[wr + 3] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 1: stride=4
inline void stockham_pass1_b(threadgroup half2* buf, uint tid) {
    uint pos = tid & 3u;
    uint grp = tid >> 2;
    float2 x0 = float2(buf[tid]);
    float2 x1 = float2(buf[tid + 1024]);
    float2 x2 = float2(buf[tid + 2048]);
    float2 x3 = float2(buf[tid + 3072]);
    {
        float a1 = TWO_PI_OVER_N_F * float(pos << 8);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3_f(x1, x2, x3, float2(c1, s1));
    }
    radix4_f(x0, x1, x2, x3);
    uint wr = grp * 16u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]      = half2(x0);
    buf[wr + 4]  = half2(x1);
    buf[wr + 8]  = half2(x2);
    buf[wr + 12] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 2: stride=16
inline void stockham_pass2_b(threadgroup half2* buf, uint tid) {
    uint pos = tid & 15u;
    uint grp = tid >> 4;
    float2 x0 = float2(buf[tid]);
    float2 x1 = float2(buf[tid + 1024]);
    float2 x2 = float2(buf[tid + 2048]);
    float2 x3 = float2(buf[tid + 3072]);
    {
        float a1 = TWO_PI_OVER_N_F * float(pos << 6);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3_f(x1, x2, x3, float2(c1, s1));
    }
    radix4_f(x0, x1, x2, x3);
    uint wr = grp * 64u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]      = half2(x0);
    buf[wr + 16] = half2(x1);
    buf[wr + 32] = half2(x2);
    buf[wr + 48] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 3: stride=64
inline void stockham_pass3_b(threadgroup half2* buf, uint tid) {
    uint pos = tid & 63u;
    uint grp = tid >> 6;
    float2 x0 = float2(buf[tid]);
    float2 x1 = float2(buf[tid + 1024]);
    float2 x2 = float2(buf[tid + 2048]);
    float2 x3 = float2(buf[tid + 3072]);
    {
        float a1 = TWO_PI_OVER_N_F * float(pos << 4);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3_f(x1, x2, x3, float2(c1, s1));
    }
    radix4_f(x0, x1, x2, x3);
    uint wr = grp * 256u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]       = half2(x0);
    buf[wr + 64]  = half2(x1);
    buf[wr + 128] = half2(x2);
    buf[wr + 192] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 4: stride=256
inline void stockham_pass4_b(threadgroup half2* buf, uint tid) {
    uint pos = tid & 255u;
    uint grp = tid >> 8;
    float2 x0 = float2(buf[tid]);
    float2 x1 = float2(buf[tid + 1024]);
    float2 x2 = float2(buf[tid + 2048]);
    float2 x3 = float2(buf[tid + 3072]);
    float a1 = TWO_PI_OVER_N_F * float(pos << 2);
    float s1, c1; s1 = sincos(a1, c1);
    apply_twiddle3_f(x1, x2, x3, float2(c1, s1));
    radix4_f(x0, x1, x2, x3);
    uint wr = grp * 1024u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]        = half2(x0);
    buf[wr + 256]  = half2(x1);
    buf[wr + 512]  = half2(x2);
    buf[wr + 768]  = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 5 (final): writes to threadgroup
inline void stockham_pass5_to_tg_b(threadgroup half2* buf, uint tid) {
    float2 x0 = float2(buf[tid]);
    float2 x1 = float2(buf[tid + 1024]);
    float2 x2 = float2(buf[tid + 2048]);
    float2 x3 = float2(buf[tid + 3072]);
    float a1 = TWO_PI_OVER_N_F * float(tid);
    float s1, c1; s1 = sincos(a1, c1);
    apply_twiddle3_f(x1, x2, x3, float2(c1, s1));
    radix4_f(x0, x1, x2, x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[tid]        = half2(x0);
    buf[tid + 1024] = half2(x1);
    buf[tid + 2048] = half2(x2);
    buf[tid + 3072] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 5: writes to device with conjugate + scale (IFFT output)
inline void stockham_pass5_conj_scale_store_b(
    threadgroup half2* buf, uint tid,
    device float2* dst, uint base
) {
    float2 x0 = float2(buf[tid]);
    float2 x1 = float2(buf[tid + 1024]);
    float2 x2 = float2(buf[tid + 2048]);
    float2 x3 = float2(buf[tid + 3072]);
    float a1 = TWO_PI_OVER_N_F * float(tid);
    float s1, c1; s1 = sincos(a1, c1);
    apply_twiddle3_f(x1, x2, x3, float2(c1, s1));
    radix4_f(x0, x1, x2, x3);
    // Conjugate + scale in FP32, write as float2
    dst[base + tid]        = float2(x0.x, -x0.y) * INV_SCALE_F;
    dst[base + tid + 1024] = float2(x1.x, -x1.y) * INV_SCALE_F;
    dst[base + tid + 2048] = float2(x2.x, -x2.y) * INV_SCALE_F;
    dst[base + tid + 3072] = float2(x3.x, -x3.y) * INV_SCALE_F;
}

// Conjugate in threadgroup (half2)
inline void conjugate_tg_b(threadgroup half2* buf, uint tid) {
    buf[tid]        = half2(buf[tid].x,        -buf[tid].y);
    buf[tid + 1024] = half2(buf[tid + 1024].x, -buf[tid + 1024].y);
    buf[tid + 2048] = half2(buf[tid + 2048].x, -buf[tid + 2048].y);
    buf[tid + 3072] = half2(buf[tid + 3072].x, -buf[tid + 3072].y);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ============================================================================
// MODE B KERNEL: FP16 Storage + FP32 Compute Fused Range Compression
// ============================================================================

kernel void fused_range_compression_fp16_storage(
    device const float2* input   [[buffer(0)]],
    device float2*       output  [[buffer(1)]],
    device const float2* filter  [[buffer(2)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_id  [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FFT;
    threadgroup half2 buf[N_FFT];  // 16 KiB

    // Forward FFT (FP16 storage, FP32 compute)
    stockham_pass0_load_b(input, base, tid, buf);
    stockham_pass1_b(buf, tid);
    stockham_pass2_b(buf, tid);
    stockham_pass3_b(buf, tid);
    stockham_pass4_b(buf, tid);
    stockham_pass5_to_tg_b(buf, tid);

    // Matched filter multiply (FP32 compute, half2 storage)
    {
        float2 f0 = filter[tid];
        float2 f1 = filter[tid + 1024];
        float2 f2 = filter[tid + 2048];
        float2 f3 = filter[tid + 3072];

        buf[tid]        = half2(cmul_f(float2(buf[tid]),        f0));
        buf[tid + 1024] = half2(cmul_f(float2(buf[tid + 1024]), f1));
        buf[tid + 2048] = half2(cmul_f(float2(buf[tid + 2048]), f2));
        buf[tid + 3072] = half2(cmul_f(float2(buf[tid + 3072]), f3));

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Inverse FFT
    conjugate_tg_b(buf, tid);
    stockham_pass0_tg_b(buf, tid);
    stockham_pass1_b(buf, tid);
    stockham_pass2_b(buf, tid);
    stockham_pass3_b(buf, tid);
    stockham_pass4_b(buf, tid);
    stockham_pass5_conj_scale_store_b(buf, tid, output, base);
}


// ============================================================================
// MODE C: FP16 Multiply + FP32 Accumulate
//
// Twiddle multiplication performed in FP16 for throughput.
// Butterfly additions (accumulation) performed in FP32 for precision.
// Results stored back as half2.
// ============================================================================

inline void radix4_mixed(thread float2& x0, thread float2& x1,
                         thread float2& x2, thread float2& x3) {
    // Additions in FP32 for accumulation precision
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

inline void apply_twiddle3_mixed(thread float2& x1, thread float2& x2, thread float2& x3,
                                 float2 w1) {
    // Twiddle multiplication: convert to half for the multiply, accumulate in float
    half2 hw1 = half2(w1);
    half2 hw2 = cmul_h(hw1, hw1);
    half2 hw3 = cmul_h(hw2, hw1);
    // Multiply in FP16, result promoted to FP32
    half2 hx1 = half2(x1);
    half2 hx2 = half2(x2);
    half2 hx3 = half2(x3);
    x1 = float2(cmul_h(hx1, hw1));
    x2 = float2(cmul_h(hx2, hw2));
    x3 = float2(cmul_h(hx3, hw3));
}

// --- Mode C: Stockham passes (FP16 multiply, FP32 accumulate) ---

// Pass 0: load from device, compute in mixed, store as half2
inline void stockham_pass0_load_c(
    device const float2* src, uint base, uint tid,
    threadgroup half2* buf
) {
    float2 x0 = src[base + tid];
    float2 x1 = src[base + tid + 1024];
    float2 x2 = src[base + tid + 2048];
    float2 x3 = src[base + tid + 3072];
    radix4_mixed(x0, x1, x2, x3);
    uint wr = tid << 2;
    buf[wr]     = half2(x0);
    buf[wr + 1] = half2(x1);
    buf[wr + 2] = half2(x2);
    buf[wr + 3] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 0: from threadgroup (for IFFT)
inline void stockham_pass0_tg_c(threadgroup half2* buf, uint tid) {
    float2 x0 = float2(buf[tid]);
    float2 x1 = float2(buf[tid + 1024]);
    float2 x2 = float2(buf[tid + 2048]);
    float2 x3 = float2(buf[tid + 3072]);
    radix4_mixed(x0, x1, x2, x3);
    uint wr = tid << 2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]     = half2(x0);
    buf[wr + 1] = half2(x1);
    buf[wr + 2] = half2(x2);
    buf[wr + 3] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 1: stride=4
inline void stockham_pass1_c(threadgroup half2* buf, uint tid) {
    uint pos = tid & 3u;
    uint grp = tid >> 2;
    float2 x0 = float2(buf[tid]);
    float2 x1 = float2(buf[tid + 1024]);
    float2 x2 = float2(buf[tid + 2048]);
    float2 x3 = float2(buf[tid + 3072]);
    {
        float a1 = TWO_PI_OVER_N_F * float(pos << 8);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3_mixed(x1, x2, x3, float2(c1, s1));
    }
    radix4_mixed(x0, x1, x2, x3);
    uint wr = grp * 16u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]      = half2(x0);
    buf[wr + 4]  = half2(x1);
    buf[wr + 8]  = half2(x2);
    buf[wr + 12] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 2: stride=16
inline void stockham_pass2_c(threadgroup half2* buf, uint tid) {
    uint pos = tid & 15u;
    uint grp = tid >> 4;
    float2 x0 = float2(buf[tid]);
    float2 x1 = float2(buf[tid + 1024]);
    float2 x2 = float2(buf[tid + 2048]);
    float2 x3 = float2(buf[tid + 3072]);
    {
        float a1 = TWO_PI_OVER_N_F * float(pos << 6);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3_mixed(x1, x2, x3, float2(c1, s1));
    }
    radix4_mixed(x0, x1, x2, x3);
    uint wr = grp * 64u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]      = half2(x0);
    buf[wr + 16] = half2(x1);
    buf[wr + 32] = half2(x2);
    buf[wr + 48] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 3: stride=64
inline void stockham_pass3_c(threadgroup half2* buf, uint tid) {
    uint pos = tid & 63u;
    uint grp = tid >> 6;
    float2 x0 = float2(buf[tid]);
    float2 x1 = float2(buf[tid + 1024]);
    float2 x2 = float2(buf[tid + 2048]);
    float2 x3 = float2(buf[tid + 3072]);
    {
        float a1 = TWO_PI_OVER_N_F * float(pos << 4);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3_mixed(x1, x2, x3, float2(c1, s1));
    }
    radix4_mixed(x0, x1, x2, x3);
    uint wr = grp * 256u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]       = half2(x0);
    buf[wr + 64]  = half2(x1);
    buf[wr + 128] = half2(x2);
    buf[wr + 192] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 4: stride=256
inline void stockham_pass4_c(threadgroup half2* buf, uint tid) {
    uint pos = tid & 255u;
    uint grp = tid >> 8;
    float2 x0 = float2(buf[tid]);
    float2 x1 = float2(buf[tid + 1024]);
    float2 x2 = float2(buf[tid + 2048]);
    float2 x3 = float2(buf[tid + 3072]);
    float a1 = TWO_PI_OVER_N_F * float(pos << 2);
    float s1, c1; s1 = sincos(a1, c1);
    apply_twiddle3_mixed(x1, x2, x3, float2(c1, s1));
    radix4_mixed(x0, x1, x2, x3);
    uint wr = grp * 1024u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]        = half2(x0);
    buf[wr + 256]  = half2(x1);
    buf[wr + 512]  = half2(x2);
    buf[wr + 768]  = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 5: writes to threadgroup
inline void stockham_pass5_to_tg_c(threadgroup half2* buf, uint tid) {
    float2 x0 = float2(buf[tid]);
    float2 x1 = float2(buf[tid + 1024]);
    float2 x2 = float2(buf[tid + 2048]);
    float2 x3 = float2(buf[tid + 3072]);
    float a1 = TWO_PI_OVER_N_F * float(tid);
    float s1, c1; s1 = sincos(a1, c1);
    apply_twiddle3_mixed(x1, x2, x3, float2(c1, s1));
    radix4_mixed(x0, x1, x2, x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[tid]        = half2(x0);
    buf[tid + 1024] = half2(x1);
    buf[tid + 2048] = half2(x2);
    buf[tid + 3072] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pass 5: writes to device with conjugate + scale
inline void stockham_pass5_conj_scale_store_c(
    threadgroup half2* buf, uint tid,
    device float2* dst, uint base
) {
    float2 x0 = float2(buf[tid]);
    float2 x1 = float2(buf[tid + 1024]);
    float2 x2 = float2(buf[tid + 2048]);
    float2 x3 = float2(buf[tid + 3072]);
    float a1 = TWO_PI_OVER_N_F * float(tid);
    float s1, c1; s1 = sincos(a1, c1);
    apply_twiddle3_mixed(x1, x2, x3, float2(c1, s1));
    radix4_mixed(x0, x1, x2, x3);
    // Final output in FP32
    dst[base + tid]        = float2(x0.x, -x0.y) * INV_SCALE_F;
    dst[base + tid + 1024] = float2(x1.x, -x1.y) * INV_SCALE_F;
    dst[base + tid + 2048] = float2(x2.x, -x2.y) * INV_SCALE_F;
    dst[base + tid + 3072] = float2(x3.x, -x3.y) * INV_SCALE_F;
}

// Conjugate (reuses Mode B conjugate since storage is identical)

// ============================================================================
// MODE C KERNEL: FP16 Multiply + FP32 Accumulate Fused Range Compression
// ============================================================================

kernel void fused_range_compression_fp16_mixed(
    device const float2* input   [[buffer(0)]],
    device float2*       output  [[buffer(1)]],
    device const float2* filter  [[buffer(2)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_id  [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FFT;
    threadgroup half2 buf[N_FFT];  // 16 KiB

    // Forward FFT (FP16 multiply, FP32 accumulate)
    stockham_pass0_load_c(input, base, tid, buf);
    stockham_pass1_c(buf, tid);
    stockham_pass2_c(buf, tid);
    stockham_pass3_c(buf, tid);
    stockham_pass4_c(buf, tid);
    stockham_pass5_to_tg_c(buf, tid);

    // Matched filter multiply (mixed: FP16 multiply, result in FP32 -> half2)
    {
        half2 hf0 = half2(filter[tid]);
        half2 hf1 = half2(filter[tid + 1024]);
        half2 hf2 = half2(filter[tid + 2048]);
        half2 hf3 = half2(filter[tid + 3072]);

        buf[tid]        = cmul_h(buf[tid],        hf0);
        buf[tid + 1024] = cmul_h(buf[tid + 1024], hf1);
        buf[tid + 2048] = cmul_h(buf[tid + 2048], hf2);
        buf[tid + 3072] = cmul_h(buf[tid + 3072], hf3);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Inverse FFT
    conjugate_tg_b(buf, tid);  // Reuse Mode B conjugate (same half2 storage)
    stockham_pass0_tg_c(buf, tid);
    stockham_pass1_c(buf, tid);
    stockham_pass2_c(buf, tid);
    stockham_pass3_c(buf, tid);
    stockham_pass4_c(buf, tid);
    stockham_pass5_conj_scale_store_c(buf, tid, output, base);
}


// ============================================================================
// Azimuth Compression Variants (all three modes)
// ============================================================================

// --- Mode A: Pure FP16 Azimuth Compression ---

kernel void fused_azimuth_compression_fp16_pure(
    device const float2* input     [[buffer(0)]],
    device float2*       output    [[buffer(1)]],
    device const float2* az_filter [[buffer(2)]],
    device const uint*   params    [[buffer(3)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_id  [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FFT;
    const uint filterSingleRow = params[0];
    const uint filter_base = filterSingleRow ? 0 : (tg_id * N_FFT);
    threadgroup half2 buf[N_FFT];

    stockham_pass0_load_h(input, base, tid, buf);
    stockham_pass1_h(buf, tid);
    stockham_pass2_h(buf, tid);
    stockham_pass3_h(buf, tid);
    stockham_pass4_h(buf, tid);
    stockham_pass5_to_tg_h(buf, tid);

    {
        half2 f0 = half2(az_filter[filter_base + tid]);
        half2 f1 = half2(az_filter[filter_base + tid + 1024]);
        half2 f2 = half2(az_filter[filter_base + tid + 2048]);
        half2 f3 = half2(az_filter[filter_base + tid + 3072]);
        buf[tid]        = cmul_h(buf[tid],        f0);
        buf[tid + 1024] = cmul_h(buf[tid + 1024], f1);
        buf[tid + 2048] = cmul_h(buf[tid + 2048], f2);
        buf[tid + 3072] = cmul_h(buf[tid + 3072], f3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    conjugate_tg_h(buf, tid);
    stockham_pass0_tg_h(buf, tid);
    stockham_pass1_h(buf, tid);
    stockham_pass2_h(buf, tid);
    stockham_pass3_h(buf, tid);
    stockham_pass4_h(buf, tid);
    stockham_pass5_conj_scale_store_h(buf, tid, output, base);
}

// --- Mode B: FP16 Storage + FP32 Compute Azimuth Compression ---

kernel void fused_azimuth_compression_fp16_storage(
    device const float2* input     [[buffer(0)]],
    device float2*       output    [[buffer(1)]],
    device const float2* az_filter [[buffer(2)]],
    device const uint*   params    [[buffer(3)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_id  [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FFT;
    const uint filterSingleRow = params[0];
    const uint filter_base = filterSingleRow ? 0 : (tg_id * N_FFT);
    threadgroup half2 buf[N_FFT];

    stockham_pass0_load_b(input, base, tid, buf);
    stockham_pass1_b(buf, tid);
    stockham_pass2_b(buf, tid);
    stockham_pass3_b(buf, tid);
    stockham_pass4_b(buf, tid);
    stockham_pass5_to_tg_b(buf, tid);

    {
        float2 f0 = az_filter[filter_base + tid];
        float2 f1 = az_filter[filter_base + tid + 1024];
        float2 f2 = az_filter[filter_base + tid + 2048];
        float2 f3 = az_filter[filter_base + tid + 3072];
        buf[tid]        = half2(cmul_f(float2(buf[tid]),        f0));
        buf[tid + 1024] = half2(cmul_f(float2(buf[tid + 1024]), f1));
        buf[tid + 2048] = half2(cmul_f(float2(buf[tid + 2048]), f2));
        buf[tid + 3072] = half2(cmul_f(float2(buf[tid + 3072]), f3));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    conjugate_tg_b(buf, tid);
    stockham_pass0_tg_b(buf, tid);
    stockham_pass1_b(buf, tid);
    stockham_pass2_b(buf, tid);
    stockham_pass3_b(buf, tid);
    stockham_pass4_b(buf, tid);
    stockham_pass5_conj_scale_store_b(buf, tid, output, base);
}

// --- Mode C: FP16 Multiply + FP32 Accumulate Azimuth Compression ---

kernel void fused_azimuth_compression_fp16_mixed(
    device const float2* input     [[buffer(0)]],
    device float2*       output    [[buffer(1)]],
    device const float2* az_filter [[buffer(2)]],
    device const uint*   params    [[buffer(3)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_id  [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FFT;
    const uint filterSingleRow = params[0];
    const uint filter_base = filterSingleRow ? 0 : (tg_id * N_FFT);
    threadgroup half2 buf[N_FFT];

    stockham_pass0_load_c(input, base, tid, buf);
    stockham_pass1_c(buf, tid);
    stockham_pass2_c(buf, tid);
    stockham_pass3_c(buf, tid);
    stockham_pass4_c(buf, tid);
    stockham_pass5_to_tg_c(buf, tid);

    {
        half2 hf0 = half2(az_filter[filter_base + tid]);
        half2 hf1 = half2(az_filter[filter_base + tid + 1024]);
        half2 hf2 = half2(az_filter[filter_base + tid + 2048]);
        half2 hf3 = half2(az_filter[filter_base + tid + 3072]);
        buf[tid]        = cmul_h(buf[tid],        hf0);
        buf[tid + 1024] = cmul_h(buf[tid + 1024], hf1);
        buf[tid + 2048] = cmul_h(buf[tid + 2048], hf2);
        buf[tid + 3072] = cmul_h(buf[tid + 3072], hf3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    conjugate_tg_b(buf, tid);
    stockham_pass0_tg_c(buf, tid);
    stockham_pass1_c(buf, tid);
    stockham_pass2_c(buf, tid);
    stockham_pass3_c(buf, tid);
    stockham_pass4_c(buf, tid);
    stockham_pass5_conj_scale_store_c(buf, tid, output, base);
}


// ============================================================================
// Fused Multiply + IFFT Variants (for azimuth processing when data is
// already in frequency domain)
// ============================================================================

// --- Mode A: Pure FP16 ---

kernel void fused_multiply_ifft_fp16_pure(
    device const float2* input     [[buffer(0)]],
    device float2*       output    [[buffer(1)]],
    device const float2* filter    [[buffer(2)]],
    device const uint*   params    [[buffer(3)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_id  [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FFT;
    const uint filterSingleRow = params[0];
    const uint filter_base = filterSingleRow ? 0 : (tg_id * N_FFT);
    threadgroup half2 buf[N_FFT];

    {
        half2 d0 = half2(input[base + tid]);
        half2 d1 = half2(input[base + tid + 1024]);
        half2 d2 = half2(input[base + tid + 2048]);
        half2 d3 = half2(input[base + tid + 3072]);
        half2 f0 = half2(filter[filter_base + tid]);
        half2 f1 = half2(filter[filter_base + tid + 1024]);
        half2 f2 = half2(filter[filter_base + tid + 2048]);
        half2 f3 = half2(filter[filter_base + tid + 3072]);
        buf[tid]        = cmul_h(d0, f0);
        buf[tid + 1024] = cmul_h(d1, f1);
        buf[tid + 2048] = cmul_h(d2, f2);
        buf[tid + 3072] = cmul_h(d3, f3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    conjugate_tg_h(buf, tid);
    stockham_pass0_tg_h(buf, tid);
    stockham_pass1_h(buf, tid);
    stockham_pass2_h(buf, tid);
    stockham_pass3_h(buf, tid);
    stockham_pass4_h(buf, tid);
    stockham_pass5_conj_scale_store_h(buf, tid, output, base);
}

// --- Mode B: FP16 Storage + FP32 Compute ---

kernel void fused_multiply_ifft_fp16_storage(
    device const float2* input     [[buffer(0)]],
    device float2*       output    [[buffer(1)]],
    device const float2* filter    [[buffer(2)]],
    device const uint*   params    [[buffer(3)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_id  [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FFT;
    const uint filterSingleRow = params[0];
    const uint filter_base = filterSingleRow ? 0 : (tg_id * N_FFT);
    threadgroup half2 buf[N_FFT];

    {
        float2 d0 = input[base + tid];
        float2 d1 = input[base + tid + 1024];
        float2 d2 = input[base + tid + 2048];
        float2 d3 = input[base + tid + 3072];
        float2 f0 = filter[filter_base + tid];
        float2 f1 = filter[filter_base + tid + 1024];
        float2 f2 = filter[filter_base + tid + 2048];
        float2 f3 = filter[filter_base + tid + 3072];
        buf[tid]        = half2(cmul_f(d0, f0));
        buf[tid + 1024] = half2(cmul_f(d1, f1));
        buf[tid + 2048] = half2(cmul_f(d2, f2));
        buf[tid + 3072] = half2(cmul_f(d3, f3));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    conjugate_tg_b(buf, tid);
    stockham_pass0_tg_b(buf, tid);
    stockham_pass1_b(buf, tid);
    stockham_pass2_b(buf, tid);
    stockham_pass3_b(buf, tid);
    stockham_pass4_b(buf, tid);
    stockham_pass5_conj_scale_store_b(buf, tid, output, base);
}

// --- Mode C: FP16 Multiply + FP32 Accumulate ---

kernel void fused_multiply_ifft_fp16_mixed(
    device const float2* input     [[buffer(0)]],
    device float2*       output    [[buffer(1)]],
    device const float2* filter    [[buffer(2)]],
    device const uint*   params    [[buffer(3)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_id  [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FFT;
    const uint filterSingleRow = params[0];
    const uint filter_base = filterSingleRow ? 0 : (tg_id * N_FFT);
    threadgroup half2 buf[N_FFT];

    {
        // Multiply in FP16, result stored as half2
        half2 d0 = half2(input[base + tid]);
        half2 d1 = half2(input[base + tid + 1024]);
        half2 d2 = half2(input[base + tid + 2048]);
        half2 d3 = half2(input[base + tid + 3072]);
        half2 f0 = half2(filter[filter_base + tid]);
        half2 f1 = half2(filter[filter_base + tid + 1024]);
        half2 f2 = half2(filter[filter_base + tid + 2048]);
        half2 f3 = half2(filter[filter_base + tid + 3072]);
        buf[tid]        = cmul_h(d0, f0);
        buf[tid + 1024] = cmul_h(d1, f1);
        buf[tid + 2048] = cmul_h(d2, f2);
        buf[tid + 3072] = cmul_h(d3, f3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    conjugate_tg_b(buf, tid);
    stockham_pass0_tg_c(buf, tid);
    stockham_pass1_c(buf, tid);
    stockham_pass2_c(buf, tid);
    stockham_pass3_c(buf, tid);
    stockham_pass4_c(buf, tid);
    stockham_pass5_conj_scale_store_c(buf, tid, output, base);
}
