// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Fused FFT → multiply → IFFT kernel — N=4096, radix-4 Stockham, 1024 threads
//
// Performs circular convolution entirely within threadgroup memory.
// Device memory traffic: 2 transfers (1 read + 1 write) vs 6 for unfused.
//
// Buffer layout:
//   0: input    float2[4096]  — time-domain signal
//   1: output   float2[4096]  — time-domain convolved result
//   2: filter   float2[4096]  — frequency-domain filter (pre-computed by host)
//
// IFFT via conjugate trick: IFFT(X) = conj(FFT(conj(X))) / N
// ============================================================================

constant uint N_FUSED  = 4096;
constant float TWO_PI_OVER_N_FUSED = -2.0f * M_PI_F / float(N_FUSED);
constant float INV_SCALE_FUSED = 1.0f / float(N_FUSED);

inline float2 cmul_f(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline void radix4_f(thread float2& x0, thread float2& x1,
                     thread float2& x2, thread float2& x3) {
    float2 t0 = x0 + x2, t1 = x1 + x3;
    float2 t2 = x0 - x2, t3 = x1 - x3;
    float2 t3r = float2(t3.y, -t3.x);
    x0 = t0 + t1; x1 = t2 + t3r;
    x2 = t0 - t1; x3 = t2 - t3r;
}

inline void apply_twiddle3_f(thread float2& x1, thread float2& x2, thread float2& x3, float2 w1) {
    float2 w2 = cmul_f(w1, w1), w3 = cmul_f(w2, w1);
    x1 = cmul_f(x1, w1); x2 = cmul_f(x2, w2); x3 = cmul_f(x3, w3);
}

// --- Stockham pass helpers (all operate on threadgroup buffer, 4 elements per thread) ---

inline void pass0_load(device const float2* src, uint base, uint tid, threadgroup float2* buf) {
    float2 x0 = src[base+tid], x1 = src[base+tid+1024], x2 = src[base+tid+2048], x3 = src[base+tid+3072];
    radix4_f(x0, x1, x2, x3);
    uint wr = tid << 2;
    buf[wr] = x0; buf[wr+1] = x1; buf[wr+2] = x2; buf[wr+3] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void pass0_tg(threadgroup float2* buf, uint tid) {
    float2 x0 = buf[tid], x1 = buf[tid+1024], x2 = buf[tid+2048], x3 = buf[tid+3072];
    radix4_f(x0, x1, x2, x3);
    uint wr = tid << 2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = x0; buf[wr+1] = x1; buf[wr+2] = x2; buf[wr+3] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void pass_mid(threadgroup float2* buf, uint tid, uint stride, uint tw_shift) {
    uint pos = tid & (stride - 1u), grp = tid >> __builtin_ctz(stride);
    float2 x0 = buf[tid], x1 = buf[tid+1024], x2 = buf[tid+2048], x3 = buf[tid+3072];
    { float a = TWO_PI_OVER_N_FUSED * float(pos << tw_shift);
      float s, c; s = sincos(a, c);
      apply_twiddle3_f(x1, x2, x3, float2(c, s)); }
    radix4_f(x0, x1, x2, x3);
    uint wr = grp * 4u * stride + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = x0; buf[wr+stride] = x1; buf[wr+2u*stride] = x2; buf[wr+3u*stride] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void pass5_tg(threadgroup float2* buf, uint tid) {
    float2 x0 = buf[tid], x1 = buf[tid+1024], x2 = buf[tid+2048], x3 = buf[tid+3072];
    float a = TWO_PI_OVER_N_FUSED * float(tid);
    float s, c; s = sincos(a, c);
    apply_twiddle3_f(x1, x2, x3, float2(c, s));
    radix4_f(x0, x1, x2, x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[tid] = x0; buf[tid+1024] = x1; buf[tid+2048] = x2; buf[tid+3072] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void pass5_conj_scale_store(threadgroup float2* buf, uint tid, device float2* dst, uint base) {
    float2 x0 = buf[tid], x1 = buf[tid+1024], x2 = buf[tid+2048], x3 = buf[tid+3072];
    float a = TWO_PI_OVER_N_FUSED * float(tid);
    float s, c; s = sincos(a, c);
    apply_twiddle3_f(x1, x2, x3, float2(c, s));
    radix4_f(x0, x1, x2, x3);
    dst[base+tid]      = float2(x0.x, -x0.y) * INV_SCALE_FUSED;
    dst[base+tid+1024] = float2(x1.x, -x1.y) * INV_SCALE_FUSED;
    dst[base+tid+2048] = float2(x2.x, -x2.y) * INV_SCALE_FUSED;
    dst[base+tid+3072] = float2(x3.x, -x3.y) * INV_SCALE_FUSED;
}

inline void conjugate_tg(threadgroup float2* buf, uint tid) {
    buf[tid]      = float2(buf[tid].x,      -buf[tid].y);
    buf[tid+1024] = float2(buf[tid+1024].x, -buf[tid+1024].y);
    buf[tid+2048] = float2(buf[tid+2048].x, -buf[tid+2048].y);
    buf[tid+3072] = float2(buf[tid+3072].x, -buf[tid+3072].y);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ============================================================================
// Fused convolve: FFT(input) → multiply by filter → IFFT → output
// One threadgroup per signal block. Dispatch batchSize threadgroups for batches.
// ============================================================================
kernel void fft_fused_convolve_4096(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    device const float2* filter [[buffer(2)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FUSED;
    threadgroup float2 buf[N_FUSED];

    // Forward FFT
    pass0_load(input, base, tid, buf);
    pass_mid(buf, tid, 4u,   8u);
    pass_mid(buf, tid, 16u,  6u);
    pass_mid(buf, tid, 64u,  4u);
    pass_mid(buf, tid, 256u, 2u);
    pass5_tg(buf, tid);

    // Pointwise multiply by pre-computed frequency-domain filter
    buf[tid]      = cmul_f(buf[tid],      filter[tid]);
    buf[tid+1024] = cmul_f(buf[tid+1024], filter[tid+1024]);
    buf[tid+2048] = cmul_f(buf[tid+2048], filter[tid+2048]);
    buf[tid+3072] = cmul_f(buf[tid+3072], filter[tid+3072]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inverse FFT: conj → FFT → conj+scale+store
    conjugate_tg(buf, tid);
    pass0_tg(buf, tid);
    pass_mid(buf, tid, 4u,   8u);
    pass_mid(buf, tid, 16u,  6u);
    pass_mid(buf, tid, 64u,  4u);
    pass_mid(buf, tid, 256u, 2u);
    pass5_conj_scale_store(buf, tid, output, base);
}
