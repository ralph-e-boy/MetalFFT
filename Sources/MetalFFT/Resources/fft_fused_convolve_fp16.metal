// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// FP16 Fused FFT → complex multiply → IFFT  (N = 4096, 1024 threads/group)
//
// Three precision modes matching the research paper:
//   Mode A (pure):    all butterfly arithmetic in FP16 — 16 KiB threadgroup,
//                     ~2× throughput, ~42 dB SQNR floor.
//   Mode B (storage): half2 threadgroup storage, FP32 butterfly — ~1.5×
//                     throughput, near-FP32 accuracy. Recommended.
//   Mode C (mixed):   FP16 twiddle multiply, FP32 accumulate — best tradeoff.
//
// Buffer layout (same as fft_fused_convolve.metal):
//   0: input  [float2 × 4096] per batch element (time domain)
//   1: output [float2 × 4096] per batch element (time domain)
//   2: filter [float2 × 4096] pre-computed frequency-domain filter (shared)
//
// Dispatch: (batchSize, 1, 1) threadgroups × (1024, 1, 1) threads
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

inline void apply_twiddle3_h(thread half2& x1, thread half2& x2, thread half2& x3, half2 w1) {
    half2 w2 = cmul_h(w1, w1);
    half2 w3 = cmul_h(w2, w1);
    x1 = cmul_h(x1, w1);
    x2 = cmul_h(x2, w2);
    x3 = cmul_h(x3, w3);
}

inline void stockham_pass0_load_h(device const float2* src, uint base, uint tid, threadgroup half2* buf) {
    half2 x0 = half2(src[base + tid]);
    half2 x1 = half2(src[base + tid + 1024]);
    half2 x2 = half2(src[base + tid + 2048]);
    half2 x3 = half2(src[base + tid + 3072]);
    radix4_h(x0, x1, x2, x3);
    uint wr = tid << 2;
    buf[wr] = x0; buf[wr+1] = x1; buf[wr+2] = x2; buf[wr+3] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass0_tg_h(threadgroup half2* buf, uint tid) {
    half2 x0 = buf[tid]; half2 x1 = buf[tid+1024]; half2 x2 = buf[tid+2048]; half2 x3 = buf[tid+3072];
    radix4_h(x0, x1, x2, x3);
    uint wr = tid << 2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = x0; buf[wr+1] = x1; buf[wr+2] = x2; buf[wr+3] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass1_h(threadgroup half2* buf, uint tid) {
    uint pos = tid & 3u, grp = tid >> 2;
    half2 x0 = buf[tid]; half2 x1 = buf[tid+1024]; half2 x2 = buf[tid+2048]; half2 x3 = buf[tid+3072];
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(pos << 8), c1);
    apply_twiddle3_h(x1, x2, x3, half2(c1, s1));
    radix4_h(x0, x1, x2, x3);
    uint wr = grp * 16u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = x0; buf[wr+4] = x1; buf[wr+8] = x2; buf[wr+12] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass2_h(threadgroup half2* buf, uint tid) {
    uint pos = tid & 15u, grp = tid >> 4;
    half2 x0 = buf[tid]; half2 x1 = buf[tid+1024]; half2 x2 = buf[tid+2048]; half2 x3 = buf[tid+3072];
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(pos << 6), c1);
    apply_twiddle3_h(x1, x2, x3, half2(c1, s1));
    radix4_h(x0, x1, x2, x3);
    uint wr = grp * 64u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = x0; buf[wr+16] = x1; buf[wr+32] = x2; buf[wr+48] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass3_h(threadgroup half2* buf, uint tid) {
    uint pos = tid & 63u, grp = tid >> 6;
    half2 x0 = buf[tid]; half2 x1 = buf[tid+1024]; half2 x2 = buf[tid+2048]; half2 x3 = buf[tid+3072];
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(pos << 4), c1);
    apply_twiddle3_h(x1, x2, x3, half2(c1, s1));
    radix4_h(x0, x1, x2, x3);
    uint wr = grp * 256u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = x0; buf[wr+64] = x1; buf[wr+128] = x2; buf[wr+192] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass4_h(threadgroup half2* buf, uint tid) {
    uint pos = tid & 255u, grp = tid >> 8;
    half2 x0 = buf[tid]; half2 x1 = buf[tid+1024]; half2 x2 = buf[tid+2048]; half2 x3 = buf[tid+3072];
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(pos << 2), c1);
    apply_twiddle3_h(x1, x2, x3, half2(c1, s1));
    radix4_h(x0, x1, x2, x3);
    uint wr = grp * 1024u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = x0; buf[wr+256] = x1; buf[wr+512] = x2; buf[wr+768] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass5_to_tg_h(threadgroup half2* buf, uint tid) {
    half2 x0 = buf[tid]; half2 x1 = buf[tid+1024]; half2 x2 = buf[tid+2048]; half2 x3 = buf[tid+3072];
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(tid), c1);
    apply_twiddle3_h(x1, x2, x3, half2(c1, s1));
    radix4_h(x0, x1, x2, x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[tid] = x0; buf[tid+1024] = x1; buf[tid+2048] = x2; buf[tid+3072] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass5_conj_scale_store_h(threadgroup half2* buf, uint tid, device float2* dst, uint base) {
    half2 x0 = buf[tid]; half2 x1 = buf[tid+1024]; half2 x2 = buf[tid+2048]; half2 x3 = buf[tid+3072];
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(tid), c1);
    apply_twiddle3_h(x1, x2, x3, half2(c1, s1));
    radix4_h(x0, x1, x2, x3);
    dst[base+tid]        = float2(half2(x0.x, -x0.y) * INV_SCALE_H);
    dst[base+tid+1024]   = float2(half2(x1.x, -x1.y) * INV_SCALE_H);
    dst[base+tid+2048]   = float2(half2(x2.x, -x2.y) * INV_SCALE_H);
    dst[base+tid+3072]   = float2(half2(x3.x, -x3.y) * INV_SCALE_H);
}

inline void conjugate_tg_h(threadgroup half2* buf, uint tid) {
    buf[tid]        = half2(buf[tid].x,        -buf[tid].y);
    buf[tid+1024]   = half2(buf[tid+1024].x,   -buf[tid+1024].y);
    buf[tid+2048]   = half2(buf[tid+2048].x,   -buf[tid+2048].y);
    buf[tid+3072]   = half2(buf[tid+3072].x,   -buf[tid+3072].y);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void fft_fused_convolve_fp16_pure(
    device const float2* input   [[buffer(0)]],
    device float2*       output  [[buffer(1)]],
    device const float2* filter  [[buffer(2)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_id  [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FFT;
    threadgroup half2 buf[N_FFT];

    stockham_pass0_load_h(input, base, tid, buf);
    stockham_pass1_h(buf, tid);
    stockham_pass2_h(buf, tid);
    stockham_pass3_h(buf, tid);
    stockham_pass4_h(buf, tid);
    stockham_pass5_to_tg_h(buf, tid);

    {
        half2 f0 = half2(filter[tid]);         half2 f1 = half2(filter[tid+1024]);
        half2 f2 = half2(filter[tid+2048]);    half2 f3 = half2(filter[tid+3072]);
        buf[tid]      = cmul_h(buf[tid],      f0);   buf[tid+1024] = cmul_h(buf[tid+1024], f1);
        buf[tid+2048] = cmul_h(buf[tid+2048], f2);   buf[tid+3072] = cmul_h(buf[tid+3072], f3);
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

// ============================================================================
// MODE B: FP16 Storage + FP32 Compute  (recommended)
// ============================================================================

inline float2 cmul_f(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline void radix4_f(thread float2& x0, thread float2& x1,
                     thread float2& x2, thread float2& x3) {
    float2 t0 = x0 + x2, t1 = x1 + x3, t2 = x0 - x2, t3 = x1 - x3;
    float2 t3r = float2(t3.y, -t3.x);
    x0 = t0+t1; x1 = t2+t3r; x2 = t0-t1; x3 = t2-t3r;
}

inline void apply_twiddle3_f(thread float2& x1, thread float2& x2, thread float2& x3, float2 w1) {
    float2 w2 = cmul_f(w1, w1), w3 = cmul_f(w2, w1);
    x1 = cmul_f(x1, w1); x2 = cmul_f(x2, w2); x3 = cmul_f(x3, w3);
}

inline void stockham_pass0_load_b(device const float2* src, uint base, uint tid, threadgroup half2* buf) {
    float2 x0 = src[base+tid]; float2 x1 = src[base+tid+1024];
    float2 x2 = src[base+tid+2048]; float2 x3 = src[base+tid+3072];
    radix4_f(x0, x1, x2, x3);
    uint wr = tid << 2;
    buf[wr] = half2(x0); buf[wr+1] = half2(x1); buf[wr+2] = half2(x2); buf[wr+3] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass0_tg_b(threadgroup half2* buf, uint tid) {
    float2 x0 = float2(buf[tid]); float2 x1 = float2(buf[tid+1024]);
    float2 x2 = float2(buf[tid+2048]); float2 x3 = float2(buf[tid+3072]);
    radix4_f(x0, x1, x2, x3);
    uint wr = tid << 2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = half2(x0); buf[wr+1] = half2(x1); buf[wr+2] = half2(x2); buf[wr+3] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass1_b(threadgroup half2* buf, uint tid) {
    uint pos = tid & 3u, grp = tid >> 2;
    float2 x0 = float2(buf[tid]); float2 x1 = float2(buf[tid+1024]);
    float2 x2 = float2(buf[tid+2048]); float2 x3 = float2(buf[tid+3072]);
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(pos << 8), c1);
    apply_twiddle3_f(x1, x2, x3, float2(c1, s1));
    radix4_f(x0, x1, x2, x3);
    uint wr = grp * 16u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = half2(x0); buf[wr+4] = half2(x1); buf[wr+8] = half2(x2); buf[wr+12] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass2_b(threadgroup half2* buf, uint tid) {
    uint pos = tid & 15u, grp = tid >> 4;
    float2 x0 = float2(buf[tid]); float2 x1 = float2(buf[tid+1024]);
    float2 x2 = float2(buf[tid+2048]); float2 x3 = float2(buf[tid+3072]);
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(pos << 6), c1);
    apply_twiddle3_f(x1, x2, x3, float2(c1, s1));
    radix4_f(x0, x1, x2, x3);
    uint wr = grp * 64u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = half2(x0); buf[wr+16] = half2(x1); buf[wr+32] = half2(x2); buf[wr+48] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass3_b(threadgroup half2* buf, uint tid) {
    uint pos = tid & 63u, grp = tid >> 6;
    float2 x0 = float2(buf[tid]); float2 x1 = float2(buf[tid+1024]);
    float2 x2 = float2(buf[tid+2048]); float2 x3 = float2(buf[tid+3072]);
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(pos << 4), c1);
    apply_twiddle3_f(x1, x2, x3, float2(c1, s1));
    radix4_f(x0, x1, x2, x3);
    uint wr = grp * 256u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = half2(x0); buf[wr+64] = half2(x1); buf[wr+128] = half2(x2); buf[wr+192] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass4_b(threadgroup half2* buf, uint tid) {
    uint pos = tid & 255u, grp = tid >> 8;
    float2 x0 = float2(buf[tid]); float2 x1 = float2(buf[tid+1024]);
    float2 x2 = float2(buf[tid+2048]); float2 x3 = float2(buf[tid+3072]);
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(pos << 2), c1);
    apply_twiddle3_f(x1, x2, x3, float2(c1, s1));
    radix4_f(x0, x1, x2, x3);
    uint wr = grp * 1024u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = half2(x0); buf[wr+256] = half2(x1); buf[wr+512] = half2(x2); buf[wr+768] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass5_to_tg_b(threadgroup half2* buf, uint tid) {
    float2 x0 = float2(buf[tid]); float2 x1 = float2(buf[tid+1024]);
    float2 x2 = float2(buf[tid+2048]); float2 x3 = float2(buf[tid+3072]);
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(tid), c1);
    apply_twiddle3_f(x1, x2, x3, float2(c1, s1));
    radix4_f(x0, x1, x2, x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[tid] = half2(x0); buf[tid+1024] = half2(x1); buf[tid+2048] = half2(x2); buf[tid+3072] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass5_conj_scale_store_b(threadgroup half2* buf, uint tid, device float2* dst, uint base) {
    float2 x0 = float2(buf[tid]); float2 x1 = float2(buf[tid+1024]);
    float2 x2 = float2(buf[tid+2048]); float2 x3 = float2(buf[tid+3072]);
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(tid), c1);
    apply_twiddle3_f(x1, x2, x3, float2(c1, s1));
    radix4_f(x0, x1, x2, x3);
    dst[base+tid]      = float2(x0.x, -x0.y) * INV_SCALE_F;
    dst[base+tid+1024] = float2(x1.x, -x1.y) * INV_SCALE_F;
    dst[base+tid+2048] = float2(x2.x, -x2.y) * INV_SCALE_F;
    dst[base+tid+3072] = float2(x3.x, -x3.y) * INV_SCALE_F;
}

inline void conjugate_tg_b(threadgroup half2* buf, uint tid) {
    buf[tid]      = half2(buf[tid].x,      -buf[tid].y);
    buf[tid+1024] = half2(buf[tid+1024].x, -buf[tid+1024].y);
    buf[tid+2048] = half2(buf[tid+2048].x, -buf[tid+2048].y);
    buf[tid+3072] = half2(buf[tid+3072].x, -buf[tid+3072].y);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void fft_fused_convolve_fp16_storage(
    device const float2* input   [[buffer(0)]],
    device float2*       output  [[buffer(1)]],
    device const float2* filter  [[buffer(2)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_id  [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FFT;
    threadgroup half2 buf[N_FFT];

    stockham_pass0_load_b(input, base, tid, buf);
    stockham_pass1_b(buf, tid);
    stockham_pass2_b(buf, tid);
    stockham_pass3_b(buf, tid);
    stockham_pass4_b(buf, tid);
    stockham_pass5_to_tg_b(buf, tid);

    {
        float2 f0 = filter[tid];          float2 f1 = filter[tid+1024];
        float2 f2 = filter[tid+2048];     float2 f3 = filter[tid+3072];
        buf[tid]      = half2(cmul_f(float2(buf[tid]),      f0));
        buf[tid+1024] = half2(cmul_f(float2(buf[tid+1024]), f1));
        buf[tid+2048] = half2(cmul_f(float2(buf[tid+2048]), f2));
        buf[tid+3072] = half2(cmul_f(float2(buf[tid+3072]), f3));
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

// ============================================================================
// MODE C: FP16 Multiply + FP32 Accumulate
// ============================================================================

inline void radix4_mixed(thread float2& x0, thread float2& x1,
                         thread float2& x2, thread float2& x3) {
    float2 t0 = x0+x2, t1 = x1+x3, t2 = x0-x2, t3 = x1-x3;
    float2 t3r = float2(t3.y, -t3.x);
    x0 = t0+t1; x1 = t2+t3r; x2 = t0-t1; x3 = t2-t3r;
}

inline void apply_twiddle3_mixed(thread float2& x1, thread float2& x2, thread float2& x3, float2 w1) {
    half2 hw1 = half2(w1), hw2 = cmul_h(hw1, hw1), hw3 = cmul_h(hw2, hw1);
    x1 = float2(cmul_h(half2(x1), hw1));
    x2 = float2(cmul_h(half2(x2), hw2));
    x3 = float2(cmul_h(half2(x3), hw3));
}

inline void stockham_pass0_load_c(device const float2* src, uint base, uint tid, threadgroup half2* buf) {
    float2 x0 = src[base+tid]; float2 x1 = src[base+tid+1024];
    float2 x2 = src[base+tid+2048]; float2 x3 = src[base+tid+3072];
    radix4_mixed(x0, x1, x2, x3);
    uint wr = tid << 2;
    buf[wr] = half2(x0); buf[wr+1] = half2(x1); buf[wr+2] = half2(x2); buf[wr+3] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass0_tg_c(threadgroup half2* buf, uint tid) {
    float2 x0 = float2(buf[tid]); float2 x1 = float2(buf[tid+1024]);
    float2 x2 = float2(buf[tid+2048]); float2 x3 = float2(buf[tid+3072]);
    radix4_mixed(x0, x1, x2, x3);
    uint wr = tid << 2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = half2(x0); buf[wr+1] = half2(x1); buf[wr+2] = half2(x2); buf[wr+3] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass1_c(threadgroup half2* buf, uint tid) {
    uint pos = tid & 3u, grp = tid >> 2;
    float2 x0 = float2(buf[tid]); float2 x1 = float2(buf[tid+1024]);
    float2 x2 = float2(buf[tid+2048]); float2 x3 = float2(buf[tid+3072]);
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(pos << 8), c1);
    apply_twiddle3_mixed(x1, x2, x3, float2(c1, s1));
    radix4_mixed(x0, x1, x2, x3);
    uint wr = grp * 16u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = half2(x0); buf[wr+4] = half2(x1); buf[wr+8] = half2(x2); buf[wr+12] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass2_c(threadgroup half2* buf, uint tid) {
    uint pos = tid & 15u, grp = tid >> 4;
    float2 x0 = float2(buf[tid]); float2 x1 = float2(buf[tid+1024]);
    float2 x2 = float2(buf[tid+2048]); float2 x3 = float2(buf[tid+3072]);
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(pos << 6), c1);
    apply_twiddle3_mixed(x1, x2, x3, float2(c1, s1));
    radix4_mixed(x0, x1, x2, x3);
    uint wr = grp * 64u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = half2(x0); buf[wr+16] = half2(x1); buf[wr+32] = half2(x2); buf[wr+48] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass3_c(threadgroup half2* buf, uint tid) {
    uint pos = tid & 63u, grp = tid >> 6;
    float2 x0 = float2(buf[tid]); float2 x1 = float2(buf[tid+1024]);
    float2 x2 = float2(buf[tid+2048]); float2 x3 = float2(buf[tid+3072]);
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(pos << 4), c1);
    apply_twiddle3_mixed(x1, x2, x3, float2(c1, s1));
    radix4_mixed(x0, x1, x2, x3);
    uint wr = grp * 256u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = half2(x0); buf[wr+64] = half2(x1); buf[wr+128] = half2(x2); buf[wr+192] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass4_c(threadgroup half2* buf, uint tid) {
    uint pos = tid & 255u, grp = tid >> 8;
    float2 x0 = float2(buf[tid]); float2 x1 = float2(buf[tid+1024]);
    float2 x2 = float2(buf[tid+2048]); float2 x3 = float2(buf[tid+3072]);
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(pos << 2), c1);
    apply_twiddle3_mixed(x1, x2, x3, float2(c1, s1));
    radix4_mixed(x0, x1, x2, x3);
    uint wr = grp * 1024u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = half2(x0); buf[wr+256] = half2(x1); buf[wr+512] = half2(x2); buf[wr+768] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass5_to_tg_c(threadgroup half2* buf, uint tid) {
    float2 x0 = float2(buf[tid]); float2 x1 = float2(buf[tid+1024]);
    float2 x2 = float2(buf[tid+2048]); float2 x3 = float2(buf[tid+3072]);
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(tid), c1);
    apply_twiddle3_mixed(x1, x2, x3, float2(c1, s1));
    radix4_mixed(x0, x1, x2, x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[tid] = half2(x0); buf[tid+1024] = half2(x1); buf[tid+2048] = half2(x2); buf[tid+3072] = half2(x3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stockham_pass5_conj_scale_store_c(threadgroup half2* buf, uint tid, device float2* dst, uint base) {
    float2 x0 = float2(buf[tid]); float2 x1 = float2(buf[tid+1024]);
    float2 x2 = float2(buf[tid+2048]); float2 x3 = float2(buf[tid+3072]);
    float s1, c1; s1 = sincos(TWO_PI_OVER_N_F * float(tid), c1);
    apply_twiddle3_mixed(x1, x2, x3, float2(c1, s1));
    radix4_mixed(x0, x1, x2, x3);
    dst[base+tid]      = float2(x0.x, -x0.y) * INV_SCALE_F;
    dst[base+tid+1024] = float2(x1.x, -x1.y) * INV_SCALE_F;
    dst[base+tid+2048] = float2(x2.x, -x2.y) * INV_SCALE_F;
    dst[base+tid+3072] = float2(x3.x, -x3.y) * INV_SCALE_F;
}

kernel void fft_fused_convolve_fp16_mixed(
    device const float2* input   [[buffer(0)]],
    device float2*       output  [[buffer(1)]],
    device const float2* filter  [[buffer(2)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_id  [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FFT;
    threadgroup half2 buf[N_FFT];

    stockham_pass0_load_c(input, base, tid, buf);
    stockham_pass1_c(buf, tid);
    stockham_pass2_c(buf, tid);
    stockham_pass3_c(buf, tid);
    stockham_pass4_c(buf, tid);
    stockham_pass5_to_tg_c(buf, tid);

    {
        half2 hf0 = half2(filter[tid]);        half2 hf1 = half2(filter[tid+1024]);
        half2 hf2 = half2(filter[tid+2048]);   half2 hf3 = half2(filter[tid+3072]);
        buf[tid]      = cmul_h(buf[tid],      hf0);
        buf[tid+1024] = cmul_h(buf[tid+1024], hf1);
        buf[tid+2048] = cmul_h(buf[tid+2048], hf2);
        buf[tid+3072] = cmul_h(buf[tid+3072], hf3);
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
