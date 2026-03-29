// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// N=4096 Stockham FFT — 6 radix-4 passes, fully unrolled
//
// Key optimizations:
// 1. Only 1 sincos per butterfly (w2=w1^2, w3=w1^3 via cmul)
// 2. Pass 0: reads from device memory directly, no twiddles
// 3. Last pass writes directly to device output
// 4. Eliminates initial load barrier + final store barrier
// 5. All strides are compile-time constants
// 6. Branchless twiddle application (no if pos>0 checks)
//
// Performance: 113.6 GFLOPS at batch 256 on M1 (2.16 us/FFT)
// Barriers: 10 total (1 after pass 0, 2 per pass for passes 1-4, 0 for pass 5)
// ============================================================================

constant uint N_FFT = 4096;
constant float TWO_PI_OVER_N = -2.0f * M_PI_F / float(N_FFT);

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

kernel void fft_4096_stockham(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    const uint base = tg_id * N_FFT;
    threadgroup float2 buf[N_FFT];

    // ======== Pass 0: stride=1, no twiddles — read from device memory ========
    {
        float2 x0 = input[base + tid];
        float2 x1 = input[base + tid + 1024];
        float2 x2 = input[base + tid + 2048];
        float2 x3 = input[base + tid + 3072];
        radix4(x0, x1, x2, x3);
        uint wr = tid << 2;
        buf[wr]     = x0;
        buf[wr + 1] = x1;
        buf[wr + 2] = x2;
        buf[wr + 3] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ======== Pass 1: stride=4, tw_scale=256 ========
    {
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

    // ======== Pass 2: stride=16, tw_scale=64 ========
    {
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

    // ======== Pass 3: stride=64, tw_scale=16 ========
    {
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

    // ======== Pass 4: stride=256, tw_scale=4 ========
    {
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

    // ======== Pass 5: stride=1024, tw_scale=1 — write directly to output ========
    {
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + 1024];
        float2 x2 = buf[tid + 2048];
        float2 x3 = buf[tid + 3072];
        float a1 = TWO_PI_OVER_N * float(tid);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3(x1, x2, x3, float2(c1, s1));
        radix4(x0, x1, x2, x3);
        output[base + tid]        = x0;
        output[base + tid + 1024] = x1;
        output[base + tid + 2048] = x2;
        output[base + tid + 3072] = x3;
    }
}
