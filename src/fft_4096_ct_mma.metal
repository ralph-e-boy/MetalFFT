// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ============================================================================
// N=4096 FFT — In-place Cooley-Tukey DIF with simdgroup_matrix MMA
// ============================================================================

constant uint N_FFT = 4096;
constant float SQRT2_2 = 0.70710678118654752f;

inline uint digit_reverse_base8_4(uint idx) {
    uint d0 = idx & 7u;
    uint d1 = (idx >> 3) & 7u;
    uint d2 = (idx >> 6) & 7u;
    uint d3 = (idx >> 9) & 7u;
    return (d0 << 9) | (d1 << 6) | (d2 << 3) | d3;
}

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline void radix8_butterfly(thread float2& x0, thread float2& x1, thread float2& x2, thread float2& x3,
                              thread float2& x4, thread float2& x5, thread float2& x6, thread float2& x7) {
    float2 t0 = x0 + x4;
    float2 t1 = x1 + x5;
    float2 t2 = x2 + x6;
    float2 t3 = x3 + x7;
    float2 t4 = x0 - x4;
    float2 t5 = x1 - x5;
    float2 t6 = x2 - x6;
    float2 t7 = x3 - x7;

    float2 t5w = float2(SQRT2_2 * (t5.x + t5.y), SQRT2_2 * (t5.y - t5.x));
    float2 t6w = float2(t6.y, -t6.x);
    float2 t7w = float2(SQRT2_2 * (-t7.x + t7.y), SQRT2_2 * (-t7.y - t7.x));

    float2 u0 = t0 + t2;
    float2 u1 = t1 + t3;
    float2 u2 = t0 - t2;
    float2 u3 = t1 - t3;
    float2 u3r = float2(u3.y, -u3.x);

    float2 v0 = t4 + t6w;
    float2 v1 = t5w + t7w;
    float2 v2 = t4 - t6w;
    float2 v3 = t5w - t7w;
    float2 v3r = float2(v3.y, -v3.x);

    x0 = u0 + u1;
    x4 = u0 - u1;
    x2 = u2 + u3r;
    x6 = u2 - u3r;
    x1 = v0 + v1;
    x5 = v0 - v1;
    x3 = v2 + v3r;
    x7 = v2 - v3r;
}

inline void apply_twiddle8(thread float2& x1, thread float2& x2, thread float2& x3,
                           thread float2& x4, thread float2& x5, thread float2& x6,
                           thread float2& x7, float2 w1) {
    float2 w2 = cmul(w1, w1);
    float2 w3 = cmul(w2, w1);
    float2 w4 = cmul(w2, w2);
    float2 w5 = cmul(w4, w1);
    float2 w6 = cmul(w4, w2);
    float2 w7 = cmul(w4, w3);
    x1 = cmul(x1, w1);
    x2 = cmul(x2, w2);
    x3 = cmul(x3, w3);
    x4 = cmul(x4, w4);
    x5 = cmul(x5, w5);
    x6 = cmul(x6, w6);
    x7 = cmul(x7, w7);
}

// ============================================================================
// MMA kernel: stages 0-2 use simdgroup_matrix, stage 3 scalar
// ============================================================================
kernel void fft_4096_ct_mma(
    device const float2* input    [[buffer(0)]],
    device float2*       output   [[buffer(1)]],
    device const float*  dft_real     [[buffer(2)]],
    device const float*  dft_imag     [[buffer(3)]],
    device const float*  dft_neg_imag [[buffer(4)]],
    device const float2* twiddles     [[buffer(5)]],  // 3 * 4096 precomputed twiddle factors
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_id    [[simdgroup_index_in_threadgroup]],
    uint lane_id    [[thread_index_in_simdgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    const uint N = N_FFT;
    const uint base_offset = tg_id * N;

    threadgroup float buf_real[4096];
    threadgroup float buf_imag[4096];

    // Load DFT_8 matrices from device memory
    simdgroup_float8x8 A_real, A_imag, A_neg_imag;
    simdgroup_load(A_real, dft_real, 8);
    simdgroup_load(A_imag, dft_imag, 8);
    simdgroup_load(A_neg_imag, dft_neg_imag, 8);

    // Load input, deinterleave
    for (uint i = 0; i < 8; i++) {
        uint idx = tid + i * 512u;
        float2 val = input[base_offset + idx];
        buf_real[idx] = val.x;
        buf_imag[idx] = val.y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // MMA stages 0-2 (stride = 512, 64, 8)
    const uint tiles_per_simd = 4u;
    uint mma_strides[3] = {512u, 64u, 8u};
    uint group_sizes[3] = {4096u, 512u, 64u};

    for (uint stage = 0; stage < 3; stage++) {
        uint S = mma_strides[stage];
        uint G = group_sizes[stage];

        for (uint tile_idx = 0; tile_idx < tiles_per_simd; tile_idx++) {
            uint global_tile = simd_id * tiles_per_simd + tile_idx;
            uint btfl_base = global_tile * 8u;

            uint group_idx = btfl_base / S;
            uint k_base = btfl_base % S;
            uint group_start = group_idx * G;
            uint tile_origin = group_start + k_base;

            // Load 8x8 tile: X[row=k][col=butterfly] at stride S
            simdgroup_float8x8 X_real, X_imag;
            simdgroup_load(X_real, &buf_real[tile_origin], S);
            simdgroup_load(X_imag, &buf_imag[tile_origin], S);

            // Complex matrix multiply: Y = F8 * X
            simdgroup_float8x8 Y_real, Y_imag;

            simdgroup_multiply(Y_real, A_real, X_real);
            simdgroup_multiply_accumulate(Y_real, A_neg_imag, X_imag, Y_real);

            simdgroup_multiply(Y_imag, A_real, X_imag);
            simdgroup_multiply_accumulate(Y_imag, A_imag, X_real, Y_imag);

            // Apply twiddle via thread_elements() before storing
            // Verified mapping on Apple M1 (32-lane SIMD):
            //   row = (lane_id / 16) * 4 + (lane_id % 8) / 2
            //   col0 = ((lane_id / 8) % 2) * 4 + (lane_id % 2) * 2
            //   elem[0] at (row, col0), elem[1] at (row, col0+1)
            {
                uint tw_base = stage * N;
                uint row = (lane_id / 16u) * 4u + (lane_id % 8u) / 2u;
                uint col0 = ((lane_id / 8u) % 2u) * 4u + (lane_id % 2u) * 2u;

                for (uint e = 0; e < 2; e++) {
                    uint c = col0 + e;
                    uint elem_addr = tile_origin + row * S + c;
                    float2 tw = twiddles[tw_base + elem_addr];
                    float re = Y_real.thread_elements()[e];
                    float im = Y_imag.thread_elements()[e];
                    Y_real.thread_elements()[e] = re * tw.x - im * tw.y;
                    Y_imag.thread_elements()[e] = re * tw.y + im * tw.x;
                }
            }

            simdgroup_store(Y_real, &buf_real[tile_origin], S);
            simdgroup_store(Y_imag, &buf_imag[tile_origin], S);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Stage 3: scalar (S=1, G=8) + digit reversal + write to device output
    // Fused: avoids writing back to threadgroup and the barrier before digit reversal
    {
        uint btfl = tid;
        uint ba = btfl * 8u;
        float2 x0 = float2(buf_real[ba], buf_imag[ba]);
        float2 x1 = float2(buf_real[ba+1], buf_imag[ba+1]);
        float2 x2 = float2(buf_real[ba+2], buf_imag[ba+2]);
        float2 x3 = float2(buf_real[ba+3], buf_imag[ba+3]);
        float2 x4 = float2(buf_real[ba+4], buf_imag[ba+4]);
        float2 x5 = float2(buf_real[ba+5], buf_imag[ba+5]);
        float2 x6 = float2(buf_real[ba+6], buf_imag[ba+6]);
        float2 x7 = float2(buf_real[ba+7], buf_imag[ba+7]);
        radix8_butterfly(x0, x1, x2, x3, x4, x5, x6, x7);

        // Write directly to device output with digit reversal
        output[base_offset + digit_reverse_base8_4(ba)]   = x0;
        output[base_offset + digit_reverse_base8_4(ba+1)] = x1;
        output[base_offset + digit_reverse_base8_4(ba+2)] = x2;
        output[base_offset + digit_reverse_base8_4(ba+3)] = x3;
        output[base_offset + digit_reverse_base8_4(ba+4)] = x4;
        output[base_offset + digit_reverse_base8_4(ba+5)] = x5;
        output[base_offset + digit_reverse_base8_4(ba+6)] = x6;
        output[base_offset + digit_reverse_base8_4(ba+7)] = x7;
    }
}

// ============================================================================
// Scalar CT DIF (debugging reference)
// ============================================================================
kernel void fft_4096_ct_scalar(
    device const float2* input    [[buffer(0)]],
    device float2*       output   [[buffer(1)]],
    device const float*  dft_real [[buffer(2)]],
    device const float*  dft_imag [[buffer(3)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    const uint N = N_FFT;
    const uint base_offset = tg_id * N;
    threadgroup float2 buf[4096];

    for (uint i = 0; i < 8; i++) {
        uint idx = tid + i * 512u;
        buf[idx] = input[base_offset + idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint strides[4] = {512u, 64u, 8u, 1u};
    uint gsizes[4]  = {4096u, 512u, 64u, 8u};

    for (uint stage = 0; stage < 4; stage++) {
        uint S = strides[stage];
        uint G = gsizes[stage];
        uint btfl = tid;
        uint group_idx = btfl / S;
        uint k = btfl % S;
        uint gs = group_idx * G;

        float2 x0 = buf[gs + k + 0u*S];
        float2 x1 = buf[gs + k + 1u*S];
        float2 x2 = buf[gs + k + 2u*S];
        float2 x3 = buf[gs + k + 3u*S];
        float2 x4 = buf[gs + k + 4u*S];
        float2 x5 = buf[gs + k + 5u*S];
        float2 x6 = buf[gs + k + 6u*S];
        float2 x7 = buf[gs + k + 7u*S];

        radix8_butterfly(x0, x1, x2, x3, x4, x5, x6, x7);

        if (k > 0) {
            float base_angle = -2.0f * M_PI_F * float(k) / float(G);
            float s1, c1;
            s1 = sincos(base_angle, c1);
            apply_twiddle8(x1, x2, x3, x4, x5, x6, x7, float2(c1, s1));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[gs + k + 0u*S] = x0;
        buf[gs + k + 1u*S] = x1;
        buf[gs + k + 2u*S] = x2;
        buf[gs + k + 3u*S] = x3;
        buf[gs + k + 4u*S] = x4;
        buf[gs + k + 5u*S] = x5;
        buf[gs + k + 6u*S] = x6;
        buf[gs + k + 7u*S] = x7;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 8; i++) {
        uint idx = tid + i * 512u;
        uint rev_idx = digit_reverse_base8_4(idx);
        output[base_offset + rev_idx] = buf[idx];
    }
}

// ============================================================================
// Hybrid: split layout + scalar butterfly (test split layout correctness)
// ============================================================================
kernel void fft_4096_ct_split_scalar(
    device const float2* input    [[buffer(0)]],
    device float2*       output   [[buffer(1)]],
    device const float*  dft_real [[buffer(2)]],
    device const float*  dft_imag [[buffer(3)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    const uint N = N_FFT;
    const uint base_offset = tg_id * N;
    threadgroup float buf_real[4096];
    threadgroup float buf_imag[4096];

    for (uint i = 0; i < 8; i++) {
        uint idx = tid + i * 512u;
        float2 val = input[base_offset + idx];
        buf_real[idx] = val.x;
        buf_imag[idx] = val.y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint strides[4] = {512u, 64u, 8u, 1u};
    uint gsizes[4]  = {4096u, 512u, 64u, 8u};

    for (uint stage = 0; stage < 4; stage++) {
        uint S = strides[stage];
        uint G = gsizes[stage];
        uint btfl = tid;
        uint group_idx = btfl / S;
        uint k = btfl % S;
        uint gs = group_idx * G;

        float2 x0 = float2(buf_real[gs+k+0u*S], buf_imag[gs+k+0u*S]);
        float2 x1 = float2(buf_real[gs+k+1u*S], buf_imag[gs+k+1u*S]);
        float2 x2 = float2(buf_real[gs+k+2u*S], buf_imag[gs+k+2u*S]);
        float2 x3 = float2(buf_real[gs+k+3u*S], buf_imag[gs+k+3u*S]);
        float2 x4 = float2(buf_real[gs+k+4u*S], buf_imag[gs+k+4u*S]);
        float2 x5 = float2(buf_real[gs+k+5u*S], buf_imag[gs+k+5u*S]);
        float2 x6 = float2(buf_real[gs+k+6u*S], buf_imag[gs+k+6u*S]);
        float2 x7 = float2(buf_real[gs+k+7u*S], buf_imag[gs+k+7u*S]);

        radix8_butterfly(x0, x1, x2, x3, x4, x5, x6, x7);

        if (k > 0) {
            float base_angle = -2.0f * M_PI_F * float(k) / float(G);
            float s1, c1;
            s1 = sincos(base_angle, c1);
            apply_twiddle8(x1, x2, x3, x4, x5, x6, x7, float2(c1, s1));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf_real[gs+k+0u*S] = x0.x; buf_imag[gs+k+0u*S] = x0.y;
        buf_real[gs+k+1u*S] = x1.x; buf_imag[gs+k+1u*S] = x1.y;
        buf_real[gs+k+2u*S] = x2.x; buf_imag[gs+k+2u*S] = x2.y;
        buf_real[gs+k+3u*S] = x3.x; buf_imag[gs+k+3u*S] = x3.y;
        buf_real[gs+k+4u*S] = x4.x; buf_imag[gs+k+4u*S] = x4.y;
        buf_real[gs+k+5u*S] = x5.x; buf_imag[gs+k+5u*S] = x5.y;
        buf_real[gs+k+6u*S] = x6.x; buf_imag[gs+k+6u*S] = x6.y;
        buf_real[gs+k+7u*S] = x7.x; buf_imag[gs+k+7u*S] = x7.y;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 8; i++) {
        uint idx = tid + i * 512u;
        uint rev_idx = digit_reverse_base8_4(idx);
        output[base_offset + rev_idx] = float2(buf_real[idx], buf_imag[idx]);
    }
}

