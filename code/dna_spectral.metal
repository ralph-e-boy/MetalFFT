#include <metal_stdlib>
using namespace metal;

// ============================================================================
// DNA 4-Channel Spectral Analysis — Metal Kernel
//
// Encodes a DNA sequence as 4 binary indicator channels (A, T, G, C) and
// computes the FFT of each channel using radix-4 Stockham decomposition.
//
// Input:  uint8 buffer where 0=A, 1=T, 2=G, 3=C
// Output: 4 complex spectra (U_A, U_T, U_G, U_C), interleaved as
//         [U_A(0..N-1), U_T(0..N-1), U_G(0..N-1), U_C(0..N-1)]
//
// Supports N = 256, 1024, 4096 via compile-time constants.
// For longer sequences, the host dispatches windowed batches.
// ============================================================================

// --- Shared FFT helpers (same as fft_stockham_4096.metal) ---

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
// Generic Stockham FFT pass — reads/writes threadgroup buffer
// stride: distance between butterfly elements in the current pass
// tw_factor: N / (4 * stride) — used to compute twiddle angles
// ============================================================================

inline void stockham_pass(threadgroup float2* buf, uint tid, uint T,
                          uint stride, float two_pi_over_n) {
    uint pos = tid & (stride - 1u);
    uint grp = tid >> __builtin_ctz(stride);
    float2 x0 = buf[tid];
    float2 x1 = buf[tid + T];
    float2 x2 = buf[tid + 2u * T];
    float2 x3 = buf[tid + 3u * T];
    if (stride > 1u) {
        uint tw_factor = T / stride;  // T = N/4, so tw_factor = N/(4*stride)
        float a1 = two_pi_over_n * float(pos * tw_factor);
        float s1, c1;
        s1 = sincos(a1, c1);
        apply_twiddle3(x1, x2, x3, float2(c1, s1));
    }
    radix4(x0, x1, x2, x3);
    uint wr = grp * 4u * stride + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr]              = x0;
    buf[wr + stride]     = x1;
    buf[wr + 2u * stride] = x2;
    buf[wr + 3u * stride] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ============================================================================
// N=1024 4-channel DNA FFT
// 5 radix-4 passes, 256 threads per channel, processes 4 channels per TG
// ============================================================================

kernel void dna_fft_1024(
    device const uchar*  dna_input  [[buffer(0)]],
    device float2*       spectra    [[buffer(1)]],
    device const uint*   params     [[buffer(2)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    const uint N = 1024;
    const uint T = 256;  // threads = N/4
    const float TWO_PI_OVER_N = -2.0f * M_PI_F / float(N);

    // params[0] = sequence_offset (base position in DNA sequence)
    uint seq_offset = params[0] + tg_id * N;

    // Threadgroup memory: 4 channels x N complex values
    threadgroup float2 buf_A[N];
    threadgroup float2 buf_T[N];
    threadgroup float2 buf_G[N];
    threadgroup float2 buf_C[N];

    // --- Encode DNA to 4 binary indicator channels ---
    // Each thread encodes 4 positions (N/T = 4)
    for (uint i = 0; i < 4u; i++) {
        uint pos = tid + i * T;
        uchar base = dna_input[seq_offset + pos];
        // Real-valued: imaginary = 0
        buf_A[pos] = float2(base == 0u ? 1.0f : 0.0f, 0.0f);
        buf_T[pos] = float2(base == 1u ? 1.0f : 0.0f, 0.0f);
        buf_G[pos] = float2(base == 2u ? 1.0f : 0.0f, 0.0f);
        buf_C[pos] = float2(base == 3u ? 1.0f : 0.0f, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Pass 0: stride=1, no twiddles ---
    // Channel A
    {
        float2 x0 = buf_A[tid];
        float2 x1 = buf_A[tid + T];
        float2 x2 = buf_A[tid + 2u * T];
        float2 x3 = buf_A[tid + 3u * T];
        radix4(x0, x1, x2, x3);
        uint wr = tid << 2;
        buf_A[wr]     = x0;
        buf_A[wr + 1] = x1;
        buf_A[wr + 2] = x2;
        buf_A[wr + 3] = x3;
    }
    // Channel T
    {
        float2 x0 = buf_T[tid];
        float2 x1 = buf_T[tid + T];
        float2 x2 = buf_T[tid + 2u * T];
        float2 x3 = buf_T[tid + 3u * T];
        radix4(x0, x1, x2, x3);
        uint wr = tid << 2;
        buf_T[wr]     = x0;
        buf_T[wr + 1] = x1;
        buf_T[wr + 2] = x2;
        buf_T[wr + 3] = x3;
    }
    // Channel G
    {
        float2 x0 = buf_G[tid];
        float2 x1 = buf_G[tid + T];
        float2 x2 = buf_G[tid + 2u * T];
        float2 x3 = buf_G[tid + 3u * T];
        radix4(x0, x1, x2, x3);
        uint wr = tid << 2;
        buf_G[wr]     = x0;
        buf_G[wr + 1] = x1;
        buf_G[wr + 2] = x2;
        buf_G[wr + 3] = x3;
    }
    // Channel C
    {
        float2 x0 = buf_C[tid];
        float2 x1 = buf_C[tid + T];
        float2 x2 = buf_C[tid + 2u * T];
        float2 x3 = buf_C[tid + 3u * T];
        radix4(x0, x1, x2, x3);
        uint wr = tid << 2;
        buf_C[wr]     = x0;
        buf_C[wr + 1] = x1;
        buf_C[wr + 2] = x2;
        buf_C[wr + 3] = x3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Passes 1-3: stride = 4, 16, 64 ---
    for (uint p = 0; p < 3u; p++) {
        uint stride = (p == 0u) ? 4u : ((p == 1u) ? 16u : 64u);
        uint pos = tid & (stride - 1u);
        uint grp = tid >> __builtin_ctz(stride);
        uint tw_factor = T / stride;
        float a1 = TWO_PI_OVER_N * float(pos * tw_factor);
        float s1, c1;
        s1 = sincos(a1, c1);
        float2 w1 = float2(c1, s1);

        // Channel A
        {
            float2 x0 = buf_A[tid];
            float2 x1 = buf_A[tid + T];
            float2 x2 = buf_A[tid + 2u * T];
            float2 x3 = buf_A[tid + 3u * T];
            apply_twiddle3(x1, x2, x3, w1);
            radix4(x0, x1, x2, x3);
            uint wr = grp * 4u * stride + pos;
            buf_A[wr]              = x0;
            buf_A[wr + stride]     = x1;
            buf_A[wr + 2u * stride] = x2;
            buf_A[wr + 3u * stride] = x3;
        }
        // Channel T
        {
            float2 x0 = buf_T[tid];
            float2 x1 = buf_T[tid + T];
            float2 x2 = buf_T[tid + 2u * T];
            float2 x3 = buf_T[tid + 3u * T];
            apply_twiddle3(x1, x2, x3, w1);
            radix4(x0, x1, x2, x3);
            uint wr = grp * 4u * stride + pos;
            buf_T[wr]              = x0;
            buf_T[wr + stride]     = x1;
            buf_T[wr + 2u * stride] = x2;
            buf_T[wr + 3u * stride] = x3;
        }
        // Channel G
        {
            float2 x0 = buf_G[tid];
            float2 x1 = buf_G[tid + T];
            float2 x2 = buf_G[tid + 2u * T];
            float2 x3 = buf_G[tid + 3u * T];
            apply_twiddle3(x1, x2, x3, w1);
            radix4(x0, x1, x2, x3);
            uint wr = grp * 4u * stride + pos;
            buf_G[wr]              = x0;
            buf_G[wr + stride]     = x1;
            buf_G[wr + 2u * stride] = x2;
            buf_G[wr + 3u * stride] = x3;
        }
        // Channel C
        {
            float2 x0 = buf_C[tid];
            float2 x1 = buf_C[tid + T];
            float2 x2 = buf_C[tid + 2u * T];
            float2 x3 = buf_C[tid + 3u * T];
            apply_twiddle3(x1, x2, x3, w1);
            radix4(x0, x1, x2, x3);
            uint wr = grp * 4u * stride + pos;
            buf_C[wr]              = x0;
            buf_C[wr + stride]     = x1;
            buf_C[wr + 2u * stride] = x2;
            buf_C[wr + 3u * stride] = x3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Pass 4 (final): stride=256, write to device memory ---
    {
        float2 x0_a = buf_A[tid]; float2 x1_a = buf_A[tid + T];
        float2 x2_a = buf_A[tid + 2u * T]; float2 x3_a = buf_A[tid + 3u * T];
        float2 x0_t = buf_T[tid]; float2 x1_t = buf_T[tid + T];
        float2 x2_t = buf_T[tid + 2u * T]; float2 x3_t = buf_T[tid + 3u * T];
        float2 x0_g = buf_G[tid]; float2 x1_g = buf_G[tid + T];
        float2 x2_g = buf_G[tid + 2u * T]; float2 x3_g = buf_G[tid + 3u * T];
        float2 x0_c = buf_C[tid]; float2 x1_c = buf_C[tid + T];
        float2 x2_c = buf_C[tid + 2u * T]; float2 x3_c = buf_C[tid + 3u * T];

        float a1 = TWO_PI_OVER_N * float(tid);
        float s1, c1;
        s1 = sincos(a1, c1);
        float2 w1 = float2(c1, s1);

        apply_twiddle3(x1_a, x2_a, x3_a, w1);
        radix4(x0_a, x1_a, x2_a, x3_a);
        apply_twiddle3(x1_t, x2_t, x3_t, w1);
        radix4(x0_t, x1_t, x2_t, x3_t);
        apply_twiddle3(x1_g, x2_g, x3_g, w1);
        radix4(x0_g, x1_g, x2_g, x3_g);
        apply_twiddle3(x1_c, x2_c, x3_c, w1);
        radix4(x0_c, x1_c, x2_c, x3_c);

        // Output layout: [A spectrum | T spectrum | G spectrum | C spectrum]
        // Each spectrum is N complex values, batch indexed by tg_id
        uint out_base = tg_id * 4u * N;
        spectra[out_base + tid]            = x0_a;
        spectra[out_base + tid + T]        = x1_a;
        spectra[out_base + tid + 2u * T]   = x2_a;
        spectra[out_base + tid + 3u * T]   = x3_a;

        spectra[out_base + N + tid]          = x0_t;
        spectra[out_base + N + tid + T]      = x1_t;
        spectra[out_base + N + tid + 2u * T] = x2_t;
        spectra[out_base + N + tid + 3u * T] = x3_t;

        spectra[out_base + 2u * N + tid]          = x0_g;
        spectra[out_base + 2u * N + tid + T]      = x1_g;
        spectra[out_base + 2u * N + tid + 2u * T] = x2_g;
        spectra[out_base + 2u * N + tid + 3u * T] = x3_g;

        spectra[out_base + 3u * N + tid]          = x0_c;
        spectra[out_base + 3u * N + tid + T]      = x1_c;
        spectra[out_base + 3u * N + tid + 2u * T] = x2_c;
        spectra[out_base + 3u * N + tid + 3u * T] = x3_c;
    }
}
