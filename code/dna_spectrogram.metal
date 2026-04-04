#include <metal_stdlib>
using namespace metal;

// ============================================================================
// DNA Spectrogram — Sliding-Window Short-Time Fourier Transform
//
// Computes a spectrogram of a DNA sequence by sliding a window along the
// sequence and computing the 4-channel FFT + total power at each position.
//
// Output: 2D spectrogram (position x frequency) where each entry is the
// total spectral power across all 4 channels.
//
// The host dispatches one threadgroup per window position. Each threadgroup:
// 1. Loads the windowed DNA segment
// 2. Applies a Hann window (for spectral leakage reduction)
// 3. Computes the 4-channel FFT using radix-4 Stockham
// 4. Writes total power spectrum for that window
//
// Window size: 1024 (hardcoded for this kernel, matching dna_fft_1024)
// Hop size: controlled by the host dispatch
// ============================================================================

constant uint SPEC_N = 1024;
constant uint SPEC_T = 256;  // threads = N/4
constant float SPEC_TWO_PI_OVER_N = -2.0f * M_PI_F / float(SPEC_N);

// --- FFT helpers (duplicated to keep this compilation unit standalone) ---

inline float2 cmul_s(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline void radix4_s(thread float2& x0, thread float2& x1,
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

inline void apply_twiddle3_s(thread float2& x1, thread float2& x2, thread float2& x3,
                             float2 w1) {
    float2 w2 = cmul_s(w1, w1);
    float2 w3 = cmul_s(w2, w1);
    x1 = cmul_s(x1, w1);
    x2 = cmul_s(x2, w2);
    x3 = cmul_s(x3, w3);
}

// Hann window: w(n) = 0.5 * (1 - cos(2*pi*n/(N-1)))
inline float hann_window(uint n, uint N) {
    return 0.5f * (1.0f - cos(2.0f * M_PI_F * float(n) / float(N - 1u)));
}

kernel void dna_spectrogram_1024(
    device const uchar*  dna_input     [[buffer(0)]],   // full DNA sequence
    device float*        spectrogram   [[buffer(1)]],   // output: num_windows * (N/2+1) floats
    device const uint*   params        [[buffer(2)]],   // [seq_length, hop_size, num_windows]
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    uint hop_size = params[1];
    uint window_start = tg_id * hop_size;

    // Threadgroup buffers for 4 channels
    threadgroup float2 buf_A[SPEC_N];
    threadgroup float2 buf_T[SPEC_N];
    threadgroup float2 buf_G[SPEC_N];
    threadgroup float2 buf_C[SPEC_N];

    // --- Encode + apply Hann window ---
    for (uint i = 0; i < 4u; i++) {
        uint pos = tid + i * SPEC_T;
        uchar base = dna_input[window_start + pos];
        float w = hann_window(pos, SPEC_N);
        buf_A[pos] = float2(base == 0u ? w : 0.0f, 0.0f);
        buf_T[pos] = float2(base == 1u ? w : 0.0f, 0.0f);
        buf_G[pos] = float2(base == 2u ? w : 0.0f, 0.0f);
        buf_C[pos] = float2(base == 3u ? w : 0.0f, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- 5 radix-4 Stockham passes for all 4 channels ---
    // Pass 0: stride=1, no twiddles
    {
        uint wr = tid << 2;
        // Channel A
        float2 a0 = buf_A[tid], a1 = buf_A[tid+SPEC_T], a2 = buf_A[tid+2*SPEC_T], a3 = buf_A[tid+3*SPEC_T];
        radix4_s(a0, a1, a2, a3);
        buf_A[wr]=a0; buf_A[wr+1]=a1; buf_A[wr+2]=a2; buf_A[wr+3]=a3;
        // Channel T
        float2 t0 = buf_T[tid], t1 = buf_T[tid+SPEC_T], t2 = buf_T[tid+2*SPEC_T], t3 = buf_T[tid+3*SPEC_T];
        radix4_s(t0, t1, t2, t3);
        buf_T[wr]=t0; buf_T[wr+1]=t1; buf_T[wr+2]=t2; buf_T[wr+3]=t3;
        // Channel G
        float2 g0 = buf_G[tid], g1 = buf_G[tid+SPEC_T], g2 = buf_G[tid+2*SPEC_T], g3 = buf_G[tid+3*SPEC_T];
        radix4_s(g0, g1, g2, g3);
        buf_G[wr]=g0; buf_G[wr+1]=g1; buf_G[wr+2]=g2; buf_G[wr+3]=g3;
        // Channel C
        float2 c0 = buf_C[tid], c1 = buf_C[tid+SPEC_T], c2 = buf_C[tid+2*SPEC_T], c3 = buf_C[tid+3*SPEC_T];
        radix4_s(c0, c1, c2, c3);
        buf_C[wr]=c0; buf_C[wr+1]=c1; buf_C[wr+2]=c2; buf_C[wr+3]=c3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Passes 1-3: stride = 4, 16, 64
    for (uint p = 0; p < 3u; p++) {
        uint stride = (p == 0u) ? 4u : ((p == 1u) ? 16u : 64u);
        uint pos = tid & (stride - 1u);
        uint grp = tid >> __builtin_ctz(stride);
        uint tw_factor = SPEC_T / stride;
        float a1 = SPEC_TWO_PI_OVER_N * float(pos * tw_factor);
        float s1, c1;
        s1 = sincos(a1, c1);
        float2 w1 = float2(c1, s1);
        uint wr = grp * 4u * stride + pos;

        // Channel A
        {
            float2 x0=buf_A[tid], x1=buf_A[tid+SPEC_T], x2=buf_A[tid+2*SPEC_T], x3=buf_A[tid+3*SPEC_T];
            apply_twiddle3_s(x1, x2, x3, w1); radix4_s(x0, x1, x2, x3);
            buf_A[wr]=x0; buf_A[wr+stride]=x1; buf_A[wr+2*stride]=x2; buf_A[wr+3*stride]=x3;
        }
        // Channel T
        {
            float2 x0=buf_T[tid], x1=buf_T[tid+SPEC_T], x2=buf_T[tid+2*SPEC_T], x3=buf_T[tid+3*SPEC_T];
            apply_twiddle3_s(x1, x2, x3, w1); radix4_s(x0, x1, x2, x3);
            buf_T[wr]=x0; buf_T[wr+stride]=x1; buf_T[wr+2*stride]=x2; buf_T[wr+3*stride]=x3;
        }
        // Channel G
        {
            float2 x0=buf_G[tid], x1=buf_G[tid+SPEC_T], x2=buf_G[tid+2*SPEC_T], x3=buf_G[tid+3*SPEC_T];
            apply_twiddle3_s(x1, x2, x3, w1); radix4_s(x0, x1, x2, x3);
            buf_G[wr]=x0; buf_G[wr+stride]=x1; buf_G[wr+2*stride]=x2; buf_G[wr+3*stride]=x3;
        }
        // Channel C
        {
            float2 x0=buf_C[tid], x1=buf_C[tid+SPEC_T], x2=buf_C[tid+2*SPEC_T], x3=buf_C[tid+3*SPEC_T];
            apply_twiddle3_s(x1, x2, x3, w1); radix4_s(x0, x1, x2, x3);
            buf_C[wr]=x0; buf_C[wr+stride]=x1; buf_C[wr+2*stride]=x2; buf_C[wr+3*stride]=x3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 4 (final): stride=256, compute power and write to spectrogram
    {
        float2 a0=buf_A[tid], a1=buf_A[tid+SPEC_T], a2=buf_A[tid+2*SPEC_T], a3=buf_A[tid+3*SPEC_T];
        float2 t0=buf_T[tid], t1=buf_T[tid+SPEC_T], t2=buf_T[tid+2*SPEC_T], t3=buf_T[tid+3*SPEC_T];
        float2 g0=buf_G[tid], g1=buf_G[tid+SPEC_T], g2=buf_G[tid+2*SPEC_T], g3=buf_G[tid+3*SPEC_T];
        float2 c0=buf_C[tid], c1=buf_C[tid+SPEC_T], c2=buf_C[tid+2*SPEC_T], c3=buf_C[tid+3*SPEC_T];

        float ang = SPEC_TWO_PI_OVER_N * float(tid);
        float s1, c1_tw;
        s1 = sincos(ang, c1_tw);
        float2 w1 = float2(c1_tw, s1);

        apply_twiddle3_s(a1, a2, a3, w1); radix4_s(a0, a1, a2, a3);
        apply_twiddle3_s(t1, t2, t3, w1); radix4_s(t0, t1, t2, t3);
        apply_twiddle3_s(g1, g2, g3, w1); radix4_s(g0, g1, g2, g3);
        apply_twiddle3_s(c1, c2, c3, w1); radix4_s(c0, c1, c2, c3);

        // Compute total power |U_A|^2 + |U_T|^2 + |U_G|^2 + |U_C|^2
        // and write to spectrogram. Only write N/2+1 unique frequencies.
        uint out_stride = SPEC_N / 2u + 1u;  // 513 frequencies per window
        uint out_base = tg_id * out_stride;

        // Each thread writes 4 frequency bins
        float2 all_a[4] = {a0, a1, a2, a3};
        float2 all_t[4] = {t0, t1, t2, t3};
        float2 all_g[4] = {g0, g1, g2, g3};
        float2 all_c[4] = {c0, c1, c2, c3};

        for (uint i = 0; i < 4u; i++) {
            uint freq = tid + i * SPEC_T;
            if (freq <= SPEC_N / 2u) {
                float power = all_a[i].x * all_a[i].x + all_a[i].y * all_a[i].y
                            + all_t[i].x * all_t[i].x + all_t[i].y * all_t[i].y
                            + all_g[i].x * all_g[i].x + all_g[i].y * all_g[i].y
                            + all_c[i].x * all_c[i].x + all_c[i].y * all_c[i].y;
                spectrogram[out_base + freq] = power;
            }
        }
    }
}

// ============================================================================
// Per-channel spectrogram variant — outputs 4 separate power channels
// Output: num_windows * (N/2+1) * 4 floats [P_A, P_T, P_G, P_C] per bin
// ============================================================================

kernel void dna_spectrogram_4ch_1024(
    device const uchar*  dna_input     [[buffer(0)]],
    device float*        spectrogram   [[buffer(1)]],   // num_windows * (N/2+1) * 4
    device const uint*   params        [[buffer(2)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    uint hop_size = params[1];
    uint window_start = tg_id * hop_size;

    threadgroup float2 buf_A[SPEC_N];
    threadgroup float2 buf_T[SPEC_N];
    threadgroup float2 buf_G[SPEC_N];
    threadgroup float2 buf_C[SPEC_N];

    // Encode + Hann window
    for (uint i = 0; i < 4u; i++) {
        uint pos = tid + i * SPEC_T;
        uchar base = dna_input[window_start + pos];
        float w = 0.5f * (1.0f - cos(2.0f * M_PI_F * float(pos) / float(SPEC_N - 1u)));
        buf_A[pos] = float2(base == 0u ? w : 0.0f, 0.0f);
        buf_T[pos] = float2(base == 1u ? w : 0.0f, 0.0f);
        buf_G[pos] = float2(base == 2u ? w : 0.0f, 0.0f);
        buf_C[pos] = float2(base == 3u ? w : 0.0f, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 0: no twiddles
    {
        uint wr = tid << 2;
        float2 a0=buf_A[tid],a1=buf_A[tid+SPEC_T],a2=buf_A[tid+2*SPEC_T],a3=buf_A[tid+3*SPEC_T];
        radix4_s(a0,a1,a2,a3); buf_A[wr]=a0;buf_A[wr+1]=a1;buf_A[wr+2]=a2;buf_A[wr+3]=a3;
        float2 t0=buf_T[tid],t1=buf_T[tid+SPEC_T],t2=buf_T[tid+2*SPEC_T],t3=buf_T[tid+3*SPEC_T];
        radix4_s(t0,t1,t2,t3); buf_T[wr]=t0;buf_T[wr+1]=t1;buf_T[wr+2]=t2;buf_T[wr+3]=t3;
        float2 g0=buf_G[tid],g1=buf_G[tid+SPEC_T],g2=buf_G[tid+2*SPEC_T],g3=buf_G[tid+3*SPEC_T];
        radix4_s(g0,g1,g2,g3); buf_G[wr]=g0;buf_G[wr+1]=g1;buf_G[wr+2]=g2;buf_G[wr+3]=g3;
        float2 c0=buf_C[tid],c1=buf_C[tid+SPEC_T],c2=buf_C[tid+2*SPEC_T],c3=buf_C[tid+3*SPEC_T];
        radix4_s(c0,c1,c2,c3); buf_C[wr]=c0;buf_C[wr+1]=c1;buf_C[wr+2]=c2;buf_C[wr+3]=c3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Passes 1-3
    for (uint p = 0; p < 3u; p++) {
        uint stride = (p == 0u) ? 4u : ((p == 1u) ? 16u : 64u);
        uint pos = tid & (stride - 1u);
        uint grp = tid >> __builtin_ctz(stride);
        uint tw_f = SPEC_T / stride;
        float a1 = SPEC_TWO_PI_OVER_N * float(pos * tw_f);
        float s1, c1; s1 = sincos(a1, c1);
        float2 w1 = float2(c1, s1);
        uint wr = grp * 4u * stride + pos;

        {float2 x0=buf_A[tid],x1=buf_A[tid+SPEC_T],x2=buf_A[tid+2*SPEC_T],x3=buf_A[tid+3*SPEC_T];
         apply_twiddle3_s(x1,x2,x3,w1);radix4_s(x0,x1,x2,x3);
         buf_A[wr]=x0;buf_A[wr+stride]=x1;buf_A[wr+2*stride]=x2;buf_A[wr+3*stride]=x3;}
        {float2 x0=buf_T[tid],x1=buf_T[tid+SPEC_T],x2=buf_T[tid+2*SPEC_T],x3=buf_T[tid+3*SPEC_T];
         apply_twiddle3_s(x1,x2,x3,w1);radix4_s(x0,x1,x2,x3);
         buf_T[wr]=x0;buf_T[wr+stride]=x1;buf_T[wr+2*stride]=x2;buf_T[wr+3*stride]=x3;}
        {float2 x0=buf_G[tid],x1=buf_G[tid+SPEC_T],x2=buf_G[tid+2*SPEC_T],x3=buf_G[tid+3*SPEC_T];
         apply_twiddle3_s(x1,x2,x3,w1);radix4_s(x0,x1,x2,x3);
         buf_G[wr]=x0;buf_G[wr+stride]=x1;buf_G[wr+2*stride]=x2;buf_G[wr+3*stride]=x3;}
        {float2 x0=buf_C[tid],x1=buf_C[tid+SPEC_T],x2=buf_C[tid+2*SPEC_T],x3=buf_C[tid+3*SPEC_T];
         apply_twiddle3_s(x1,x2,x3,w1);radix4_s(x0,x1,x2,x3);
         buf_C[wr]=x0;buf_C[wr+stride]=x1;buf_C[wr+2*stride]=x2;buf_C[wr+3*stride]=x3;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 4 (final)
    {
        float2 a0=buf_A[tid],a1=buf_A[tid+SPEC_T],a2=buf_A[tid+2*SPEC_T],a3=buf_A[tid+3*SPEC_T];
        float2 t0=buf_T[tid],t1=buf_T[tid+SPEC_T],t2=buf_T[tid+2*SPEC_T],t3=buf_T[tid+3*SPEC_T];
        float2 g0=buf_G[tid],g1=buf_G[tid+SPEC_T],g2=buf_G[tid+2*SPEC_T],g3=buf_G[tid+3*SPEC_T];
        float2 c0=buf_C[tid],c1=buf_C[tid+SPEC_T],c2=buf_C[tid+2*SPEC_T],c3=buf_C[tid+3*SPEC_T];
        float ang = SPEC_TWO_PI_OVER_N * float(tid);
        float s1, c1_tw; s1 = sincos(ang, c1_tw);
        float2 w1 = float2(c1_tw, s1);
        apply_twiddle3_s(a1,a2,a3,w1);radix4_s(a0,a1,a2,a3);
        apply_twiddle3_s(t1,t2,t3,w1);radix4_s(t0,t1,t2,t3);
        apply_twiddle3_s(g1,g2,g3,w1);radix4_s(g0,g1,g2,g3);
        apply_twiddle3_s(c1,c2,c3,w1);radix4_s(c0,c1,c2,c3);

        uint out_freqs = SPEC_N / 2u + 1u;
        uint out_base = tg_id * out_freqs * 4u;
        float2 aa[4]={a0,a1,a2,a3}, tt[4]={t0,t1,t2,t3}, gg[4]={g0,g1,g2,g3}, cc[4]={c0,c1,c2,c3};
        for (uint i = 0; i < 4u; i++) {
            uint freq = tid + i * SPEC_T;
            if (freq <= SPEC_N / 2u) {
                uint idx = out_base + freq * 4u;
                spectrogram[idx]     = aa[i].x*aa[i].x + aa[i].y*aa[i].y;
                spectrogram[idx + 1] = tt[i].x*tt[i].x + tt[i].y*tt[i].y;
                spectrogram[idx + 2] = gg[i].x*gg[i].x + gg[i].y*gg[i].y;
                spectrogram[idx + 3] = cc[i].x*cc[i].x + cc[i].y*cc[i].y;
            }
        }
    }
}
