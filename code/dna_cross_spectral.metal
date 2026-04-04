#include <metal_stdlib>
using namespace metal;

// ============================================================================
// DNA Cross-Spectral Analysis Kernel
//
// Takes 4-channel FFT output (U_A, U_T, U_G, U_C) and computes:
// 1. Power spectrum per channel: P_X(k) = |U_X(k)|^2
// 2. Total power: P_total(k) = sum of all channels
// 3. Cross-spectral matrix: S_XY(k) = U_X(k) * conj(U_Y(k))
// 4. Coherence: gamma^2_XY(k) = |S_XY|^2 / (P_X * P_Y)
// 5. Period-3 score at k = N/3
//
// Input:  4*N complex values [U_A | U_T | U_G | U_C] from dna_fft_1024
// Output: Power spectra, cross-spectra, coherence values
// ============================================================================

inline float2 cmul_conj(float2 a, float2 b) {
    // a * conj(b) = (ar*br + ai*bi, ai*br - ar*bi)
    return float2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}

inline float mag_sq(float2 c) {
    return c.x * c.x + c.y * c.y;
}

// Output layout per frequency k:
// power_out[4*k + 0..3] = P_A, P_T, P_G, P_C
// cross_out[6*k + 0..5] = S_AT, S_AG, S_AC, S_TG, S_TC, S_GC (upper triangle, complex)
// coherence_out[6*k + 0..5] = gamma^2_AT, gamma^2_AG, gamma^2_AC, gamma^2_TG, gamma^2_TC, gamma^2_GC

kernel void dna_cross_spectral(
    device const float2* spectra       [[buffer(0)]],   // 4*N complex from FFT
    device float*        power_out     [[buffer(1)]],   // 4*N floats (power per channel)
    device float*        total_power   [[buffer(2)]],   // N floats
    device float2*       cross_out     [[buffer(3)]],   // 6*N complex (upper triangle)
    device float*        coherence_out [[buffer(4)]],   // 6*N floats
    device const uint*   params        [[buffer(5)]],   // params[0] = N
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    uint N = params[0];
    uint k = tg_id * 256u + tid;  // frequency index
    if (k >= N) return;

    // Read the 4 spectra at frequency k
    // Input layout per batch: [U_A(0..N-1) | U_T(0..N-1) | U_G(0..N-1) | U_C(0..N-1)]
    float2 ua = spectra[k];
    float2 ut = spectra[N + k];
    float2 ug = spectra[2u * N + k];
    float2 uc = spectra[3u * N + k];

    // Power spectra
    float pa = mag_sq(ua);
    float pt = mag_sq(ut);
    float pg = mag_sq(ug);
    float pc = mag_sq(uc);
    float ptotal = pa + pt + pg + pc;

    power_out[4u * k]     = pa;
    power_out[4u * k + 1] = pt;
    power_out[4u * k + 2] = pg;
    power_out[4u * k + 3] = pc;
    total_power[k] = ptotal;

    // Cross-spectra (upper triangle of 4x4 Hermitian matrix)
    // S_XY(k) = U_X(k) * conj(U_Y(k))
    float2 s_at = cmul_conj(ua, ut);
    float2 s_ag = cmul_conj(ua, ug);
    float2 s_ac = cmul_conj(ua, uc);
    float2 s_tg = cmul_conj(ut, ug);
    float2 s_tc = cmul_conj(ut, uc);
    float2 s_gc = cmul_conj(ug, uc);

    cross_out[6u * k]     = s_at;
    cross_out[6u * k + 1] = s_ag;
    cross_out[6u * k + 2] = s_ac;
    cross_out[6u * k + 3] = s_tg;
    cross_out[6u * k + 4] = s_tc;
    cross_out[6u * k + 5] = s_gc;

    // Coherence: gamma^2_XY = |S_XY|^2 / (P_X * P_Y)
    // Guard against division by zero
    float eps = 1e-30f;
    coherence_out[6u * k]     = mag_sq(s_at) / max(pa * pt, eps);
    coherence_out[6u * k + 1] = mag_sq(s_ag) / max(pa * pg, eps);
    coherence_out[6u * k + 2] = mag_sq(s_ac) / max(pa * pc, eps);
    coherence_out[6u * k + 3] = mag_sq(s_tg) / max(pt * pg, eps);
    coherence_out[6u * k + 4] = mag_sq(s_tc) / max(pt * pc, eps);
    coherence_out[6u * k + 5] = mag_sq(s_gc) / max(pg * pc, eps);
}

// ============================================================================
// Period-3 Detector
//
// For a window at position p, extract the spectral power at k = N/3.
// This is the signature of coding regions (triplet periodicity).
// Also extracts the ratio P(N/3) / mean(P) as a normalized score.
// ============================================================================

kernel void dna_period3_detect(
    device const float* total_power  [[buffer(0)]],   // N floats per window
    device float*       period3_out  [[buffer(1)]],   // 2 floats per window: [raw_power, normalized_score]
    device const uint*  params       [[buffer(2)]],   // params[0] = N, params[1] = num_windows
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    uint N = params[0];
    uint num_windows = params[1];
    uint window_idx = tg_id * 256u + tid;
    if (window_idx >= num_windows) return;

    uint base = window_idx * N;
    uint k3 = N / 3u;

    float p3 = total_power[base + k3];

    // Compute mean power (exclude DC and Nyquist)
    float sum_power = 0.0f;
    for (uint k = 1u; k < N / 2u; k++) {
        sum_power += total_power[base + k];
    }
    float mean_power = sum_power / float(N / 2u - 1u);

    period3_out[2u * window_idx]     = p3;
    period3_out[2u * window_idx + 1] = p3 / max(mean_power, 1e-30f);
}
