// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Generic N-channel cross-spectral analysis kernel
//
// Input layout:  spectra[C * N] — C channels, each N complex values, channels
//                concatenated: [ch0(0..N-1) | ch1(0..N-1) | ... | ch(C-1)(0..N-1)]
// Output layout:
//   power_out[C * N]     — real power per channel per bin
//   cross_out[P * N]     — complex cross-spectra, upper triangle only, P = C*(C-1)/2
//                          pair ordering: (0,1),(0,2),...,(0,C-1),(1,2),...,(C-2,C-1)
//   coh_out[P * N]       — magnitude-squared coherence [0, 1] per pair per bin
//
// params[0] = N (FFT size)
// params[1] = C (channel count, max 16)
//
// Dispatch: ceil(N / 256) threadgroups × 256 threads; each thread handles one bin.
// ============================================================================

inline float mag_sq_cs(float2 c) { return c.x * c.x + c.y * c.y; }

kernel void fft_cross_spectral(
    device const float2* spectra   [[buffer(0)]],
    device float*        power_out [[buffer(1)]],
    device float2*       cross_out [[buffer(2)]],
    device float*        coh_out   [[buffer(3)]],
    device const uint*   params    [[buffer(4)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    const uint N = params[0];
    const uint C = min(params[1], 16u);
    const uint k = tg_id * 256u + tid;
    if (k >= N) return;

    // Load all channel spectra at bin k into registers (max 16 channels)
    float2 u[16];
    float  p[16];
    for (uint c = 0; c < C; c++) {
        u[c] = spectra[c * N + k];
        p[c] = mag_sq_cs(u[c]);
        power_out[c * N + k] = p[c];
    }

    // Upper-triangle cross-spectra and coherence
    uint pair = 0;
    for (uint i = 0; i < C; i++) {
        for (uint j = i + 1u; j < C; j++) {
            // S_ij = u[i] * conj(u[j])
            float2 s = float2(u[i].x * u[j].x + u[i].y * u[j].y,
                              u[i].y * u[j].x - u[i].x * u[j].y);
            cross_out[pair * N + k] = s;
            coh_out[pair * N + k]   = mag_sq_cs(s) / max(p[i] * p[j], 1e-30f);
            pair++;
        }
    }
}
