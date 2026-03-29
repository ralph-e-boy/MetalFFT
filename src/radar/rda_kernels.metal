// =============================================================================
// AppleSiliconFFT
// Copyright (c) 2026 Mohamed Amine Bergach <mbergach@illumina.com>
// Licensed under the MIT License. See LICENSE file in the project root.
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// RDA Pipeline Utility Kernels
//
// Complex multiply, transpose, RCMC interpolation, azimuth matched filter
// generation, and other helpers for the Range Doppler Algorithm.
// ============================================================================

// --- Complex multiplication helpers ---

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline float2 cconj(float2 a) {
    return float2(a.x, -a.y);
}

// ============================================================================
// Complex element-wise multiply
// a[i] * b[i] for matching arrays, or a[i] * b[i % rowSize] if b is single row
// ============================================================================

kernel void complex_multiply(
    device const float2* a       [[buffer(0)]],
    device const float2* b       [[buffer(1)]],
    device float2*       output  [[buffer(2)]],
    device const uint*   params  [[buffer(3)]],  // [rowSize, numRows, bIsSingleRow]
    uint gid                     [[thread_position_in_grid]]
) {
    uint rowSize = params[0];
    uint numRows = params[1];
    uint bSingleRow = params[2];
    uint totalCount = rowSize * numRows;

    if (gid >= totalCount) return;

    uint bIdx = bSingleRow ? (gid % rowSize) : gid;
    output[gid] = cmul(a[gid], b[bIdx]);
}

// ============================================================================
// Complex multiply with conjugate of b: a[i] * conj(b[i])
// ============================================================================

kernel void complex_multiply_conjugate(
    device const float2* a       [[buffer(0)]],
    device const float2* b       [[buffer(1)]],
    device float2*       output  [[buffer(2)]],
    device const uint*   params  [[buffer(3)]],  // [rowSize, numRows, bIsSingleRow]
    uint gid                     [[thread_position_in_grid]]
) {
    uint rowSize = params[0];
    uint numRows = params[1];
    uint bSingleRow = params[2];
    uint totalCount = rowSize * numRows;

    if (gid >= totalCount) return;

    uint bIdx = bSingleRow ? (gid % rowSize) : gid;
    output[gid] = cmul(a[gid], cconj(b[bIdx]));
}

// ============================================================================
// 2D Transpose: input[row * cols + col] -> output[col * rows + row]
// ============================================================================

kernel void transpose_2d(
    device const float2* input   [[buffer(0)]],
    device float2*       output  [[buffer(1)]],
    device const uint*   params  [[buffer(2)]],  // [rows, cols]
    uint gid                     [[thread_position_in_grid]]
) {
    uint rows = params[0];
    uint cols = params[1];
    uint totalCount = rows * cols;

    if (gid >= totalCount) return;

    uint row = gid / cols;
    uint col = gid % cols;
    output[col * rows + row] = input[gid];
}

// ============================================================================
// RCMC via sinc interpolation in range-Doppler domain
//
// For each azimuth frequency bin f_a, the range migration is:
//   deltaR(f_a) = R0 * (1/sqrt(1 - (lambda*f_a/(2V))^2) - 1)
//
// We shift each range-Doppler cell by deltaR / rangePixelSpacing using
// 8-point sinc interpolation.
// ============================================================================

kernel void rcmc_sinc_interp(
    device const float2* input   [[buffer(0)]],
    device float2*       output  [[buffer(1)]],
    device const float*  params  [[buffer(2)]],  // [lambda, V, R0, rangePixel, Nr, Na]
    uint gid                     [[thread_position_in_grid]]
) {
    float lambda     = params[0];
    float V          = params[1];
    float R0         = params[2];
    float rangePixel = params[3];
    uint  Nr         = uint(params[4]);
    uint  Na         = uint(params[5]);
    uint  totalCount = Nr * Na;

    if (gid >= totalCount) return;

    uint azIdx = gid / Nr;  // Row index (azimuth frequency bin)
    uint rgIdx = gid % Nr;  // Column index (range bin)

    // Azimuth frequency for this bin
    // Frequency axis: f_a = (azIdx - Na/2) * PRF/Na, but we're already in FFT order
    // so azIdx 0 = DC, azIdx Na/2 = Nyquist
    float fa_norm;
    if (azIdx < Na / 2) {
        fa_norm = float(azIdx) / float(Na);
    } else {
        fa_norm = float(int(azIdx) - int(Na)) / float(Na);
    }
    // fa_norm is in [-0.5, 0.5), actual f_a = fa_norm * PRF
    // But we need lambda * f_a / (2V) = lambda * fa_norm * PRF / (2V)
    // Since PRF = V / azimuthPixelSpacing, this simplifies
    // We compute the ratio directly: lambda * f_a / (2V)

    // PRF is implicit: f_a ranges from -PRF/2 to PRF/2
    // For our FFT output, PRF = Na * (1/Na) = 1 in normalized freq
    // We need the actual PRF which we can derive from V and azimuth spacing
    // But we passed lambda, V directly, so:
    // Actually f_a = fa_norm * PRF (Hz), but PRF not passed. Let's compute ratio differently.
    // ratio = lambda * f_a / (2*V)
    // With f_a in range [-PRF/2, PRF/2] and PRF = 1000 Hz (from params),
    // we can bound this. But PRF not in params, so compute from geometry:
    // The max unambiguous Doppler is PRF, and PRF ≈ 2*V/lambda * sin(beamwidth/2)
    // For simplicity, use ratio = fa_norm * lambda * 1000.0 / (2.0 * V)
    // where 1000 is PRF (hardcoded to match SARParameters)
    float PRF = 1000.0;
    float f_a = fa_norm * PRF;
    float ratio = lambda * f_a / (2.0 * V);

    // Range migration
    float deltaR = 0.0;
    if (abs(ratio) < 0.95) {  // Avoid singularity near ±1
        deltaR = R0 * (1.0 / sqrt(1.0 - ratio * ratio) - 1.0);
    }

    // Shift in range bins
    float shift = deltaR / rangePixel;
    float srcIdx = float(rgIdx) - shift;

    // 8-point sinc interpolation
    int srcCenter = int(floor(srcIdx));
    float frac = srcIdx - float(srcCenter);

    float2 result = float2(0.0);
    for (int k = -3; k <= 4; k++) {
        int idx = srcCenter + k;
        if (idx >= 0 && idx < int(Nr)) {
            float x = frac - float(k);
            // sinc(x) * Hamming window
            float sincVal;
            if (abs(x) < 1e-6) {
                sincVal = 1.0;
            } else {
                sincVal = sin(M_PI_F * x) / (M_PI_F * x);
            }
            // Hamming window over interpolation kernel
            float t = float(k + 3) / 7.0;
            float window = 0.54 - 0.46 * cos(2.0 * M_PI_F * t);
            result += input[azIdx * Nr + uint(idx)] * sincVal * window;
        }
    }

    output[gid] = result;
}

// ============================================================================
// Generate azimuth matched filter in range-Doppler domain
//
// H_a(f_a, R0) = exp(j * 4*pi*R0/lambda * sqrt(1 - (lambda*f_a/(2V))^2))
//
// Applied as conjugate for compression (matched filter = conj(signal spectrum))
// ============================================================================

kernel void generate_azimuth_matched_filter(
    device float2*       filter  [[buffer(0)]],
    device const float*  params  [[buffer(1)]],  // [lambda, V, R0_center, PRF, Nr, Na, rangePixel, rangeTimeStart_range0]
    uint gid                     [[thread_position_in_grid]]
) {
    float lambda     = params[0];
    float V          = params[1];
    float R0_center  = params[2];
    float PRF        = params[3];
    uint  Nr         = uint(params[4]);
    uint  Na         = uint(params[5]);
    float rangePixel = params[6];
    uint  totalCount = Nr * Na;

    if (gid >= totalCount) return;

    uint azIdx = gid / Nr;
    uint rgIdx = gid % Nr;

    // Range-dependent R0: each range bin corresponds to a different slant range
    // R0(rgIdx) = R0_center + (rgIdx - Nr/2) * rangePixel
    float R0 = R0_center + (float(rgIdx) - float(Nr / 2)) * rangePixel;

    // Azimuth frequency
    float fa_norm;
    if (azIdx < Na / 2) {
        fa_norm = float(azIdx) / float(Na);
    } else {
        fa_norm = float(int(azIdx) - int(Na)) / float(Na);
    }
    float f_a = fa_norm * PRF;
    float ratio = lambda * f_a / (2.0 * V);

    float2 h;
    if (abs(ratio) < 0.95) {
        float phase = 4.0 * M_PI_F * R0 / lambda * sqrt(1.0 - ratio * ratio);
        // Matched filter is conjugate of signal spectrum
        h = float2(cos(phase), -sin(phase));
    } else {
        h = float2(0.0, 0.0);
    }

    filter[gid] = h;
}

// ============================================================================
// FFT shift along columns (swap first and second halves of azimuth)
// ============================================================================

kernel void fftshift_columns(
    device float2*       data    [[buffer(0)]],
    device const uint*   params  [[buffer(1)]],  // [Nr, Na]
    uint gid                     [[thread_position_in_grid]]
) {
    uint Nr = params[0];
    uint Na = params[1];
    uint halfNa = Na / 2;

    if (gid >= Nr * halfNa) return;

    uint rgIdx = gid % Nr;
    uint azIdx = gid / Nr;

    uint idx1 = azIdx * Nr + rgIdx;
    uint idx2 = (azIdx + halfNa) * Nr + rgIdx;

    float2 tmp = data[idx1];
    data[idx1] = data[idx2];
    data[idx2] = tmp;
}

// ============================================================================
// Magnitude detection: output[i] = |input[i]|^2
// ============================================================================

kernel void magnitude_detect(
    device const float2* input   [[buffer(0)]],
    device float*        output  [[buffer(1)]],
    uint gid                     [[thread_position_in_grid]]
) {
    float2 val = input[gid];
    output[gid] = val.x * val.x + val.y * val.y;
}
