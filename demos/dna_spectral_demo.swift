// =============================================================================
// DNA Spectral Analysis Demo — MetalFFT Package
//
// Demonstrates MultiChannelFFT and PSD.crossSpectral on a synthetic DNA
// sequence encoded as four binary channels: A, T, G, C.
//
// Biological background:
//   Genomic DNA has a known period-3 spectral signature in coding regions
//   (every 3rd base position is biased in codons). This demo generates a
//   synthetic sequence with embedded period-3 structure and verifies that
//   the cross-spectral analysis detects elevated power at bin N/3.
// =============================================================================

import Metal
import MetalFFT
import Accelerate

// MARK: - Synthetic DNA sequence

let seqLen = 32768   // 32 K bases
let fftSize = 1024
let hopSize = 512
let sampleRate: Double = 1.0  // 1 sample per base

// Bases: 0=A, 1=T, 2=G, 3=C
var sequence = [UInt8](repeating: 0, count: seqLen)

// Background: random uniform
for i in 0..<seqLen { sequence[i] = UInt8.random(in: 0..<4) }

// Inject period-3 codon bias in the middle third: A favored at codon pos 0
let codingStart = seqLen / 3
let codingEnd   = 2 * seqLen / 3
for i in stride(from: codingStart, to: codingEnd, by: 3) {
    if Float.random(in: 0...1) < 0.7 { sequence[i] = 0 }       // A at pos 0
    if Float.random(in: 0...1) < 0.5 { sequence[i+1] = 3 }     // C at pos 1
}

// Encode as 4 binary channels
let A = sequence.map { Float($0 == 0 ? 1 : 0) }
let T = sequence.map { Float($0 == 1 ? 1 : 0) }
let G = sequence.map { Float($0 == 2 ? 1 : 0) }
let C = sequence.map { Float($0 == 3 ? 1 : 0) }

print("=== DNA Spectral Analysis Demo ===")
print("Sequence length: \(seqLen) bases")
print("Coding region (period-3): bases \(codingStart)–\(codingEnd)")
print("FFT size: \(fftSize), hop: \(hopSize)\n")

// MARK: - MultiChannelFFT: single-frame 4-channel FFT

do {
    print("--- MultiChannelFFT (single frame) ---")
    let mfft = try MultiChannelFFT(channels: 4, size: fftSize)
    let frame = [A, T, G, C].map { ch in
        ch.prefix(fftSize).map { SIMD2<Float>($0, 0) }
    }
    let spectra = try mfft.forward(frame)

    // Find the dominant bin for each channel (skip DC)
    for (idx, name) in ["A","T","G","C"].enumerated() {
        var maxPow: Float = 0
        var maxBin = 0
        for k in 1..<(fftSize/2) {
            let p = spectra[idx][k].x * spectra[idx][k].x + spectra[idx][k].y * spectra[idx][k].y
            if p > maxPow { maxPow = p; maxBin = k }
        }
        print("  Channel \(name): peak bin \(maxBin)  (freq = \(Float(maxBin)/Float(fftSize))/base)")
    }
}

// MARK: - PSD.crossSpectral: full Welch cross-spectral matrix

do {
    print("\n--- PSD.crossSpectral (Welch, 4 channels) ---")
    let result = try PSD.crossSpectral(
        channels: [A, T, G, C],
        fftSize: fftSize,
        hopSize: hopSize,
        sampleRate: sampleRate
    )

    print("Pair count: \(result.pairCount)")

    // Find max-power bin across all channels (period-3 should show at N/3)
    let period3Bin = fftSize / 3
    print("\nPower at period-3 bin (\(period3Bin)) vs DC:")
    for c in 0..<4 {
        let name = ["A","T","G","C"][c]
        let dcPow = result.power[c][1]
        let p3Pow = result.power[c][period3Bin]
        print("  Channel \(name): P(1)=\(String(format:"%.4f", dcPow))  P(\(period3Bin))=\(String(format:"%.4f", p3Pow))")
    }

    // Coherence between A and C (both involved in codon bias) at period-3
    let acPair = result.pairIndex(0, 3)  // A=0, C=3
    let coh3   = result.coherence[acPair][period3Bin]
    print("\nA–C coherence at bin \(period3Bin): \(String(format:"%.4f", coh3))")
    print("(> 0.1 indicates shared period-3 structure)")
}

print("\nDone.")
