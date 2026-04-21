# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Two largely independent things live here:

1. **`MetalFFT` Swift Package** (repo root) — the primary active codebase. A GPU-accelerated FFT library with a layered DSP API, published via Swift Package Manager.
2. **Research subsystems** (`src/`, `benchmarks/`, `code/`) — standalone paper-supporting code with manual builds. Not part of the package.

---

## MetalFFT Package

### Build & Test

```bash
swift build --target MetalFFT          # build library only
swift test                             # run all tests
swift test --filter DemoComparisonTests  # run a specific test class
swift test --filter testPackageVsRadix8Demo  # run a single test
make docs                              # generate DocC archive → docs/MetalFFT.doccarchive
make docs-llm                          # emit .swiftinterface → docs/MetalFFT.swiftinterface
```

### Architecture

All GPU work flows through a single shared `MetalContext` (singleton, `Internal/MetalContext.swift`), which owns the `MTLDevice`, `MTLCommandQueue`, and pre-compiled `MTLComputePipelineState` map. Metal source is `Sources/MetalFFT/Resources/fft_multisize.metal`, compiled once at init and bundled as a package resource.

**`MetalFFT` class** (`FFT/MetalFFT.swift`) is the only type that touches Metal directly. It selects a dispatch strategy at init via `FFTDescriptor`:
- **Single-pass** (N ≤ 4096): one threadgroup, one dispatch.
- **Four-step** (N = 8192, 16384): transpose → row-FFTs → twiddle+transpose → row-FFTs → transpose, all in one `MTLCommandBuffer` with 5 encoder passes.

IFFT uses the conjugate trick: `IFFT(X) = conj(FFT(conj(X))) / N` — no separate kernel.

**DSP layer** (`DSP/`) builds on `MetalFFT` and `Accelerate` only — no Metal calls:
- `FFTAnalyzer` — stateful all-in-one: window → pack complex → GPU FFT → `AnalysisResult`
- `STFT` — sliding-window FFT; `STFTFrame` has `.magnitudes`, `.magnitudesDB`, `.phase`
- `Convolver` — overlap-add FIR; pre-computes kernel spectrum at init
- `FusedConvolver` — fuses FFT → multiply → IFFT into a single GPU dispatch (N=4096 only). 32 KiB→16 KiB threadgroup memory with FP16 modes. `Precision` enum: `.float32` (default), `.float16Pure`, `.float16Storage` (recommended FP16), `.float16Mixed`.
- `Correlator` — cross/auto-correlation via FFT multiply + IFFT
- `OnsetDetector` — half-wave rectified spectral flux with adaptive threshold
- `PSD` — Welch's method, magnitude-squared coherence, and `crossSpectral(channels:fftSize:hopSize:sampleRate:)` for multi-channel cross-spectral density matrices
- `FrequencyBands` / `OctaveBands` — perceptual and ISO 266 1/3-octave band energy
- `Spectrum`, `PeakDetection`, `Window` — stateless utilities; all `Float`, all `Accelerate`

**FFT layer additions** (`FFT/`):
- `MultiChannelFFT` — typed wrapper for batch dispatch of C independent FFTs of the same size. `forward(_:) -> [[SIMD2<Float>]]`, `inverse(_:)`.

**Cross-spectral analysis** (`DSP/CrossSpectral.swift`):
- `CrossSpectralResult` — `power[c][k]`, `crossSpectra[pair][k]`, `coherence[pair][k]`, `pairIndex(_:_:)`. Pairs ordered as upper triangle: (0,1),(0,2),…
- GPU kernel `fft_cross_spectral` supports 1–16 channels via runtime `params[C]`. Min buffer size `max(1, pairs)*N` for single-channel (0 pairs) case.

**Pitch layer** (`Pitch/`) — `Float` throughout, `Accelerate` not `Foundation`:
- `Pitch` — scalar and batch (`vvlog2f` + `vDSP_vsmsa`) frequency ↔ MIDI/note conversion
- `FrequencyTracker` — fixed-capacity ring buffer; mean via `vDSP_meanv`

### Key Conventions
- All public APIs use `Float` (GPU-native precision). `Double` appears only where `PeakDetection` / `sampleRate` haven't been migrated yet.
- Prefer `Accelerate`/`vDSP` over scalar loops. `Foundation` is not imported in DSP or Pitch files.
- `MetalFFT` is stateful and not thread-safe — serialize all calls.
- Errors surface as `FFTError` (defined in `Internal/MetalContext.swift`).

### Build & Demo Commands

```bash
swift run DNASpectralDemo              # 4-channel genomic spectral analysis demo
make demo-dna                          # same via Makefile
```

### Tests (`Tests/MetalFFTTests/`)
- `MetalFFTTests.swift` — core correctness (impulse, sine, round-trip, batch, IFFT)
- `DemoComparisonTests.swift` — compares package output against the raw `src/` demo Metal kernels (Stockham, Radix-8, CT MMA). The CT MMA kernel requires 4 precomputed buffers at indices 2–5 (DFT matrices + twiddle table); `ctmmaKernel()` handles this.
- `CrossSpectralTests.swift` — MultiChannelFFT correctness, PSD.crossSpectral coherence, pair ordering.
- `FusedConvolverTests.swift` — delta-filter identity, FusedConvolver vs Convolver, all three FP16 modes vs FP32.

---

## Research Subsystems (read-only reference)

### `src/` — Standalone FFT & SAR demos
Manual build; `make demo` runs all demos via precompiled binaries in `bin/`.
```bash
make demo-fft        # Radix-4 Stockham (113.6 GFLOPS)
make demo-ct         # CT DIF + simdgroup MMA (128 GFLOPS)
make demo-batched    # Batched FFT
make demo-multisize  # N=256–16384
make demo-radar      # SAR Range-Doppler fused pipeline
make demo-dna        # 4-channel genomic spectral analysis (MetalFFT package)
```
`src/` kernels compile at runtime — `.metal` files must be co-located with the binary (the Makefile copies them).

The CT MMA kernel (`fft_4096_ct_mma.metal`, function `fft_4096_ct_mma`) needs:
- Index 2: 8×8 DFT real matrix
- Index 3: 8×8 DFT imag matrix
- Index 4: negated imag matrix
- Index 5: 3×4096 twiddle table for strides [512, 64, 8]

### `benchmarks/` — GPU microbenchmarks
```bash
cd benchmarks && make && make run
```

### `code/` — DNA genomic spectral analysis (Paper 3)
Separate SPM package (`DNASpectralAnalysis`). 4-channel A/T/G/C quaternion FFT, cross-spectral coherence for genomic sequences. **Note**: `Package.swift` declares targets under `Sources/` but source files live directly in `code/DNALib/` etc. — fix path declarations before `swift build` will work.

---

## Two-Tier GPU Memory Model (core FFT insight)
Apple Silicon has two tiers of fast local memory per threadgroup:
- **Register file** (208 KiB) — primary data store across FFT passes
- **Threadgroup memory** (32 KiB) — inter-SIMD-group exchange only (~2 cycle barriers)

The Stockham algorithm is chosen for its naturally coalesced access pattern that keeps data in this hierarchy. For N≥8192, the four-step algorithm decomposes the FFT into two smaller FFTs with explicit transposes to maintain coalesced access.
