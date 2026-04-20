# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research codebase accompanying three papers on GPU-accelerated FFT on Apple Silicon (M1+, macOS 13+). Three independent subsystems, each with its own build system.

## Build Commands

### FFT Kernels & SAR Radar (`src/`)
Manual build — no Makefile or SPM:
```bash
cd src
# Compile a Metal shader
xcrun metal -o fft_r8.air fft_4096_radix8.metal
xcrun metallib -o default.metallib fft_r8.air
# Compile and run a Swift host
swiftc -O -framework Metal -framework Accelerate -o fft_host fft_host.swift
./fft_host
```
SAR radar requires additional Metal and Swift files — see README for exact multi-file commands.

### Benchmarks (`benchmarks/`)
```bash
cd benchmarks && make && make run
```
Named targets: `bench-tgmem`, `bench-shuffle`, `bench-regcopy`, `bench-occupancy`, `bench-threadcount`, `bench-vdsp`.

### DNA Genomic Analysis (`code/`)
```bash
cd code
swift build --product MultilevelAnalysis
swift build --product NullModelTest
swift build --product PfGenomeWide
```
**Note**: `Package.swift` declares targets under `Sources/DNALib`, etc., but source files currently live at `code/DNALib/`, `code/MultilevelAnalysis/`, etc. — no `Sources/` subdirectory exists. Fix the layout or the path declarations before `swift build` will succeed.

## Architecture

### Two-Tier GPU Memory Model (core FFT insight)
All FFT kernels exploit Apple Silicon's fast local memory hierarchy:
- **Register file** (208 KiB/threadgroup) — primary data store across FFT passes
- **Threadgroup memory** (32 KiB) — only for inter-SIMD-group exchange (~2 cycle barriers)

The Stockham algorithm is chosen specifically for its naturally coalesced access pattern that keeps data in this hierarchy and avoids scattered device-memory traffic.

### FFT Kernels (`src/`)
| File | Algorithm | Threads | Throughput |
|---|---|---|---|
| `fft_stockham_4096.metal` | Radix-4 Stockham, 6 passes | 1024 | 113.6 GFLOPS |
| `fft_4096_radix8.metal` | Radix-8 split-radix DIT, 4 passes | 512 | 138.45 GFLOPS (best) |
| `fft_4096_ct_mma.metal` | Cooley-Tukey DIF via `simdgroup_float8x8` MMA | — | 128 GFLOPS |
| `fft_multisize.metal` | N=256–16384; four-step algorithm for N≥8192 | — | — |
| `fft_sar_fused.metal` | Fused FFT+multiply+IFFT for SAR range compression (FP32) | — | — |
| `fft_sar_fused_fp16.metal` | Same, three FP16 modes (A/B/C) | — | — |

Each Swift host file creates `MTLDevice`, compiles its Metal source at runtime via `MTLDevice.makeLibrary(source:)`, dispatches compute kernels, and validates against Apple's vDSP (Accelerate framework).

### SAR Radar Pipeline (`src/radar/`)
- `sar_simulator.swift` — generates synthetic SAR raw data (point targets)
- `rda_pipeline.swift` — unfused Range-Doppler Algorithm baseline (~8 s)
- `rda_fused_pipeline.swift` — fused single-dispatch RDA (~0.37 s, 22× speedup)
- `rda_kernels.metal` — GPU utility kernels: multiply, transpose, RCMC
- `radar_metrics.swift` — PSLR, ISLR, SNR measurement
- `main.swift` — CLI entry point; flags `--fused`, `--precision`

### DNA Spectral Analysis (`code/`)
**DNALib** (shared library): `fasta_reader.swift`, `dna_analyzer.swift` (4-channel A/T/G/C Metal FFT pipeline), `welch_coherence.swift`, `multilevel_spectral.swift`, `dinucleotide_shuffle.swift`

Metal shaders (`dna_spectral.metal`, `dna_cross_spectral.metal`, `dna_spectrogram.metal`) implement a 4-channel quaternion FFT and cross-spectral coherence matrix; compiled at runtime — must be adjacent to the executable.

### Benchmarks (`benchmarks/`)
Microbenchmarks measuring GPU memory hierarchy bandwidth and vDSP baselines. `memory_bandwidth.metal` + `memory_bandwidth_bench.swift` + `vdsp_baseline_bench.swift`.

## Key Patterns
- **No test suite** — correctness validated inline via vDSP cross-checks inside each host binary.
- **Runtime Metal compilation** — Metal shaders in `src/` and `code/` are compiled at runtime; the `.metal` source files must be co-located with or reachable from the executable's working directory.
- **No Xcode project** — pure command-line builds throughout all three subsystems.
