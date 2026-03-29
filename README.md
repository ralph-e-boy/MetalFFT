# AppleSiliconFFT

High-performance FFT and kernel-fused SAR processing for Apple Silicon GPUs using Metal compute shaders.

**Paper 1**: 138.45 GFLOPS FFT at N=4096 -- 29% faster than Apple's vDSP/Accelerate.
**Paper 2**: Kernel-fused SAR Range Doppler in 370 ms -- 22x speedup over unfused baseline.

Companion code for:

> M. A. Bergach, "Beating vDSP: A 138 GFLOPS Radix-8 Stockham FFT on Apple Silicon via Two-Tier Register-Threadgroup Memory Decomposition," 2026.

> M. A. Bergach, "From 8 Seconds to 370 ms: Kernel-Fused SAR Imaging on Apple Silicon via Single-Dispatch FFT Pipelines," 2026.

## Key Results (Apple M1)

### FFT Kernels

| Kernel | N=4096, batch=256 | vs vDSP |
|--------|-------------------|---------|
| Radix-8 Stockham (best) | **138.45 GFLOPS** (1.78 us/FFT) | **+29%** |
| In-place CT MMA (simdgroup_matrix) | 128 GFLOPS (1.92 us/FFT) | +20% |
| Radix-4 Stockham | 113.6 GFLOPS (2.16 us/FFT) | +6% |
| Apple vDSP (Accelerate) | 107.0 GFLOPS (2.29 us/FFT) | baseline |

### SAR Range Doppler Processing (4096x4096)

| Pipeline | Time | Speedup |
|----------|------|---------|
| Unfused baseline (separate dispatches) | 8.16 s | 1x |
| **Fused pipeline (single-dispatch FFT+multiply+IFFT)** | **0.37 s** | **22x** |

### Multi-Size FFT (batch=64)

| Size | GFLOPS | Type |
|------|--------|------|
| 256 | 53 | Single threadgroup |
| 512 | 75 | Single threadgroup |
| 1024 | 112 | Single threadgroup |
| 2048 | 104 | Single threadgroup |
| 4096 | 138 | Single threadgroup |
| 8192 | 84 | Four-step (multi-threadgroup) |
| 16384 | 103 | Four-step (multi-threadgroup) |

## Requirements

- macOS 13.0+ (Ventura or later)
- Apple Silicon (M1 or later)
- Xcode Command Line Tools (for `swiftc` and Metal compiler)

## Building and Running

### FFT Kernels (N=4096)

```bash
cd src
# Radix-4 Stockham (113.6 GFLOPS)
xcrun metal -o fft.air fft_stockham_4096.metal
xcrun metallib -o default.metallib fft.air
swiftc -O -framework Metal -framework Accelerate -o fft_host fft_host.swift
./fft_host

# Radix-8 split-radix DIT (138.45 GFLOPS, best)
xcrun metal -o fft_r8.air fft_4096_radix8.metal
xcrun metallib -o default.metallib fft_r8.air
swiftc -O -framework Metal -framework Accelerate -o fft_host fft_host.swift
./fft_host

# In-place Cooley-Tukey with simdgroup_matrix MMA (128 GFLOPS)
xcrun metal -o fft_ct.air fft_4096_ct_mma.metal
xcrun metallib -o default.metallib fft_ct.air
swiftc -O -framework Metal -framework Accelerate -o fft_ct_host fft_ct_mma_host.swift
./fft_ct_host
```

### Multi-Size FFT (N=256 through N=16384)

```bash
cd src
xcrun metal -o fft_multi.air fft_multisize.metal
xcrun metallib -o default.metallib fft_multi.air
swiftc -O -framework Metal -framework Accelerate -o fft_multi_host fft_multisize_host.swift
./fft_multi_host
```

### SAR Range Doppler Processing

```bash
cd src/radar
# Compile all Metal kernels needed
xcrun metal -o rda.air rda_kernels.metal ../fft_sar_fused.metal ../fft_multisize.metal
xcrun metallib -o default.metallib rda.air
# Build SAR pipeline
swiftc -O -framework Metal -framework Accelerate -o sar \
  main.swift sar_simulator.swift rda_pipeline.swift rda_fused_pipeline.swift \
  radar_metrics.swift precision_comparison.swift
# Run unfused baseline
./sar 4096
# Run fused pipeline (22x speedup)
./sar 4096 --fused
# Run mixed-precision comparison
./sar 4096 --precision
```

### Microbenchmarks

```bash
cd benchmarks
make && make run
```

## Architecture

### Two-Tier Local Memory Model

Apple Silicon GPU has two tiers of fast local memory:

1. **Register file** (208 KiB per threadgroup): primary data-resident storage
2. **Threadgroup memory** (32 KiB): inter-SIMD-group exchange scratchpad

### FFT Kernel Variants

- **Radix-4 Stockham** -- 6 passes, 1024 threads, coalesced access
- **Radix-8 Split-Radix DIT** -- 4 passes, 512 threads, split-radix butterfly (~32 FLOPs vs ~320 naive)
- **In-Place CT DIF + MMA** -- 4 stages, `simdgroup_float8x8` hardware matrix multiply for radix-8 DFT
- **Batched** -- multiple FFTs per dispatch for throughput workloads

### Kernel Fusion (SAR)

Fused range compression: FFT + matched filter multiply + IFFT in a single Metal dispatch. Data stays in 32 KiB threadgroup memory between operations, eliminating 4 of 6 device-memory round-trips.

**Key insight**: On Apple GPU, threadgroup memory barriers are cheap (~2 cycles) while scattered access is expensive. The Stockham algorithm's coalesced access pattern is optimal despite more barriers.

### Mixed Precision (FP16)

Three modes for fused SAR kernels:
- **Mode A**: Pure FP16 (2x throughput, ~42 dB SQNR floor)
- **Mode B**: FP16 storage + FP32 compute (1.5x throughput, near-FP32 accuracy)
- **Mode C**: FP16 multiply + FP32 accumulate (best accuracy/throughput tradeoff)

## File Structure

```
src/
  fft_stockham_4096.metal      -- Radix-4 Stockham FFT (113.6 GFLOPS)
  fft_4096_radix8.metal        -- Radix-8 split-radix DIT (138.45 GFLOPS, best)
  fft_4096_ct_mma.metal        -- In-place CT DIF with simdgroup_matrix MMA (128 GFLOPS)
  fft_4096_batched.metal       -- Batched FFT for throughput workloads
  fft_multisize.metal          -- Multi-size kernels (N=256 through N=16384)
  fft_sar_fused.metal          -- Fused FFT+multiply+IFFT for SAR (FP32)
  fft_sar_fused_fp16.metal     -- Fused SAR kernels (FP16 modes A/B/C)
  fft_host.swift               -- Host for N=4096 kernels + validation
  fft_ct_mma_host.swift        -- Host for CT MMA kernel + validation
  fft_batched_host.swift       -- Host for batched FFT + validation
  fft_multisize_host.swift     -- Host for multi-size + validation
  radar/
    sar_simulator.swift        -- Point-target SAR raw data generator
    rda_pipeline.swift         -- Unfused Range Doppler Algorithm baseline
    rda_fused_pipeline.swift   -- Fused RDA pipeline (22x speedup)
    rda_kernels.metal          -- GPU utility kernels (multiply, transpose, RCMC)
    radar_metrics.swift        -- PSLR, ISLR, SNR measurement
    precision_comparison.swift -- FP32 vs FP16 accuracy comparison
    main.swift                 -- Entry point (supports --fused, --precision flags)
benchmarks/
  Sources/
    memory_bandwidth.metal     -- GPU microbenchmark shaders
    memory_bandwidth_bench.swift -- Threadgroup/SIMD/register benchmarks
    vdsp_baseline_bench.swift  -- vDSP performance baseline
    main.swift                 -- Benchmark runner
    utils.swift                -- Benchmark utilities
  Makefile                     -- Build with: make && make run
```

## Citation

```bibtex
@article{bergach2026beating,
  title={Beating vDSP: A 138 GFLOPS Radix-8 Stockham FFT on Apple Silicon
         via Two-Tier Register-Threadgroup Memory Decomposition},
  author={Bergach, Mohamed Amine},
  year={2026}
}

@article{bergach2026fused,
  title={From 8 Seconds to 370 ms: Kernel-Fused SAR Imaging on Apple Silicon
         via Single-Dispatch FFT Pipelines},
  author={Bergach, Mohamed Amine},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

Mohamed Amine Bergach -- mbergach@illumina.com
