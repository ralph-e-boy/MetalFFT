# AppleSiliconFFT

High-performance FFT (Fast Fourier Transform) implementation for Apple Silicon GPUs using Metal compute shaders. Achieves **138.45 GFLOPS** at N=4096 on Apple M1 -- **29% faster than Apple's vDSP/Accelerate**.

This is the companion code for the paper:

> M. A. Bergach, "Beating vDSP: A 138 GFLOPS Radix-8 Stockham FFT on Apple Silicon via Two-Tier Register-Threadgroup Memory Decomposition," 2026.

## Key Results (Apple M1)

| Kernel | N=4096, batch=256 | vs vDSP |
|--------|-------------------|---------|
| Radix-8 Stockham (best) | **138.45 GFLOPS** (1.78 us/FFT) | **+29%** |
| Radix-4 Stockham | 113.6 GFLOPS (2.16 us/FFT) | +6% |
| Apple vDSP (Accelerate) | 107.0 GFLOPS (2.29 us/FFT) | baseline |

### Multi-Size Performance (batch=64)

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

### FFT Kernels (N=4096, radix-4 + radix-8)

```bash
cd src
# Compile Metal shaders
xcrun metal -o fft.air fft_stockham_4096.metal
xcrun metal -o fft_r8.air fft_4096_radix8.metal
xcrun metallib -o default.metallib fft.air fft_r8.air

# Build and run host (validates against vDSP + benchmarks)
swiftc -O -framework Metal -framework Accelerate -o fft_host fft_host.swift
./fft_host
```

### Multi-Size FFT (N=256 through N=16384)

```bash
cd src
xcrun metal -o fft_multi.air fft_multisize.metal
xcrun metallib -o default.metallib fft_multi.air

swiftc -O -framework Metal -framework Accelerate -o fft_multi_host fft_multisize_host.swift
./fft_multi_host
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

The key insight is that threadgroup memory is small (32 KiB = 4096 complex float32), so the FFT must be designed to minimize threadgroup memory traffic and maximize register-resident computation.

### Kernel Design

**Radix-4 Stockham** (113.6 GFLOPS):
- 6 fully-unrolled radix-4 passes, 1024 threads
- Single sincos per butterfly (w2 = w1^2, w3 = w1^3 via complex multiply)
- Direct device I/O (first pass reads from input, last writes to output)
- 10 threadgroup barriers

**Radix-8 Split-Radix DIT** (138.45 GFLOPS):
- 4 radix-8 Stockham passes, 512 threads
- Split-radix butterfly: DFT_8 decomposed as radix-2 + radix-4 (~32 FLOPs vs ~320 naive)
- Same sincos optimization and direct I/O
- 6 threadgroup barriers

**Multi-threadgroup (N > 4096)**:
- Four-step FFT decomposition through device memory
- Sub-FFTs of size 64 or 128 in single threadgroups
- Twiddle+transpose kernels between passes

### Key Finding

On Apple GPU, **threadgroup memory barriers are cheap (~2 cycles)** while **scattered threadgroup access is expensive** (bank conflicts). This means the Stockham approach (coalesced access, more barriers) significantly outperforms SIMD shuffle approaches (scattered access, fewer barriers).

## File Structure

```
src/
  fft_stockham_4096.metal    -- Radix-4 Stockham FFT (N=4096)
  fft_4096_radix8.metal      -- Radix-8 split-radix DIT FFT (N=4096, best)
  fft_multisize.metal        -- Multi-size kernels (N=256 through N=16384)
  fft_host.swift             -- Host code for N=4096 kernels + validation
  fft_multisize_host.swift   -- Host code for multi-size kernels + validation
benchmarks/
  Sources/
    memory_bandwidth.metal   -- GPU microbenchmark shaders
    memory_bandwidth_bench.swift -- Threadgroup/SIMD/register benchmarks
    vdsp_baseline_bench.swift    -- vDSP performance baseline
    main.swift               -- Benchmark runner
    utils.swift              -- Benchmark utilities
  Makefile                   -- Build with: make && make run
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bergach2026beating,
  title={Beating vDSP: A 138 GFLOPS Radix-8 Stockham FFT on Apple Silicon via Two-Tier Register-Threadgroup Memory Decomposition},
  author={Bergach, Mohamed Amine},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

Mohamed Amine Bergach -- mbergach@illumina.com
