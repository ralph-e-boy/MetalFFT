# MetalFFT — LLM reference

Auto-generated from symbol graph. Public surface only; signatures + doc-comments verbatim. Regenerate via the `generate-llm-ref` SPM command plugin.

**Module:** MetalFFT  •  **Symbols:** 172

---

## AnalysisResult  _(struct)_

`struct AnalysisResult`

All computed properties are lazy over the stored complex spectrum — call only what you need.

- `let complex: [SIMD2<Float>]`
- `var dominantFreq: Float? { get }`
  Parabolic-interpolated dominant frequency in Hz, or `nil` if the spectrum looks like noise.
- `var dominantNote: (name: String, octave: Int)? { get }`
  Nearest piano note for the dominant frequency, or `nil`.
- `let fftSize: Int`
- `var isNoise: Bool { get }`
  `true` if the spectrum matches environmental noise heuristics.
- `var magnitudes: [Float] { get }`
  Squared magnitudes (x²+y²) for each bin — same as `vDSP_zvmags`.
- `var magnitudesDB: [Float] { get }`
  Magnitudes in dB (10·log₁₀ of squared magnitudes). Floor at –120 dB.
- `var phase: [Float] { get }`
  Per-bin phase in radians (–π to π).
- `var rms: Float { get }`
  RMS amplitude via Parseval's theorem: √(Σ|X[k]|²) / N.
- `let sampleRate: Double`
- `func binFrequency(_ bin: Int) -> Double`
  Frequency in Hz for a given bin index.

## BandEnergy  _(struct)_

`struct BandEnergy`

Per-band summed energy from a magnitude spectrum.

- `let air: Float`
- `var all: [(name: String, energy: Float)] { get }`
- `let bass: Float`
- `let low: Float`
- `let mid: Float`
- `let sub: Float`
- `var total: Float { get }`
- `let upper: Float`

## Convolver  _(class)_

`final class Convolver`

FFT-based overlap-add FIR convolution. Pre-computes the kernel spectrum at init.
Holds a `MetalFFT` instance — create once per (kernel, fftSize) and reuse for any signal.

- `init(kernel: [Float], fftSize: Int) throws`
  - Parameters:
    - kernel: FIR filter coefficients.
    - fftSize: Must be a supported MetalFFT size (64–16384) and strictly greater than `kernel.count`.
- `let blockSize: Int`
- `let hopSize: Int`
- `static func recommendedFFTSize(forKernelCount kernelCount: Int) -> Int?`
  Returns the smallest supported `MetalFFT` size > `kernelCount`, or `nil` if none.
- `func apply(to signal: [Float]) throws -> [Float]`
  Convolves `signal` with the kernel using overlap-add.
  Output length is `signal.count + kernelLen - 1`.

## Correlator  _(class)_

`final class Correlator`

GPU-accelerated cross-correlation and autocorrelation via FFT.
Holds a `MetalFFT` instance — create once per `fftSize` and reuse.

- `init(fftSize: Int) throws`
- `let fftSize: Int`
- `func auto(_ signal: [Float]) throws -> [Float]`
  Circular autocorrelation of `signal` (zero-padded to `fftSize`).
  Peak at lag 0 equals signal energy; peaks at lag m indicate periodicity at m samples.
- `func cross(_ a: [Float], _ b: [Float]) throws -> [Float]`
  Circular cross-correlation R_ab[m] = Σ a[n] · b[n+m] (zero-padded to `fftSize`).
  Via FFT: IFFT(FFT(a) · conj(FFT(b))).

## CrossSpectralResult  _(struct)_

`struct CrossSpectralResult`

Result of a multi-channel cross-spectral analysis.

Pairs are ordered as the upper triangle of the channel × channel matrix:
(0,1), (0,2), …, (0,C-1), (1,2), …, (C-2, C-1).
Use `pairIndex(_:_:)` to look up by channel indices.

- `let channels: Int`
- `let coherence: [[Float]]`
  Magnitude-squared coherence per pair per bin. Values in [0, 1].
- `let crossSpectra: [[SIMD2<Float>]]`
  Complex cross-spectra for each pair. `crossSpectra[pair][k]` = X_i(k) · conj(X_j(k)).
- `let fftSize: Int`
- `var pairCount: Int { get }`
  Number of channel pairs (upper triangle count = C*(C-1)/2).
- `let power: [[Float]]`
  Power spectrum per channel. `power[c][k]` = |X_c(k)|².
- `func pairIndex(_ i: Int, _ j: Int) -> Int`
  Returns the flat pair index for channels `i` and `j` (i < j).

## FFTAnalyzer  _(class)_

`final class FFTAnalyzer`

Stateful one-stop analyzer: window → pack → GPU FFT → AnalysisResult.
Reuses internal buffers across calls. Not thread-safe.

- `init(size: Int, sampleRate: Double, window windowType: WindowType = .hann) throws`
- `let sampleRate: Double`
- `let size: Int`
- `let windowType: WindowType`
- `func analyze(_ samples: [Float]) throws -> AnalysisResult`
  Analyze `samples` (must have `count == size`). Returns a lazy result struct.
- `func analyze(_ samples: [Float], offset: Int) throws -> AnalysisResult`
  Analyze a sub-range of `samples` starting at `offset`, without allocating a slice.
- `func analyze(samples: UnsafeBufferPointer<Float>, intoMagnitudes output: UnsafeMutableBufferPointer<Float>, binCount: Int? = nil) throws`
  Zero-allocation analyze: windows the input, runs the GPU FFT, and writes
  squared magnitudes (x²+y²) of the first `binCount` bins into `output`.
  
  `binCount` defaults to `N/2 + 1` (rfft convention: DC through Nyquist
  inclusive). Callers that only need the unique-frequency half of a real-input
  spectrum should pass `binCount: N/2`.
  
  `output` must have capacity ≥ `binCount`. No allocations occur per call.
- `func analyze(samples: UnsafeBufferPointer<Float>, intoMagnitudesDB output: UnsafeMutableBufferPointer<Float>, binCount: Int? = nil, floorDB: Float = -120) throws`
  Zero-allocation analyze: writes dB-scaled magnitudes (10·log₁₀(x²+y²))
  of the first `binCount` bins into `output`, clamped at `floorDB`.
  
  See `analyze(samples:intoMagnitudes:binCount:)` for `binCount` semantics.

## FFTError  _(enum)_

`enum FFTError`

- `var description: String { get }`
  A textual representation of this instance.
  
  Calling this property directly is discouraged. Instead, convert an
  instance of any type to a string by using the `String(describing:)`
  initializer. This initializer works with any type, and uses the custom
  `description` property for types that conform to
  `CustomStringConvertible`:
  
      struct Point: CustomStringConvertible {
          let x: Int, y: Int
  
          var description: String {
              return "(\(x), \(y))"
          }
      }
  
      let p = Point(x: 21, y: 30)
      let s = String(describing: p)
      print(s)
      // Prints "(21, 30)"
  
  The conversion of `p` to a string in the assignment to `s` uses the
  `Point` type's `description` property.
- `case batchInputSize(expected: Int, got: Int, batchIndex: Int)`
- `case bufferAllocationFailed`
- `case commandBufferFailed(String)`
- `case invalidInputSize(expected: Int, got: Int)`
- `case kernelNotFound(String)`
- `case libraryBuildFailed(any Error)`
- `case noCommandQueue`
- `case noMetalDevice`
- `case unsupportedFFTSize(Int)`

## FrequencyBands  _(struct)_

`struct FrequencyBands`

Splits a magnitude spectrum into named perceptual bands.
Construct once per (sampleRate, fftSize) pair and reuse.

- `init(sampleRate: Double, fftSize: Int)`
- `let fftSize: Int`
- `let sampleRate: Double`
- `func analyze(_ magnitudes: [Float]) -> BandEnergy`
  Summed energy per band from squared magnitudes (as returned by `Spectrum.magnitudes`).
- `func analyzeNormalized(_ magnitudes: [Float]) -> BandEnergy`
  Same as `analyze` but each band is divided by total energy → values in [0, 1].

## FrequencyTracker  _(class)_

`final class FrequencyTracker`

Stateful frequency smoother using a fixed-capacity ring buffer.
Construct once and reuse. Not thread-safe.

- `init(smoothingWindow: Int = 5)`
  - Parameter smoothingWindow: Number of frames to average. Clamped to [1, 64].
- `let smoothingWindow: Int`
- `func reset()`
  Clears the smoothing buffer.
- `func track(_ frequency: Float) -> Float`
  Pushes `frequency` into the ring buffer and returns the smoothed mean.
  Returns `frequency` unchanged if it is ≤ 0.

## FusedConvolver  _(class)_

`final class FusedConvolver`

GPU convolution that fuses FFT → multiply → IFFT into a single Metal dispatch.

Compared to `Convolver` (which issues three separate GPU dispatches per block),
`FusedConvolver` keeps all intermediate data in threadgroup memory and issues
one dispatch per block. Device memory traffic drops from 6 transfers to 2 per block.

**Constraint**: only supports `fftSize == 4096`. For other sizes or for linear
convolution over longer signals, use `Convolver` (overlap-add).

Create once per filter and reuse — the kernel spectrum is pre-computed at init
and stored in a Metal buffer.

- `init(kernel: [Float], fftSize: Int = 4096, precision: FusedConvolver.Precision = .float32) throws`
  - Parameters:
    - kernel: Real-valued FIR filter coefficients (time domain). Must have `count < 4096`.
    - fftSize: Must be `4096`.
    - precision: Arithmetic precision. Default `.float32`.
- `static let supportedSize: Int`
  The only FFT size supported by the fused kernel.
- `let hopSize: Int`
- `let kernelLen: Int`
- `let precision: FusedConvolver.Precision`
- `func apply(to signal: [Float]) throws -> [Float]`
  Convolves `signal` with the kernel using overlap-add.
  Each block dispatches a single fused GPU kernel (FFT → multiply → IFFT).
  Output length is `signal.count + kernelLen - 1`.
- `enum Precision`
  Arithmetic precision of the fused FFT → multiply → IFFT pipeline.
  
  All three modes use 16 KiB of threadgroup memory (half2[4096]) versus 32 KiB
  for `.float32`, potentially doubling threadgroup occupancy on M1/M2.
  
  - Note: FP16 modes have a ~42 dB SQNR floor. Use `.float32` when accuracy matters.

## MetalFFT  _(class)_

`final class MetalFFT`

Metal-accelerated complex FFT. Sizes: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384.

Input/output format: interleaved SIMD2<Float> where .x = real, .y = imaginary.
Not thread-safe: serialize all calls on a single writer.

- `init(size: Int) throws`
- `let size: Int`
- `func forward(_ input: [SIMD2<Float>]) throws -> [SIMD2<Float>]`
- `func forward(batch input: [[SIMD2<Float>]]) throws -> [[SIMD2<Float>]]`
  Batch FFT: all `input` elements must have `count == size`.
  Single-pass sizes use one GPU dispatch; four-step uses one command buffer per element.
- `func forward(input: UnsafeBufferPointer<SIMD2<Float>>, output: inout [SIMD2<Float>]) throws`
  Zero-copy variant: copies input from caller-managed buffer pointer.
- `func forward(real input: [Float]) throws -> [SIMD2<Float>]`
  Forward FFT of a real-valued signal. The samples are packed as complex with
  imag = 0 internally; the returned spectrum is the full N-point complex output
  (Hermitian-symmetric for real input — bins above N/2 mirror bins below).
  
  Allocates the output array; use `forward(real:output:)` for the zero-allocation variant.
- `func forward(real input: UnsafeBufferPointer<Float>, output: inout [SIMD2<Float>]) throws`
  Zero-copy real-input forward FFT. Writes both real and imag slots of the
  internal input buffer on every call (imag := 0), so interleaving with the
  complex `forward(input:output:)` path is safe.
- `func inverse(_ input: [SIMD2<Float>]) throws -> [SIMD2<Float>]`
  Inverse FFT via the conjugate trick: IFFT(X) = conj(FFT(conj(X))) / N.

## MultiChannelFFT  _(class)_

`final class MultiChannelFFT`

GPU-accelerated FFT over multiple independent channels of the same size.

All channels share a single GPU dispatch — one threadgroup per channel — so
the overhead of issuing separate dispatches is eliminated. For N=4096 the
dispatch uses the radix-8 Stockham kernel (138 GFLOPS on M1).

Input/output format: one `[SIMD2<Float>]` array per channel, where `.x` = real,
`.y` = imaginary.  Channels are independent; there is no cross-channel coupling
in the FFT itself.

Useful for: stereo/surround audio, multi-sensor arrays, multi-antenna radar,
multi-channel biological spectral analysis.

Not thread-safe. Not intended for sizes that change between calls.

- `init(channels: Int, size: Int) throws`
  - Parameters:
    - channels: Number of independent FFT channels to transform per call.
    - size: FFT length. Must be a supported `MetalFFT` size (64–16384).
- `let channels: Int`
- `let size: Int`
- `func forward(_ inputs: [[SIMD2<Float>]]) throws -> [[SIMD2<Float>]]`
  Forward FFT on all channels in a single GPU dispatch.
  
  - Parameter inputs: Exactly `channels` arrays, each of length `size`.
  - Returns: `channels` complex spectra in the same order as `inputs`.
- `func inverse(_ inputs: [[SIMD2<Float>]]) throws -> [[SIMD2<Float>]]`
  Inverse FFT on all channels in a single GPU dispatch.
  
  - Parameter inputs: Exactly `channels` complex spectra, each of length `size`.
  - Returns: `channels` time-domain signals.

## OctaveBands  _(enum)_

`enum OctaveBands`

Standard 1/3-octave band analysis (31 bands, ISO 266 center frequencies).

- `static let centerFrequencies: [Double]`
  ISO 266 nominal 1/3-octave center frequencies from 16 Hz to 16 kHz.
- `static func analyze(magnitudes: [Float], sampleRate: Double, fftSize: Int) -> [(center: Double, energy: Float)]`
  Returns summed energy for each 1/3-octave band.
  Input `magnitudes` is squared (as returned by `Spectrum.magnitudes`).
- `static func energies(magnitudes: [Float], sampleRate: Double, fftSize: Int) -> [Float]`
  Returns only the energy values (same order as `centerFrequencies`).

## OnsetDetector  _(class)_

`final class OnsetDetector`

Streaming onset/beat detector using half-wave rectified spectral flux.
Feed short audio buffers repeatedly; read `onsets` and `isPeak` after each call.
Not thread-safe.

- `init(sampleRate: Double, fftSize: Int = 1024, hopSize: Int = 256) throws`
  - Parameters:
    - sampleRate: Input signal sample rate.
    - fftSize: FFT size for spectral analysis. Default 1024.
    - hopSize: Hop between frames. Default 256 (75% overlap at fftSize=1024).
- `let fftSize: Int`
- `let hopSize: Int`
- `var isPeak: Bool { get }`
  `true` if the most recent `feed` call ended on a detected onset frame.
- `var onsets: [Double] { get }`
  Timestamps in seconds where onsets were detected (cumulative across all `feed` calls).
- `let sampleRate: Double`
- `func feed(_ samples: [Float]) throws`
  Feed the next chunk of samples. Detects onsets and appends to `onsets`.
- `func reset()`
  Clear all accumulated state.

## PeakDetection  _(enum)_

`enum PeakDetection`

Stateless frequency-domain peak detection utilities.

- `static func fundamentalFrequency(magnitudes: [Float], sampleRate: Double, fftSize: Int, minFreq: Double, maxFreq: Double, magnitudeThreshold: Float = 0.12) -> (index: Int, rawFrequency: Double)?`
  Finds the fundamental frequency bin using harmonic scoring.
  
  Scores each candidate bin by its magnitude plus 50% credit for harmonics 2–5.
  Low-frequency candidates (< 300 Hz) receive a 20% boost.
  
  - Parameters:
    - magnitudes: Squared-magnitude spectrum (from `Spectrum.magnitudes`).
    - sampleRate: Original signal sample rate.
    - fftSize: Original real signal length (= 2 × `magnitudes.count`).
    - minFreq: Lowest candidate frequency in Hz.
    - maxFreq: Highest candidate frequency in Hz.
    - magnitudeThreshold: Bins below this power are skipped. Default 0.12 (for squared magnitudes).
  - Returns: The winning bin index and its raw frequency, or `nil` if none qualify.
- `static func parabolicInterpolation(magnitudes: [Float], peakIndex: Int, sampleRate: Double, fftSize: Int) -> Double`
  Sub-bin frequency via 3-point parabolic interpolation around `peakIndex`.
  
  - Parameters:
    - magnitudes: Squared-magnitude spectrum.
    - peakIndex: Bin index of the peak.
    - sampleRate: Original signal sample rate.
    - fftSize: Original real signal length (= 2 × `magnitudes.count`).
  - Returns: Interpolated frequency in Hz.
- `static func peak(in magnitudes: [Float], range: Range<Int>? = nil) -> (index: Int, value: Float)?`
  Returns the index and value of the maximum in `magnitudes`, optionally restricted to `range`.
- `static func topPeaks(in magnitudes: [Float], count: Int, minSpacing: Int = 1) -> [(index: Int, value: Float)]`
  Returns up to `count` local maxima, each separated by at least `minSpacing` bins.
  Results are sorted by bin index ascending.

## Pitch  _(enum)_

`enum Pitch`

Music-theory utilities: frequency ↔ note name, MIDI note, and cents deviation.
All frequencies are `Float` to match the library's GPU-native precision.

- `static let noteNames: [String]`
- `static func centsDeviation(frequency: Float, referenceA: Float = 440.0) -> Float?`
  Cents deviation from the nearest equal-tempered semitone. Range: –50 to +50 cents.
  
  - Parameters:
    - frequency: Frequency in Hz (must be > 0).
    - referenceA: Tuning reference for A4. Default: 440 Hz.
- `static func frequency(midiNote: Int, referenceA: Float = 440.0) -> Float`
  Frequency in Hz for a MIDI note number. A4 = MIDI 69 = 440 Hz.
- `static func midiNote(frequency: Float, referenceA: Float = 440.0) -> Int?`
  MIDI note number (0–127) for `frequency`. A4 = 440 Hz = MIDI 69.
- `static func midiNotes(frequencies: [Float], referenceA: Float = 440.0) -> [Int?]`
  Batch MIDI note lookup for an array of frequencies.
  Uses `vvlog2f` + `vDSP` for vectorised log2 and affine scaling.
- `static func note(frequency: Float, referenceA: Float = 440.0) -> (name: String, octave: Int)?`
  Maps a frequency to the nearest piano note.
  
  - Parameters:
    - frequency: Frequency in Hz (must be > 0).
    - referenceA: Tuning reference for A4. Default: 440 Hz.
  - Returns: Note name and octave, or `nil` if outside the 88-key piano range.

## PSD  _(enum)_

`enum PSD`

Power spectral density and coherence estimation.

- `static func coherence(a: [Float], b: [Float], fftSize: Int, hopSize: Int, sampleRate: Double, window windowType: WindowType = .hann) throws -> [Float]`
  Magnitude-squared coherence between two signals: C(f) = |S_xy(f)|² / (S_xx(f)·S_yy(f)).
  
  Output is in [0, 1] per bin: 1 = fully linearly coherent at that frequency.
  Useful for measuring coupling, common excitation, or signal similarity.
  
  - Parameters:
    - a: First real-valued signal.
    - b: Second real-valued signal (same length as `a`).
    - fftSize: Must be a supported MetalFFT size (64–16384).
    - hopSize: Hop between analysis frames.
    - sampleRate: Signal sample rate.
    - windowType: Spectral window. Default `.hann`.
- `static func crossSpectral(channels signals: [[Float]], fftSize: Int, hopSize: Int, sampleRate: Double, window windowType: WindowType = .hann) throws -> CrossSpectralResult`
  Multi-channel cross-spectral density matrix via Welch's method.
  
  Slides a window over each channel, FFTs each window, computes cross-spectra
  on the GPU, and averages over windows.
  
  - Parameters:
    - channels: Time-domain signals, all the same length. Maximum 16 channels.
    - fftSize: Analysis window length. Must be a supported `MetalFFT` size.
    - hopSize: Window advance per frame.
    - sampleRate: Sample rate in Hz (used only for bin-frequency labelling).
    - window: Spectral window applied before each FFT.
  - Returns: Averaged power, cross-spectra, and coherence per bin.
- `static func welch(signal: [Float], fftSize: Int, hopSize: Int, sampleRate: Double, window windowType: WindowType = .hann) throws -> [Float]`
  Power spectral density via Welch's method (averaged overlapping periodograms).
  
  Returns PSD in linear units (squared magnitude per Hz). Take `Spectrum.toDecibels`
  of the result to convert to dB/Hz.
  
  - Parameters:
    - signal: Real-valued input signal.
    - fftSize: Must be a supported MetalFFT size (64–16384).
    - hopSize: Hop between analysis frames (controls overlap).
    - sampleRate: Signal sample rate.
    - windowType: Spectral window. Default `.hann`.

## Spectrum  _(enum)_

`enum Spectrum`

Stateless spectral analysis utilities.

- `static func isNoise(_ magnitudes: [Float], sampleRate: Double, fftSize: Int, config: Spectrum.NoiseConfig = .default) -> Bool`
  Returns `true` if the power spectrum `magnitudes` looks like environmental noise.
  
  `magnitudes` is expected to be squared magnitudes as returned by `Spectrum.magnitudes(_:)`.
  `sampleRate` is the original signal's sample rate.
  `fftSize` is the length of the real signal (2 × `magnitudes.count`).
- `static func magnitudes(_ complex: [SIMD2<Float>], count: Int? = nil) -> [Float]`
  Returns the squared magnitude (x²+y²) of each complex bin — matches `vDSP_zvmags` output.
  `count` limits the number of bins returned; defaults to `complex.count`.
- `static func normalize(_ magnitudes: inout [Float])`
  Normalizes `magnitudes` in-place to [0, 1] by the peak value. No-op if peak is zero.
- `static func phase(_ complex: [SIMD2<Float>]) -> [Float]`
  Returns the instantaneous phase (atan2) of each complex bin, in radians (–π to π).
- `static func rms(_ samples: UnsafeBufferPointer<Float>) -> Float`
  RMS amplitude of a real sample buffer (equivalent to `vDSP_rmsqv`).
- `static func rms(_ samples: [Float]) -> Float`
  Convenience overload for Array.
- `static func toDecibels(_ magnitudes: [Float], floorDB: Float = -120) -> [Float]`
  Converts squared magnitudes to dB (10·log₁₀). `floorDB` clamps -∞ from zero-valued bins.
- `struct NoiseConfig`
  Configuration for `isNoise`.

## STFT  _(class)_

`final class STFT`

Short-Time Fourier Transform: sliding-window FFT over a long signal.
Construct once and reuse — holds internal Metal and scratch buffers. Not thread-safe.

- `init(fftSize: Int, hopSize: Int, window windowType: WindowType = .hann, sampleRate: Double) throws`
  - Parameters:
    - fftSize: Must be a supported MetalFFT size (64–16384).
    - hopSize: Samples advanced per frame. Overlap = `fftSize - hopSize`.
               Typical choices: `fftSize/2` (50%) or `fftSize/4` (75%).
    - windowType: Spectral window applied before each FFT. Default `.hann`.
    - sampleRate: Input signal sample rate in Hz.
- `let fftSize: Int`
- `let hopSize: Int`
- `let sampleRate: Double`
- `func analyze(_ signal: [Float]) throws -> [STFTFrame]`
  Analyze `signal`, returning one `STFTFrame` per hop.
- `func binFrequency(_ bin: Int) -> Double`
  Frequency in Hz for a given bin index.
- `func frameCount(for sampleCount: Int) -> Int`
  Number of frames that will be produced for a signal of `sampleCount` samples.
- `func frameTime(_ frameIndex: Int) -> Double`
  Start time in seconds for a given frame index.
- `func spectrogram(_ signal: [Float]) throws -> [[Float]]`
  2-D spectrogram as dB magnitudes: `[time][frequency]`.

## STFTFrame  _(struct)_

`struct STFTFrame`

- `let complex: [SIMD2<Float>]`
- `let magnitudes: [Float]`
- `var magnitudesDB: [Float] { get }`
- `var phase: [Float] { get }`

## Window  _(enum)_

`enum Window`

Windowing functions and application helpers.

- `static func apply(_ window: [Float], input: UnsafeBufferPointer<Float>, output: inout [Float])`
  Element-wise multiply `window` by `input`, writing into `output`.
- `static func apply(_ window: [Float], to samples: [Float]) -> [Float]`
  Convenience: apply window to `samples` returning a new windowed array.
- `static func blackman(_ size: Int) -> [Float]`
  Blackman window — lower sidelobes than Hann, good general-purpose choice.
- `static func flatTop(_ size: Int) -> [Float]`
  Flat-top window — maximally flat passband for accurate amplitude measurement.
- `static func hamming(_ size: Int) -> [Float]`
- `static func hann(_ size: Int) -> [Float]`
- `static func kaiser(_ size: Int, beta: Double = 6.0) -> [Float]`
  Kaiser window — tunable sidelobe attenuation via `beta` (common values: 5–10).

## WindowType  _(enum)_

`enum WindowType`

Selects a windowing function for use with `FFTAnalyzer`, `STFT`, and other components.

- `case blackman`
- `case flatTop`
- `case hamming`
- `case hann`
- `case kaiser(beta: Double)`
- `case rectangular`
- `func coefficients(_ size: Int) -> [Float]`

## FusedConvolver.Precision

- `case float16Mixed`
  FP16 twiddle multiply, FP32 butterfly accumulate.
- `case float16Pure`
  All butterfly arithmetic in FP16.
- `case float16Storage`
  FP16 threadgroup storage, FP32 butterfly compute. Recommended FP16 mode.
- `case float32`
  Full FP32 throughout (default).

## Spectrum.NoiseConfig

- `init(lowFreqDominanceThreshold: Float = 0.8, flatnessVarianceThreshold: Float = 0.005, peakToMeanRatioThreshold: Float = 2.5)`
- `static let `default`: Spectrum.NoiseConfig`
- `var flatnessVarianceThreshold: Float`
  Spectral variance below this → noise. Default: 0.005.
- `var lowFreqDominanceThreshold: Float`
  Fraction of total power in 0–200 Hz that classifies as noise. Default: 0.8.
- `var peakToMeanRatioThreshold: Float`
  Peak-to-mean ratio below this → noise. Default: 2.5.

