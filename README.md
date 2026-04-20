# MetalFFT

GPU-accelerated FFT for Apple Silicon — a Swift Package built on Metal compute shaders, with a layered DSP API for spectrum analysis, pitch detection, convolution, time-frequency analysis, and beat detection.

## Requirements

- macOS 13.0+ (Ventura or later)
- Apple Silicon (M1 or later)
- Swift 5.9+

## Installation

Add the package via Swift Package Manager:

```swift
.package(url: "https://github.com/ralphseaman/AppleSiliconFFT", from: "1.0.0"),
```

Then add `"MetalFFT"` to your target dependencies.

## Quick Example

```swift
import MetalFFT

// All-in-one: window → GPU FFT → analysis in one call
let analyzer = try FFTAnalyzer(size: 4096, sampleRate: 44100, window: .hann)
let result   = try analyzer.analyze(samples)

print(result.dominantNote!)   // e.g. ("A", 4)
print(result.rms)             // RMS amplitude
print(result.magnitudesDB)    // [Float] in dB

// Frequency band energies — great for audio-reactive visuals
let bands  = FrequencyBands(sampleRate: 44100, fftSize: 4096)
let energy = bands.analyze(result.magnitudes)
print(energy.bass, energy.mid, energy.air)
```

## API

### Core FFT — `MetalFFT`

Metal GPU-accelerated complex FFT. Input/output is interleaved `SIMD2<Float>` (`.x` = real, `.y` = imaginary).

| Method | Description |
|---|---|
| `init(size:)` | Supported sizes: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 |
| `forward([SIMD2<Float>])` | Forward FFT |
| `forward(input:output:)` | Zero-copy forward FFT with caller-managed buffer |
| `forward(batch:)` | Batch forward FFT over multiple signals |
| `inverse([SIMD2<Float>])` | Inverse FFT (IFFT) |

`MetalFFT` is stateful — not thread-safe. Errors surface as `FFTError`.

---

### All-in-One — `FFTAnalyzer` + `AnalysisResult`

Hides the window → pack → FFT → magnitudes pipeline. Reuses all internal buffers across calls.

```swift
let analyzer = try FFTAnalyzer(size: 2048, sampleRate: 44100, window: .blackman)
let result   = try analyzer.analyze(samples)

result.magnitudes       // [Float] — squared magnitudes (x²+y²)
result.magnitudesDB     // [Float] — dB scale (10·log₁₀)
result.phase            // [Float] — per-bin phase in radians
result.binFrequency(5)  // Double  — Hz for bin 5
result.dominantFreq     // Double? — parabolic-interpolated fundamental
result.dominantNote     // (name: String, octave: Int)?
result.isNoise          // Bool
result.rms              // Float   — via Parseval's theorem
```

---

### Windowing — `Window` + `WindowType`

```swift
let hann     = Window.hann(1024)
let hamming  = Window.hamming(1024)
let blackman = Window.blackman(1024)      // lower sidelobes than Hann
let flatTop  = Window.flatTop(1024)       // accurate amplitude measurement
let kaiser   = Window.kaiser(1024, beta: 8.0)  // tunable sidelobe attenuation

let windowed = Window.apply(hann, to: samples)
```

Pass windows by type to `FFTAnalyzer`, `STFT`, `Convolver`, and `PSD`:

```swift
WindowType.hann / .hamming / .blackman / .flatTop / .kaiser(beta:) / .rectangular
```

---

### Spectrum Analysis — `Spectrum` & `PeakDetection`

```swift
var mags = Spectrum.magnitudes(complexOutput)   // squared magnitudes
Spectrum.normalize(&mags)                        // scale to [0, 1]
let db   = Spectrum.toDecibels(mags)            // dB with –120 dB floor
let phi  = Spectrum.phase(complexOutput)        // per-bin phase
let rms  = Spectrum.rms(samples)
let quiet = Spectrum.isNoise(mags, sampleRate: 44100, fftSize: 4096)

// Single dominant peak
let peak = PeakDetection.peak(in: mags)
let f0   = PeakDetection.fundamentalFrequency(
    magnitudes: mags, sampleRate: 44100, fftSize: 4096,
    minFreq: 80, maxFreq: 1200
)
let refined = PeakDetection.parabolicInterpolation(
    magnitudes: mags, peakIndex: f0!.index, sampleRate: 44100, fftSize: 4096
)

// Top N peaks (for chord detection, harmonic analysis, visual displays)
let peaks = PeakDetection.topPeaks(in: mags, count: 5, minSpacing: 10)
```

---

### Frequency Bands — `FrequencyBands` & `OctaveBands`

```swift
// Named perceptual bands — ideal for audio-reactive visuals and games
let bands  = FrequencyBands(sampleRate: 44100, fftSize: 4096)
let energy = bands.analyze(mags)         // or analyzeNormalized for [0,1] values

energy.sub    // 20–60 Hz   — rumble, kick fundamental
energy.bass   // 60–250 Hz  — bass guitar, kick body
energy.low    // 250–500 Hz — lower mids
energy.mid    // 500–2 kHz  — vocals, guitars
energy.upper  // 2–4 kHz    — presence, attack
energy.air    // 4–20 kHz   — cymbals, sibilance, sheen

// ISO 266 standard 1/3-octave bands (31 bands, 16 Hz–16 kHz)
let octave = OctaveBands.analyze(magnitudes: mags, sampleRate: 44100, fftSize: 4096)
// → [(center: Double, energy: Float)]
```

---

### Pitch Recognition — `Pitch` & `FrequencyTracker`

```swift
// Note lookup
if let note = Pitch.note(frequency: 440.0) { print("\(note.name)\(note.octave)") }  // A4
let cents = Pitch.centsDeviation(frequency: 441.5)   // +5.9 cents

// MIDI
let midi = Pitch.midiNote(frequency: 440.0)          // 69
let freq = Pitch.frequency(midiNote: 69)             // 440.0 Hz

// Stateful real-time smoothing with harmonic-jump correction
let tracker = FrequencyTracker(smoothingWindow: 5)
let smooth  = tracker.track(rawFrequency)
tracker.reset()
```

---

### Convolution — `Convolver`

FFT-based overlap-add FIR convolution. Pre-computes the kernel spectrum at init — efficient for repeated use with a fixed filter (reverb IR, EQ, matched filter).

```swift
let fftSize  = Convolver.recommendedFFTSize(forKernelCount: impulseResponse.count)!
let convolver = try Convolver(kernel: impulseResponse, fftSize: fftSize)
let wet       = try convolver.apply(to: drySignal)
// output length = drySignal.count + impulseResponse.count - 1
```

---

### Correlation — `Correlator`

GPU-accelerated circular cross-correlation and autocorrelation via FFT.

```swift
let correlator = try Correlator(fftSize: 4096)

let acorr = try correlator.auto(signal)        // peaks at periodic lags
let xcorr = try correlator.cross(signalA, signalB)  // time-delay estimation, similarity
```

---

### Time-Frequency — `STFT`

Sliding-window FFT over a long signal. Returns one frame per hop.

```swift
let stft = try STFT(fftSize: 2048, hopSize: 512, window: .hann, sampleRate: 44100)

let frames = try stft.analyze(signal)   // [STFTFrame]
frames[0].magnitudes                    // [Float]
frames[0].magnitudesDB                  // [Float] in dB
frames[0].phase                         // [Float] in radians

let spectrogram = try stft.spectrogram(signal)  // [[Float]] time × freq in dB

stft.binFrequency(10)    // Hz for bin 10
stft.frameTime(3)        // seconds for frame 3
```

---

### Beat Detection — `OnsetDetector`

Streaming onset detector using half-wave rectified spectral flux with an adaptive threshold. Feed short audio buffers in real time.

```swift
let detector = try OnsetDetector(sampleRate: 44100, fftSize: 1024, hopSize: 256)

try detector.feed(audioBuffer)
print(detector.isPeak)    // Bool — onset detected in this buffer
print(detector.onsets)    // [Double] — timestamps in seconds (cumulative)

detector.reset()
```

---

### Power Spectral Density — `PSD`

```swift
// Welch's method — averaged overlapping periodograms
let psd = try PSD.welch(
    signal: signal, fftSize: 2048, hopSize: 512,
    sampleRate: 44100, window: .hann
)

// Magnitude-squared coherence between two signals: [0, 1] per bin
let coh = try PSD.coherence(
    a: signalA, b: signalB,
    fftSize: 2048, hopSize: 512, sampleRate: 44100
)
```

---

## Performance

Underlying Metal kernels were benchmarked on Apple M1 at N=4096, batch=256:

| Kernel | GFLOPS | vs vDSP |
|---|---|---|
| Radix-8 Stockham (best) | 138.45 | +29% |
| CT DIF + simdgroup MMA | 128.0 | +20% |
| Radix-4 Stockham | 113.6 | +6% |
| Apple vDSP (baseline) | 107.0 | — |

## Credits

Based on original research and Metal kernel implementations by **Mohamed Amine Bergach** (mbergach@illumina.com):

- *Beating vDSP: A 138 GFLOPS Radix-8 Stockham FFT on Apple Silicon via Two-Tier Register-Threadgroup Memory Decomposition*, 2026
- *From 8 Seconds to 370 ms: Kernel-Fused SAR Imaging on Apple Silicon via Single-Dispatch FFT Pipelines*, 2026
- *Quaternion Spectral Fingerprinting of DNA: GPU-Accelerated Multi-Channel Fourier Analysis for Alignment-Free Genomics*, 2026

## License

MIT License. See [LICENSE](LICENSE) for details.
