import Foundation
import Metal

// ============================================================================
// Multilevel Spectral Filtering of Genomic Sequences
//
// Decomposes the 4-channel DNA cross-spectral analysis into biologically-
// motivated frequency bands, revealing scale-dependent inter-nucleotide
// coupling. Inspired by SAR radar's Range-Doppler multilevel processing.
//
// Bands:
//   B0 (long-range):   periods > 102 bp   — compositional domains
//   B1 (gene-scale):   periods 20-102 bp  — regulatory/structural elements
//   B2 (structural):   periods 8-20 bp    — helical repeat, nucleosomes
//   B3 (coding):       periods 2.8-8 bp   — codon periodicity, splice sites
//   B4 (dinucleotide): periods 2.0-2.8 bp — CpG, purine-pyrimidine patterns
//   B5 (broadband):    all frequencies     — full-spectrum reference
//
// For each band at each genomic window:
//   - 6 pairwise coherences (AT, AG, AC, TG, TC, GC)
//   - 6 phases
//   - 4 channel powers
//   - spectral entropy within band
//   - condition number within band
// ============================================================================

/// Definition of a frequency band for multilevel analysis
public struct SpectralBand {
    public let name: String
    public let shortName: String
    public let binLow: Int // lowest frequency bin (inclusive)
    public let binHigh: Int // highest frequency bin (inclusive)
    public let description: String

    public init(name: String, shortName: String, binLow: Int, binHigh: Int, description: String) {
        self.name = name
        self.shortName = shortName
        self.binLow = binLow
        self.binHigh = binHigh
        self.description = description
    }

    /// Period range in base pairs for N=1024 FFT
    public var periodRange: (low: Float, high: Float) {
        let high = binLow > 0 ? 1024.0 / Float(binLow) : Float.infinity
        let low = 1024.0 / Float(binHigh)
        return (low, high)
    }

    /// Number of frequency bins in this band
    public var numBins: Int {
        binHigh - binLow + 1
    }
}

/// Standard biologically-motivated bands for N=1024 FFT
public let standardGenomicBands: [SpectralBand] = [
    SpectralBand(
        name: "Long-range", shortName: "B0",
        binLow: 1, binHigh: 10,
        description: "Periods >102 bp: compositional domains, gene boundaries"
    ),
    SpectralBand(
        name: "Gene-scale", shortName: "B1",
        binLow: 10, binHigh: 51,
        description: "Periods 20-102 bp: regulatory elements, repeat units"
    ),
    SpectralBand(
        name: "Structural", shortName: "B2",
        binLow: 51, binHigh: 128,
        description: "Periods 8-20 bp: helical repeat (~10.5 bp), nucleosome signals"
    ),
    SpectralBand(
        name: "Coding", shortName: "B3",
        binLow: 128, binHigh: 366,
        description: "Periods 2.8-8 bp: codon (period-3), splice sites"
    ),
    SpectralBand(
        name: "Dinucleotide", shortName: "B4",
        binLow: 366, binHigh: 512,
        description: "Periods 2.0-2.8 bp: CpG, purine-pyrimidine alternation"
    ),
    SpectralBand(
        name: "Broadband", shortName: "B5",
        binLow: 1, binHigh: 512,
        description: "All frequencies: full-spectrum reference"
    )
]

/// Per-band coherence features for a single genomic window
public struct BandFeatures {
    public let coherence: [Float] // 6 pairwise coherences
    public let phase: [Float] // 6 pairwise phases
    public let power: [Float] // 4 channel powers (band-integrated)
    public let entropy: Float // spectral entropy within band
    public let conditionNumber: Float // cross-spectral matrix condition number
}

/// Result of multilevel spectral analysis for one window
public struct MultilevelWindowResult {
    public let position: Int // start position in genome
    public let bandFeatures: [BandFeatures] // one per band
}

/// Full multilevel spectral analysis result
public struct MultilevelSpectralResult {
    public let bands: [SpectralBand]
    public let windowSize: Int
    public let hopSize: Int
    public let numWindows: Int
    public let windowResults: [MultilevelWindowResult]

    // Genome-wide averaged per-band coherence
    public let avgBandCoherence: [[Float]] // [bandIdx][6 pairs]
    public let avgBandPhase: [[Float]] // [bandIdx][6 pairs]

    /// Cross-band correlation matrix
    /// crossBandCorrelation[b1][b2][pair] = correlation of coherence between bands b1 and b2
    public let crossBandCorrelation: [[[Float]]] // [band1][band2][6 pairs]
}

// ============================================================================
// Multilevel Spectral Analyzer
// ============================================================================

public extension DNASpectralAnalyzer {
    /// Compute position-resolved multilevel spectral features.
    ///
    /// For each overlapping N=1024 window, computes 4-channel FFT, then
    /// extracts per-band cross-spectral coherence features.
    ///
    /// - Parameters:
    ///   - dna: Encoded DNA (0=A, 1=T, 2=G, 3=C)
    ///   - bands: Frequency bands to analyze (default: standard genomic bands)
    ///   - overlap: Fractional overlap between windows (default: 0.5)
    func computeMultilevelSpectral(
        dna: [UInt8],
        bands: [SpectralBand] = standardGenomicBands,
        overlap: Float = 0.5
    ) -> MultilevelSpectralResult {
        let N = 1024
        let numFreqs = N / 2 + 1 // 513
        let hopSize = Int(Float(N) * (1.0 - overlap))
        let numWindows = max(0, (dna.count - N) / hopSize + 1)
        let numBands = bands.count
        print("Multilevel spectral analysis:")
        print("  Windows: \(numWindows), N=\(N), hop=\(hopSize)")
        print("  Bands: \(bands.map { $0.shortName }.joined(separator: ", "))")
        for band in bands {
            let (pLo, pHi) = band.periodRange
            print("    \(band.shortName) [\(band.name)]: bins \(band.binLow)-\(band.binHigh), periods \(String(format: "%.1f", pLo))-\(pHi.isInfinite ? "inf" : String(format: "%.1f", pHi)) bp")
        }

        // Per-window, per-band accumulators for genome-wide averages
        var sumBandCoherence = [[Double]](repeating: [Double](repeating: 0, count: 6), count: numBands)
        var sumBandPhaseReal = [[Double]](repeating: [Double](repeating: 0, count: 6), count: numBands)
        var sumBandPhaseImag = [[Double]](repeating: [Double](repeating: 0, count: 6), count: numBands)

        // Per-window band coherence storage for cross-band correlation
        var allWindowBandCoh = [[Float]](repeating: [Float](repeating: 0, count: 6), count: numWindows * numBands)

        var windowResults = [MultilevelWindowResult]()
        windowResults.reserveCapacity(numWindows)

        let pairs: [(Int, Int)] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        for w in 0 ..< numWindows {
            let offset = w * hopSize
            let segmentDNA = Array(dna[offset ..< (offset + N)])

            // GPU: 4-channel FFT
            let spectra = runFFT(dna: segmentDNA)
            // spectra: [U_A(0..N-1), U_T(0..N-1), U_G(0..N-1), U_C(0..N-1)] as SIMD2<Float>

            // For each band, compute coherence from this window's spectrum
            var bandFeatures = [BandFeatures]()

            for (bIdx, band) in bands.enumerated() {
                // Accumulate auto-spectra and cross-spectra within band
                var bandAutoP = [Double](repeating: 0, count: 4)
                var bandCrossReal = [Double](repeating: 0, count: 6)
                var bandCrossImag = [Double](repeating: 0, count: 6)

                let kLo = max(band.binLow, 1)
                let kHi = min(band.binHigh, numFreqs - 1)

                for k in kLo ... kHi {
                    let ua = spectra[k]
                    let ut = spectra[N + k]
                    let ug = spectra[2 * N + k]
                    let uc = spectra[3 * N + k]

                    // Auto-spectra
                    bandAutoP[0] += Double(ua.x * ua.x + ua.y * ua.y)
                    bandAutoP[1] += Double(ut.x * ut.x + ut.y * ut.y)
                    bandAutoP[2] += Double(ug.x * ug.x + ug.y * ug.y)
                    bandAutoP[3] += Double(uc.x * uc.x + uc.y * uc.y)

                    // Cross-spectra
                    let channels: [SIMD2<Float>] = [ua, ut, ug, uc]
                    for (pIdx, (i, j)) in pairs.enumerated() {
                        let x = channels[i]
                        let y = channels[j]
                        bandCrossReal[pIdx] += Double(x.x * y.x + x.y * y.y)
                        bandCrossImag[pIdx] += Double(x.y * y.x - x.x * y.y)
                    }
                }

                // Compute per-band coherence: |<S_XY>|^2 / (<S_XX> * <S_YY>)
                // Here "averaging" is over frequency bins within the band
                let invBins = 1.0 / Double(kHi - kLo + 1)
                var coh = [Float](repeating: 0, count: 6)
                var ph = [Float](repeating: 0, count: 6)

                for (pIdx, (chX, chY)) in pairs.enumerated() {
                    let avgCR = bandCrossReal[pIdx] * invBins
                    let avgCI = bandCrossImag[pIdx] * invBins
                    let avgPX = bandAutoP[chX] * invBins
                    let avgPY = bandAutoP[chY] * invBins

                    let magSq = avgCR * avgCR + avgCI * avgCI
                    let denom = avgPX * avgPY
                    coh[pIdx] = denom > 1e-30 ? Float(magSq / denom) : 0
                    ph[pIdx] = Float(atan2(avgCI, avgCR))
                }

                // Band-integrated channel powers
                let pow = bandAutoP.map { Float($0) }

                // Spectral entropy within band
                let totalP = bandAutoP.reduce(0, +)
                var H: Float = 0
                if totalP > 1e-30 {
                    for p in bandAutoP {
                        let px = p / totalP
                        if px > 1e-30 { H -= Float(px * log(px)) }
                    }
                }

                // Condition number (simplified: ratio of max to min channel power)
                let sortedP = bandAutoP.sorted()
                let cn = sortedP[0] > 1e-30 ? Float(sortedP[3] / sortedP[0]) : Float.infinity

                bandFeatures.append(BandFeatures(
                    coherence: coh, phase: ph, power: pow,
                    entropy: H, conditionNumber: cn
                ))

                // Accumulate for genome-wide averages
                for p in 0 ..< 6 {
                    sumBandCoherence[bIdx][p] += Double(coh[p])
                    sumBandPhaseReal[bIdx][p] += Double(cos(ph[p]))
                    sumBandPhaseImag[bIdx][p] += Double(sin(ph[p]))
                }

                // Store for cross-band correlation
                allWindowBandCoh[w * numBands + bIdx] = coh
            }

            windowResults.append(MultilevelWindowResult(
                position: offset,
                bandFeatures: bandFeatures
            ))

            if (w + 1) % 2000 == 0 || w == numWindows - 1 {
                print("  Processed \(w + 1)/\(numWindows) windows")
            }
        }

        // Compute genome-wide averages
        let invW = 1.0 / Double(max(numWindows, 1))
        var avgBandCoh = [[Float]](repeating: [Float](repeating: 0, count: 6), count: numBands)
        var avgBandPh = [[Float]](repeating: [Float](repeating: 0, count: 6), count: numBands)

        for b in 0 ..< numBands {
            for p in 0 ..< 6 {
                avgBandCoh[b][p] = Float(sumBandCoherence[b][p] * invW)
                avgBandPh[b][p] = Float(atan2(
                    sumBandPhaseImag[b][p] * invW,
                    sumBandPhaseReal[b][p] * invW
                ))
            }
        }

        // Cross-band correlation: for each pair, compute Pearson r between
        // band b1's per-window coherence and band b2's per-window coherence
        var crossBandCorr = [[[Float]]](
            repeating: [[Float]](repeating: [Float](repeating: 0, count: 6), count: numBands),
            count: numBands
        )

        if numWindows > 10 {
            for pIdx in 0 ..< 6 {
                // Extract per-band time series of coherence for this pair
                var series = [[Double]](repeating: [Double](repeating: 0, count: numWindows), count: numBands)
                for w in 0 ..< numWindows {
                    for b in 0 ..< numBands {
                        series[b][w] = Double(allWindowBandCoh[w * numBands + b][pIdx])
                    }
                }

                // Compute means
                var means = [Double](repeating: 0, count: numBands)
                for b in 0 ..< numBands {
                    means[b] = series[b].reduce(0, +) / Double(numWindows)
                }

                // Compute correlations
                for b1 in 0 ..< numBands {
                    for b2 in b1 ..< numBands {
                        var sumXY: Double = 0
                        var sumX2: Double = 0
                        var sumY2: Double = 0
                        for w in 0 ..< numWindows {
                            let dx = series[b1][w] - means[b1]
                            let dy = series[b2][w] - means[b2]
                            sumXY += dx * dy
                            sumX2 += dx * dx
                            sumY2 += dy * dy
                        }
                        let denom = sqrt(sumX2 * sumY2)
                        let r = denom > 1e-30 ? Float(sumXY / denom) : 0
                        crossBandCorr[b1][b2][pIdx] = r
                        crossBandCorr[b2][b1][pIdx] = r
                    }
                }
            }
        }

        return MultilevelSpectralResult(
            bands: bands,
            windowSize: N,
            hopSize: hopSize,
            numWindows: numWindows,
            windowResults: windowResults,
            avgBandCoherence: avgBandCoh,
            avgBandPhase: avgBandPh,
            crossBandCorrelation: crossBandCorr
        )
    }
}

// ============================================================================
// TSV Output for Multilevel Results
// ============================================================================

public extension MultilevelSpectralResult {
    /// Write genome-wide average per-band coherence table
    func writeBandCoherenceTSV(path: String) throws {
        let pairNames = ["AT", "AG", "AC", "TG", "TC", "GC"]
        var header = "band\tname\tbins\tperiod_low\tperiod_high"
        for name in pairNames {
            header += "\tcoh_\(name)"
        }
        for name in pairNames {
            header += "\tphase_\(name)"
        }

        var lines = [header]
        for (bIdx, band) in bands.enumerated() {
            let (pLo, pHi) = band.periodRange
            var line = "\(band.shortName)\t\(band.name)\t\(band.binLow)-\(band.binHigh)"
            line += "\t\(String(format: "%.1f", pLo))\t\(pHi.isInfinite ? "inf" : String(format: "%.1f", pHi))"
            for p in 0 ..< 6 {
                line += "\t\(String(format: "%.6f", avgBandCoherence[bIdx][p]))"
            }
            for p in 0 ..< 6 {
                line += "\t\(String(format: "%.4f", avgBandPhase[bIdx][p]))"
            }
            lines.append(line)
        }
        try lines.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
    }

    /// Write cross-band correlation matrix
    func writeCrossBandCorrelationTSV(path: String) throws {
        let pairNames = ["AT", "AG", "AC", "TG", "TC", "GC"]
        var lines = [String]()

        for (pIdx, pairName) in pairNames.enumerated() {
            lines.append("# Cross-band correlation for \(pairName) coherence")
            var header = "band"
            for band in bands {
                header += "\t\(band.shortName)"
            }
            lines.append(header)

            for (b1, band1) in bands.enumerated() {
                var line = band1.shortName
                for b2 in 0 ..< bands.count {
                    line += "\t\(String(format: "%.4f", crossBandCorrelation[b1][b2][pIdx]))"
                }
                lines.append(line)
            }
            lines.append("")
        }
        try lines.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
    }

    /// Write position-resolved per-band coherence (for spectrogram visualization)
    func writePositionBandTSV(path: String, maxWindows: Int = 50000) throws {
        let pairNames = ["AT", "AG", "AC", "TG", "TC", "GC"]
        var header = "position"
        for band in bands {
            for name in pairNames {
                header += "\t\(band.shortName)_coh_\(name)"
            }
            header += "\t\(band.shortName)_entropy"
        }

        let stride = max(1, numWindows / maxWindows)
        var lines = [header]

        for w in Swift.stride(from: 0, to: numWindows, by: stride) {
            let wr = windowResults[w]
            var line = "\(wr.position)"
            for bf in wr.bandFeatures {
                for p in 0 ..< 6 {
                    line += "\t\(String(format: "%.4f", bf.coherence[p]))"
                }
                line += "\t\(String(format: "%.4f", bf.entropy))"
            }
            lines.append(line)
        }
        try lines.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
    }

    /// Print summary report to stdout
    func printReport(organismName: String) {
        let pairNames = ["AT", "AG", "AC", "TG", "TC", "GC"]

        print("\n" + String(repeating: "=", count: 72))
        print("MULTILEVEL SPECTRAL ANALYSIS: \(organismName)")
        print(String(repeating: "=", count: 72))
        print("Windows: \(numWindows), size: \(windowSize), hop: \(hopSize)")
        print("Genome length: ~\(numWindows * hopSize + windowSize) bp")

        // Per-band coherence summary
        print("\n--- Genome-Wide Average Coherence by Band ---")
        let header = "Band".padding(toLength: 14, withPad: " ", startingAt: 0)
            + pairNames.map { $0.padding(toLength: 8, withPad: " ", startingAt: 0) }.joined()
        print(header)
        print(String(repeating: "-", count: 62))

        for (bIdx, band) in bands.enumerated() {
            let (pLo, pHi) = band.periodRange
            let periodStr = "\(String(format: "%.0f", pLo))-\(pHi.isInfinite ? "inf" : String(format: "%.0f", pHi))"
            var line = "\(band.shortName) (\(periodStr))".padding(toLength: 14, withPad: " ", startingAt: 0)
            for p in 0 ..< 6 {
                let v = avgBandCoherence[bIdx][p]
                line += String(format: "  %6.4f", v)
            }
            print(line)

            // Mark dominant pair
            let maxIdx = avgBandCoherence[bIdx].enumerated().max(by: { $0.element < $1.element })!.offset
            let minIdx = avgBandCoherence[bIdx].enumerated().min(by: { $0.element < $1.element })!.offset
            print("".padding(toLength: 14, withPad: " ", startingAt: 0) + "  Dominant: \(pairNames[maxIdx]) (\(String(format: "%.4f", avgBandCoherence[bIdx][maxIdx]))), Weakest: \(pairNames[minIdx]) (\(String(format: "%.4f", avgBandCoherence[bIdx][minIdx])))")
        }

        // Cross-band correlation highlights
        print("\n--- Cross-Band Correlation Highlights ---")
        print("(Pearson r between per-window coherence of band pairs)")

        for (pIdx, pairName) in pairNames.enumerated() {
            // Find strongest cross-band correlations (excluding self and broadband)
            var correlations: [(String, Float)] = []
            for b1 in 0 ..< (bands.count - 1) { // exclude broadband
                for b2 in (b1 + 1) ..< (bands.count - 1) {
                    let r = crossBandCorrelation[b1][b2][pIdx]
                    correlations.append(("\(bands[b1].shortName)-\(bands[b2].shortName)", r))
                }
            }
            correlations.sort { abs($0.1) > abs($1.1) }

            let top = correlations.prefix(3).map { "\($0.0)=\(String(format: "%.3f", $0.1))" }.joined(separator: ", ")
            print("  \(pairName): \(top)")
        }

        // Band-specific biological interpretations
        print("\n--- Biological Interpretation ---")
        for (bIdx, band) in bands.enumerated() {
            if band.shortName == "B5" { continue } // skip broadband summary
            let coh = avgBandCoherence[bIdx]
            let maxIdx = coh.enumerated().max(by: { $0.element < $1.element })!.offset
            let maxVal = coh[maxIdx]

            var interpretation = ""
            switch band.shortName {
            case "B0":
                if coh[0] > coh[1], coh[0] > 0.1 {
                    interpretation = "A-T dominance at long range → AT-rich domain structure"
                } else if coh[5] > 0.1 {
                    interpretation = "G-C coupling at long range → isochore/GC-content domains"
                } else {
                    interpretation = "Weak long-range coupling → no strong compositional domains"
                }
            case "B1":
                interpretation = "Gene-scale dominant pair: \(pairNames[maxIdx]) → reflects repeat/regulatory structure"
            case "B2":
                if coh[0] > 0.15 {
                    interpretation = "Strong A-T structural coherence → nucleosome positioning signal"
                } else if coh[1] > 0.15 {
                    interpretation = "Strong A-G structural coherence → base-stacking preferences"
                }
                if maxVal < 0.1 {
                    interpretation = "Weak structural coherence → prokaryotic (no nucleosomes?)"
                }
            case "B3":
                if coh[2] > 0.3 || coh[3] > 0.3 {
                    interpretation = "Strong A-C/T-G coding coherence → genetic code signature (universal)"
                } else {
                    interpretation = "Weak coding coherence → low gene density or weak codon bias"
                }
            case "B4":
                if maxVal > 0.1 {
                    interpretation = "Dinucleotide coupling detected → CpG/base-stacking patterns"
                } else {
                    interpretation = "Low dinucleotide coherence → near-random base alternation"
                }
            default: break
            }
            if !interpretation.isEmpty {
                print("  \(band.shortName) [\(band.name)]: \(interpretation)")
            }
        }
    }
}
