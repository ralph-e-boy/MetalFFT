import Foundation
import DNALib

// ============================================================================
// Biological Validation: Per-Window Multilevel Features vs Gene Annotations
//
// The critical test: does B3 AT coherence (which the null model showed is
// depleted in real genomes vs shuffled) actually track coding regions?
//
// For S. cerevisiae:
// 1. Parse GFF3 gene annotations → mark each base as coding or non-coding
// 2. Compute per-window multilevel spectral features
// 3. Classify each window by coding fraction (0-100%)
// 4. Compare per-band coherence between coding and non-coding windows
// 5. Report effect sizes and statistical significance
// ============================================================================

struct GFFFeature {
    let seqid: String
    let featureType: String
    let start: Int  // 1-based
    let end: Int    // 1-based, inclusive
    let strand: Character
}

/// Parse CDS features from a GFF3 file
func parseCDS(gffPath: String) throws -> [GFFFeature] {
    let content = try String(contentsOfFile: gffPath, encoding: .utf8)
    var features = [GFFFeature]()

    for line in content.split(separator: "\n") {
        if line.hasPrefix("#") { continue }
        let fields = line.split(separator: "\t", omittingEmptySubsequences: false)
        guard fields.count >= 9 else { continue }

        let featureType = String(fields[2])
        guard featureType == "CDS" else { continue }

        guard let start = Int(fields[3]), let end = Int(fields[4]) else { continue }
        let strand = fields[6].first ?? "+"

        features.append(GFFFeature(
            seqid: String(fields[0]),
            featureType: featureType,
            start: start,
            end: end,
            strand: strand
        ))
    }
    return features
}

/// Build per-base coding mask for a set of sequences
func buildCodingMask(sequences: [FASTASequence], cdsFeatures: [GFFFeature]) -> [Bool] {
    // Map seqid → offset in concatenated sequence
    var seqOffsets = [String: Int]()
    var offset = 0
    for seq in sequences {
        // Extract accession from header (first word)
        let accession = String(seq.header.split(separator: " ").first ?? "")
        seqOffsets[accession] = offset
        offset += seq.sequence.filter { $0 <= 3 }.count
    }

    let totalLen = offset
    var mask = [Bool](repeating: false, count: totalLen)

    var mapped = 0
    for cds in cdsFeatures {
        guard let seqOffset = seqOffsets[cds.seqid] else { continue }
        let start = seqOffset + cds.start - 1  // Convert 1-based to 0-based
        let end = seqOffset + cds.end - 1
        guard start >= 0 && end < totalLen else { continue }
        for i in start...end {
            mask[i] = true
        }
        mapped += 1
    }

    let codingBases = mask.filter { $0 }.count
    print("  Coding mask: \(mapped) CDS features mapped, \(codingBases) coding bases (\(String(format: "%.1f%%", Double(codingBases) / Double(totalLen) * 100)))")

    return mask
}

func main() throws {
    print(String(repeating: "=", count: 72))
    print("BIOLOGICAL VALIDATION: Multilevel Spectral Features vs Gene Annotations")
    print(String(repeating: "=", count: 72))

    let analyzer = try DNASpectralAnalyzer()

    // Find data directory
    let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
    let searchDirs = [
        "\(execDir)/../data/genomes", "\(execDir)/../../data/genomes",
        "\(execDir)/../../../data/genomes", "\(execDir)/../../../../data/genomes",
        "\(execDir)/../../../../../data/genomes",
        "\(FileManager.default.currentDirectoryPath)/data/genomes",
    ]
    var genomesDir = ""
    for dir in searchDirs {
        let path = (dir as NSString).standardizingPath
        if FileManager.default.fileExists(atPath: path + "/s_cerevisiae.fasta") {
            genomesDir = path
            break
        }
    }

    let fastaPath = "\(genomesDir)/s_cerevisiae.fasta"
    let gffPath = "\(genomesDir)/s_cerevisiae_R64.gff"

    guard FileManager.default.fileExists(atPath: fastaPath),
          FileManager.default.fileExists(atPath: gffPath) else {
        print("ERROR: Need s_cerevisiae.fasta and s_cerevisiae_R64.gff in data/genomes/")
        return
    }

    // 1. Read genome
    print("\n1. Reading S. cerevisiae genome...")
    let sequences = try readFASTA(path: fastaPath)
    var allDNA = [UInt8]()
    for seq in sequences {
        for base in seq.sequence where base <= 3 {
            allDNA.append(base)
        }
    }
    print("   Total: \(allDNA.count) bp across \(sequences.count) chromosomes")

    // 2. Parse gene annotations
    print("\n2. Parsing CDS annotations...")
    let cdsFeatures = try parseCDS(gffPath: gffPath)
    print("   Found \(cdsFeatures.count) CDS features")

    // 3. Build coding mask
    print("\n3. Building coding mask...")
    let codingMask = buildCodingMask(sequences: sequences, cdsFeatures: cdsFeatures)

    // 4. Compute per-window multilevel features
    print("\n4. Computing per-window multilevel spectral features...")
    let result = analyzer.computeMultilevelSpectral(dna: allDNA)

    let N = 1024
    let hopSize = 512
    let bands = standardGenomicBands
    let pairNames = ["AT", "AG", "AC", "TG", "TC", "GC"]

    // 5. Classify each window by coding fraction
    print("\n5. Classifying windows by coding fraction...")

    struct WindowStats {
        var codingFraction: Float
        var bandCoherence: [[Float]]  // [band][pair]
    }

    var windowStats = [WindowStats]()
    for w in 0..<result.numWindows {
        let startPos = w * hopSize
        let endPos = min(startPos + N, codingMask.count)
        guard endPos <= codingMask.count else { continue }

        var codingCount = 0
        for i in startPos..<endPos {
            if codingMask[i] { codingCount += 1 }
        }
        let codingFrac = Float(codingCount) / Float(endPos - startPos)

        let bandCoh = result.windowResults[w].bandFeatures.map { $0.coherence }
        windowStats.append(WindowStats(codingFraction: codingFrac, bandCoherence: bandCoh))
    }

    // 6. Compare coding vs non-coding windows
    print("\n6. Comparing coding (>70% CDS) vs non-coding (<10% CDS) windows...")

    let codingWindows = windowStats.filter { $0.codingFraction > 0.7 }
    let noncodingWindows = windowStats.filter { $0.codingFraction < 0.1 }

    print("   Coding windows (>70% CDS): \(codingWindows.count)")
    print("   Non-coding windows (<10% CDS): \(noncodingWindows.count)")

    print("\n" + String(repeating: "=", count: 72))
    print("RESULTS: Per-Band Coherence in Coding vs Non-Coding Regions")
    print(String(repeating: "=", count: 72))

    var reportLines = [String]()
    reportLines.append("# Biological Validation: Coding vs Non-Coding Multilevel Coherence")
    reportLines.append("# S. cerevisiae S288C, N=1024 windows, 50% overlap")
    reportLines.append("# Coding: >70% CDS overlap, Non-coding: <10% CDS overlap")
    reportLines.append("")

    let header = "Band".padding(toLength: 8, withPad: " ", startingAt: 0) +
        "Pair".padding(toLength: 6, withPad: " ", startingAt: 0) +
        "Coding".padding(toLength: 10, withPad: " ", startingAt: 0) +
        "NonCod".padding(toLength: 10, withPad: " ", startingAt: 0) +
        "Delta".padding(toLength: 10, withPad: " ", startingAt: 0) +
        "Cohen_d".padding(toLength: 10, withPad: " ", startingAt: 0) +
        "Interpretation"
    print(header)
    print(String(repeating: "-", count: 90))

    reportLines.append("| Band | Pair | Coding | Non-coding | Delta | Cohen's d | Significance |")
    reportLines.append("|------|------|--------|-----------|-------|-----------|-------------|")

    // TSV output
    var tsvLines = ["band\tpair\tmean_coding\tmean_noncoding\tdelta\tcohen_d\tstd_coding\tstd_noncoding"]

    for (bIdx, band) in bands.enumerated() {
        for pIdx in 0..<6 {
            // Collect values for coding and non-coding
            let codingVals = codingWindows.map { $0.bandCoherence[bIdx][pIdx] }
            let noncodingVals = noncodingWindows.map { $0.bandCoherence[bIdx][pIdx] }

            guard !codingVals.isEmpty && !noncodingVals.isEmpty else { continue }

            let meanCoding = codingVals.reduce(0, +) / Float(codingVals.count)
            let meanNoncoding = noncodingVals.reduce(0, +) / Float(noncodingVals.count)

            var varCoding: Float = 0
            for v in codingVals { varCoding += (v - meanCoding) * (v - meanCoding) }
            varCoding /= Float(max(codingVals.count - 1, 1))
            let stdCoding = sqrt(varCoding)

            var varNoncoding: Float = 0
            for v in noncodingVals { varNoncoding += (v - meanNoncoding) * (v - meanNoncoding) }
            varNoncoding /= Float(max(noncodingVals.count - 1, 1))
            let stdNoncoding = sqrt(varNoncoding)

            let delta = meanCoding - meanNoncoding
            let pooledStd = sqrt((varCoding + varNoncoding) / 2)
            let cohenD = pooledStd > 1e-6 ? delta / pooledStd : 0

            let sig: String
            if abs(cohenD) > 0.8 { sig = "LARGE ***" }
            else if abs(cohenD) > 0.5 { sig = "MEDIUM **" }
            else if abs(cohenD) > 0.2 { sig = "SMALL *" }
            else { sig = "negligible" }

            let direction = delta > 0 ? "coding>" : "coding<"

            let line = band.shortName.padding(toLength: 8, withPad: " ", startingAt: 0) +
                pairNames[pIdx].padding(toLength: 6, withPad: " ", startingAt: 0) +
                String(format: "%.4f", meanCoding).padding(toLength: 10, withPad: " ", startingAt: 0) +
                String(format: "%.4f", meanNoncoding).padding(toLength: 10, withPad: " ", startingAt: 0) +
                String(format: "%+.4f", delta).padding(toLength: 10, withPad: " ", startingAt: 0) +
                String(format: "%+.3f", cohenD).padding(toLength: 10, withPad: " ", startingAt: 0) +
                sig
            print(line)

            reportLines.append("| \(band.shortName) | \(pairNames[pIdx]) | \(String(format: "%.4f", meanCoding)) | \(String(format: "%.4f", meanNoncoding)) | \(String(format: "%+.4f", delta)) | \(String(format: "%+.3f", cohenD)) | \(sig) |")

            tsvLines.append("\(band.shortName)\t\(pairNames[pIdx])\t\(String(format: "%.6f", meanCoding))\t\(String(format: "%.6f", meanNoncoding))\t\(String(format: "%.6f", delta))\t\(String(format: "%.4f", cohenD))\t\(String(format: "%.6f", stdCoding))\t\(String(format: "%.6f", stdNoncoding))")
        }
        print(String(repeating: "-", count: 90))
    }

    // 7. Coding fraction correlation
    print("\n\nCODING FRACTION CORRELATION (Pearson r)")
    print("Band   Pair    r(coding_frac, coherence)")
    print(String(repeating: "-", count: 50))

    reportLines.append("")
    reportLines.append("## Continuous Correlation: Coding Fraction vs Coherence")
    reportLines.append("| Band | Pair | Pearson r |")
    reportLines.append("|------|------|-----------|")

    for (bIdx, band) in bands.enumerated() {
        for pIdx in 0..<6 {
            let xs = windowStats.map { Double($0.codingFraction) }
            let ys = windowStats.map { Double($0.bandCoherence[bIdx][pIdx]) }

            let n = Double(xs.count)
            let meanX = xs.reduce(0, +) / n
            let meanY = ys.reduce(0, +) / n

            var sumXY: Double = 0, sumX2: Double = 0, sumY2: Double = 0
            for i in 0..<xs.count {
                let dx = xs[i] - meanX
                let dy = ys[i] - meanY
                sumXY += dx * dy
                sumX2 += dx * dx
                sumY2 += dy * dy
            }
            let denom = sqrt(sumX2 * sumY2)
            let r = denom > 1e-30 ? sumXY / denom : 0

            if abs(r) > 0.05 {
                let sig = abs(r) > 0.2 ? "***" : abs(r) > 0.1 ? "**" : "*"
                print("  \(band.shortName.padding(toLength: 6, withPad: " ", startingAt: 0)) \(pairNames[pIdx].padding(toLength: 6, withPad: " ", startingAt: 0))  r = \(String(format: "%+.4f", r)) \(sig)")
                reportLines.append("| \(band.shortName) | \(pairNames[pIdx]) | \(String(format: "%+.4f", r)) |")
            }
        }
    }

    // Write outputs
    let reportPath = "\(genomesDir)/biological_validation_report.md"
    try reportLines.joined(separator: "\n").write(toFile: reportPath, atomically: true, encoding: .utf8)
    print("\nReport: \(reportPath)")

    let tsvPath = "\(genomesDir)/biological_validation.tsv"
    try tsvLines.joined(separator: "\n").write(toFile: tsvPath, atomically: true, encoding: .utf8)
    print("TSV: \(tsvPath)")

    print("\n" + String(repeating: "=", count: 72))
    print("INTERPRETATION:")
    print("  Cohen's d > 0.8: LARGE effect — this band clearly separates coding/non-coding")
    print("  Cohen's d 0.5-0.8: MEDIUM effect — meaningful but overlapping distributions")
    print("  Cohen's d 0.2-0.5: SMALL effect — detectable but weak separation")
    print("  Cohen's d < 0.2: negligible — band does not distinguish coding/non-coding")
    print(String(repeating: "=", count: 72))
}

try main()
