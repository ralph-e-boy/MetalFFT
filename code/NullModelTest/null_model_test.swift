import Foundation
import DNALib

// ============================================================================
// Null Model Test: Real Genome vs Dinucleotide-Preserving Shuffle
//
// The critical experiment: does multilevel spectral coherence contain
// information beyond what dinucleotide composition alone explains?
//
// For each genome:
//   1. Compute multilevel spectral features on real sequence
//   2. Generate N shuffled sequences (preserving dinucleotide frequencies)
//   3. Compute multilevel spectral features on each shuffle
//   4. Compare: z-score = (real - mean_shuffle) / std_shuffle
//
// z > 3: genuine structural signal beyond dinucleotide composition
// z ~ 0: explained entirely by dinucleotide frequencies
// z < -3: real genome has LESS coherence than expected (depletion signal)
// ============================================================================

struct GenomeSpec {
    let name: String
    let path: String
    let domain: String
}

func findGenomesDir() -> String {
    let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
    let searchDirs = [
        "\(execDir)/../data/genomes",
        "\(execDir)/../../data/genomes",
        "\(execDir)/../../../data/genomes",
        "\(execDir)/../../../../data/genomes",
        "\(execDir)/../../../../../data/genomes",
        "\(FileManager.default.currentDirectoryPath)/data/genomes",
    ]
    for dir in searchDirs {
        let path = (dir as NSString).standardizingPath
        if FileManager.default.fileExists(atPath: path + "/s_cerevisiae.fasta") {
            return path
        }
    }
    return FileManager.default.currentDirectoryPath
}

func main() throws {
    print(String(repeating: "=", count: 72))
    print("NULL MODEL TEST: Real Genome vs Dinucleotide-Preserving Shuffle")
    print("Testing whether multilevel spectral coherence contains signal")
    print("beyond what dinucleotide composition alone explains.")
    print(String(repeating: "=", count: 72))

    let analyzer = try DNASpectralAnalyzer()
    let genomesDir = findGenomesDir()
    let numShuffles = 5  // 5 shuffles per genome (enough for z-scores)

    // Representative genomes spanning the diversity
    let genomes: [GenomeSpec] = [
        GenomeSpec(name: "B. subtilis", path: "\(genomesDir)/b_subtilis.fasta", domain: "Bacteria"),
        GenomeSpec(name: "M. tuberculosis", path: "\(genomesDir)/m_tuberculosis.fasta", domain: "Bacteria"),
        GenomeSpec(name: "T. thermophilus", path: "\(genomesDir)/t_thermophilus.fasta", domain: "Bacteria"),
        GenomeSpec(name: "M. jannaschii", path: "\(genomesDir)/m_jannaschii.fasta", domain: "Archaea"),
        GenomeSpec(name: "H. salinarum", path: "\(genomesDir)/h_salinarum.fasta", domain: "Archaea"),
        GenomeSpec(name: "S. cerevisiae", path: "\(genomesDir)/s_cerevisiae.fasta", domain: "Eukarya"),
        GenomeSpec(name: "Drosophila chr2L", path: "\(genomesDir)/drosophila_chr2L.fna", domain: "Eukarya"),
        GenomeSpec(name: "C. elegans chrI", path: "\(genomesDir)/c_elegans_chrI.fasta", domain: "Eukarya"),
        GenomeSpec(name: "P. falciparum chr13", path: "\(genomesDir)/p_falciparum_chr13.fasta", domain: "Eukarya"),
    ]

    let pairNames = ["AT", "AG", "AC", "TG", "TC", "GC"]
    let bands = standardGenomicBands

    var allResults = [String]()
    allResults.append("# Null Model Test Results")
    allResults.append("# z-score = (real - mean_shuffle) / std_shuffle")
    allResults.append("# |z| > 3: genuine signal beyond dinucleotide composition")
    allResults.append("# Shuffles: \(numShuffles) per genome, dinucleotide-preserving")
    allResults.append("")

    // Summary TSV
    var summaryLines = [String]()
    summaryLines.append("organism\tdomain\tband\tpair\treal_coh\tmean_shuffle\tstd_shuffle\tz_score\tsignificant")

    for genome in genomes {
        guard FileManager.default.fileExists(atPath: genome.path) else {
            print("SKIP: \(genome.name) (file not found)")
            continue
        }

        print("\n" + String(repeating: "━", count: 72))
        print("Testing: \(genome.name) [\(genome.domain)]")

        // Read genome
        let sequences = try readFASTA(path: genome.path)
        var realDNA = [UInt8]()
        for seq in sequences {
            for base in seq.sequence where base <= 3 {
                realDNA.append(base)
            }
        }
        print("  Genome: \(realDNA.count) bp")

        // Quick dinucleotide frequency report
        var dinucCounts = [Int](repeating: 0, count: 16)
        for i in 0..<(realDNA.count - 1) {
            dinucCounts[Int(realDNA[i]) * 4 + Int(realDNA[i+1])] += 1
        }
        let totalDinuc = dinucCounts.reduce(0, +)
        let nucNames = ["A", "T", "G", "C"]
        print("  Top dinucleotides:")
        let sortedDinuc = dinucCounts.enumerated().sorted { $0.element > $1.element }
        for (idx, count) in sortedDinuc.prefix(4) {
            let d1 = nucNames[idx / 4]
            let d2 = nucNames[idx % 4]
            print("    \(d1)\(d2): \(String(format: "%.2f%%", Double(count) / Double(totalDinuc) * 100))")
        }

        // 1. Real genome multilevel analysis
        print("  Computing real genome multilevel spectral features...")
        let realResult = analyzer.computeMultilevelSpectral(dna: realDNA)

        // 2. Shuffled genome multilevel analyses
        var shuffleCoherences = [[[Float]]](
            repeating: [[Float]](repeating: [Float](repeating: 0, count: 6), count: bands.count),
            count: numShuffles
        )

        for s in 0..<numShuffles {
            print("  Shuffle \(s + 1)/\(numShuffles)...")
            var rng = SeededRNG(seed: UInt64(42 + s * 137))
            let shuffledDNA = dinucleotideShuffle(dna: realDNA, rng: &rng)

            // Verify dinucleotide preservation
            if s == 0 {
                let preserved = verifyDinucleotidePreservation(realDNA, shuffledDNA)
                print("    Dinucleotide preservation verified: \(preserved)")
            }

            let shuffResult = analyzer.computeMultilevelSpectral(dna: shuffledDNA)
            for b in 0..<bands.count {
                shuffleCoherences[s][b] = shuffResult.avgBandCoherence[b]
            }
        }

        // 3. Compute z-scores
        print("\n  --- Z-SCORES (real vs shuffled) ---")

        allResults.append("## \(genome.name) [\(genome.domain)]")
        allResults.append("")

        var headerLine = "  Band         "
        for name in pairNames { headerLine += "  \(name.padding(toLength: 8, withPad: " ", startingAt: 0))" }
        print(headerLine)
        print("  " + String(repeating: "-", count: 66))

        allResults.append("| Band | " + pairNames.joined(separator: " | ") + " |")
        allResults.append("|------|" + pairNames.map { _ in "------|" }.joined())

        for (bIdx, band) in bands.enumerated() {
            var line = "  \(band.shortName.padding(toLength: 13, withPad: " ", startingAt: 0))"
            var mdLine = "| \(band.shortName) |"

            for p in 0..<6 {
                let realVal = realResult.avgBandCoherence[bIdx][p]

                // Mean and std of shuffle values
                var shuffVals = [Float]()
                for s in 0..<numShuffles {
                    shuffVals.append(shuffleCoherences[s][bIdx][p])
                }
                let mean = shuffVals.reduce(0, +) / Float(numShuffles)
                var variance: Float = 0
                for v in shuffVals { variance += (v - mean) * (v - mean) }
                let std = sqrt(variance / Float(max(numShuffles - 1, 1)))

                let z = std > 1e-6 ? (realVal - mean) / std : 0

                let marker: String
                if abs(z) > 5 { marker = "***" }
                else if abs(z) > 3 { marker = "** " }
                else if abs(z) > 2 { marker = "*  " }
                else { marker = "   " }

                line += String(format: "  %+6.1f%@", z, marker as NSString)
                mdLine += " \(String(format: "%+.1f", z)) |"

                let sig = abs(z) > 3 ? "YES" : "no"
                summaryLines.append("\(genome.name)\t\(genome.domain)\t\(band.shortName)\t\(pairNames[p])\t\(String(format: "%.6f", realVal))\t\(String(format: "%.6f", mean))\t\(String(format: "%.6f", std))\t\(String(format: "%.2f", z))\t\(sig)")
            }
            print(line)
            allResults.append(mdLine)
        }

        // Highlight significant findings
        print("\n  Key findings:")
        var foundSignificant = false
        for (bIdx, band) in bands.enumerated() {
            for p in 0..<6 {
                let realVal = realResult.avgBandCoherence[bIdx][p]
                var shuffVals = [Float]()
                for s in 0..<numShuffles { shuffVals.append(shuffleCoherences[s][bIdx][p]) }
                let mean = shuffVals.reduce(0, +) / Float(numShuffles)
                var variance: Float = 0
                for v in shuffVals { variance += (v - mean) * (v - mean) }
                let std = sqrt(variance / Float(max(numShuffles - 1, 1)))
                let z = std > 1e-6 ? (realVal - mean) / std : 0

                if abs(z) > 3 {
                    let direction = z > 0 ? "ENRICHED" : "DEPLETED"
                    print("    \(band.shortName) \(pairNames[p]): z=\(String(format: "%+.1f", z)) (\(direction), real=\(String(format: "%.4f", realVal)) vs shuffle=\(String(format: "%.4f", mean)))")
                    foundSignificant = true
                }
            }
        }
        if !foundSignificant {
            print("    NO significant differences found — coherence fully explained by dinucleotide composition")
        }

        allResults.append("")
        allResults.append("---")
        allResults.append("")
    }

    // Write results
    let outDir = genomesDir
    let summaryPath = "\(outDir)/null_model_summary.tsv"
    try summaryLines.joined(separator: "\n").write(toFile: summaryPath, atomically: true, encoding: .utf8)
    print("\n\nSummary TSV: \(summaryPath)")

    let reportPath = "\(outDir)/null_model_report.md"
    try allResults.joined(separator: "\n").write(toFile: reportPath, atomically: true, encoding: .utf8)
    print("Full report: \(reportPath)")

    print("\n" + String(repeating: "=", count: 72))
    print("INTERPRETATION GUIDE:")
    print("  |z| > 3: GENUINE signal beyond dinucleotide composition")
    print("  |z| ~ 0: Explained by dinucleotide frequencies alone")
    print("  z > 0: Real genome has MORE coherence than shuffled")
    print("  z < 0: Real genome has LESS coherence than shuffled")
    print(String(repeating: "=", count: 72))
}

try main()
