import Foundation
import DNALib

// ============================================================================
// Multilevel Spectral Analysis of Genomic DNA
//
// Runs the 6-band multilevel spectral filter bank on multiple genomes,
// computing per-band cross-spectral coherence and cross-band correlations.
//
// Usage: MultilevelAnalysis [genome1.fasta genome2.fasta ...]
//        If no arguments, runs on all available genomes in data/genomes/
// ============================================================================

struct GenomeSpec {
    let name: String
    let path: String
    let domain: String  // Bacteria, Archaea, Eukarya
}

func findGenomes() -> [GenomeSpec] {
    let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
    let searchDirs = [
        "\(execDir)/../data/genomes",
        "\(execDir)/../../data/genomes",
        "\(execDir)/../../../data/genomes",
        "\(execDir)/../../../../data/genomes",
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

    guard !genomesDir.isEmpty else {
        print("ERROR: Could not find data/genomes directory")
        return []
    }

    // Curated list spanning all domains of life
    return [
        // Bacteria
        GenomeSpec(name: "B. subtilis 168", path: "\(genomesDir)/b_subtilis.fasta",
                   domain: "Bacteria"),
        GenomeSpec(name: "M. tuberculosis", path: "\(genomesDir)/m_tuberculosis.fasta",
                   domain: "Bacteria"),
        GenomeSpec(name: "C. crescentus", path: "\(genomesDir)/c_crescentus.fasta",
                   domain: "Bacteria"),
        GenomeSpec(name: "T. thermophilus", path: "\(genomesDir)/t_thermophilus.fasta",
                   domain: "Bacteria"),

        // Archaea
        GenomeSpec(name: "M. jannaschii", path: "\(genomesDir)/m_jannaschii.fasta",
                   domain: "Archaea"),
        GenomeSpec(name: "H. salinarum", path: "\(genomesDir)/h_salinarum.fasta",
                   domain: "Archaea"),
        GenomeSpec(name: "S. acidocaldarius", path: "\(genomesDir)/s_acidocaldarius.fasta",
                   domain: "Archaea"),

        // Eukarya
        GenomeSpec(name: "S. cerevisiae", path: "\(genomesDir)/s_cerevisiae.fasta",
                   domain: "Eukarya"),
        GenomeSpec(name: "S. pombe chr1", path: "\(genomesDir)/s_pombe_chr1.fasta",
                   domain: "Eukarya"),
        GenomeSpec(name: "C. elegans chrI", path: "\(genomesDir)/c_elegans_chrI.fasta",
                   domain: "Eukarya"),
        GenomeSpec(name: "Drosophila chr2L", path: "\(genomesDir)/drosophila_chr2L.fna",
                   domain: "Eukarya"),
        GenomeSpec(name: "Arabidopsis chr1", path: "\(genomesDir)/arabidopsis_chr1.fna",
                   domain: "Eukarya"),
        GenomeSpec(name: "Zebrafish chr1", path: "\(genomesDir)/d_rerio_chr1.fasta",
                   domain: "Eukarya"),
        GenomeSpec(name: "Mouse chr19", path: "\(genomesDir)/m_musculus_chr19.fasta",
                   domain: "Eukarya"),
        GenomeSpec(name: "P. falciparum chr13", path: "\(genomesDir)/p_falciparum_chr13.fasta",
                   domain: "Eukarya"),
    ]
}

func main() throws {
    print("=" .padding(toLength: 72, withPad: "=", startingAt: 0))
    print("MULTILEVEL SPECTRAL FILTERING OF GENOMIC DNA")
    print("6-band decomposition with cross-spectral coherence")
    print("=" .padding(toLength: 72, withPad: "=", startingAt: 0))

    let analyzer = try DNASpectralAnalyzer()

    // Determine which genomes to process
    var genomes = findGenomes()

    // Filter by command-line args if provided
    if CommandLine.arguments.count > 1 {
        let argPaths = Set(CommandLine.arguments.dropFirst())
        genomes = genomes.filter { argPaths.contains($0.path) || argPaths.contains($0.name) }
        if genomes.isEmpty {
            // Treat args as direct FASTA paths
            genomes = CommandLine.arguments.dropFirst().map {
                GenomeSpec(name: ($0 as NSString).lastPathComponent, path: $0, domain: "Unknown")
            }
        }
    }

    // Filter to genomes that exist
    genomes = genomes.filter { FileManager.default.fileExists(atPath: $0.path) }
    print("\nFound \(genomes.count) genomes to analyze")

    // Output directory
    let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
    let outDirs = [
        "\(execDir)/../data/genomes",
        "\(execDir)/../../data/genomes",
        "\(execDir)/../../../data/genomes",
        "\(execDir)/../../../../data/genomes",
        "\(execDir)/../../../../../data/genomes",
        "\(FileManager.default.currentDirectoryPath)/data/genomes",
    ]
    var outDir = FileManager.default.currentDirectoryPath
    for dir in outDirs {
        let path = (dir as NSString).standardizingPath
        if FileManager.default.fileExists(atPath: path) {
            outDir = path
            break
        }
    }

    // Comparative table across all organisms
    let pairNames = ["AT", "AG", "AC", "TG", "TC", "GC"]
    let bands = standardGenomicBands

    var comparativeLines = [String]()
    comparativeLines.append("# Multilevel Spectral Coherence — Cross-Species Comparison")
    comparativeLines.append("# Generated: \(ISO8601DateFormatter().string(from: Date()))")
    comparativeLines.append("")

    // Process each genome
    for genome in genomes {
        print("\n" + String(repeating: "━", count: 72))
        print("Processing: \(genome.name) [\(genome.domain)]")
        print("  File: \(genome.path)")

        // Read FASTA
        let fastaSequences: [FASTASequence]
        do {
            fastaSequences = try readFASTA(path: genome.path)
        } catch {
            print("  ERROR reading FASTA: \(error)")
            continue
        }

        // Concatenate all sequences (for multi-chromosome genomes), skip N bases
        var allDNA = [UInt8]()
        for seq in fastaSequences {
            print("  Sequence: \(seq.header.prefix(60))... (\(seq.length) bp)")
            // Filter out N bases (encoded as 4) — replace with random to avoid spectral artifacts
            for base in seq.sequence {
                if base <= 3 {
                    allDNA.append(base)
                }
                // Skip N bases entirely (removes gaps)
            }
        }
        print("  Total: \(allDNA.count) bp")

        guard allDNA.count >= 2048 else {
            print("  SKIP: genome too short for multilevel analysis")
            continue
        }

        // Run multilevel analysis
        let result = analyzer.computeMultilevelSpectral(dna: allDNA)

        // Print report
        result.printReport(organismName: "\(genome.name) [\(genome.domain)]")

        // Save per-band coherence
        let safeName = genome.name.lowercased()
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: ".", with: "")
        let bandFile = "\(outDir)/\(safeName)_multilevel_bands.tsv"
        try result.writeBandCoherenceTSV(path: bandFile)
        print("\n  Saved: \(bandFile)")

        // Save cross-band correlations
        let crossFile = "\(outDir)/\(safeName)_crossband_corr.tsv"
        try result.writeCrossBandCorrelationTSV(path: crossFile)
        print("  Saved: \(crossFile)")

        // Save position-resolved data (capped at 50k windows for file size)
        let posFile = "\(outDir)/\(safeName)_multilevel_positions.tsv"
        try result.writePositionBandTSV(path: posFile)
        print("  Saved: \(posFile)")

        // Add to comparative table
        comparativeLines.append("## \(genome.name) [\(genome.domain)]")
        comparativeLines.append("Genome: \(allDNA.count) bp, \(result.numWindows) windows")
        comparativeLines.append("")
        comparativeLines.append("| Band | " + pairNames.joined(separator: " | ") + " | Dominant |")
        comparativeLines.append("|------|" + pairNames.map { _ in "------|" }.joined())
        for (bIdx, band) in bands.enumerated() {
            let coh = result.avgBandCoherence[bIdx]
            let maxIdx = coh.enumerated().max(by: { $0.element < $1.element })!.offset
            var line = "| \(band.shortName) |"
            for p in 0..<6 {
                line += " \(String(format: "%.4f", coh[p])) |"
            }
            line += " **\(pairNames[maxIdx])** |"
            comparativeLines.append(line)
        }
        comparativeLines.append("")

        // Cross-band highlights
        comparativeLines.append("Cross-band correlations (top):")
        for (pIdx, pairName) in pairNames.enumerated() {
            var corrs: [(String, Float)] = []
            for b1 in 0..<(bands.count - 1) {
                for b2 in (b1+1)..<(bands.count - 1) {
                    corrs.append(("\(bands[b1].shortName)-\(bands[b2].shortName)",
                                  result.crossBandCorrelation[b1][b2][pIdx]))
                }
            }
            corrs.sort { abs($0.1) > abs($1.1) }
            let top3 = corrs.prefix(3).map { "\($0.0)=\(String(format: "%.3f", $0.1))" }.joined(separator: ", ")
            comparativeLines.append("- \(pairName): \(top3)")
        }
        comparativeLines.append("")
        comparativeLines.append("---")
        comparativeLines.append("")
    }

    // Write comparative report
    let compPath = "\(outDir)/multilevel_cross_species.md"
    try comparativeLines.joined(separator: "\n").write(toFile: compPath, atomically: true, encoding: .utf8)
    print("\n" + String(repeating: "=", count: 72))
    print("Cross-species comparison: \(compPath)")
    print(String(repeating: "=", count: 72))
}

try main()
