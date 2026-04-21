import DNALib
import Foundation

// ============================================================================
// Genome-Wide Pseudogene Reannotation Screen for P. falciparum
//
// For every annotated pseudogene, compute multilevel spectral features and
// compare to coding vs intergenic centroids. Flag pseudogenes whose spectral
// signature resembles coding DNA — potential misannotated functional genes
// like EBL-1 (PF3D7_1371600).
//
// Also identify intergenic regions with coding-like spectral signatures
// (potential unannotated genes) and intergenic regions with anomalous
// spectral structure (potential functional non-coding elements).
// ============================================================================

func main() throws {
    print(String(repeating: "=", count: 72))
    print("GENOME-WIDE SPECTRAL SCREEN OF P. FALCIPARUM 3D7")
    print("Hunting for misannotated pseudogenes and unannotated elements")
    print(String(repeating: "=", count: 72))

    let analyzer = try DNASpectralAnalyzer()

    let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
    let searchDirs = [
        "\(execDir)/../data/genomes", "\(execDir)/../../data/genomes",
        "\(execDir)/../../../data/genomes", "\(execDir)/../../../../data/genomes",
        "\(execDir)/../../../../../data/genomes",
        "\(FileManager.default.currentDirectoryPath)/data/genomes"
    ]
    var genomesDir = ""
    for dir in searchDirs {
        let path = (dir as NSString).standardizingPath
        if FileManager.default.fileExists(atPath: path + "/p_falciparum_full.fasta") {
            genomesDir = path; break
        }
    }

    // 1. Read full genome
    print("\n1. Reading full P. falciparum genome (14 chromosomes)...")
    let sequences = try readFASTA(path: "\(genomesDir)/p_falciparum_full.fasta")
    var allDNA = [UInt8]()
    var chrOffsets = [(accession: String, offset: Int, length: Int)]()
    var rng: UInt64 = 42

    for seq in sequences {
        let acc = String(seq.header.split(separator: " ").first ?? "")
        let startOffset = allDNA.count
        for base in seq.sequence {
            if base <= 3 {
                allDNA.append(base)
            } else {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
                allDNA.append(UInt8(rng % 4))
            }
        }
        let len = allDNA.count - startOffset
        chrOffsets.append((acc, startOffset, len))
        print("  \(acc): \(len) bp (offset \(startOffset))")
    }
    print("  Total: \(allDNA.count) bp")

    // 2. Parse all GFF features
    print("\n2. Parsing GFF annotations...")
    let gffContent = try String(contentsOfFile: "\(genomesDir)/p_falciparum.gff", encoding: .utf8)

    struct GFFEntry {
        let seqid: String; let type: String; let start: Int; let end: Int
        let name: String; let product: String; let geneId: String
    }
    var gffEntries = [GFFEntry]()

    for line in gffContent.split(separator: "\n") {
        if line.hasPrefix("#") { continue }
        let f = line.split(separator: "\t", omittingEmptySubsequences: false)
        guard f.count >= 9, let start = Int(f[3]), let end = Int(f[4]) else { continue }
        let attrs = String(f[8])
        var name = String(f[2]); var product = ""; var geneId = ""

        if let r = attrs.range(of: ";gene=") {
            name = String(attrs[r.upperBound...].prefix(while: { $0 != ";" }))
        } else if let r = attrs.range(of: "Name=") {
            name = String(attrs[r.upperBound...].prefix(while: { $0 != ";" }))
            name = name.removingPercentEncoding ?? name
        }
        if let r = attrs.range(of: "product=") {
            product = String(attrs[r.upperBound...].prefix(while: { $0 != ";" }))
            product = product.removingPercentEncoding ?? product
        }
        if let r = attrs.range(of: "ID=") {
            geneId = String(attrs[r.upperBound...].prefix(while: { $0 != ";" }))
        }

        gffEntries.append(GFFEntry(
            seqid: String(f[0]), type: String(f[2]),
            start: start, end: end,
            name: name, product: product, geneId: geneId
        ))
    }

    // Count pseudogenes genome-wide
    let pseudogenes = gffEntries.filter { $0.type == "pseudogene" }
    print("  Total pseudogenes: \(pseudogenes.count)")
    for pg in pseudogenes {
        print("    \(pg.seqid): \(pg.name) (\(pg.start)-\(pg.end), \(pg.end - pg.start + 1) bp)")
    }

    // 3. Compute multilevel spectral features for full genome
    print("\n3. Computing multilevel spectral features (full genome)...")
    let result = analyzer.computeMultilevelSpectral(dna: allDNA)
    let bands = standardGenomicBands
    let pairNames = ["AT", "AG", "AC", "TG", "TC", "GC"]
    let N = 1024; let hopSize = 512
    let nf = bands.count * 6

    // 4. Build coding and intergenic centroids
    print("\n4. Building coding and intergenic centroids...")

    struct WinData {
        let idx: Int; let gpos: Int; let features: [Float]; let codingFrac: Float
    }

    var windows = [WinData]()
    for w in 0 ..< result.numWindows {
        let gpos = w * hopSize
        let gend = gpos + N

        var feats = [Float]()
        for bf in result.windowResults[w].bandFeatures {
            feats.append(contentsOf: bf.coherence)
        }

        var codingBases = 0
        for e in gffEntries where e.type == "CDS" {
            guard let info = chrOffsets.first(where: { $0.accession == e.seqid }) else { continue }
            let fStart = info.offset + e.start - 1
            let fEnd = info.offset + e.end
            let oS = max(gpos, fStart); let oE = min(gend, fEnd)
            if oS < oE { codingBases += oE - oS }
        }

        windows.append(WinData(idx: w, gpos: gpos, features: feats,
                               codingFrac: Float(codingBases) / Float(N)))
    }

    let codingWins = windows.filter { $0.codingFrac > 0.7 }
    let intergenicWins = windows.filter { $0.codingFrac < 0.05 }

    func centroid(_ ws: [WinData]) -> (m: [Float], s: [Float]) {
        let n = Float(ws.count)
        var m = [Float](repeating: 0, count: nf)
        for w in ws {
            for i in 0 ..< nf {
                m[i] += w.features[i]
            }
        }
        for i in 0 ..< nf {
            m[i] /= n
        }
        var v = [Float](repeating: 0, count: nf)
        for w in ws {
            for i in 0 ..< nf {
                let d = w.features[i] - m[i]; v[i] += d * d
            }
        }
        return (m, v.map { sqrt($0 / max(n - 1, 1)) })
    }

    let (codM, codS) = centroid(codingWins)
    let (intM, intS) = centroid(intergenicWins)
    print("  Coding: \(codingWins.count) windows, Intergenic: \(intergenicWins.count) windows")

    func rmsZ(_ f: [Float], _ m: [Float], _ s: [Float]) -> Float {
        var sum: Float = 0; var c: Float = 0
        for i in 0 ..< min(f.count, m.count) {
            if s[i] > 1e-6 { let z = (f[i] - m[i]) / s[i]; sum += z * z; c += 1 }
        }
        return c > 0 ? sqrt(sum / c) : 0
    }

    // 5. Score every pseudogene
    print("\n" + String(repeating: "=", count: 72))
    print("PSEUDOGENE SPECTRAL SCREENING")
    print("(comparing each pseudogene's spectral profile to coding centroid)")
    print(String(repeating: "=", count: 72))

    struct PseudogeneScore {
        let entry: GFFEntry
        let distCoding: Float
        let distIntergenic: Float
        let codingLikeness: Float // how much more like coding than intergenic
        let gcContent: Float
        let maxOrfLen: Int
        let meanFeatures: [Float]
    }

    var pgScores = [PseudogeneScore]()

    for pg in pseudogenes {
        guard let info = chrOffsets.first(where: { $0.accession == pg.seqid }) else { continue }
        let pgStart = info.offset + pg.start - 1
        let pgEnd = info.offset + pg.end

        // Find all overlapping windows
        var overlappingFeatures = [[Float]]()
        for w in windows {
            let wEnd = w.gpos + N
            if w.gpos < pgEnd, wEnd > pgStart {
                overlappingFeatures.append(w.features)
            }
        }
        guard !overlappingFeatures.isEmpty else { continue }

        // Mean feature vector
        var meanF = [Float](repeating: 0, count: nf)
        for f in overlappingFeatures {
            for i in 0 ..< nf {
                meanF[i] += f[i]
            }
        }
        for i in 0 ..< nf {
            meanF[i] /= Float(overlappingFeatures.count)
        }

        let dCod = rmsZ(meanF, codM, codS)
        let dInt = rmsZ(meanF, intM, intS)
        let codingLikeness = dInt - dCod // positive = more like coding

        // GC content
        let seqSlice = Array(allDNA[pgStart ..< min(pgEnd, allDNA.count)])
        let gc = Float(seqSlice.count(where: { $0 == 2 || $0 == 3 })) / Float(seqSlice.count) * 100

        // ORF scan
        let dnaStr = seqSlice.map { ["A", "T", "G", "C"][$0 <= 3 ? Int($0) : 0] }.joined()
        var maxOrf = 0
        for frame in 0 ..< 3 {
            var orfStart = -1; var i = frame
            while i + 2 < dnaStr.count {
                let idx = dnaStr.index(dnaStr.startIndex, offsetBy: i)
                let codon = String(dnaStr[idx ..< dnaStr.index(idx, offsetBy: 3)])
                if codon == "ATG", orfStart < 0 { orfStart = i }
                if codon == "TAA" || codon == "TAG" || codon == "TGA", orfStart >= 0 {
                    maxOrf = max(maxOrf, (i + 3 - orfStart) / 3)
                    orfStart = -1
                }
                i += 3
            }
            if orfStart >= 0 { maxOrf = max(maxOrf, (dnaStr.count - orfStart) / 3) }
        }

        pgScores.append(PseudogeneScore(
            entry: pg, distCoding: dCod, distIntergenic: dInt,
            codingLikeness: codingLikeness, gcContent: gc,
            maxOrfLen: maxOrf, meanFeatures: meanF
        ))
    }

    // Sort by coding likeness (most coding-like first)
    pgScores.sort { $0.codingLikeness > $1.codingLikeness }

    print("\nAll pseudogenes ranked by coding-likeness:")
    print("(positive = more like coding than intergenic)")
    print("")
    print("Rank  Gene ID                   Chr  Size    GC%    MaxORF   CodLike  dCod   dInt")
    print(String(repeating: "-", count: 95))

    for (i, pg) in pgScores.enumerated() {
        let chr = pg.entry.seqid.suffix(4)
        let size = pg.entry.end - pg.entry.start + 1
        let marker = pg.codingLikeness > 0 ? " ***" : ""
        print(String(format: "  %2d   %-24s %@  %5d  %5.1f%%  %4d aa  %+6.2f  %5.2f  %5.2f%@",
                     i + 1, pg.entry.name, String(chr), size, pg.gcContent, pg.maxOrfLen,
                     pg.codingLikeness, pg.distCoding, pg.distIntergenic, marker))
    }

    // Highlight candidates
    let candidates = pgScores.filter { $0.codingLikeness > 0 }
    print("\n\n" + String(repeating: "=", count: 72))
    print("REANNOTATION CANDIDATES: \(candidates.count) pseudogenes with coding-like spectra")
    print(String(repeating: "=", count: 72))

    for (i, pg) in candidates.enumerated() {
        print("\n  Candidate #\(i + 1): \(pg.entry.name)")
        print("    \(pg.entry.seqid):\(pg.entry.start)-\(pg.entry.end) (\(pg.entry.end - pg.entry.start + 1) bp)")
        print("    GC content: \(String(format: "%.1f%%", pg.gcContent)) (genome avg: 19.4%)")
        print("    Largest ORF: \(pg.maxOrfLen) codons")
        print("    Coding likeness: \(String(format: "%+.2f", pg.codingLikeness)) (dist_intergenic - dist_coding)")
        print("    Distance to coding centroid: \(String(format: "%.2f", pg.distCoding))")
        print("    Distance to intergenic centroid: \(String(format: "%.2f", pg.distIntergenic))")

        // Top spectral deviations from intergenic
        var devs = [(String, Float)]()
        for (bIdx, band) in bands.enumerated() {
            for pIdx in 0 ..< 6 {
                let idx = bIdx * 6 + pIdx
                if idx < pg.meanFeatures.count, idx < intS.count, intS[idx] > 1e-6 {
                    let z = (pg.meanFeatures[idx] - intM[idx]) / intS[idx]
                    devs.append(("\(band.shortName)_\(pairNames[pIdx])", z))
                }
            }
        }
        devs.sort { abs($0.1) > abs($1.1) }
        let topDevs = devs.prefix(5).map { "\($0.0)=\(String(format: "%+.1f", $0.1))" }.joined(separator: ", ")
        print("    Spectral deviations: \(topDevs)")

        // Nearby genes
        var nearby = [String]()
        for e in gffEntries where e.type == "CDS" || e.type == "gene" {
            guard e.seqid == pg.entry.seqid else { continue }
            let dist = min(abs(e.start - pg.entry.end), abs(pg.entry.start - e.end))
            if dist < 10000, e.type == "CDS" {
                let desc = e.product.isEmpty ? e.name : String(e.product.prefix(50))
                nearby.append("\(desc) (\(dist)bp)")
            }
        }
        if !nearby.isEmpty {
            print("    Nearby genes: \(nearby.prefix(3).joined(separator: "; "))")
        }
    }

    // Save results
    var tsvLines = ["gene_id\tchr\tstart\tend\tsize\tgc_pct\tmax_orf_codons\tcoding_likeness\tdist_coding\tdist_intergenic\tcandidate"]
    for pg in pgScores {
        let isCand = pg.codingLikeness > 0 ? "YES" : "no"
        tsvLines.append("\(pg.entry.name)\t\(pg.entry.seqid)\t\(pg.entry.start)\t\(pg.entry.end)\t\(pg.entry.end - pg.entry.start + 1)\t\(String(format: "%.1f", pg.gcContent))\t\(pg.maxOrfLen)\t\(String(format: "%.3f", pg.codingLikeness))\t\(String(format: "%.3f", pg.distCoding))\t\(String(format: "%.3f", pg.distIntergenic))\t\(isCand)")
    }
    let outPath = "\(genomesDir)/pf_pseudogene_screen.tsv"
    try tsvLines.joined(separator: "\n").write(toFile: outPath, atomically: true, encoding: .utf8)
    print("\n\nResults saved: \(outPath)")
}

try main()
