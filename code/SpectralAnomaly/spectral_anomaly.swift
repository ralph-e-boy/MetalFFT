import Foundation
import DNALib

// ============================================================================
// Spectral Anomaly Detection in S. cerevisiae
//
// Goal: Find genomic regions with spectral signatures that don't match
// typical coding or non-coding profiles. These "spectral anomalies" may
// represent functional elements with distinctive sequence organization.
//
// Approach:
// 1. Compute per-window 30-feature multilevel spectral vectors
// 2. Build coding and non-coding spectral centroids
// 3. For each window, compute distance from both centroids
// 4. Windows far from both = spectral anomalies
// 5. Overlay comprehensive annotations (CDS, tRNA, ARS, LTR, telomere, etc.)
// 6. Characterize what makes anomalous windows distinct
// ============================================================================

struct AnnotatedFeature {
    let seqid: String
    let featureType: String
    let start: Int
    let end: Int
    let name: String
    let strand: Character
}

/// Parse all interesting feature types from GFF3
func parseAllFeatures(gffPath: String) throws -> [AnnotatedFeature] {
    let interestingTypes: Set<String> = [
        "CDS", "tRNA", "rRNA", "snoRNA", "snRNA", "ncRNA",
        "long_terminal_repeat", "mobile_genetic_element",
        "origin_of_replication", "centromere", "telomere",
        "regulatory_region", "pseudogene", "antisense_RNA",
        "telomerase_RNA", "RNase_P_RNA", "RNase_MRP_RNA", "SRP_RNA"
    ]

    let content = try String(contentsOfFile: gffPath, encoding: .utf8)
    var features = [AnnotatedFeature]()

    for line in content.split(separator: "\n") {
        if line.hasPrefix("#") { continue }
        let fields = line.split(separator: "\t", omittingEmptySubsequences: false)
        guard fields.count >= 9 else { continue }

        let featureType = String(fields[2])
        guard interestingTypes.contains(featureType) else { continue }

        guard let start = Int(fields[3]), let end = Int(fields[4]) else { continue }

        // Extract Name from attributes
        let attrs = String(fields[8])
        var name = featureType
        if let nameRange = attrs.range(of: "Name=") {
            let nameStart = attrs[nameRange.upperBound...]
            if let semiIdx = nameStart.firstIndex(of: ";") {
                name = String(nameStart[..<semiIdx])
            } else {
                name = String(nameStart)
            }
            name = name.removingPercentEncoding ?? name
        }

        features.append(AnnotatedFeature(
            seqid: String(fields[0]),
            featureType: featureType,
            start: start, end: end,
            name: name,
            strand: fields[6].first ?? "+"
        ))
    }
    return features
}

/// Classify a window by its dominant annotation
func classifyWindow(windowStart: Int, windowEnd: Int,
                    annotations: [AnnotatedFeature],
                    seqOffsets: [String: (offset: Int, length: Int)]) -> (String, Float) {
    // Count bases overlapping each feature type
    var typeCounts = [String: Int]()
    let windowLen = windowEnd - windowStart

    for feat in annotations {
        guard let info = seqOffsets[feat.seqid] else { continue }
        let fStart = info.offset + feat.start - 1
        let fEnd = info.offset + feat.end

        let overlapStart = max(windowStart, fStart)
        let overlapEnd = min(windowEnd, fEnd)
        if overlapStart < overlapEnd {
            typeCounts[feat.featureType, default: 0] += (overlapEnd - overlapStart)
        }
    }

    if typeCounts.isEmpty {
        return ("intergenic", 0.0)
    }

    let dominant = typeCounts.max(by: { $0.value < $1.value })!
    let fraction = Float(dominant.value) / Float(windowLen)

    // Simplify categories
    let category: String
    switch dominant.key {
    case "CDS": category = "CDS"
    case "tRNA": category = "tRNA"
    case "rRNA": category = "rRNA"
    case "snoRNA", "snRNA", "ncRNA", "antisense_RNA",
         "telomerase_RNA", "RNase_P_RNA", "RNase_MRP_RNA", "SRP_RNA":
        category = "ncRNA"
    case "long_terminal_repeat": category = "LTR"
    case "mobile_genetic_element": category = "transposon"
    case "origin_of_replication": category = "ARS"
    case "centromere": category = "centromere"
    case "telomere": category = "telomere"
    case "regulatory_region": category = "regulatory"
    case "pseudogene": category = "pseudogene"
    default: category = dominant.key
    }

    return (category, fraction)
}

func main() throws {
    print(String(repeating: "=", count: 72))
    print("SPECTRAL ANOMALY DETECTION IN S. CEREVISIAE")
    print("Finding genomic regions with unusual spectral signatures")
    print(String(repeating: "=", count: 72))

    let analyzer = try DNASpectralAnalyzer()

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
            genomesDir = path; break
        }
    }

    // 1. Read genome and build sequence offset map
    print("\n1. Reading genome...")
    let sequences = try readFASTA(path: "\(genomesDir)/s_cerevisiae.fasta")
    var allDNA = [UInt8]()
    var seqOffsets = [String: (offset: Int, length: Int)]()

    for seq in sequences {
        let accession = String(seq.header.split(separator: " ").first ?? "")
        let filtered = seq.sequence.filter { $0 <= 3 }
        seqOffsets[accession] = (offset: allDNA.count, length: filtered.count)
        allDNA.append(contentsOf: filtered)
    }
    print("   \(allDNA.count) bp, \(sequences.count) chromosomes")

    // 2. Parse comprehensive annotations
    print("\n2. Parsing annotations...")
    let features = try parseAllFeatures(gffPath: "\(genomesDir)/s_cerevisiae_R64.gff")
    var typeCounts = [String: Int]()
    for f in features { typeCounts[f.featureType, default: 0] += 1 }
    for (t, c) in typeCounts.sorted(by: { $0.value > $1.value }) {
        print("   \(t): \(c)")
    }

    // 3. Compute multilevel features
    print("\n3. Computing multilevel spectral features...")
    let result = analyzer.computeMultilevelSpectral(dna: allDNA)
    let bands = standardGenomicBands
    let pairNames = ["AT", "AG", "AC", "TG", "TC", "GC"]
    let numFeatures = bands.count * 6  // 30 features per window

    // 4. Extract feature vectors and classify windows
    print("\n4. Extracting feature vectors and classifying windows...")

    struct WindowData {
        let position: Int
        let features: [Float]       // 30-dim vector
        let category: String
        let categoryFraction: Float
        let codingFraction: Float
    }

    var windows = [WindowData]()
    let N = 1024
    let hopSize = 512

    for w in 0..<result.numWindows {
        let startPos = w * hopSize
        let endPos = startPos + N

        // Extract 30-dim feature vector
        var featureVec = [Float]()
        for bf in result.windowResults[w].bandFeatures {
            featureVec.append(contentsOf: bf.coherence)
        }

        // Classify by annotation
        let (cat, catFrac) = classifyWindow(
            windowStart: startPos, windowEnd: endPos,
            annotations: features, seqOffsets: seqOffsets)

        // Compute coding fraction specifically
        var codingCount = 0
        for feat in features where feat.featureType == "CDS" {
            guard let info = seqOffsets[feat.seqid] else { continue }
            let fStart = info.offset + feat.start - 1
            let fEnd = info.offset + feat.end
            let oStart = max(startPos, fStart)
            let oEnd = min(endPos, fEnd)
            if oStart < oEnd { codingCount += oEnd - oStart }
        }
        let codFrac = Float(codingCount) / Float(N)

        windows.append(WindowData(
            position: startPos,
            features: featureVec,
            category: cat,
            categoryFraction: catFrac,
            codingFraction: codFrac
        ))
    }

    // Category counts
    var catCounts = [String: Int]()
    for w in windows { catCounts[w.category, default: 0] += 1 }
    print("   Window categories:")
    for (cat, count) in catCounts.sorted(by: { $0.value > $1.value }) {
        print("     \(cat): \(count) windows (\(String(format: "%.1f%%", Float(count) / Float(windows.count) * 100)))")
    }

    // 5. Build coding and intergenic centroids
    print("\n5. Building spectral centroids...")

    func computeCentroid(_ subset: [WindowData]) -> (mean: [Float], std: [Float]) {
        let n = Float(subset.count)
        var mean = [Float](repeating: 0, count: numFeatures)
        for w in subset {
            for i in 0..<numFeatures { mean[i] += w.features[i] }
        }
        for i in 0..<numFeatures { mean[i] /= n }

        var variance = [Float](repeating: 0, count: numFeatures)
        for w in subset {
            for i in 0..<numFeatures {
                let d = w.features[i] - mean[i]
                variance[i] += d * d
            }
        }
        let std = variance.map { sqrt($0 / max(n - 1, 1)) }
        return (mean, std)
    }

    let codingWindows = windows.filter { $0.codingFraction > 0.7 }
    let intergenicWindows = windows.filter { $0.category == "intergenic" && $0.codingFraction < 0.05 }

    let (codingMean, codingStd) = computeCentroid(codingWindows)
    let (intergenicMean, intergenicStd) = computeCentroid(intergenicWindows)

    print("   Coding centroid from \(codingWindows.count) windows")
    print("   Intergenic centroid from \(intergenicWindows.count) windows")

    // 6. Compute anomaly scores for every window
    print("\n6. Computing anomaly scores...")

    func normalizedDistance(_ features: [Float], _ mean: [Float], _ std: [Float]) -> Float {
        var sumSq: Float = 0
        var count: Float = 0
        for i in 0..<features.count {
            if std[i] > 1e-6 {
                let z = (features[i] - mean[i]) / std[i]
                sumSq += z * z
                count += 1
            }
        }
        return count > 0 ? sqrt(sumSq / count) : 0  // RMS z-score
    }

    struct ScoredWindow {
        let data: WindowData
        let distCoding: Float
        let distIntergenic: Float
        let anomalyScore: Float  // min distance to any centroid
    }

    var scored = windows.map { w -> ScoredWindow in
        let dc = normalizedDistance(w.features, codingMean, codingStd)
        let di = normalizedDistance(w.features, intergenicMean, intergenicStd)
        return ScoredWindow(data: w, distCoding: dc, distIntergenic: di,
                           anomalyScore: min(dc, di))
    }

    scored.sort { $0.anomalyScore > $1.anomalyScore }

    // 7. Analyze the most anomalous windows
    print("\n" + String(repeating: "=", count: 72))
    print("TOP SPECTRAL ANOMALIES")
    print("(Windows most distant from both coding and intergenic centroids)")
    print(String(repeating: "=", count: 72))

    // Category breakdown of top anomalies
    let topN = 500
    let topAnomalies = Array(scored.prefix(topN))

    var topCatCounts = [String: Int]()
    for w in topAnomalies { topCatCounts[w.data.category, default: 0] += 1 }

    print("\nTop \(topN) anomalous windows by category:")
    let totalWindows = Float(windows.count)
    for (cat, count) in topCatCounts.sorted(by: { $0.value > $1.value }) {
        let pctInTop = Float(count) / Float(topN) * 100
        let pctInGenome = Float(catCounts[cat] ?? 0) / totalWindows * 100
        let enrichment = pctInTop / max(pctInGenome, 0.01)
        let enrichStr = enrichment > 2 ? " *** ENRICHED \(String(format: "%.1fx", enrichment))" :
                        enrichment > 1.5 ? " ** enriched \(String(format: "%.1fx", enrichment))" :
                        enrichment < 0.5 ? " (depleted)" : ""
        print("  \(cat.padding(toLength: 14, withPad: " ", startingAt: 0)) \(count) (\(String(format: "%.1f%%", pctInTop)) of anomalies vs \(String(format: "%.1f%%", pctInGenome)) of genome)\(enrichStr)")
    }

    // 8. Spectral profile of each category
    print("\n" + String(repeating: "=", count: 72))
    print("SPECTRAL PROFILES BY GENOMIC ELEMENT TYPE")
    print("(Mean per-band coherence for each annotation category)")
    print(String(repeating: "=", count: 72))

    let categories = ["CDS", "intergenic", "tRNA", "ARS", "LTR", "transposon",
                       "centromere", "telomere", "ncRNA", "regulatory"]

    // Header
    var profileHeader = "Category".padding(toLength: 14, withPad: " ", startingAt: 0) + "  N  "
    for band in bands {
        if band.shortName == "B5" { continue }
        for pair in pairNames {
            profileHeader += " \(band.shortName)\(pair) "
        }
    }

    // For each category, compute mean features
    var categoryProfiles = [(String, Int, [Float])]()

    for cat in categories {
        let catWindows = windows.filter { $0.category == cat }
        guard catWindows.count >= 10 else { continue }

        var mean = [Float](repeating: 0, count: numFeatures)
        for w in catWindows {
            for i in 0..<numFeatures { mean[i] += w.features[i] }
        }
        for i in 0..<numFeatures { mean[i] /= Float(catWindows.count) }
        categoryProfiles.append((cat, catWindows.count, mean))
    }

    // Print comparison table (selected features only)
    print("\nKey discriminating features by element type:")
    print("")
    let keyFeatures = [
        (0, 0, "B0_AT"), (2, 0, "B2_AT"), (2, 5, "B2_GC"),
        (3, 0, "B3_AT"), (3, 2, "B3_AC"), (3, 3, "B3_TG"),
        (4, 0, "B4_AT"), (4, 1, "B4_AG"), (4, 4, "B4_TC"),
        (1, 5, "B1_GC"),
    ]

    var tableHeader = "Category".padding(toLength: 14, withPad: " ", startingAt: 0) + "  N     "
    for (_, _, name) in keyFeatures {
        tableHeader += name.padding(toLength: 8, withPad: " ", startingAt: 0)
    }
    print(tableHeader)
    print(String(repeating: "-", count: 14 + 6 + keyFeatures.count * 8))

    for (cat, n, mean) in categoryProfiles {
        var line = cat.padding(toLength: 14, withPad: " ", startingAt: 0)
        line += String(format: "%5d ", n)
        for (bIdx, pIdx, _) in keyFeatures {
            let val = mean[bIdx * 6 + pIdx]
            line += String(format: " %6.4f ", val)
        }
        print(line)
    }

    // 9. Statistical comparison: each category vs CDS
    print("\n\n" + String(repeating: "=", count: 72))
    print("EFFECT SIZES: Each Element Type vs CDS Windows")
    print("(Cohen's d for each feature; positive = higher in element type)")
    print(String(repeating: "=", count: 72))

    guard let cdsProfile = categoryProfiles.first(where: { $0.0 == "CDS" }) else {
        print("ERROR: No CDS windows found")
        return
    }
    let cdsN = cdsProfile.1

    for (cat, catN, catMean) in categoryProfiles {
        if cat == "CDS" { continue }
        guard catN >= 10 else { continue }

        // Compute Cohen's d for each feature
        let catWindows = windows.filter { $0.category == cat }
        let cdsWindowsSample = codingWindows

        print("\n  \(cat) (\(catN) windows) vs CDS (\(cdsN) windows):")

        var significantEffects = [(String, Float, Float, Float)]()  // name, d, catVal, cdsVal

        for (bIdx, band) in bands.enumerated() {
            if band.shortName == "B5" { continue }
            for pIdx in 0..<6 {
                let fIdx = bIdx * 6 + pIdx

                let catVals = catWindows.map { $0.features[fIdx] }
                let cdsVals = cdsWindowsSample.map { $0.features[fIdx] }

                let catMeanV = catVals.reduce(0, +) / Float(catVals.count)
                let cdsMeanV = cdsVals.reduce(0, +) / Float(cdsVals.count)

                var catVar: Float = 0
                for v in catVals { catVar += (v - catMeanV) * (v - catMeanV) }
                catVar /= Float(max(catVals.count - 1, 1))

                var cdsVar: Float = 0
                for v in cdsVals { cdsVar += (v - cdsMeanV) * (v - cdsMeanV) }
                cdsVar /= Float(max(cdsVals.count - 1, 1))

                let pooledStd = sqrt((catVar + cdsVar) / 2)
                let d = pooledStd > 1e-6 ? (catMeanV - cdsMeanV) / pooledStd : 0

                if abs(d) > 0.3 {
                    significantEffects.append(("\(band.shortName)_\(pairNames[pIdx])", d, catMeanV, cdsMeanV))
                }
            }
        }

        significantEffects.sort { abs($0.1) > abs($1.1) }
        for (name, d, catVal, cdsVal) in significantEffects.prefix(8) {
            let stars = abs(d) > 0.8 ? "***" : abs(d) > 0.5 ? "** " : "*  "
            print("    \(name.padding(toLength: 8, withPad: " ", startingAt: 0)) d=\(String(format: "%+.2f", d)) \(stars) (\(cat)=\(String(format: "%.4f", catVal)) vs CDS=\(String(format: "%.4f", cdsVal)))")
        }
        if significantEffects.isEmpty {
            print("    No features with |d| > 0.3")
        }
    }

    // 10. Write comprehensive output
    var report = [String]()
    report.append("# Spectral Anomaly Detection in S. cerevisiae")
    report.append("")

    // Category profiles as TSV
    var tsvLines = [String]()
    var tsvHeader = "category\tn_windows"
    for (bIdx, band) in bands.enumerated() {
        for pIdx in 0..<6 {
            tsvHeader += "\t\(band.shortName)_\(pairNames[pIdx])"
        }
    }
    tsvLines.append(tsvHeader)
    for (cat, n, mean) in categoryProfiles {
        var line = "\(cat)\t\(n)"
        for val in mean {
            line += "\t\(String(format: "%.6f", val))"
        }
        tsvLines.append(line)
    }

    let tsvPath = "\(genomesDir)/spectral_element_profiles.tsv"
    try tsvLines.joined(separator: "\n").write(toFile: tsvPath, atomically: true, encoding: .utf8)
    print("\n\nElement profiles: \(tsvPath)")

    // Per-window anomaly scores
    var windowTSV = ["position\tcategory\tcoding_frac\tdist_coding\tdist_intergenic\tanomaly_score"]
    for sw in scored.prefix(2000) {
        windowTSV.append("\(sw.data.position)\t\(sw.data.category)\t\(String(format: "%.3f", sw.data.codingFraction))\t\(String(format: "%.3f", sw.distCoding))\t\(String(format: "%.3f", sw.distIntergenic))\t\(String(format: "%.3f", sw.anomalyScore))")
    }
    let windowPath = "\(genomesDir)/spectral_anomaly_windows.tsv"
    try windowTSV.joined(separator: "\n").write(toFile: windowPath, atomically: true, encoding: .utf8)
    print("Anomaly windows: \(windowPath)")
}

try main()
