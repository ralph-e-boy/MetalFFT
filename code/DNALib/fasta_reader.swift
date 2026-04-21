import Foundation

// ============================================================================
// FASTA File Reader
// ============================================================================

public struct FASTASequence {
    public let header: String
    public let sequence: [UInt8] // 0=A, 1=T, 2=G, 3=C
    public var length: Int {
        sequence.count
    }

    public init(header: String, sequence: [UInt8]) {
        self.header = header
        self.sequence = sequence
    }
}

public enum FASTAError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case emptySequence
    case readError(String)

    public var description: String {
        switch self {
        case let .fileNotFound(path): "FASTA file not found: \(path)"
        case .emptySequence: "No valid nucleotide bases found in FASTA file"
        case let .readError(msg): "FASTA read error: \(msg)"
        }
    }
}

public func readFASTA(path: String) throws -> [FASTASequence] {
    guard FileManager.default.fileExists(atPath: path) else {
        throw FASTAError.fileNotFound(path)
    }

    let content = try String(contentsOfFile: path, encoding: .utf8)
    var sequences: [FASTASequence] = []
    var currentHeader = ""
    var currentBases: [UInt8] = []

    for line in content.split(separator: "\n", omittingEmptySubsequences: false) {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { continue }

        if trimmed.hasPrefix(">") {
            if !currentBases.isEmpty {
                sequences.append(FASTASequence(header: currentHeader, sequence: currentBases))
                currentBases = []
            }
            currentHeader = String(trimmed.dropFirst())
        } else {
            for ch in trimmed {
                switch ch {
                case "A", "a": currentBases.append(0)
                case "T", "t": currentBases.append(1)
                case "G", "g": currentBases.append(2)
                case "C", "c": currentBases.append(3)
                case "N", "n": currentBases.append(4) // Unknown base
                default: break
                }
            }
        }
    }

    if !currentBases.isEmpty {
        sequences.append(FASTASequence(header: currentHeader, sequence: currentBases))
    }

    if sequences.isEmpty {
        throw FASTAError.emptySequence
    }

    return sequences
}

// ============================================================================
// Synthetic DNA Generator
// ============================================================================

public struct SyntheticDNA {
    public static func random(length: Int) -> [UInt8] {
        (0 ..< length).map { _ in UInt8.random(in: 0 ... 3) }
    }

    public static func withCodingRegion(totalLength: Int, codingStart: Int, codingLength: Int) -> [UInt8] {
        var seq = random(length: totalLength)
        let codons: [[UInt8]] = [
            [0, 1, 2], // ATG
            [2, 0, 1], // GAT
            [3, 1, 0], // CTA
            [0, 2, 3] // AGC
        ]
        for i in stride(from: 0, to: codingLength, by: 3) {
            let codon = codons[(i / 3) % codons.count]
            for j in 0 ..< min(3, codingLength - i) {
                let pos = codingStart + i + j
                if pos < totalLength {
                    seq[pos] = codon[j]
                }
            }
        }
        return seq
    }

    public static func withTandemRepeat(totalLength: Int, repeatStart: Int, repeatLength: Int, period: Int) -> [UInt8] {
        var seq = random(length: totalLength)
        let motif = random(length: period)
        for i in 0 ..< repeatLength {
            let pos = repeatStart + i
            if pos < totalLength {
                seq[pos] = motif[i % period]
            }
        }
        return seq
    }
}

public func writeFASTA(sequences: [FASTASequence], path: String) throws {
    let nucleotides: [Character] = ["A", "T", "G", "C"]
    var output = ""
    for seq in sequences {
        output += ">\(seq.header)\n"
        for i in stride(from: 0, to: seq.sequence.count, by: 80) {
            let end = min(i + 80, seq.sequence.count)
            let line = seq.sequence[i ..< end].map { nucleotides[Int($0)] }
            output += String(line) + "\n"
        }
    }
    try output.write(toFile: path, atomically: true, encoding: .utf8)
}
