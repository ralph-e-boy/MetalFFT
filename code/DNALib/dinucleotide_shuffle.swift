import Foundation

// ============================================================================
// Dinucleotide-Preserving Shuffle (Altschul-Erickson Algorithm)
//
// Generates a random permutation of a DNA sequence that preserves:
// - Exact single-nucleotide frequencies (trivial)
// - Exact dinucleotide frequencies (non-trivial)
// - First and last nucleotide
//
// This is the correct null model for spectral analysis: any spectral
// feature that also appears in the shuffled sequence is explained by
// dinucleotide composition alone. Only features ABSENT in shuffled
// sequences are genuine higher-order structure.
//
// Algorithm: Euler path through the dinucleotide graph.
// Reference: Altschul & Erickson (1985) "Significance of nucleotide
// sequence alignments: A method for random sequence permutation that
// preserves dinucleotide and codon usage." Mol Biol Evol 2(6):526-538.
// Also: Kandel, Matias & Unger (1996) for the Euler path formulation.
// ============================================================================

/// Shuffle a DNA sequence preserving dinucleotide frequencies.
/// Uses Euler path through the directed dinucleotide multigraph.
///
/// - Parameters:
///   - dna: Encoded DNA (0=A, 1=T, 2=G, 3=C). Values >3 are skipped.
///   - rng: Random number generator (for reproducibility)
/// - Returns: Shuffled sequence with identical dinucleotide frequencies
public func dinucleotideShuffle<R: RandomNumberGenerator>(
    dna: [UInt8],
    rng: inout R
) -> [UInt8] {
    // Filter to valid bases only
    let seq = dna.filter { $0 <= 3 }
    let n = seq.count
    guard n >= 3 else { return seq }

    // Build the directed multigraph:
    // Node = nucleotide (0-3)
    // Edge = dinucleotide occurrence (from seq[i] to seq[i+1])
    // We need to find a random Euler path starting at seq[0] and ending at seq[n-1]

    // Collect edges: for each node, list of successor nodes
    var edges: [[UInt8]] = [[], [], [], []]  // edges[from] = [to1, to2, ...]
    for i in 0..<(n - 1) {
        edges[Int(seq[i])].append(seq[i + 1])
    }

    // Shuffle each edge list randomly (this randomizes the Euler path)
    for i in 0..<4 {
        edges[i].shuffle(using: &rng)
    }

    // The last edge from each node to the final nucleotide must be preserved
    // to ensure the path ends correctly. We need to ensure the Euler path
    // ends at seq[n-1].
    //
    // Strategy: For each node, hold back one edge leading toward the final
    // node's predecessor chain. The simplest correct approach: use
    // Hierholzer's algorithm with the shuffled adjacency lists.

    // Track which edge index to use next for each source node
    var edgeIdx = [Int](repeating: 0, count: 4)

    // Hierholzer's algorithm for Euler path
    var stack = [Int(seq[0])]
    var path = [UInt8]()

    while !stack.isEmpty {
        let v = stack.last!
        if edgeIdx[v] < edges[v].count {
            let u = Int(edges[v][edgeIdx[v]])
            edgeIdx[v] += 1
            stack.append(u)
        } else {
            path.append(UInt8(stack.removeLast()))
        }
    }

    // Hierholzer gives the path in reverse
    path.reverse()

    // Verify length
    guard path.count == n else {
        // If Euler path doesn't cover all edges (disconnected graph),
        // fall back to simple shuffle preserving first/last base
        var fallback = seq
        let mid = Array(fallback[1..<(n-1)])
        let shuffled = mid.shuffled(using: &rng)
        for i in 0..<shuffled.count {
            fallback[i + 1] = shuffled[i]
        }
        return fallback
    }

    return path
}

/// Convenience: shuffle with system RNG
public func dinucleotideShuffle(dna: [UInt8]) -> [UInt8] {
    var rng = SystemRandomNumberGenerator()
    return dinucleotideShuffle(dna: dna, rng: &rng)
}

/// Verify that two sequences have identical dinucleotide frequencies
public func verifyDinucleotidePreservation(_ a: [UInt8], _ b: [UInt8]) -> Bool {
    func dinucCounts(_ seq: [UInt8]) -> [Int] {
        var counts = [Int](repeating: 0, count: 16)
        for i in 0..<(seq.count - 1) {
            if seq[i] <= 3 && seq[i+1] <= 3 {
                counts[Int(seq[i]) * 4 + Int(seq[i+1])] += 1
            }
        }
        return counts
    }
    return dinucCounts(a) == dinucCounts(b)
}

/// Seeded RNG for reproducible shuffles
public struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64

    public init(seed: UInt64) {
        self.state = seed
    }

    public mutating func next() -> UInt64 {
        // xorshift64
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}
