import Accelerate

// MARK: - Pitch

/// Music-theory utilities: frequency ↔ note name, MIDI note, and cents deviation.
/// All frequencies are `Float` to match the library's GPU-native precision.
public enum Pitch {
    public static let noteNames: [String] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    ]

    /// Maps a frequency to the nearest piano note.
    ///
    /// - Parameters:
    ///   - frequency: Frequency in Hz (must be > 0).
    ///   - referenceA: Tuning reference for A4. Default: 440 Hz.
    /// - Returns: Note name and octave, or `nil` if outside the 88-key piano range.
    public static func note(
        frequency: Float,
        referenceA: Float = 440.0
    ) -> (name: String, octave: Int)? {
        guard frequency > 0 else { return nil }
        let semitonesFromA4 = 12.0 * log2f(frequency / referenceA)
        let noteIndex = Int(semitonesFromA4.rounded()) + 49
        guard noteIndex >= 0, noteIndex < 88 else { return nil }
        return (name: noteNames[(noteIndex + 8) % 12], octave: (noteIndex + 8) / 12)
    }

    /// MIDI note number (0–127) for `frequency`. A4 = 440 Hz = MIDI 69.
    public static func midiNote(
        frequency: Float,
        referenceA: Float = 440.0
    ) -> Int? {
        guard frequency > 0 else { return nil }
        let semitones = 12.0 * log2f(frequency / referenceA)
        let note = Int((69.0 + semitones).rounded())
        guard note >= 0, note <= 127 else { return nil }
        return note
    }

    /// Batch MIDI note lookup for an array of frequencies.
    /// Uses `vvlog2f` + `vDSP` for vectorised log2 and affine scaling.
    public static func midiNotes(
        frequencies: [Float],
        referenceA: Float = 440.0
    ) -> [Int?] {
        guard !frequencies.isEmpty else { return [] }
        var n = Int32(frequencies.count)

        // log2(f) for every frequency
        var logs = [Float](repeating: 0, count: frequencies.count)
        vvlog2f(&logs, frequencies, &n)

        // semitones = 12 * (log2(f) - log2(referenceA)); midi = 69 + semitones
        // 12*(log2(f) - log2(refA)) + 69  ≡  12*log2(f) + (69 - 12*log2(refA))
        var scale: Float = 12.0
        var offset = 69.0 - 12.0 * log2f(referenceA)
        vDSP_vsmsa(logs, 1, &scale, &offset, &logs, 1, vDSP_Length(frequencies.count))

        return logs.map { m in
            let note = Int(m.rounded())
            guard note >= 0, note <= 127 else { return nil }
            return note
        }
    }

    /// Frequency in Hz for a MIDI note number. A4 = MIDI 69 = 440 Hz.
    public static func frequency(midiNote: Int, referenceA: Float = 440.0) -> Float {
        referenceA * powf(2.0, Float(midiNote - 69) / 12.0)
    }

    /// Cents deviation from the nearest equal-tempered semitone. Range: –50 to +50 cents.
    ///
    /// - Parameters:
    ///   - frequency: Frequency in Hz (must be > 0).
    ///   - referenceA: Tuning reference for A4. Default: 440 Hz.
    public static func centsDeviation(
        frequency: Float,
        referenceA: Float = 440.0
    ) -> Float? {
        guard frequency > 0 else { return nil }
        let centsFromA4 = 1200.0 * log2f(frequency / referenceA)
        let nearestSemitone = (centsFromA4 / 100.0).rounded()
        return centsFromA4 - nearestSemitone * 100.0
    }
}
