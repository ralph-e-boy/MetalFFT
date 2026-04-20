import Foundation

// MARK: - Pitch

/// Music-theory utilities: frequency ↔ note name and cents deviation.
public enum Pitch {

    public static let noteNames: [String] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ]

    /// Maps a frequency to the nearest piano note.
    ///
    /// - Parameters:
    ///   - frequency: Frequency in Hz (must be > 0).
    ///   - referenceA: Tuning reference for A4. Default: 440.0 Hz.
    /// - Returns: Note name and octave number, or `nil` if frequency is outside the 88-key piano range.
    public static func note(
        frequency: Double,
        referenceA: Double = 440.0
    ) -> (name: String, octave: Int)? {
        guard frequency > 0 else { return nil }
        let semitonesFromA4 = 12.0 * log2(frequency / referenceA)
        let noteIndex = Int(semitonesFromA4.rounded()) + 49
        guard noteIndex >= 0, noteIndex < 88 else { return nil }
        let octave = (noteIndex + 8) / 12
        let name = noteNames[(noteIndex + 8) % 12]
        return (name: name, octave: octave)
    }

    /// MIDI note number (0–127) for `frequency`. A4 = 440 Hz = MIDI 69.
    public static func midiNote(
        frequency: Double,
        referenceA: Double = 440.0
    ) -> Int? {
        guard frequency > 0 else { return nil }
        let semitones = 12.0 * log2(frequency / referenceA)
        let note = Int((69.0 + semitones).rounded())
        guard note >= 0, note <= 127 else { return nil }
        return note
    }

    /// Frequency in Hz for a MIDI note number. A4 = MIDI 69 = 440 Hz.
    public static func frequency(midiNote: Int, referenceA: Double = 440.0) -> Double {
        referenceA * pow(2.0, Double(midiNote - 69) / 12.0)
    }

    /// Returns the cents deviation from the nearest equal-tempered semitone.
    ///
    /// Range: –50 to +50 cents.
    ///
    /// - Parameters:
    ///   - frequency: Frequency in Hz (must be > 0).
    ///   - referenceA: Tuning reference for A4. Default: 440.0 Hz.
    public static func centsDeviation(
        frequency: Double,
        referenceA: Double = 440.0
    ) -> Double? {
        guard frequency > 0 else { return nil }
        let centsFromA4 = 1200.0 * log2(frequency / referenceA)
        let nearestSemitone = (centsFromA4 / 100.0).rounded()
        return centsFromA4 - nearestSemitone * 100.0
    }
}
