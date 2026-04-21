import Foundation

/// Simple frequency smoother using a running average.
public final class FrequencyTracker {
    public let smoothingWindow: Int
    
    private var buffer: [Double] = []
    
    public init(smoothingWindow: Int = 5) {
        self.smoothingWindow = max(1, smoothingWindow)
    }
    
    /// Smooths the frequency using a running average.
    /// Returns the input unchanged if it's <= 0.
    public func track(_ frequency: Double) -> Double {
        guard frequency > 0 else { return frequency }
        
        buffer.append(frequency)
        if buffer.count > smoothingWindow {
            buffer.removeFirst()
        }
        
        return buffer.reduce(0.0, +) / Double(buffer.count)
    }
    
    /// Clears the smoothing buffer.
    public func reset() {
        buffer.removeAll()
    }
}
