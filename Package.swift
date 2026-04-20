// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MetalFFT",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "MetalFFT", targets: ["MetalFFT"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "MetalFFT",
            path: "Sources/MetalFFT",
            resources: [.copy("Resources/fft_multisize.metal")]
        ),
        .testTarget(
            name: "MetalFFTTests",
            dependencies: ["MetalFFT"],
            path: "Tests/MetalFFTTests"
        ),
    ]
)
