// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MetalFFT",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "MetalFFT", targets: ["MetalFFT"]),
        .executable(name: "DNASpectralDemo", targets: ["DNASpectralDemo"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "MetalFFT",
            path: "Sources/MetalFFT",
            resources: [
                .copy("Resources/fft_multisize.metal"),
                .copy("Resources/fft_4096_batched.metal"),
                .copy("Resources/fft_fused_convolve.metal"),
                .copy("Resources/fft_cross_spectral.metal"),
                .copy("Resources/fft_fused_convolve_fp16.metal"),
            ]
        ),
        .executableTarget(
            name: "DNASpectralDemo",
            dependencies: ["MetalFFT"],
            path: "demos",
            sources: ["dna_spectral_demo.swift"]
        ),
        .testTarget(
            name: "MetalFFTTests",
            dependencies: ["MetalFFT"],
            path: "Tests/MetalFFTTests"
        ),
    ]
)
