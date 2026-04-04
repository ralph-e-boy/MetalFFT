// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DNASpectralAnalysis",
    platforms: [.macOS(.v13)],
    targets: [
        .target(
            name: "DNALib",
            path: "Sources/DNALib",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "DNASpectralAnalysis",
            dependencies: ["DNALib"],
            path: "Sources/DNASpectralAnalysis",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "EcoliAnalysis",
            dependencies: ["DNALib"],
            path: "Sources/EcoliAnalysis",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "WelchCoherenceAnalysis",
            dependencies: ["DNALib"],
            path: "Sources/WelchCoherenceAnalysis",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "SpectralVariant",
            dependencies: ["DNALib"],
            path: "Sources/SpectralVariant",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "HumanAnalysis",
            dependencies: ["DNALib"],
            path: "Sources/HumanAnalysis",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "Period4Investigation",
            dependencies: ["DNALib"],
            path: "Sources/Period4Investigation",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "CrossSpeciesAnalysis",
            dependencies: ["DNALib"],
            path: "Sources/CrossSpeciesAnalysis",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "CrossSpeciesAnalysis2",
            dependencies: ["DNALib"],
            path: "Sources/CrossSpeciesAnalysis2",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "NatureCrossSpecies",
            dependencies: ["DNALib"],
            path: "Sources/NatureCrossSpecies",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "MultilevelAnalysis",
            dependencies: ["DNALib"],
            path: "Sources/MultilevelAnalysis",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "NullModelTest",
            dependencies: ["DNALib"],
            path: "Sources/NullModelTest",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "BiologicalValidation",
            dependencies: ["DNALib"],
            path: "Sources/BiologicalValidation",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "SpectralAnomaly",
            dependencies: ["DNALib"],
            path: "Sources/SpectralAnomaly",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "InvestigateAnomalies",
            dependencies: ["DNALib"],
            path: "Sources/InvestigateAnomalies",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "PfalciparumAnomaly",
            dependencies: ["DNALib"],
            path: "Sources/PfalciparumAnomaly",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "PfDeepInvestigation",
            dependencies: ["DNALib"],
            path: "Sources/PfDeepInvestigation",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "PfGenomeWide",
            dependencies: ["DNALib"],
            path: "Sources/PfGenomeWide",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
    ]
)
