// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VideoUpscaler",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "UpscaleKit", targets: ["UpscaleKit"]),
        .executable(name: "upscale", targets: ["UpscaleCLI"]),
        .executable(name: "UpscalerApp", targets: ["UpscalerApp"]),
    ],
    targets: [
        .target(
            name: "UpscaleKit",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "UpscaleCLI",
            dependencies: ["UpscaleKit"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "UpscalerApp",
            dependencies: ["UpscaleKit"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
