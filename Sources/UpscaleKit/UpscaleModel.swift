import CoreML
import CoreVideo
import Foundation

/// 超解像モデルの実行先。
public enum ComputeDevice: String, CaseIterable, Sendable {
    case all
    case cpuAndGPU
    case cpuAndNeuralEngine
    case cpuOnly

    var mlComputeUnits: MLComputeUnits {
        switch self {
        case .all: .all
        case .cpuAndGPU: .cpuAndGPU
        case .cpuAndNeuralEngine: .cpuAndNeuralEngine
        case .cpuOnly: .cpuOnly
        }
    }
}

/// 画像 1 枚を入力・画像 1 枚を出力する超解像 Core ML モデルのラッパー。
///
/// 入出力の名前・解像度はモデルから読み取るので、Real-ESRGAN でも
/// 自前の SR モデルでも「画像 in / 画像 out」であればそのまま差し替えられる。
public final class UpscaleModel: @unchecked Sendable {
    private let model: MLModel
    private let inputName: String
    private let outputName: String

    /// モデルが受け付ける入力解像度（固定）。
    public let inputWidth: Int
    public let inputHeight: Int
    /// 実測した出力解像度。
    public let outputWidth: Int
    public let outputHeight: Int

    /// 拡大倍率（横方向）。x2 モデルなら 2、x4 モデルなら 4。
    public var scale: Double { Double(outputWidth) / Double(inputWidth) }

    public var summary: String {
        "\(inputWidth)x\(inputHeight) -> \(outputWidth)x\(outputHeight) (x\(String(format: "%.2g", scale)))"
    }

    public init(url: URL, device: ComputeDevice = .all) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw UpscaleError.modelNotFound(url.path)
        }
        let compiledURL = try Self.compiledModelURL(for: url)

        let configuration = MLModelConfiguration()
        configuration.computeUnits = device.mlComputeUnits
        let model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        self.model = model

        let description = model.modelDescription
        guard let input = description.inputDescriptionsByName.first(where: { $0.value.type == .image }) else {
            throw UpscaleError.modelInterfaceMismatch("画像入力が見つかりません")
        }
        guard let output = description.outputDescriptionsByName.first(where: { $0.value.type == .image }) else {
            throw UpscaleError.modelInterfaceMismatch("画像出力が見つかりません")
        }
        guard let constraint = input.value.imageConstraint else {
            throw UpscaleError.modelInterfaceMismatch("入力の解像度制約が取得できません")
        }
        inputName = input.key
        outputName = output.key
        inputWidth = constraint.pixelsWide
        inputHeight = constraint.pixelsHigh

        // 出力サイズは imageConstraint が空のこともあるため、黒フレームを 1 回通して実測する。
        // 初回推論のウォームアップも兼ねる。
        let probePool = try PixelBufferPool(width: inputWidth, height: inputHeight)
        let probe = try probePool.makeBuffer()
        try probe.fillBlack()
        let result = try Self.predict(model: model, inputName: inputName, outputName: outputName, pixelBuffer: probe)
        outputWidth = result.pixelWidth
        outputHeight = result.pixelHeight

        guard outputWidth > inputWidth || outputHeight > inputHeight else {
            throw UpscaleError.modelInterfaceMismatch(
                "出力 (\(outputWidth)x\(outputHeight)) が入力 (\(inputWidth)x\(inputHeight)) より大きくありません"
            )
        }
    }

    /// `inputWidth` x `inputHeight` のバッファを 1 枚推論する。
    public func upscale(_ pixelBuffer: CVPixelBuffer) throws -> CVPixelBuffer {
        try Self.predict(model: model, inputName: inputName, outputName: outputName, pixelBuffer: pixelBuffer)
    }

    private static func predict(
        model: MLModel,
        inputName: String,
        outputName: String,
        pixelBuffer: CVPixelBuffer
    ) throws -> CVPixelBuffer {
        let features = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(pixelBuffer: pixelBuffer)
        ])
        let prediction = try model.prediction(from: features)
        guard let buffer = prediction.featureValue(for: outputName)?.imageBufferValue else {
            throw UpscaleError.modelInterfaceMismatch("出力 '\(outputName)' が画像として取得できません")
        }
        return buffer
    }

    // MARK: - コンパイル済みモデルのキャッシュ

    /// .mlpackage / .mlmodel は実行前にコンパイルが要る。毎回やると数秒かかるのでキャッシュする。
    public static func compiledModelURL(for url: URL) throws -> URL {
        if url.pathExtension == "mlmodelc" { return url }

        let fm = FileManager.default
        let cacheDirectory = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CoreMLVideoUpscaler", isDirectory: true)
        try fm.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)

        let cached = cacheDirectory
            .appendingPathComponent("\(url.deletingPathExtension().lastPathComponent)-\(fnv1a(url.path)).mlmodelc")

        if fm.fileExists(atPath: cached.path), try isUpToDate(cached: cached, source: url) {
            return cached
        }

        let compiled = try MLModel.compileModel(at: url)
        if fm.fileExists(atPath: cached.path) {
            try fm.removeItem(at: cached)
        }
        // compileModel の出力は一時ディレクトリに置かれ、いつ消えるか保証がないので退避する。
        try fm.moveItem(at: compiled, to: cached)
        return cached
    }

    private static func isUpToDate(cached: URL, source: URL) throws -> Bool {
        let keys: Set<URLResourceKey> = [.contentModificationDateKey]
        guard
            let cachedDate = try cached.resourceValues(forKeys: keys).contentModificationDate,
            let sourceDate = try source.resourceValues(forKeys: keys).contentModificationDate
        else { return false }
        return cachedDate >= sourceDate
    }

    private static func fnv1a(_ string: String) -> String {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return String(hash, radix: 36)
    }
}
