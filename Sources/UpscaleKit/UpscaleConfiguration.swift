import AVFoundation
import Foundation

public enum VideoCodec: String, CaseIterable, Sendable {
    case hevc
    case h264
    case proRes422

    var avCodec: AVVideoCodecType {
        switch self {
        case .hevc: .hevc
        case .h264: .h264
        case .proRes422: .proRes422
        }
    }

    /// 自動ビットレート算出に使う「1 ピクセル 1 フレームあたりのビット数」。
    var bitsPerPixel: Double {
        switch self {
        case .hevc: 0.16
        case .h264: 0.24
        case .proRes422: 0  // 固定ビットレート指定を持たない
        }
    }
}

public struct UpscaleConfiguration: Sendable {
    /// 超解像に使う Core ML モデル (.mlpackage / .mlmodelc)。
    public var modelURL: URL
    /// 書き出し解像度。片方を nil にすると、元の縦横比からもう片方に合わせて決まる。
    /// 既定は「縦 1080・横は元の比率のまま」。
    public var outputWidth: Int?
    public var outputHeight: Int? = 1080
    /// 指定すると `outputWidth` / `outputHeight` より優先し、短辺をこの長さに合わせる。
    /// 横長なら縦が、縦長なら横がこの値になるので、向きが混ざっていても同じ基準で揃う。
    public var shortSide: Int?
    /// 幅と高さを両方指定したときの、縦横比の合わせ方。既定は枠に収める `.contain`。
    public var fitMode: FitMode = .contain
    public var codec: VideoCodec = .hevc
    /// nil なら解像度とフレームレートから自動算出する。
    public var bitrateMbps: Double?
    public var device: ComputeDevice = .all
    /// 目標フレームレート。nil なら元のフレームレートのまま（既定）。
    /// 指定すると VideoToolbox の VTFrameRateConversion（GPU 上の ML 補間）でフレームを
    /// 生成してから超解像にかける。macOS 15.4 以降が必要。
    public var targetFrameRate: Double?
    /// タイルの重なり幅（入力ピクセル）。nil ならタイルサイズから自動。
    public var tileOverlap: Int?
    /// モデルを何回重ねて適用するか。2 にすると 2x モデルで 4 倍まで拡大してから
    /// 目標解像度へ縮小する（スーパーサンプリング）。その分、処理は重くなる。
    public var passes: Int = 1
    /// 音声トラックを再エンコードせずにコピーする。
    public var copyAudio: Bool = true

    public init(modelURL: URL) {
        self.modelURL = modelURL
    }
}

public struct UpscaleProgress: Sendable {
    public let framesProcessed: Int
    /// 元動画から推定した総フレーム数。推定できなければ nil。
    public let totalFrames: Int?
    public let framesPerSecond: Double

    public var fractionCompleted: Double? {
        guard let totalFrames, totalFrames > 0 else { return nil }
        return min(1.0, Double(framesProcessed) / Double(totalFrames))
    }
}

public struct UpscaleReport: Sendable {
    public let outputURL: URL
    public let sourceSize: CGSize
    public let outputSize: CGSize
    public let frameCount: Int
    public let tilesPerFrame: Int
    public let elapsed: TimeInterval
    /// 実行した推論の回数（フレーム数 × 1 フレームあたりのタイル数）。
    public let modelSummary: String
    /// 元動画のフレームレート。
    public let sourceFrameRate: Double
    /// 書き出したフレームレート（`--fps` 未指定なら `sourceFrameRate` と同じ）。
    public let outputFrameRate: Double

    public var framesPerSecond: Double { elapsed > 0 ? Double(frameCount) / elapsed : 0 }
    /// 1 秒あたりの推論回数。`--benchmark` が出す上限と直接比べられる。
    public var inferencesPerSecond: Double {
        elapsed > 0 ? Double(frameCount * tilesPerFrame) / elapsed : 0
    }
}
