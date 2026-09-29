import AVFoundation
import Foundation
import Observation
import UpscaleKit

/// フレームレート変換（VTFrameRateConversion）のプリセット。macOS 15.4 以降が必要。
enum FrameRatePreset: String, CaseIterable, Identifiable {
    case original = "元のまま"
    case fps30 = "30"
    case fps60 = "60"

    var id: String { rawValue }
    /// `UpscaleConfiguration.targetFrameRate` に渡す値。「元のまま」は nil。
    var targetFrameRate: Double? {
        switch self {
        case .original: nil
        case .fps30: 30
        case .fps60: 60
        }
    }
}

enum OutputPreset: String, CaseIterable, Identifiable {
    case fullHD = "フル HD（短辺 1080）"
    case wqhd = "WQHD（短辺 1440）"
    case uhd4K = "4K UHD（短辺 2160）"

    var id: String { rawValue }
    /// 基準にする短辺の長さ。横長なら縦が、縦長なら横がこの値になる。
    var shortSide: Int {
        switch self {
        case .fullHD: 1080
        case .wqhd: 1440
        case .uhd4K: 2160
        }
    }
}

@MainActor
@Observable
final class UpscaleController {
    /// バッチキューの 1 件。
    struct QueueItem: Identifiable {
        let id = UUID()
        let url: URL
        var status: Status = .pending
        /// 入力動画の情報（解像度・尺）。読み込み後に入る。
        var mediaSummary: String?
        /// 回転を反映した表示上の解像度。
        var displaySize: CGSize?
        /// 完了時間・エラー内容など、行に添える短い説明。
        var detail: String?
        var fraction: Double?
        var outputURL: URL?

        enum Status {
            case pending, running, done, failed, skipped, cancelled
        }
    }

    var queue: [QueueItem] = []
    var modelURL: URL?
    var availableModels: [URL] = []
    var preset: OutputPreset = .fullHD
    /// 既定は元の縦横比を保つ。黒帯を出したい場合だけ .fit などに変える。
    var fitMode: FitMode = .contain
    var codec: VideoCodec = .hevc
    /// フレームレート変換のプリセット。既定は元のフレームレートのまま。
    var frameRatePreset: FrameRatePreset = .original
    /// モデルを 2 回重ねて適用する（スーパーサンプリング）。高画質・低速。
    var doublePass = false
    /// 既定は Neural Engine 固定。`.all` は Core ML 任せで GPU に振られることがあり、
    /// 実測でも回によって速度が落ち込んだため。
    var device: ComputeDevice = .cpuAndNeuralEngine {
        didSet { refreshDeviceBreakdown() }
    }

    var isRunning = false
    var statusText = ""
    var errorMessage: String?
    /// 選択中のモデルの演算が、どの実行先に割り当てられるかの要約。
    var deviceBreakdown: String?

    private var task: Task<Void, Never>?

    private static let lastModelKey = "lastModelPath"

    init() {
        reloadModels()
        if let saved = UserDefaults.standard.string(forKey: Self.lastModelKey),
           FileManager.default.fileExists(atPath: saved) {
            modelURL = URL(fileURLWithPath: saved)
        } else {
            modelURL = availableModels.first
        }
        refreshDeviceBreakdown()
    }

    var canStart: Bool { !queue.isEmpty && modelURL != nil && !isRunning }

    // MARK: キュー操作

    func addInputs(_ urls: [URL]) {
        guard !isRunning else { return }
        for url in urls {
            let standardized = url.standardizedFileURL
            guard !queue.contains(where: { $0.url.path == standardized.path }) else { continue }
            let item = QueueItem(url: standardized)
            queue.append(item)
            Task { await loadSummary(for: item.id) }
        }
    }

    func removeItem(_ id: QueueItem.ID) {
        guard !isRunning else { return }
        queue.removeAll { $0.id == id }
    }

    func clearQueue() {
        guard !isRunning else { return }
        queue.removeAll()
    }

    private func loadSummary(for id: QueueItem.ID) async {
        guard let index = queue.firstIndex(where: { $0.id == id }) else { return }
        let asset = AVURLAsset(url: queue[index].url)
        guard let track = try? await asset.loadTracks(withMediaType: .video).first,
              let loaded = try? await track.load(.naturalSize, .preferredTransform),
              let duration = try? await asset.load(.duration)
        else {
            update(id) { $0.mediaSummary = "動画情報を読み取れませんでした" }
            return
        }
        let (size, transform) = loaded
        // 縦動画などは transform に 90/270 度の回転が入るので、見た目の向きに直して扱う。
        let isRotatedQuarterTurn = abs(transform.b) > 0.5 && abs(transform.c) > 0.5
        let displaySize = isRotatedQuarterTurn
            ? CGSize(width: size.height, height: size.width)
            : size
        update(id) {
            $0.displaySize = displaySize
            $0.mediaSummary = String(
                format: "%d x %d ・ %.1f 秒",
                Int(displaySize.width), Int(displaySize.height), duration.seconds
            )
        }
    }

    private func update(_ id: QueueItem.ID, _ change: (inout QueueItem) -> Void) {
        guard let index = queue.firstIndex(where: { $0.id == id }) else { return }
        change(&queue[index])
    }

    // MARK: モデル

    /// Models/ を探してモデル候補を集める。
    func reloadModels() {
        let directories = [
            // アプリバンドルに同梱したもの
            Bundle.main.resourceURL?.appendingPathComponent("Models"),
            // リポジトリ内で直接動かしたとき
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("Models"),
            URL(fileURLWithPath: CommandLine.arguments[0])
                .deletingLastPathComponent().appendingPathComponent("Models"),
        ].compactMap { $0 }
        var found: [URL] = []
        for directory in directories {
            let contents = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
            found += contents.filter { ["mlpackage", "mlmodelc", "mlmodel"].contains($0.pathExtension) }
        }
        // 同じ実体を指すパスが複数出るので、ファイル名で重複を落とす。
        var seen = Set<String>()
        availableModels = found.filter { seen.insert($0.lastPathComponent).inserted }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    func selectModel(_ url: URL) {
        modelURL = url
        UserDefaults.standard.set(url.path, forKey: Self.lastModelKey)
        if !availableModels.contains(where: { $0.path == url.path }) {
            availableModels.append(url)
        }
        refreshDeviceBreakdown()
    }

    /// MLComputePlan で演算の割り当てを読み、Neural Engine に載る割合を出す。
    private func refreshDeviceBreakdown() {
        guard let modelURL else {
            deviceBreakdown = nil
            return
        }
        guard #available(macOS 14.4, *) else {
            deviceBreakdown = nil
            return
        }
        let device = self.device
        deviceBreakdown = "実行先を確認しています…"
        Task {
            do {
                let report = try await ModelInspector.report(modelURL: modelURL, device: device)
                deviceBreakdown = String(
                    format: "Neural Engine %d / %d 演算 ・ ほかに回る演算の推定コスト %.2f%%",
                    report.neuralEngineOperations, report.totalOperations, report.offloadedCost * 100
                )
            } catch {
                deviceBreakdown = nil
            }
        }
    }

    // MARK: 出力先

    /// 縦横比を変える指定のときに使う枠（16:9 換算）。
    /// `.contain` のときは短辺基準で決めるのでこちらは使わない。
    private var framedBounds: (width: Int?, height: Int?) {
        let height = preset.shortSide
        return (height * 16 / 9, height)
    }

    /// 1 件ぶんの書き出し解像度を求める。
    private func outputSize(for displaySize: CGSize) -> (width: Int, height: Int) {
        let width = Int(displaySize.width.rounded())
        let height = Int(displaySize.height.rounded())
        guard fitMode == .contain else {
            let bounds = framedBounds
            return OutputGeometry.resolve(
                sourceWidth: width, sourceHeight: height,
                boundsWidth: bounds.width, boundsHeight: bounds.height,
                fitMode: fitMode
            )
        }
        // 縦動画でも横動画でも短辺を揃える。
        return OutputGeometry.resolveShortSide(sourceWidth: width, sourceHeight: height, length: preset.shortSide)
    }

    /// 実際に書き出される解像度。キューが 1 件で読み込み済みのときだけ出す。
    var plannedOutputSize: String? {
        guard queue.count == 1, let size = queue[0].displaySize,
              size.width > 0, size.height > 0 else { return nil }
        let resolved = outputSize(for: size)
        return "\(resolved.width) x \(resolved.height)"
    }

    /// 既定の出力ファイル名（入力名 + 解像度タグ）。
    func defaultOutputName(for item: QueueItem) -> String {
        item.url.deletingPathExtension().lastPathComponent + ".upscaled-\(preset.shortSide)p.mov"
    }

    // MARK: 実行

    /// キューを順番に処理する。`outputs` は項目 ID → 書き出し先。
    /// `overwriteExisting` が false のとき、既にあるファイルは消さずにスキップする。
    func start(outputs: [QueueItem.ID: URL], overwriteExisting: Bool) {
        guard canStart, !outputs.isEmpty else { return }

        guard let modelURL else { return }
        var configuration = UpscaleConfiguration(modelURL: modelURL)
        if fitMode == .contain {
            configuration.shortSide = preset.shortSide
        } else {
            configuration.outputWidth = framedBounds.width
            configuration.outputHeight = framedBounds.height
        }
        configuration.fitMode = fitMode
        configuration.codec = codec
        configuration.device = device
        configuration.targetFrameRate = frameRatePreset.targetFrameRate
        configuration.passes = doublePass ? 2 : 1

        isRunning = true
        errorMessage = nil
        statusText = "モデルを読み込んでいます…"
        for index in queue.indices {
            queue[index].status = .pending
            queue[index].fraction = nil
            queue[index].detail = nil
            queue[index].outputURL = nil
        }
        let total = queue.count

        task = Task {
            var doneCount = 0
            var failedCount = 0
            var skippedCount = 0

            for index in queue.indices {
                guard !Task.isCancelled else {
                    queue[index].status = .cancelled
                    continue
                }
                let item = queue[index]
                guard let output = outputs[item.id] else {
                    queue[index].status = .skipped
                    queue[index].detail = "出力先が決まっていません"
                    skippedCount += 1
                    continue
                }

                if FileManager.default.fileExists(atPath: output.path) {
                    if overwriteExisting {
                        try? FileManager.default.removeItem(at: output)
                    } else {
                        queue[index].status = .skipped
                        queue[index].detail = "出力先に既にファイルがあります: \(output.lastPathComponent)"
                        skippedCount += 1
                        continue
                    }
                }

                queue[index].status = .running
                queue[index].fraction = nil
                let position = "\(doneCount + failedCount + skippedCount + 1) / \(total) 件目"
                statusText = "\(position): \(item.url.lastPathComponent)"

                do {
                    let id = item.id
                    let report = try await VideoUpscaler(configuration: configuration)
                        .upscale(input: item.url, output: output) { progress in
                            Task { @MainActor [weak self] in
                                guard let self else { return }
                                self.update(id) { $0.fraction = progress.fractionCompleted }
                                self.statusText = String(
                                    format: "%@: %@ ・ %.1f fps",
                                    position, item.url.lastPathComponent, progress.framesPerSecond
                                )
                            }
                        }
                    queue[index].status = .done
                    queue[index].fraction = 1
                    queue[index].outputURL = report.outputURL
                    queue[index].detail = String(
                        format: "%d x %d ・ %.1f 秒 (%.1f fps)",
                        Int(report.outputSize.width), Int(report.outputSize.height),
                        report.elapsed, report.framesPerSecond
                    )
                    doneCount += 1
                } catch is CancellationError {
                    queue[index].status = .cancelled
                    queue[index].fraction = nil
                } catch {
                    queue[index].status = .failed
                    queue[index].fraction = nil
                    queue[index].detail = error.localizedDescription
                    failedCount += 1
                }
            }

            if Task.isCancelled {
                statusText = "キャンセルしました（完了 \(doneCount) 件）"
            } else {
                var parts = ["完了 \(doneCount) 件"]
                if skippedCount > 0 { parts.append("スキップ \(skippedCount) 件") }
                if failedCount > 0 { parts.append("エラー \(failedCount) 件") }
                statusText = parts.joined(separator: " ・ ")
            }
            isRunning = false
        }
    }

    func cancel() {
        task?.cancel()
    }

    /// バッチ全体の進み具合（完了件数 + 現在のファイルの進捗）。
    var overallFraction: Double? {
        guard isRunning, !queue.isEmpty else { return nil }
        var finished = 0.0
        for item in queue {
            switch item.status {
            case .done, .failed, .skipped, .cancelled: finished += 1
            case .running: finished += item.fraction ?? 0
            case .pending: break
            }
        }
        return finished / Double(queue.count)
    }
}
