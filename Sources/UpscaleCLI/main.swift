import Foundation
import UpscaleKit

let usage = """
使い方: upscale <入力動画> [出力動画] [オプション]

Core ML の超解像モデルで動画を拡大し、指定解像度（既定はフル HD）で書き出します。

オプション:
  --model <path>      Core ML モデル (.mlpackage / .mlmodelc)
                      省略時は ./Models と実行ファイル隣の Models から探します
  --width <n>         出力幅。省略すると高さと元の縦横比から決まる
  --height <n>        出力高さ (既定: 1080)。省略すると幅と元の縦横比から決まる
                      既定は「縦 1080・横は元の比率のまま」なので、
                      16:9 でない素材でも黒帯なしで書き出します
  --short-side <n>    短辺をこの長さに合わせる（--width / --height より優先）
                      横長なら縦が、縦長なら横がこの値になる。
                      縦横が混ざった素材を同じ基準で揃えたいときに
  --fit <mode>        幅と高さを両方指定したときの合わせ方 (既定: contain)
                      contain=元の縦横比のまま、その枠に収まる解像度に縮める（余白なし）
                      fit    =指定解像度に固定し、余白を黒で埋める
                      fill   =指定解像度に固定し、はみ出しを切る
                      stretch=指定解像度に固定し、引き伸ばす
  --fps <n>           出力フレームレート。省略すると元のフレームレートのまま
                      VideoToolbox のフレーム補間 (VTFrameRateConversion) を GPU で
                      使ってフレームを生成してから超解像にかける（macOS 15.4 以降が必要）
  --codec <name>      hevc | h264 | proRes422 (既定: hevc)
  --bitrate <Mbps>    ビットレート        (既定: 解像度とfpsから自動)
  --device <name>     all | cpuAndGPU | cpuAndNeuralEngine | cpuOnly (既定: all)
  --overlap <px>      タイルの重なり幅    (既定: タイルサイズの 1/8)
  --passes <n>        モデルを重ねて適用する回数 (既定: 1)
                      2x モデルで --passes 2 にすると 4 倍まで拡大してから
                      目標解像度へ縮小する（スーパーサンプリング。高画質・低速）
  --no-audio          音声トラックを引き継がない
  -f, --force         出力先が既にあれば上書きする
  --inspect           変換せず、モデルの演算が CPU / GPU / Neural Engine の
                      どれに割り当てられるかを表示する
  --benchmark         変換せず、推論だけを繰り返して素のスループットを測る
                      （並行度を変えて、実行先が埋まっているかを確認できる）
  -h, --help          このヘルプ

例:
  upscale input.mp4                              # 縦 1080・横は元の比率のまま
  upscale input.mp4 --height 1440                # 縦 1440 基準
  upscale input.mp4 --width 1920 --height 1080   # 1920x1080 の枠に収める
  upscale input.mp4 out.mov --model Models/realesr-general-x4v3-256.mlpackage --codec proRes422
  upscale input.mp4 out60.mp4 --fps 60                      # 24fps 素材を 60fps へ補間してから拡大
  upscale --inspect --device cpuAndNeuralEngine
"""

struct ArgumentError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

/// 依存を増やさないための最小限のパーサ。`--key value` と単独フラグだけ扱う。
struct Arguments {
    private var positional: [String] = []
    private var options: [String: String] = [:]
    private var flags: Set<String> = []

    private static let valueOptions: Set<String> = [
        "--model", "--width", "--height", "--fit", "--codec", "--bitrate", "--device", "--overlap",
        "--passes", "--short-side", "--fps",
    ]

    init(_ raw: [String]) throws {
        var iterator = raw.makeIterator()
        while let token = iterator.next() {
            if Self.valueOptions.contains(token) {
                guard let value = iterator.next() else {
                    throw ArgumentError(message: "\(token) には値が必要です")
                }
                options[token] = value
            } else if token.hasPrefix("-") {
                flags.insert(token)
            } else {
                positional.append(token)
            }
        }
    }

    func positional(_ index: Int) -> String? {
        index < positional.count ? positional[index] : nil
    }
    func has(_ flag: String...) -> Bool { flag.contains { flags.contains($0) } }
    func string(_ key: String) -> String? { options[key] }

    func int(_ key: String) throws -> Int? {
        guard let raw = options[key] else { return nil }
        guard let value = Int(raw), value > 0 else { throw ArgumentError(message: "\(key) は正の整数で指定してください: \(raw)") }
        return value
    }

    func double(_ key: String) throws -> Double? {
        guard let raw = options[key] else { return nil }
        guard let value = Double(raw), value > 0 else { throw ArgumentError(message: "\(key) は正の数で指定してください: \(raw)") }
        return value
    }

    func enumValue<T: RawRepresentable & CaseIterable>(_ key: String, as type: T.Type) throws -> T?
    where T.RawValue == String {
        guard let raw = options[key] else { return nil }
        guard let value = T(rawValue: raw) else {
            let choices = T.allCases.map(\.rawValue).joined(separator: " | ")
            throw ArgumentError(message: "\(key) には次のいずれかを指定してください: \(choices)")
        }
        return value
    }
}

/// --model 省略時にモデルを探す場所。
func findDefaultModel() -> URL? {
    let candidates = [
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("Models"),
        URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("Models"),
    ]
    for directory in candidates {
        let found = ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? [])
            .filter { ["mlpackage", "mlmodelc", "mlmodel"].contains($0.pathExtension) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        if let first = found.first { return first }
    }
    return nil
}

/// モデルの演算がどの実行先に割り当てられるかを表示する。
func inspect(modelPath: URL, device: ComputeDevice) async throws {
    guard #available(macOS 14.4, *) else {
        throw ArgumentError(message: "--inspect には macOS 14.4 以降が必要です")
    }
    print("モデル      : \(modelPath.lastPathComponent)")
    print("実行先の指定: \(device.rawValue)")

    let model = try UpscaleModel(url: modelPath, device: device)
    print("入出力      : \(model.summary)")

    let report = try await ModelInspector.report(modelURL: modelPath, device: device)
    print("\n演算の割り当て (全 \(report.totalOperations) 個):")
    for entry in report.preferredCounts {
        let share = Double(entry.count) / Double(max(report.totalOperations, 1)) * 100
        print(String(format: "  %-14@ %4d 個 (%.1f%%)", entry.device as NSString, entry.count, share))
    }

    if report.offloadedOperations.isEmpty {
        print("\nすべての演算が Neural Engine で実行されます。")
    } else {
        print(String(format: "\nNeural Engine 以外に回る演算 (推定コスト合計 %.2f%%):", report.offloadedCost * 100))
        for entry in report.offloadedOperations {
            print(String(format: "  %-16@ %-14@ コスト %.3f%%",
                         entry.operatorName as NSString, entry.device as NSString, entry.cost * 100))
        }
    }

    if !report.notSupportedOnNeuralEngine.isEmpty {
        print("\nNeural Engine が対応していない演算:")
        for entry in report.notSupportedOnNeuralEngine {
            print("  \(entry.operatorName)  \(entry.count) 個")
        }
    }
}

/// 推論だけを回して、実行先が埋まりきっているかを確かめる。
func benchmark(modelPath: URL, device: ComputeDevice) async throws {
    print("モデル      : \(modelPath.lastPathComponent)")
    print("実行先の指定: \(device.rawValue)")
    let model = try UpscaleModel(url: modelPath, device: device)
    print("入出力      : \(model.summary)")
    print("\n推論だけを繰り返したときのスループット:")
    print("  並行度   1秒あたり   1推論あたり   逐次比")

    var baseline: Double = 0
    for concurrency in [1, 2, 3, 4] {
        let result = try await ModelBenchmark.run(model: model, iterations: 60, concurrency: concurrency)
        if concurrency == 1 { baseline = result.inferencesPerSecond }
        let ratio = baseline > 0 ? result.inferencesPerSecond / baseline : 1
        print(String(format: "  %6d   %7.2f 回   %8.1f ms   x%.2f",
                     result.concurrency, result.inferencesPerSecond,
                     result.millisecondsPerInference, ratio))
    }
    print("\n並行度を上げても伸びなければ、その実行先はすでに埋まっています。")
}

func run() async throws {
    let arguments = try Arguments(Array(CommandLine.arguments.dropFirst()))

    if arguments.has("-h", "--help") {
        print(usage)
        return
    }

    let wantsInspect = arguments.has("--inspect")
    let wantsBenchmark = arguments.has("--benchmark")
    if !wantsInspect && !wantsBenchmark && arguments.positional(0) == nil {
        print(usage)
        return
    }

    guard let modelPath = arguments.string("--model").map({ URL(fileURLWithPath: $0) }) ?? findDefaultModel() else {
        throw ArgumentError(message: """
            Core ML モデルが見つかりません。--model でパスを指定するか、Models/ に置いてください。
            モデルは tools/convert_realesrgan.py で作れます:
              python3 tools/convert_realesrgan.py --arch compact --tile 256
            """)
    }

    let device = try arguments.enumValue("--device", as: ComputeDevice.self) ?? .all

    if wantsInspect {
        try await inspect(modelPath: modelPath.standardizedFileURL, device: device)
        return
    }

    if wantsBenchmark {
        try await benchmark(modelPath: modelPath.standardizedFileURL, device: device)
        return
    }

    let input = URL(fileURLWithPath: arguments.positional(0)!).standardizedFileURL
    guard FileManager.default.fileExists(atPath: input.path) else {
        throw ArgumentError(message: "入力ファイルが見つかりません: \(input.path)")
    }

    let shortSide = try arguments.int("--short-side")
    let requestedWidth = try arguments.int("--width")
    let requestedHeight = try arguments.int("--height")
    // どちらも指定がなければ「縦 1080・横は元の比率のまま」。
    let height = shortSide == nil ? (requestedHeight ?? (requestedWidth == nil ? 1080 : nil)) : nil
    let width = shortSide == nil ? requestedWidth : nil

    let output: URL
    if let raw = arguments.positional(1) {
        output = URL(fileURLWithPath: raw).standardizedFileURL
    } else {
        let tag = shortSide.map { "upscaled-\($0)p" }
            ?? height.map { "upscaled-\($0)p" } ?? width.map { "upscaled-w\($0)" } ?? "upscaled"
        output = input.deletingPathExtension()
            .appendingPathExtension(tag)
            .appendingPathExtension("mov")
    }

    if FileManager.default.fileExists(atPath: output.path) {
        guard arguments.has("-f", "--force") else {
            throw ArgumentError(message: "出力先に既にファイルがあります: \(output.path)\n上書きするなら --force を付けてください。")
        }
        try FileManager.default.removeItem(at: output)
    }

    var configuration = UpscaleConfiguration(modelURL: modelPath.standardizedFileURL)
    configuration.outputWidth = width
    configuration.outputHeight = height
    configuration.fitMode = try arguments.enumValue("--fit", as: FitMode.self) ?? .contain
    configuration.codec = try arguments.enumValue("--codec", as: VideoCodec.self) ?? .hevc
    configuration.device = device
    configuration.bitrateMbps = try arguments.double("--bitrate")
    configuration.shortSide = shortSide
    configuration.targetFrameRate = try arguments.double("--fps")
    configuration.tileOverlap = try arguments.int("--overlap")
    let passes = try arguments.int("--passes") ?? 1
    guard passes <= 4 else { throw ArgumentError(message: "--passes は 4 以下で指定してください") }
    configuration.passes = passes
    configuration.copyAudio = !arguments.has("--no-audio")

    print("入力  : \(input.lastPathComponent)")
    print("モデル: \(modelPath.lastPathComponent)")
    var sizeNote: String
    switch (width, height) {
    case let (w?, h?): sizeNote = configuration.fitMode == .contain ? "\(w)x\(h) に収める" : "\(w)x\(h)"
    case let (nil, h?): sizeNote = "縦 \(h)・横は元の比率"
    case let (w?, nil): sizeNote = "横 \(w)・縦は元の比率"
    case (nil, nil): sizeNote = "元の解像度"
    }
    if let shortSide {
        sizeNote = "短辺 \(shortSide)・長辺は元の比率"
    }
    if let targetFrameRate = configuration.targetFrameRate {
        sizeNote += ", \(String(format: "%.3g", targetFrameRate)) fps に変換"
    }
    print("出力  : \(output.path) (\(sizeNote), \(configuration.codec.rawValue))")

    let report = try await VideoUpscaler(configuration: configuration).upscale(input: input, output: output) { progress in
        let percent = progress.fractionCompleted.map { String(format: "%5.1f%%", $0 * 100) } ?? "  --  "
        let line = String(format: "\r  %@  %d フレーム  %.2f fps", percent, progress.framesProcessed, progress.framesPerSecond)
        FileHandle.standardError.write(line.data(using: .utf8)!)
    }
    FileHandle.standardError.write("\r\u{1B}[K".data(using: .utf8)!)

    let fpsNote = report.sourceFrameRate == report.outputFrameRate
        ? "\(String(format: "%.3g", report.outputFrameRate)) fps"
        : "\(String(format: "%.3g", report.sourceFrameRate)) -> \(String(format: "%.3g", report.outputFrameRate)) fps"
    print("""
        完了 ✅
          モデル      : \(report.modelSummary)
          解像度      : \(Int(report.sourceSize.width))x\(Int(report.sourceSize.height)) \
        -> \(Int(report.outputSize.width))x\(Int(report.outputSize.height))
          フレームレート: \(fpsNote)
          フレーム数  : \(report.frameCount) (1 フレーム \(report.tilesPerFrame) タイル)
          所要時間    : \(String(format: "%.1f", report.elapsed)) 秒 (\(String(format: "%.2f", report.framesPerSecond)) fps)
          推論        : \(report.frameCount * report.tilesPerFrame) 回 (\(String(format: "%.1f", report.inferencesPerSecond)) 回/秒)
        """)
}

do {
    try await run()
} catch {
    FileHandle.standardError.write("エラー: \(error.localizedDescription)\n".data(using: .utf8)!)
    exit(1)
}
