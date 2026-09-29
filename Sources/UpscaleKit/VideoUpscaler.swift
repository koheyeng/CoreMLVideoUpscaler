import AVFoundation
import CoreVideo
import Foundation

/// 1 フレーム分の拡大処理。結果は Sendable でない CVPixelBuffer なので箱に入れて渡す。
private typealias FrameTask = Task<UnsafeSendableBox<CVPixelBuffer>, Error>

/// 動画ファイルを 1 フレームずつ Core ML で拡大し、指定解像度で書き出す。
public final class VideoUpscaler {
    private let configuration: UpscaleConfiguration

    public init(configuration: UpscaleConfiguration) {
        self.configuration = configuration
    }

    /// `input` を拡大して `output` に書き出す。
    ///
    /// `output` に既にファイルがある場合はエラーにする（消すかどうかは呼び出し側の判断）。
    /// キャンセルは Swift の Task キャンセルで行う。
    @discardableResult
    public func upscale(
        input: URL,
        output: URL,
        progress: (@Sendable (UpscaleProgress) -> Void)? = nil
    ) async throws -> UpscaleReport {
        let fileManager = FileManager.default
        guard !fileManager.fileExists(atPath: output.path) else {
            throw UpscaleError.writerFailed("出力先に既にファイルがあります: \(output.path)")
        }

        let startedAt = Date()
        let asset = AVURLAsset(url: input)
        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
            throw UpscaleError.noVideoTrack
        }
        let (naturalSize, transform, nominalFrameRate, formatDescriptions) = try await videoTrack.load(
            .naturalSize, .preferredTransform, .nominalFrameRate, .formatDescriptions
        )
        let duration = try await asset.load(.duration)
        let frameRate = nominalFrameRate > 0 ? Double(nominalFrameRate) : 30
        // 書き出しの色タグは常にソースへ合わせる（タグが無ければ BT.709 にフォールバック）。
        // 合わせないと常に BT.709 として書き出され、sRGB タグの素材などで再生側の解釈がずれる。
        let colorTags = SourceColorTags.from(formatDescriptions.first)

        let usesFrameRateConversion = configuration.targetFrameRate != nil
        if let targetFrameRate = configuration.targetFrameRate {
            guard targetFrameRate > 0 else {
                throw UpscaleError.featureUnavailable("フレームレートは正の値で指定してください")
            }
            var supported = false
            if #available(macOS 15.4, *) {
                supported = FrameRateConverter.isSupported
            }
            guard supported else {
                throw UpscaleError.featureUnavailable(
                    "フレームレート変換 (--fps) には macOS 15.4 以降と対応 GPU が必要です"
                )
            }
        }
        // --fps 指定時は出力フレーム数・進捗も変換後の値で数える。
        let outputFrameRate = configuration.targetFrameRate ?? frameRate
        let estimatedFrames = duration.seconds.isFinite ? Int((duration.seconds * outputFrameRate).rounded()) : nil

        let sourceWidth = Int(naturalSize.width.rounded())
        let sourceHeight = Int(naturalSize.height.rounded())

        // 縦動画などは preferredTransform に 90/270 度の回転が入る。デコード後のバッファは
        // 回転前の向きなので、目標解像度の方を入れ替えて、回転はメタデータとして引き継ぐ。
        let isRotatedQuarterTurn = abs(transform.b) > 0.5 && abs(transform.c) > 0.5
        let (targetWidth, targetHeight): (Int, Int)
        if let shortSide = configuration.shortSide {
            // 短辺基準は向きに依存しないので、回転の有無で入れ替える必要がない。
            (targetWidth, targetHeight) = OutputGeometry.resolveShortSide(
                sourceWidth: sourceWidth,
                sourceHeight: sourceHeight,
                length: shortSide
            )
        } else {
            (targetWidth, targetHeight) = OutputGeometry.resolve(
                sourceWidth: sourceWidth,
                sourceHeight: sourceHeight,
                boundsWidth: isRotatedQuarterTurn ? configuration.outputHeight : configuration.outputWidth,
                boundsHeight: isRotatedQuarterTurn ? configuration.outputWidth : configuration.outputHeight,
                fitMode: configuration.fitMode
            )
        }

        let model = try UpscaleModel(url: configuration.modelURL, device: configuration.device)

        // モデルを passes 回重ねる。段ごとに入力解像度が変わるので TiledUpscaler は段別に持つ。
        // MLModel 自体はスレッドセーフなので全段で共有する。
        var stages: [TiledUpscaler] = []
        var stageWidth = sourceWidth
        var stageHeight = sourceHeight
        for _ in 0..<max(1, configuration.passes) {
            let stage = try TiledUpscaler(
                model: model,
                sourceWidth: stageWidth,
                sourceHeight: stageHeight,
                overlap: configuration.tileOverlap
            )
            stages.append(stage)
            stageWidth = stage.upscaledWidth
            stageHeight = stage.upscaledHeight
        }
        let tiled = stages[stages.count - 1]

        // MARK: リーダー
        let reader = try AVAssetReader(asset: asset)
        let videoOutput = AVAssetReaderTrackOutput(
            track: videoTrack,
            outputSettings: [
                // フレームレート変換ありなら VTFrameRateConversion が要求する 64RGBAHalf で
                // 読み出す（TiledUpscaler へ渡す前に 32BGRA へ変換する）。
                kCVPixelBufferPixelFormatTypeKey as String: usesFrameRateConversion
                    ? kCVPixelFormatType_64RGBAHalf : kUpscalePixelFormat,
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
            ]
        )
        videoOutput.alwaysCopiesSampleData = false
        guard reader.canAdd(videoOutput) else { throw UpscaleError.readerFailed("映像トラックを追加できません") }
        reader.add(videoOutput)

        var audioOutput: AVAssetReaderTrackOutput?
        if configuration.copyAudio, let audioTrack = try await asset.loadTracks(withMediaType: .audio).first {
            // outputSettings が nil なら圧縮済みのまま取り出せる＝再エンコードなしでコピーできる。
            let output = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: nil)
            output.alwaysCopiesSampleData = false
            if reader.canAdd(output) {
                reader.add(output)
                audioOutput = output
            }
        }

        // MARK: ライター
        let fileType: AVFileType = output.pathExtension.lowercased() == "mp4" ? .mp4 : .mov
        let writer = try AVAssetWriter(outputURL: output, fileType: fileType)
        let videoInput = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: videoSettings(
                width: targetWidth, height: targetHeight, frameRate: outputFrameRate, colorTags: colorTags
            )
        )
        videoInput.expectsMediaDataInRealTime = false
        videoInput.transform = transform
        guard writer.canAdd(videoInput) else { throw UpscaleError.writerFailed("映像入力を追加できません") }
        writer.add(videoInput)

        var audioInput: AVAssetWriterInput?
        if audioOutput != nil {
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: nil)
            input.expectsMediaDataInRealTime = false
            if writer.canAdd(input) {
                writer.add(input)
                audioInput = input
            } else {
                audioOutput = nil
            }
        }

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: videoInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kUpscalePixelFormat,
                kCVPixelBufferWidthKey as String: targetWidth,
                kCVPixelBufferHeightKey as String: targetHeight,
            ]
        )

        guard writer.startWriting() else {
            throw UpscaleError.writerFailed(writer.error?.localizedDescription ?? "startWriting に失敗しました")
        }
        guard reader.startReading() else {
            throw UpscaleError.readerFailed(reader.error?.localizedDescription ?? "startReading に失敗しました")
        }

        // 出力タイムラインの基準時刻。音声はソースの PTS をそのまま引き継ぐため、
        // 映像側もこれに合わせないと音ズレする。
        //   - 変換なし: 実際に読めた最初のフレームの PTS
        //     （先頭サンプルの PTS は 0 とは限らない。この素材も 0.023 秒から始まる。
        //      セッションを 0 から開くと冒頭に空白ができ、再生時に 1 フレーム余分に見える）
        //   - フレームレート変換あり: FrameRateConverter が同じ基準で計算する
        var lastFrameEnd: CMTime?
        let sessionStartTime: CMTime
        let nextFrame: () async throws -> (buffer: CVPixelBuffer, time: CMTime)?

        if usesFrameRateConversion {
            guard #available(macOS 15.4, *) else {
                // 冒頭で確認済みなので実際には来ない。
                reader.cancelReading()
                writer.cancelWriting()
                try? fileManager.removeItem(at: output)
                throw UpscaleError.featureUnavailable("フレームレート変換 (--fps) には macOS 15.4 以降が必要です")
            }
            let converter: FrameRateConverter
            do {
                converter = try FrameRateConverter(
                    sourceOutput: videoOutput,
                    frameWidth: sourceWidth,
                    frameHeight: sourceHeight,
                    sourceDuration: duration,
                    targetFrameRate: configuration.targetFrameRate!,
                    allowInterpolation: configuration.targetFrameRate! > frameRate,
                    colorTags: colorTags
                )
            } catch {
                reader.cancelReading()
                writer.cancelWriting()
                try? fileManager.removeItem(at: output)
                throw error
            }
            sessionStartTime = converter.sessionStart
            lastFrameEnd = converter.endTime
            nextFrame = { try await converter.next() }
        } else {
            guard let firstSample = videoOutput.copyNextSampleBuffer() else {
                reader.cancelReading()
                writer.cancelWriting()
                try? fileManager.removeItem(at: output)
                throw UpscaleError.readerFailed("映像フレームを 1 枚も読み出せませんでした")
            }
            sessionStartTime = CMSampleBufferGetPresentationTimeStamp(firstSample)
            var pendingSample: CMSampleBuffer? = firstSample
            nextFrame = {
                while let sample = pendingSample {
                    pendingSample = videoOutput.copyNextSampleBuffer()
                    guard let frame = CMSampleBufferGetImageBuffer(sample) else { continue }
                    let presentationTime = CMSampleBufferGetPresentationTimeStamp(sample)
                    let sampleDuration = CMSampleBufferGetDuration(sample)
                    if sampleDuration.isNumeric {
                        lastFrameEnd = CMTimeAdd(presentationTime, sampleDuration)
                    }
                    return (frame, presentationTime)
                }
                return nil
            }
        }
        writer.startSession(atSourceTime: sessionStartTime)

        // 音声は別キューで並行にコピーする。映像側は推論待ちが長いので、待たせておく理由がない。
        let audioTask: Task<Void, Never>? = audioInput.flatMap { input in
            audioOutput.map { output in
                let copier = AudioCopier(output: output, input: input)
                return Task { await copier.run() }
            }
        }

        // MARK: 映像ループ
        let needsResize = tiled.upscaledWidth != targetWidth || tiled.upscaledHeight != targetHeight
        let frameWriter = FrameWriter(
            input: videoInput,
            adaptor: adaptor,
            outputPool: needsResize ? try PixelBufferPool(width: targetWidth, height: targetHeight) : nil,
            fitMode: configuration.fitMode
        )
        var frameCount = 0
        var lastReportedAt = startedAt
        var inFlight: (task: FrameTask, time: CMTime)?

        /// 次のフレームを読み出し（またはフレームレート変換で生成し）、その推論をすぐ走らせる。
        /// フレームの切れ目で Neural Engine のキューが空にならないよう、
        /// 手前のフレームを待つ前にこれを呼んでおく。
        func startNextFrame() async throws -> (task: FrameTask, time: CMTime)? {
            guard let (frame, presentationTime) = try await nextFrame() else { return nil }
            let source = UnsafeSendableBox(frame)
            let task = FrameTask {
                var buffer = source.value
                for stage in stages {
                    buffer = try await stage.upscale(buffer)
                }
                return UnsafeSendableBox(buffer)
            }
            return (task, presentationTime)
        }

        do {
            inFlight = try await startNextFrame()
            while let current = inFlight {
                try Task.checkCancellation()
                // 手前のフレームの結果を待つ前に、次のフレームの推論を始めておく。
                let next = try await startNextFrame()
                let upscaled = try await current.task.value
                // リサイズと書き出しも、さらに次のフレームの推論と並行に進む。
                try await frameWriter.enqueue(upscaled.value, at: current.time)
                inFlight = next

                frameCount += 1
                if let progress {
                    let now = Date()
                    if now.timeIntervalSince(lastReportedAt) > 0.2 {
                        lastReportedAt = now
                        let elapsed = now.timeIntervalSince(startedAt)
                        progress(UpscaleProgress(
                            framesProcessed: frameCount,
                            totalFrames: estimatedFrames,
                            framesPerSecond: elapsed > 0 ? Double(frameCount) / elapsed : 0
                        ))
                    }
                }
            }
            // 最後に投げた書き出しが終わるまで待つ。
            try await frameWriter.finish()
        } catch {
            inFlight?.task.cancel()
            frameWriter.cancel()
            videoInput.markAsFinished()
            audioTask?.cancel()
            reader.cancelReading()
            writer.cancelWriting()
            try? fileManager.removeItem(at: output)
            throw error
        }

        videoInput.markAsFinished()
        await audioTask?.value
        // 尺を確定させる。呼ばないと最終フレームの表示時間が推測値になり、尺が伸びる。
        if let lastFrameEnd {
            writer.endSession(atSourceTime: lastFrameEnd)
        }

        if reader.status == .failed {
            writer.cancelWriting()
            try? fileManager.removeItem(at: output)
            throw UpscaleError.readerFailed(reader.error?.localizedDescription ?? "読み込みに失敗しました")
        }

        await writer.finishWriting()
        guard writer.status == .completed else {
            try? fileManager.removeItem(at: output)
            throw UpscaleError.writerFailed(writer.error?.localizedDescription ?? "書き出しに失敗しました")
        }

        return UpscaleReport(
            outputURL: output,
            sourceSize: CGSize(width: sourceWidth, height: sourceHeight),
            outputSize: CGSize(width: targetWidth, height: targetHeight),
            frameCount: frameCount,
            tilesPerFrame: stages.reduce(0) { $0 + $1.tilesPerFrame },
            elapsed: Date().timeIntervalSince(startedAt),
            modelSummary: model.summary,
            sourceFrameRate: frameRate,
            outputFrameRate: outputFrameRate
        )
    }

    private func videoSettings(width: Int, height: Int, frameRate: Double, colorTags: SourceColorTags) -> [String: Any] {
        var settings: [String: Any] = [
            AVVideoCodecKey: configuration.codec.avCodec,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoColorPropertiesKey: colorTags.avVideoColorProperties,
        ]
        if configuration.codec != .proRes422 {
            let bitrate = configuration.bitrateMbps.map { $0 * 1_000_000 }
                ?? Double(width * height) * frameRate * configuration.codec.bitsPerPixel
            settings[AVVideoCompressionPropertiesKey] = [
                AVVideoAverageBitRateKey: Int(bitrate),
                AVVideoExpectedSourceFrameRateKey: Int(frameRate.rounded()),
            ]
        }
        return settings
    }

    /// 拡大済みフレームのリサイズと書き出しを、次のフレームの推論と並行して進める。
    ///
    /// 書き出しは 1 本ずつ直列に流す（PTS の順序を崩さないため）が、その 1 本が走っている間に
    /// 呼び出し側は次のフレームの推論へ進めるので、Neural Engine が CPU 処理を待たずに済む。
    private final class FrameWriter: @unchecked Sendable {
        private let input: AVAssetWriterInput
        private let adaptor: AVAssetWriterInputPixelBufferAdaptor
        private let outputPool: PixelBufferPool?
        private let fitMode: FitMode
        private let scaler = Scaler()
        private var pending: Task<Void, Error>?

        init(
            input: AVAssetWriterInput,
            adaptor: AVAssetWriterInputPixelBufferAdaptor,
            outputPool: PixelBufferPool?,
            fitMode: FitMode
        ) {
            self.input = input
            self.adaptor = adaptor
            self.outputPool = outputPool
            self.fitMode = fitMode
        }

        func enqueue(_ buffer: CVPixelBuffer, at time: CMTime) async throws {
            // 直前のフレームを書き終えてから次を始める。並行に流すと順序が崩れる。
            try await pending?.value
            let boxed = UnsafeSendableBox(buffer)
            pending = Task.detached(priority: .userInitiated) { [self] in
                try await write(boxed.value, at: time)
            }
        }

        func finish() async throws {
            try await pending?.value
            pending = nil
        }

        func cancel() {
            pending?.cancel()
            pending = nil
        }

        private func write(_ buffer: CVPixelBuffer, at time: CMTime) async throws {
            let frame: CVPixelBuffer
            if let outputPool {
                let resized = try outputPool.makeBuffer()
                try scaler.place(buffer, into: resized, mode: fitMode)
                frame = resized
            } else {
                frame = buffer
            }
            while !input.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 2_000_000)
            }
            guard adaptor.append(frame, withPresentationTime: time) else {
                throw UpscaleError.writerFailed("フレームの追加に失敗しました")
            }
        }
    }

    /// 音声サンプルを無変換のまま流し込む。
    ///
    /// AVFoundation のリーダー／ライターは Sendable ではないが、ここで触るのは
    /// requestMediaDataWhenReady が使うシリアルキューの上だけなので、まとめて
    /// @unchecked Sendable のボックスに入れて扱う。
    private final class AudioCopier: @unchecked Sendable {
        private let output: AVAssetReaderTrackOutput
        private let input: AVAssetWriterInput
        private var finished = false

        init(output: AVAssetReaderTrackOutput, input: AVAssetWriterInput) {
            self.output = output
            self.input = input
        }

        func run() async {
            let queue = DispatchQueue(label: "UpscaleKit.audio")
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                input.requestMediaDataWhenReady(on: queue) { [self] in
                    while input.isReadyForMoreMediaData {
                        guard !finished else { return }
                        guard let sample = output.copyNextSampleBuffer(), input.append(sample) else {
                            finished = true
                            input.markAsFinished()
                            continuation.resume()
                            return
                        }
                    }
                }
            }
        }
    }
}
