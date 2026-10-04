import AVFoundation
import CoreMedia
import CoreVideo
import VideoToolbox

/// ソース映像トラックの色タグ（primaries / transfer function / YCbCr matrix）。
/// タグが無い項目は BT.709 で補う。
///
/// 書き出し設定（`AVVideoColorPropertiesKey`）と、フレームレート変換時のピクセル
/// フォーマット変換（64RGBAHalf -> 32BGRA）の変換先タグの、両方をこれ 1 つで揃える。
/// 揃えないと常に BT.709 として書き出されたり（メタデータのずれ）、変換時に
/// 別の transfer function へ再ガンマされて画素値そのものが変わったりする
/// （sRGB タグの素材が ITU_R_709_2 前提で暗くなる、など。本パッケージで実際に踏んだ不具合）。
struct SourceColorTags: Sendable {
    let primaries: String
    let transferFunction: String
    let yCbCrMatrix: String
    /// ソースに実際に付いていたタグ（BT.709 で補った項目は含まない）。
    /// ピクセル変換の変換先には、これに含まれる項目だけを指定する。タグの無い素材に
    /// BT.709 を押し付けると、デコーダが付けた既定値（BT.601 など）との差で再ガンマされるため。
    var explicitPrimaries: String? = nil
    var explicitTransferFunction: String? = nil
    var explicitYCbCrMatrix: String? = nil

    static let bt709 = SourceColorTags(
        primaries: AVVideoColorPrimaries_ITU_R_709_2,
        transferFunction: AVVideoTransferFunction_ITU_R_709_2,
        yCbCrMatrix: AVVideoYCbCrMatrix_ITU_R_709_2
    )

    /// トラックのフォーマット記述から読み取る。タグが無い項目は BT.709 で補う。
    static func from(_ formatDescription: CMFormatDescription?) -> SourceColorTags {
        guard let formatDescription else { return .bt709 }
        func tag(_ key: CFString) -> String? {
            CMFormatDescriptionGetExtension(formatDescription, extensionKey: key) as? String
        }
        let primaries = tag(kCVImageBufferColorPrimariesKey)
        let transferFunction = tag(kCVImageBufferTransferFunctionKey)
        let yCbCrMatrix = tag(kCVImageBufferYCbCrMatrixKey)
        return SourceColorTags(
            primaries: primaries ?? bt709.primaries,
            transferFunction: transferFunction ?? bt709.transferFunction,
            yCbCrMatrix: yCbCrMatrix ?? bt709.yCbCrMatrix,
            explicitPrimaries: primaries,
            explicitTransferFunction: transferFunction,
            explicitYCbCrMatrix: yCbCrMatrix
        )
    }

    var avVideoColorProperties: [String: Any] {
        [
            AVVideoColorPrimariesKey: primaries,
            AVVideoTransferFunctionKey: transferFunction,
            AVVideoYCbCrMatrixKey: yCbCrMatrix,
        ]
    }
}

/// FRC の出力先スクラッチバッファ用の薄いプール（`PixelBufferSupport.swift` の
/// `PixelBufferPool` は 32BGRA 固定なので、64RGBAHalf 用に別で持つ）。
private final class FRCScratchPool: @unchecked Sendable {
    private let pool: CVPixelBufferPool

    init(width: Int, height: Int, pixelFormat: OSType) throws {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ]
        var pool: CVPixelBufferPool?
        let status = CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool)
        guard status == kCVReturnSuccess, let pool else {
            throw UpscaleError.pixelBufferAllocationFailed(status)
        }
        self.pool = pool
    }

    func makeBuffer() throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer)
        guard status == kCVReturnSuccess, let buffer else {
            throw UpscaleError.pixelBufferAllocationFailed(status)
        }
        return buffer
    }
}

/// フレームレート変換（`--fps`）がこの Mac で使えるか（macOS 15.4 以降かつ対応ハードウェア）。
public enum FrameRateConversion {
    public static var isAvailable: Bool {
        if #available(macOS 15.4, *) { return FrameRateConverter.isSupported }
        return false
    }
}

/// `VTFrameProcessor` の `VTFrameRateConversion`（VideoToolbox の ML ベースのフレーム補間）を
/// 使って、映像フレームを目標フレームレートへ変換しながら 1 枚ずつ供給する。
///
/// GPU で動くため、Neural Engine で動く超解像とは別の実行資源を使う。フレーム補間は
/// 64RGBAHalf でしか受け付けないため、`sourceOutput` はそのフォーマットで読み出し設定
/// 済みである必要がある。取り出したフレームは呼び出し側（`TiledUpscaler`）がそのまま
/// 使える 32BGRA へ、`VTPixelTransferSession` で変換してから返す。
///
/// - Important: `VTFrameRateConversionConfiguration` のヘッダ上の注釈は
///   `macos(15.4), ios(26.0)` 。iOS 側のメジャー番号（26）につられて macOS も 26 以降が
///   要ると誤解しやすいので注意（実際の最小要件は macOS 15.4）。
@available(macOS 15.4, *)
final class FrameRateConverter: @unchecked Sendable {
    /// FRC が要求するソース側のピクセルフォーマット。手元の実機ではこれ以外は対応していない。
    private static let processingPixelFormat = kCVPixelFormatType_64RGBAHalf

    /// この Mac／OS が VTFrameRateConversion に対応しているか。
    static var isSupported: Bool { VTFrameRateConversionConfiguration.isSupported }

    private let sourceOutput: AVAssetReaderTrackOutput
    private let processor: VTFrameProcessor
    private let transferSession: VTPixelTransferSession
    private let scratchPool: FRCScratchPool
    private let outputPool: PixelBufferPool
    /// 目標フレームレートが元のフレームレート以下なら補間せず、直前のソースフレームを使い回す
    /// （フレームを間引くだけなので ML 補間は不要）。
    private let needsInterpolation: Bool
    /// 出力 1 フレームぶんの時間（`CMTimeMultiply` で正確な有理数のまま n 倍する）。
    private let frameDuration: CMTime

    /// 出力の基準時刻（ソース映像トラックの実際の最初のフレームの PTS）。
    /// 音声はソースの PTS をそのまま引き継ぐので、映像側もここを基準に合わせて同期を保つ。
    let sessionStart: CMTime
    /// 書き出すべき出力フレームの総数。
    let totalOutputFrames: Int
    /// 最後の出力フレームの終了時刻（`writer.endSession` に渡す）。
    let endTime: CMTime

    private var srcFrames: [VTFrameProcessorFrame] = []
    private var srcBase = 0
    private var readerExhausted = false
    private var cursor = 0
    private var outputIndex = 0
    private var queue: [(buffer: CVPixelBuffer, time: CMTime)] = []
    private var sessionStarted = false

    /// - Parameters:
    ///   - sourceOutput: 64RGBAHalf で出力設定済みの映像トラック出力（呼び出し側の
    ///     `AVAssetReader` に追加済み・`startReading()` 済みであること）。
    ///   - frameWidth/frameHeight: ソースの解像度（回転前）。
    ///   - sourceDuration: 出力フレーム数の算出に使う尺（通常はアセットの尺）。
    ///   - targetFrameRate: 目標フレームレート。
    ///   - allowInterpolation: `targetFrameRate` が元のフレームレートより高いか
    ///     （＝補間が必要か）。呼び出し側で計算して渡す。
    ///   - colorTags: ピクセルフォーマット変換時に付け直す色タグ（ソースの再ガンマ防止）。
    init(
        sourceOutput: AVAssetReaderTrackOutput,
        frameWidth: Int,
        frameHeight: Int,
        sourceDuration: CMTime,
        targetFrameRate: Double,
        allowInterpolation: Bool,
        colorTags: SourceColorTags
    ) throws {
        guard Self.isSupported else {
            throw UpscaleError.featureUnavailable("この Mac は VTFrameRateConversion に対応していません")
        }
        self.sourceOutput = sourceOutput
        self.needsInterpolation = allowInterpolation

        guard let config = VTFrameRateConversionConfiguration(
            frameWidth: frameWidth,
            frameHeight: frameHeight,
            usePrecomputedFlow: false,
            qualityPrioritization: .quality,
            revision: .revision1
        ) else {
            throw UpscaleError.featureUnavailable("VTFrameRateConversionConfiguration の作成に失敗しました")
        }
        guard config.supportedPixelFormats.contains(Self.processingPixelFormat) else {
            throw UpscaleError.featureUnavailable("VTFrameRateConversion が 64RGBAHalf に対応していません")
        }

        // セッション開始は init の最後で行う（途中で throw したときに、終了されない
        // セッションを残さないため）。終了は deinit。
        self.processor = VTFrameProcessor()

        var transferSessionOut: VTPixelTransferSession?
        let transferStatus = VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transferSessionOut)
        guard transferStatus == noErr, let transferSession = transferSessionOut else {
            throw UpscaleError.featureUnavailable("VTPixelTransferSession の作成に失敗しました (\(transferStatus))")
        }
        // 変換先の色タグをソースに合わせる。既定（未設定）のままだと ITU_R_709_2 とみなされて
        // 再ガンマされ、画素値が変わってしまう（sRGB 素材が暗くなる、など）。
        // ソースにタグが無い項目は指定しない（補った BT.709 を押し付けると逆に再ガンマされる）。
        if let value = colorTags.explicitPrimaries {
            VTSessionSetProperty(transferSession, key: kVTPixelTransferPropertyKey_DestinationColorPrimaries, value: value as CFString)
        }
        if let value = colorTags.explicitTransferFunction {
            VTSessionSetProperty(transferSession, key: kVTPixelTransferPropertyKey_DestinationTransferFunction, value: value as CFString)
        }
        if let value = colorTags.explicitYCbCrMatrix {
            VTSessionSetProperty(transferSession, key: kVTPixelTransferPropertyKey_DestinationYCbCrMatrix, value: value as CFString)
        }
        self.transferSession = transferSession

        scratchPool = try FRCScratchPool(width: frameWidth, height: frameHeight, pixelFormat: Self.processingPixelFormat)
        outputPool = try PixelBufferPool(width: frameWidth, height: frameHeight)

        frameDuration = Self.rationalFrameDuration(fps: targetFrameRate)

        // 最初のソースフレームを読んで、出力の基準時刻にする（音声はソースの PTS を
        // そのまま引き継ぐため、映像側もここに合わせないと音ズレする）。
        while srcFrames.isEmpty {
            guard let sample = sourceOutput.copyNextSampleBuffer() else {
                throw UpscaleError.readerFailed("映像フレームを 1 枚も読み出せませんでした")
            }
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            guard let frame = VTFrameProcessorFrame(buffer: pixelBuffer, presentationTimeStamp: pts) else { continue }
            srcFrames.append(frame)
        }
        sessionStart = srcFrames[0].presentationTimeStamp

        let frameCount = sourceDuration.seconds.isFinite
            ? max(0, Int((sourceDuration.seconds * targetFrameRate).rounded())) : 0
        totalOutputFrames = frameCount
        endTime = CMTimeAdd(sessionStart, CMTimeMultiply(frameDuration, multiplier: Int32(frameCount)))

        try processor.startSession(configuration: config)
        sessionStarted = true
    }

    deinit {
        // ヘッダの指示どおり、使い終わったセッションは明示的に終了する
        // （アプリで複数ファイルを続けて処理するとき、GPU 側のセッションを溜めないため）。
        if sessionStarted { processor.endSession() }
        VTPixelTransferSessionInvalidate(transferSession)
    }

    /// 整数 fps は `1/fps` の有理数、NTSC 系 (29.97 / 59.94 等) は `1001/(1000*n)` の有理数、
    /// それ以外は十分に細かい共通タイムスケールで近似する。
    private static func rationalFrameDuration(fps: Double) -> CMTime {
        let roundedInteger = fps.rounded()
        if abs(fps - roundedInteger) < 0.001, roundedInteger > 0 {
            return CMTime(value: 1, timescale: CMTimeScale(roundedInteger))
        }
        let ntscBase = (fps * 1001 / 1000).rounded()
        if ntscBase > 0, abs(fps - ntscBase * 1000 / 1001) < 0.001 {
            return CMTime(value: 1001, timescale: CMTimeScale(ntscBase * 1000))
        }
        let timescale: CMTimeScale = 1_000_000
        return CMTime(value: Int64((Double(timescale) / fps).rounded()), timescale: timescale)
    }

    /// 出力フレーム番号 n の絶対 PTS（`sessionStart` 基準）。
    private func outputTime(_ n: Int) -> CMTime {
        CMTimeAdd(sessionStart, CMTimeMultiply(frameDuration, multiplier: Int32(n)))
    }

    private func decodeNextSourceFrameIfNeeded(upTo index: Int) {
        while srcBase + srcFrames.count <= index && !readerExhausted {
            guard let sample = sourceOutput.copyNextSampleBuffer() else {
                readerExhausted = true
                break
            }
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            guard let frame = VTFrameProcessorFrame(buffer: pixelBuffer, presentationTimeStamp: pts) else { continue }
            srcFrames.append(frame)
        }
    }

    private func srcFrame(at index: Int) -> VTFrameProcessorFrame { srcFrames[index - srcBase] }
    private func srcFrameAvailable(_ index: Int) -> Bool { index >= srcBase && index - srcBase < srcFrames.count }

    /// メモリを増やし続けないよう、もう使わないソースフレームを捨てる。
    private func trimFrames(keepFrom index: Int) {
        let drop = min(max(0, index - srcBase), srcFrames.count)
        if drop > 0 {
            srcFrames.removeFirst(drop)
            srcBase += drop
        }
    }

    /// 出力時刻 `t` 以下で最大のソースフレーム番号まで `cursor` を進める。
    private func advanceCursor(to t: CMTime) {
        while true {
            decodeNextSourceFrameIfNeeded(upTo: cursor + 1)
            guard srcFrameAvailable(cursor + 1) else { break }
            let nextPTS = srcFrame(at: cursor + 1).presentationTimeStamp
            guard CMTimeCompare(nextPTS, t) <= 0 else { break }
            cursor += 1
        }
    }

    /// 半チック分ほどの許容差で時刻が一致するか。
    private func isNearlyEqual(_ a: CMTime, _ b: CMTime) -> Bool {
        abs(a.seconds - b.seconds) < 0.0005
    }

    /// 変換元（64RGBAHalf）のフレームを、ソースの色タグを保ったまま 32BGRA に変換する。
    private func convertToOutputFormat(_ frame: VTFrameProcessorFrame) throws -> CVPixelBuffer {
        let destination = try outputPool.makeBuffer()
        let status = VTPixelTransferSessionTransferImage(transferSession, from: frame.buffer, to: destination)
        guard status == noErr else {
            throw UpscaleError.featureUnavailable("VTPixelTransferSessionTransferImage に失敗しました (\(status))")
        }
        return destination
    }

    /// 次の出力フレームを 32BGRA で返す。全て出力し終えたら nil。
    func next() async throws -> (buffer: CVPixelBuffer, time: CMTime)? {
        if queue.isEmpty {
            try await refill()
        }
        guard !queue.isEmpty else { return nil }
        return queue.removeFirst()
    }

    /// キューが空になったら、次のブラケット分をまとめて生成して詰める。
    private func refill() async throws {
        guard outputIndex < totalOutputFrames else { return }
        let t = outputTime(outputIndex)
        advanceCursor(to: t)
        let i = cursor
        guard srcFrameAvailable(i) else { return }

        decodeNextSourceFrameIfNeeded(upTo: i + 1)
        let hasNext = srcFrameAvailable(i + 1)
        let tI = srcFrame(at: i).presentationTimeStamp

        guard needsInterpolation, hasNext, !isNearlyEqual(t, tI) else {
            // ダウンコンバート／ちょうど一致／末尾（次のソースフレームが無い）:
            // 直前のソースフレームをそのまま使う（補間しない）。
            let converted = try convertToOutputFormat(srcFrame(at: i))
            queue.append((converted, t))
            outputIndex += 1
            trimFrames(keepFrom: i)
            return
        }

        // t は t_i と t_{i+1} の間にある。同じ (i, i+1) の組を使う出力フレームをまとめて
        // 1 回の呼び出しに束ねる（呼び出し回数を減らすため。VTFrameRateConversion 自体は
        // 十分速いので必須ではないが、Neural Engine 側の推論に対しては誤差の範囲）。
        let tNext = srcFrame(at: i + 1).presentationTimeStamp
        var ns: [Int] = [outputIndex]
        var n = outputIndex + 1
        while n < totalOutputFrames {
            let tn = outputTime(n)
            guard CMTimeCompare(tn, tNext) < 0 else { break }
            ns.append(n)
            n += 1
        }

        let phases: [Float] = ns.map { nn in
            let tn = outputTime(nn)
            return Float((tn.seconds - tI.seconds) / (tNext.seconds - tI.seconds))
        }
        let destFrames: [VTFrameProcessorFrame] = try ns.map { nn in
            guard let frame = VTFrameProcessorFrame(buffer: try scratchPool.makeBuffer(), presentationTimeStamp: outputTime(nn)) else {
                throw UpscaleError.featureUnavailable("VTFrameProcessorFrame の作成に失敗しました")
            }
            return frame
        }
        guard let params = VTFrameRateConversionParameters(
            sourceFrame: srcFrame(at: i),
            nextFrame: srcFrame(at: i + 1),
            opticalFlow: nil,
            interpolationPhase: phases,
            // .sequential だと動きの大きい素材で補間フレームが破綻する（明るさが落ちたり、
            // どのソースフレームにも似ない絵になる）。.random なら破綻せず、速度もほぼ同じ。
            submissionMode: .random,
            destinationFrames: destFrames
        ) else {
            throw UpscaleError.featureUnavailable("VTFrameRateConversionParameters の作成に失敗しました")
        }
        _ = try await processor.process(parameters: params)

        for (index, nn) in ns.enumerated() {
            let converted = try convertToOutputFormat(destFrames[index])
            queue.append((converted, outputTime(nn)))
        }
        outputIndex = n
        trimFrames(keepFrom: i)
    }
}
