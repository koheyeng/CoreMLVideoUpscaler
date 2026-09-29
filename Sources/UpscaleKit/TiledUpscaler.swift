import CoreVideo
import Foundation

/// Core ML モデルは固定解像度しか受け取れないため、フレームをタイルに割って推論し、
/// 拡大後のキャンバスへ貼り合わせる。
///
/// タイルはラスタ順に貼る。各タイルは「左辺・上辺」だけを既存ピクセルへフェードイン
/// させる。右辺・下辺は後続タイルが自分でフェードインしてくるので、こちら側で
/// フェードアウトさせる必要はない。
///
/// 推論は 1 枚先まで走らせておく。Neural Engine は 1 回の推論を待つ間に次の入力を
/// 受け取れるため、逐次に投げるより速い（貼り合わせの CPU 処理も推論と重なる）。
public final class TiledUpscaler: @unchecked Sendable {
    /// 1 枚のタイルをどこから切り出し、どこへ貼るか。フレームごとに変わらないので先に作っておく。
    private struct TilePlan {
        let sourceX: Int
        let sourceY: Int
        let copyWidth: Int
        let copyHeight: Int
        let destinationX: Int
        let destinationY: Int
        let featherLeft: Int
        let featherTop: Int
    }

    private let model: UpscaleModel
    private let tilePool: PixelBufferPool
    private let canvasPool: PixelBufferPool
    private let plans: [TilePlan]

    public let sourceWidth: Int
    public let sourceHeight: Int
    /// 拡大直後（最終リサイズ前）の解像度。
    public let upscaledWidth: Int
    public let upscaledHeight: Int
    /// 1 フレームあたりの推論回数。
    public var tilesPerFrame: Int { plans.count }

    public init(model: UpscaleModel, sourceWidth: Int, sourceHeight: Int, overlap: Int? = nil) throws {
        self.model = model
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight

        let tileW = model.inputWidth
        let tileH = model.inputHeight
        let scaleX = Double(model.outputWidth) / Double(model.inputWidth)
        let scaleY = Double(model.outputHeight) / Double(model.inputHeight)
        let singleTile = tileW >= sourceWidth && tileH >= sourceHeight

        let placementsX: [(origin: Int, feather: Int)]
        let placementsY: [(origin: Int, feather: Int)]

        if singleTile {
            // 1 枚で収まるならタイル継ぎ目が無いので、倍率が整数でなくても構わない。
            placementsX = [(0, 0)]
            placementsY = [(0, 0)]
            upscaledWidth = Int((Double(sourceWidth) * scaleX).rounded())
            upscaledHeight = Int((Double(sourceHeight) * scaleY).rounded())
        } else {
            guard
                model.outputWidth % model.inputWidth == 0,
                model.outputHeight % model.inputHeight == 0
            else {
                throw UpscaleError.modelInterfaceMismatch(
                    "タイル分割には整数倍のモデルが必要です (\(model.summary))"
                )
            }
            // 重なりが広いほど継ぎ目は目立たないが、その分タイル枚数＝推論回数が増える。
            placementsX = Self.placements(
                total: sourceWidth, tile: tileW, overlap: min(overlap ?? max(8, tileW / 8), tileW / 2)
            )
            placementsY = Self.placements(
                total: sourceHeight, tile: tileH, overlap: min(overlap ?? max(8, tileH / 8), tileH / 2)
            )
            upscaledWidth = sourceWidth * (model.outputWidth / model.inputWidth)
            upscaledHeight = sourceHeight * (model.outputHeight / model.inputHeight)
        }

        let canvasWidth = upscaledWidth
        let canvasHeight = upscaledHeight
        plans = placementsY.flatMap { py in
            placementsX.map { px in
                let copyWidth = min(tileW, sourceWidth - px.origin)
                let copyHeight = min(tileH, sourceHeight - py.origin)
                let destinationX = Int((Double(px.origin) * scaleX).rounded())
                let destinationY = Int((Double(py.origin) * scaleY).rounded())
                return TilePlan(
                    sourceX: px.origin,
                    sourceY: py.origin,
                    copyWidth: copyWidth,
                    copyHeight: copyHeight,
                    destinationX: destinationX,
                    destinationY: destinationY,
                    featherLeft: min(
                        Int((Double(px.feather) * scaleX).rounded()),
                        canvasWidth - destinationX
                    ),
                    featherTop: min(
                        Int((Double(py.feather) * scaleY).rounded()),
                        canvasHeight - destinationY
                    )
                )
            }
        }

        tilePool = try PixelBufferPool(width: tileW, height: tileH)
        canvasPool = try PixelBufferPool(width: upscaledWidth, height: upscaledHeight)
    }

    /// フレーム 1 枚をタイル推論して、拡大後のバッファを返す。
    public func upscale(_ frame: CVPixelBuffer) async throws -> CVPixelBuffer {
        let canvas = try canvasPool.makeBuffer()
        let source = UnsafeSendableBox(frame)

        // 1 枚目を先に走らせ、以降は「今のタイルを貼る間に次のタイルを推論する」形で回す。
        var running: Task<UnsafeSendableBox<CVPixelBuffer>, Error>? = plans.isEmpty ? nil : infer(0, source)

        for index in plans.indices {
            guard let current = running else { break }
            running = index + 1 < plans.count ? infer(index + 1, source) : nil

            let output = try await current.value.value

            let plan = plans[index]
            // モデルはタイル全面を返すが、端のタイルはパディング分を含む。有効領域だけ貼る。
            let validWidth = min(
                Int((Double(plan.copyWidth) * Double(model.outputWidth) / Double(model.inputWidth)).rounded()),
                upscaledWidth - plan.destinationX,
                output.pixelWidth
            )
            let validHeight = min(
                Int((Double(plan.copyHeight) * Double(model.outputHeight) / Double(model.inputHeight)).rounded()),
                upscaledHeight - plan.destinationY,
                output.pixelHeight
            )

            try output.withBGRA(readOnly: true) { tile in
                try canvas.withBGRA(readOnly: false) { destination in
                    ImageOps.blend(
                        tile: tile.region(x: 0, y: 0, width: validWidth, height: validHeight),
                        into: destination,
                        atX: plan.destinationX,
                        atY: plan.destinationY,
                        featherLeft: min(plan.featherLeft, validWidth),
                        featherTop: min(plan.featherTop, validHeight)
                    )
                }
            }
        }
        return canvas
    }

    /// タイル 1 枚の切り出しと推論をバックグラウンドで始める。
    private func infer(
        _ index: Int,
        _ source: UnsafeSendableBox<CVPixelBuffer>
    ) -> Task<UnsafeSendableBox<CVPixelBuffer>, Error> {
        let plan = plans[index]
        let pool = tilePool
        let model = self.model
        return Task.detached(priority: .userInitiated) {
            let input = try pool.makeBuffer()
            try source.value.withBGRA(readOnly: true) { frame in
                try input.withBGRA(readOnly: false) { tile in
                    Self.fillTile(from: frame, plan: plan, into: tile)
                }
            }
            return UnsafeSendableBox(try model.upscale(input))
        }
    }

    /// タイル入力バッファを埋める。フレーム端でタイルがはみ出す分は端のピクセルを複製して伸ばす
    /// （黒で埋めると、その黒がモデルに拡大されて縁に滲む）。
    private static func fillTile(from source: BGRAView, plan: TilePlan, into tile: BGRAView) {
        for y in 0..<plan.copyHeight {
            let destination = tile.row(y)
            memcpy(destination, source.row(plan.sourceY + y).advanced(by: plan.sourceX * 4), plan.copyWidth * 4)
            if plan.copyWidth < tile.width {
                let edge = destination.advanced(by: (plan.copyWidth - 1) * 4)
                for x in plan.copyWidth..<tile.width {
                    memcpy(destination.advanced(by: x * 4), edge, 4)
                }
            }
        }
        if plan.copyHeight < tile.height {
            let edge = tile.row(plan.copyHeight - 1)
            for y in plan.copyHeight..<tile.height {
                memcpy(tile.row(y), edge, tile.width * 4)
            }
        }
    }

    /// タイルの開始位置を左から並べる。最後の 1 枚は必ず右端にぴったり合わせるため、
    /// 直前との重なりが `overlap` より広くなることがある。
    private static func placements(total: Int, tile: Int, overlap: Int) -> [(origin: Int, feather: Int)] {
        guard total > tile else { return [(0, 0)] }
        let step = max(1, tile - overlap)

        var origins: [Int] = []
        var origin = 0
        while origin + tile < total {
            origins.append(origin)
            origin += step
        }
        origins.append(total - tile)

        return origins.enumerated().map { index, origin in
            guard index > 0 else { return (origin, 0) }
            return (origin, (origins[index - 1] + tile) - origin)
        }
    }
}
