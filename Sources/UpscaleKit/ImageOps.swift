import Accelerate
import CoreVideo
import Foundation

/// vImage のスケーリング。テンポラリバッファを使い回すため参照型で保持する。
final class Scaler: @unchecked Sendable {
    private var temp: UnsafeMutableRawPointer?
    private var tempSize: Int = 0

    deinit { temp?.deallocate() }

    /// `src` を `dst` の矩形いっぱいに拡大・縮小する。
    func scale(_ src: BGRAView, into dst: BGRAView) throws {
        var s = src.vImageBuffer
        var d = dst.vImageBuffer
        let flags = vImage_Flags(kvImageHighQualityResampling | kvImageEdgeExtend)

        let required = vImageScale_ARGB8888(&s, &d, nil, flags | vImage_Flags(kvImageGetTempBufferSize))
        if required > 0 && required != tempSize {
            temp?.deallocate()
            temp = UnsafeMutableRawPointer.allocate(byteCount: Int(required), alignment: 16)
            tempSize = Int(required)
        }
        let error = vImageScale_ARGB8888(&s, &d, temp, flags)
        guard error == kvImageNoError else { throw UpscaleError.vImageFailed(error) }
    }

    /// `src` を `mode` に従って `dst` 全体に配置する。
    func place(_ src: CVPixelBuffer, into dst: CVPixelBuffer, mode: FitMode) throws {
        try src.withBGRA(readOnly: true) { s in
            try dst.withBGRA(readOnly: false) { d in
                switch mode {
                case .contain, .stretch:
                    // contain は書き出し解像度の側を縦横比に合わせてあるので、全面へ伸ばすだけでよい。
                    try scale(s, into: d)

                case .fill:
                    // src 側を切り出してから dst 全面へ伸ばす。
                    let scaleFactor = max(Double(d.width) / Double(s.width), Double(d.height) / Double(s.height))
                    let cropW = min(s.width, Int((Double(d.width) / scaleFactor).rounded()))
                    let cropH = min(s.height, Int((Double(d.height) / scaleFactor).rounded()))
                    let crop = s.region(
                        x: (s.width - cropW) / 2, y: (s.height - cropH) / 2,
                        width: max(cropW, 1), height: max(cropH, 1)
                    )
                    try scale(crop, into: d)

                case .fit:
                    let scaleFactor = min(Double(d.width) / Double(s.width), Double(d.height) / Double(s.height))
                    let fitW = max(1, min(d.width, Int((Double(s.width) * scaleFactor).rounded())))
                    let fitH = max(1, min(d.height, Int((Double(s.height) * scaleFactor).rounded())))
                    if fitW != d.width || fitH != d.height {
                        for y in 0..<d.height { memset(d.row(y), 0, d.width * 4) }
                    }
                    // 偶数境界に寄せておくと YUV エンコード時にクロマがずれにくい。
                    let ox = ((d.width - fitW) / 2) & ~1
                    let oy = ((d.height - fitH) / 2) & ~1
                    try scale(s, into: d.region(x: ox, y: oy, width: fitW, height: fitH))
                }
            }
        }
    }
}

enum ImageOps {
    /// タイル 1 枚をキャンバスへ合成する。
    ///
    /// `featherLeft` / `featherTop` は直前のタイルと重なっている幅。その帯だけ
    /// 0→255 で線形に立ち上げて既存ピクセルと混ぜ、継ぎ目を消す。右辺・下辺は
    /// 後から来るタイルがフェードインしてくるので、こちらでは何もしない。
    static func blend(
        tile: BGRAView,
        into canvas: BGRAView,
        atX ox: Int,
        atY oy: Int,
        featherLeft: Int,
        featherTop: Int
    ) {
        let width = tile.width
        let feather = min(featherLeft, width)
        let columnWeights = (0..<feather).map { weight($0, fade: feather) }

        columnWeights.withUnsafeBufferPointer { wx in
            for ty in 0..<tile.height {
                let wy = ty < featherTop ? weight(ty, fade: featherTop) : 255
                let source = tile.row(ty)
                let destination = canvas.row(oy + ty).advanced(by: ox * 4)

                if wy == 255 {
                    // 縦は不透明。横のフェード帯だけ混ぜて、残りはそのまま流し込む。
                    blendSpan(source: source, destination: destination, from: 0, to: feather, wx: wx, wy: 255)
                    if feather < width {
                        memcpy(destination + feather * 4, source + feather * 4, (width - feather) * 4)
                    }
                } else {
                    blendSpan(source: source, destination: destination, from: 0, to: feather, wx: wx, wy: wy)
                    blendSpan(source: source, destination: destination, from: feather, to: width, wx: nil, wy: wy)
                }
            }
        }
    }

    private static func blendSpan(
        source: UnsafeMutablePointer<UInt8>,
        destination: UnsafeMutablePointer<UInt8>,
        from: Int,
        to: Int,
        wx: UnsafeBufferPointer<Int>?,
        wy: Int
    ) {
        for x in from..<to {
            let w = wx.map { $0[x] * wy / 255 } ?? wy
            let i = x * 4
            if w >= 255 {
                memcpy(destination + i, source + i, 4)
            } else if w > 0 {
                let inverse = 255 - w
                for c in 0..<4 {
                    destination[i + c] = UInt8((Int(destination[i + c]) * inverse + Int(source[i + c]) * w + 127) / 255)
                }
            }
        }
    }

    /// 帯の中で 0 → 255 に線形に立ち上がる重み。
    private static func weight(_ i: Int, fade: Int) -> Int {
        guard fade > 0 else { return 255 }
        return min(255, (i + 1) * 255 / (fade + 1))
    }
}
