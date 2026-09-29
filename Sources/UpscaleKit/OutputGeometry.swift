import Foundation

/// 出力解像度に合わせ込むときの縦横比の扱い。
public enum FitMode: String, CaseIterable, Sendable {
    /// 縦横比を保ったまま、指定した枠に収まるよう出力解像度そのものを決める。余白は出ない。
    case contain
    /// 出力解像度は指定どおりに固定し、縦横比を保って内接させて余白を黒で埋める（レターボックス）。
    case fit
    /// 出力解像度は指定どおりに固定し、縦横比を保って外接させ、はみ出した部分を切り落とす。
    case fill
    /// 出力解像度は指定どおりに固定し、縦横比を無視して引き伸ばす。
    case stretch

    public var label: String {
        switch self {
        case .contain: "元の比率のまま"
        case .fit: "余白を黒で埋める"
        case .fill: "はみ出しを切る"
        case .stretch: "引き伸ばす"
        }
    }
}

/// 入力サイズ・指定した出力寸法・合わせ方から、実際の書き出し解像度を決める。
public enum OutputGeometry {
    /// 短辺を `length` に合わせ、長辺は元の縦横比から決める。
    ///
    /// 横長素材なら縦が、縦長素材なら横が `length` になる。縦横が混ざった素材を
    /// 同じ「1080p 相当」の基準で揃えたいときに使う。
    public static func resolveShortSide(
        sourceWidth: Int,
        sourceHeight: Int,
        length: Int
    ) -> (width: Int, height: Int) {
        guard sourceWidth > 0, sourceHeight > 0 else { return (evenDown(length), evenDown(length)) }
        let isPortrait = sourceHeight > sourceWidth
        return resolve(
            sourceWidth: sourceWidth,
            sourceHeight: sourceHeight,
            // 縦長なら横を、そうでなければ縦を基準にする。
            boundsWidth: isPortrait ? length : nil,
            boundsHeight: isPortrait ? nil : length,
            fitMode: .contain
        )
    }

    /// `boundsWidth` / `boundsHeight` は片方を nil にできる。nil の側は元の縦横比から求める
    /// （「縦を 1080 に、横は元の比率のまま」といった指定ができる）。
    /// 両方指定した場合は `fitMode` に従い、`.contain` のときだけその枠に収まるよう縮める。
    public static func resolve(
        sourceWidth: Int,
        sourceHeight: Int,
        boundsWidth: Int?,
        boundsHeight: Int?,
        fitMode: FitMode
    ) -> (width: Int, height: Int) {
        guard sourceWidth > 0, sourceHeight > 0 else {
            return (evenDown(boundsWidth ?? 1920), evenDown(boundsHeight ?? 1080))
        }
        let aspect = Double(sourceWidth) / Double(sourceHeight)

        switch (boundsWidth, boundsHeight) {
        case let (width?, nil):
            // 横を基準に、縦は元の比率から。
            return (evenDown(width), clampEven(Double(width) / aspect, limit: nil))
        case let (nil, height?):
            // 縦を基準に、横は元の比率から。
            return (clampEven(Double(height) * aspect, limit: nil), evenDown(height))
        case let (width?, height?):
            guard fitMode == .contain else { return (evenDown(width), evenDown(height)) }
            // 枠に収まる最大の大きさへ、比率を保ったまま縮める。
            let scale = min(Double(width) / Double(sourceWidth), Double(height) / Double(sourceHeight))
            return (
                clampEven(Double(sourceWidth) * scale, limit: width),
                clampEven(Double(sourceHeight) * scale, limit: height)
            )
        case (nil, nil):
            return (evenDown(sourceWidth), evenDown(sourceHeight))
        }
    }

    /// 映像コーデックは 4:2:0 のクロマを持つため、幅・高さは偶数でなければならない。
    private static func clampEven(_ value: Double, limit: Int?) -> Int {
        let rounded = evenDown(Int(value.rounded()))
        guard let limit else { return rounded }
        return min(rounded, evenDown(limit))
    }

    private static func evenDown(_ value: Int) -> Int {
        max(2, value - (value % 2))
    }
}
