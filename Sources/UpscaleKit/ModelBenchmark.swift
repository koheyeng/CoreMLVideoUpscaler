import CoreVideo
import Foundation

/// 動画パイプラインを通さず、モデルの推論だけを繰り返して素のスループットを測る。
///
/// パイプライン全体の fps と比べることで、「Neural Engine が計算に使われている時間」と
/// 「貼り合わせ・エンコードを待って遊んでいる時間」を切り分けられる。
public enum ModelBenchmark {
    public struct Result: Sendable {
        public let concurrency: Int
        public let iterations: Int
        public let elapsed: TimeInterval

        /// 1 秒あたりの推論回数。
        public var inferencesPerSecond: Double { elapsed > 0 ? Double(iterations) / elapsed : 0 }
        /// 1 推論あたりの実時間。並行度を上げると、待ち時間が重なって短く見える。
        public var millisecondsPerInference: Double { elapsed / Double(iterations) * 1000 }
    }

    /// `concurrency` 本の並行ループで合計 `iterations` 回の推論を回す。
    public static func run(model: UpscaleModel, iterations: Int, concurrency: Int) async throws -> Result {
        let pool = try PixelBufferPool(width: model.inputWidth, height: model.inputHeight)

        // 初回はモデルのロードや ANE 側の準備が入るので、計測から外す。
        let warmup = try pool.makeBuffer()
        try warmup.fillBlack()
        _ = try model.upscale(warmup)

        let perTask = max(1, iterations / concurrency)
        let total = perTask * concurrency

        let startedAt = CFAbsoluteTimeGetCurrent()
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<concurrency {
                group.addTask {
                    // 入力バッファはタスクごとに持つ。共有すると書き込みが競合する。
                    let buffer = try pool.makeBuffer()
                    try buffer.fillBlack()
                    for _ in 0..<perTask {
                        _ = try model.upscale(buffer)
                    }
                }
            }
            try await group.waitForAll()
        }
        return Result(
            concurrency: concurrency,
            iterations: total,
            elapsed: CFAbsoluteTimeGetCurrent() - startedAt
        )
    }
}
