import CoreML
import Foundation

/// 演算がどの実行先に割り当てられたかの内訳。
public struct ModelDeviceReport: Sendable {
    /// 実行先ごとの演算数（多い順）。
    public let preferredCounts: [(device: String, count: Int)]
    /// Neural Engine では実行できない演算の種類と数（多い順）。
    public let notSupportedOnNeuralEngine: [(operatorName: String, count: Int)]
    /// Neural Engine 以外に落ちた演算の、推定コスト込みの内訳（重い順）。
    public let offloadedOperations: [(operatorName: String, device: String, cost: Double)]
    public let totalOperations: Int
    /// Neural Engine 以外に落ちた演算の推定コスト合計（モデル全体を 1.0 とした割合）。
    public let offloadedCost: Double

    public var neuralEngineOperations: Int {
        preferredCounts.first { $0.device == "Neural Engine" }?.count ?? 0
    }
    /// Neural Engine に載った演算の割合。
    public var neuralEngineShare: Double {
        totalOperations > 0 ? Double(neuralEngineOperations) / Double(totalOperations) : 0
    }
}

/// Core ML がモデルをどの実行先に割り当てるかを、実行前に調べる。
///
/// 速度比較だけでは「Neural Engine が本当に使われたか」は分からないため、
/// MLComputePlan で演算ごとの割り当てを直接読む。
@available(macOS 14.4, *)
public enum ModelInspector {
    public static func report(modelURL: URL, device: ComputeDevice = .all) async throws -> ModelDeviceReport {
        let compiledURL = try UpscaleModel.compiledModelURL(for: modelURL)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = device.mlComputeUnits

        let plan = try await MLComputePlan.load(contentsOf: compiledURL, configuration: configuration)
        guard case let .program(program) = plan.modelStructure else {
            throw UpscaleError.modelInterfaceMismatch("mlprogram 形式のモデルではないため解析できません")
        }

        var state = WalkState()
        for (_, function) in program.functions {
            walk(function.block, plan: plan, state: &state)
        }

        return ModelDeviceReport(
            preferredCounts: state.preferred.map { (device: $0.key, count: $0.value) }.sorted { $0.count > $1.count },
            notSupportedOnNeuralEngine: state.missingOnANE.map { (operatorName: $0.key, count: $0.value) }
                .sorted { $0.count > $1.count },
            offloadedOperations: state.offloaded.sorted { $0.cost > $1.cost },
            totalOperations: state.total,
            offloadedCost: state.offloaded.reduce(0) { $0 + $1.cost }
        )
    }

    private struct WalkState {
        var preferred: [String: Int] = [:]
        var missingOnANE: [String: Int] = [:]
        var offloaded: [(operatorName: String, device: String, cost: Double)] = []
        var total = 0
    }

    private static func walk(
        _ block: MLModelStructure.Program.Block,
        plan: MLComputePlan,
        state: inout WalkState
    ) {
        for operation in block.operations {
            // const は重みの読み出しで、実行先の話ではないので数えない。
            guard operation.operatorName != "const" else { continue }
            state.total += 1

            if let usage = plan.deviceUsage(for: operation) {
                let deviceName = name(of: usage.preferred)
                state.preferred[deviceName, default: 0] += 1
                if !usage.supported.contains(where: { isNeuralEngine($0) }) {
                    state.missingOnANE[operation.operatorName, default: 0] += 1
                }
                if !isNeuralEngine(usage.preferred) {
                    let cost = plan.estimatedCost(of: operation)?.weight ?? 0
                    state.offloaded.append((operation.operatorName, deviceName, cost))
                }
            }
            for nested in operation.blocks {
                walk(nested, plan: plan, state: &state)
            }
        }
    }

    private static func isNeuralEngine(_ device: MLComputeDevice) -> Bool {
        if case .neuralEngine = device { return true }
        return false
    }

    private static func name(of device: MLComputeDevice) -> String {
        switch device {
        case .cpu: "CPU"
        case .gpu: "GPU"
        case .neuralEngine: "Neural Engine"
        @unknown default: "不明"
        }
    }
}
