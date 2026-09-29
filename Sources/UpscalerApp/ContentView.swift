import AppKit
import SwiftUI
import UniformTypeIdentifiers
import UpscaleKit

struct ContentView: View {
    @State private var controller = UpscaleController()
    @State private var isDropTargeted = false

    var body: some View {
        VStack(spacing: 14) {
            inputArea
            settings
            statusArea
            Spacer(minLength: 0)
            actions
        }
        .padding(20)
    }

    // MARK: 入力（キュー）

    @ViewBuilder
    private var inputArea: some View {
        Group {
            if controller.queue.isEmpty {
                emptyDropZone
            } else {
                queueList
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: 190)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(isDropTargeted ? Color.accentColor.opacity(0.12) : Color.secondary.opacity(0.06))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(isDropTargeted ? Color.accentColor : Color.secondary.opacity(0.35),
                              style: StrokeStyle(lineWidth: 1.5, dash: [6]))
        )
        .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers in
            loadDroppedFiles(from: providers)
        }
    }

    private var emptyDropZone: some View {
        VStack(spacing: 6) {
            Image(systemName: "arrow.down.doc")
                .font(.system(size: 32))
                .foregroundStyle(.secondary)
            Text("動画をここにドロップ（複数可）").font(.headline)
            Text("またはクリックして選択").font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        .onTapGesture { chooseInputs() }
        .disabled(controller.isRunning)
    }

    private var queueList: some View {
        VStack(spacing: 0) {
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(controller.queue) { item in
                        queueRow(item)
                        Divider().opacity(0.4)
                    }
                }
            }
            HStack(spacing: 12) {
                Button("追加…") { chooseInputs() }
                Button("すべて削除") { controller.clearQueue() }
                Spacer()
                Text("\(controller.queue.count) 件")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .controlSize(.small)
            .disabled(controller.isRunning)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
        }
        .padding(4)
    }

    private func queueRow(_ item: UpscaleController.QueueItem) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "film")
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(item.url.lastPathComponent)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(item.detail ?? item.mediaSummary ?? "読み込み中…")
                    .font(.caption)
                    .foregroundStyle(item.status == .failed ? AnyShapeStyle(.red) : AnyShapeStyle(.secondary))
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer()
            statusBadge(item)
            if controller.isRunning {
                EmptyView()
            } else {
                Button {
                    controller.removeItem(item.id)
                } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("リストから外す")
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    @ViewBuilder
    private func statusBadge(_ item: UpscaleController.QueueItem) -> some View {
        switch item.status {
        case .running:
            HStack(spacing: 6) {
                ProgressView(value: item.fraction ?? 0)
                    .frame(width: 70)
                if let fraction = item.fraction {
                    Text(String(format: "%.0f%%", fraction * 100))
                        .font(.caption)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            }
        case .done:
            HStack(spacing: 6) {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                if let output = item.outputURL {
                    Button("表示") {
                        NSWorkspace.shared.activateFileViewerSelecting([output])
                    }
                    .buttonStyle(.link)
                    .font(.caption)
                }
            }
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
                .help(item.detail ?? "エラー")
        case .skipped:
            Text("スキップ").font(.caption).foregroundStyle(.orange)
        case .cancelled:
            Text("キャンセル").font(.caption).foregroundStyle(.secondary)
        case .pending:
            Text("待機中").font(.caption).foregroundStyle(.secondary)
        }
    }

    // MARK: 設定

    private var settings: some View {
        Form {
            Picker("モデル", selection: modelBinding) {
                if controller.availableModels.isEmpty {
                    Text("モデルがありません").tag(Optional<URL>.none)
                }
                ForEach(controller.availableModels, id: \.path) { url in
                    Text(url.deletingPathExtension().lastPathComponent).tag(Optional(url))
                }
            }
            HStack(alignment: .firstTextBaseline) {
                if let breakdown = controller.deviceBreakdown {
                    Label(breakdown, systemImage: "cpu")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                Button("別のモデルを選ぶ…") { chooseModel() }
                    .buttonStyle(.link)
                    .font(.caption)
                    .fixedSize()
            }

            Picker("出力解像度", selection: $controller.preset) {
                ForEach(OutputPreset.allCases) { Text($0.rawValue).tag($0) }
            }
            Picker("縦横比の合わせ方", selection: $controller.fitMode) {
                ForEach(FitMode.allCases, id: \.self) { Text($0.label).tag($0) }
            }
            if let planned = controller.plannedOutputSize {
                LabeledContent("書き出しサイズ", value: planned)
            }
            Picker("コーデック", selection: $controller.codec) {
                Text("HEVC").tag(VideoCodec.hevc)
                Text("H.264").tag(VideoCodec.h264)
                Text("ProRes 422").tag(VideoCodec.proRes422)
            }
            Picker("フレームレート", selection: $controller.frameRatePreset) {
                ForEach(FrameRatePreset.allCases) { Text($0.rawValue).tag($0) }
            }
            .disabled(!FrameRateConversion.isAvailable)
            .help(FrameRateConversion.isAvailable ? "" : "フレームレート変換には macOS 15.4 以降が必要です")
            Toggle("スーパーサンプリング（2回適用・高画質だが約5倍遅い）", isOn: $controller.doublePass)
            Picker("実行先", selection: $controller.device) {
                Text("Neural Engine").tag(ComputeDevice.cpuAndNeuralEngine)
                Text("自動 (Core ML 任せ)").tag(ComputeDevice.all)
                Text("GPU").tag(ComputeDevice.cpuAndGPU)
                Text("CPU のみ").tag(ComputeDevice.cpuOnly)
            }
        }
        .formStyle(.grouped)
        .disabled(controller.isRunning)
    }

    private var modelBinding: Binding<URL?> {
        Binding(
            get: { controller.modelURL },
            set: { if let url = $0 { controller.selectModel(url) } }
        )
    }

    // MARK: 状態

    @ViewBuilder
    private var statusArea: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let fraction = controller.overallFraction {
                ProgressView(value: fraction) { Text(controller.statusText) }
            } else if !controller.statusText.isEmpty {
                Text(controller.statusText).font(.callout)
            }

            if let message = controller.errorMessage {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: 操作

    private var actions: some View {
        HStack {
            if controller.isRunning {
                Button("キャンセル", role: .cancel) { controller.cancel() }
            }
            Spacer()
            Button(controller.queue.count > 1 ? "\(controller.queue.count) 件をアップスケール" : "アップスケール開始") {
                startBatch()
            }
            .keyboardShortcut(.defaultAction)
            .disabled(!controller.canStart)
        }
    }

    // MARK: パネル

    private func chooseInputs() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.movie, .video, .mpeg4Movie, .quickTimeMovie]
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK {
            controller.addInputs(panel.urls)
        }
    }

    private func chooseModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true  // .mlpackage はディレクトリ
        panel.allowsMultipleSelection = false
        panel.message = "Core ML モデル (.mlpackage / .mlmodelc) を選んでください"
        if panel.runModal() == .OK, let url = panel.url {
            controller.selectModel(url)
        }
    }

    private func startBatch() {
        let queue = controller.queue
        guard !queue.isEmpty else { return }

        if queue.count == 1 {
            // 1 件のときはこれまでどおり保存パネルで場所と名前を決める（上書きもパネルが確認する）。
            let item = queue[0]
            let panel = NSSavePanel()
            panel.allowedContentTypes = [.quickTimeMovie, .mpeg4Movie]
            panel.directoryURL = item.url.deletingLastPathComponent()
            panel.nameFieldStringValue = controller.defaultOutputName(for: item)
            panel.message = "書き出し先を選んでください"
            if panel.runModal() == .OK, let url = panel.url {
                controller.start(outputs: [item.id: url], overwriteExisting: true)
            }
        } else {
            // 複数のときはフォルダを 1 回選び、名前は自動。既にあるファイルはスキップする。
            let panel = NSOpenPanel()
            panel.canChooseFiles = false
            panel.canChooseDirectories = true
            panel.canCreateDirectories = true
            panel.allowsMultipleSelection = false
            panel.prompt = "この場所に書き出す"
            panel.message = "書き出し先フォルダを選んでください（ファイル名は自動。既にあるものはスキップ）"
            panel.directoryURL = queue[0].url.deletingLastPathComponent()
            if panel.runModal() == .OK, let directory = panel.url {
                var outputs: [UpscaleController.QueueItem.ID: URL] = [:]
                for item in queue {
                    outputs[item.id] = directory.appendingPathComponent(controller.defaultOutputName(for: item))
                }
                controller.start(outputs: outputs, overwriteExisting: false)
            }
        }
    }

    private func loadDroppedFiles(from providers: [NSItemProvider]) -> Bool {
        guard !providers.isEmpty else { return false }
        for provider in providers {
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                Task { @MainActor in controller.addInputs([url]) }
            }
        }
        return true
    }
}
