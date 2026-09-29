import AppKit
import SwiftUI

@main
struct UpscalerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        WindowGroup("動画アップスケーラー") {
            ContentView()
                .frame(minWidth: 620, minHeight: 680)
        }
        .windowResizability(.contentMinSize)
    }
}

/// SwiftPM の実行ファイルはアプリバンドルではないので、Dock に出てフォアグラウンドに
/// 来るように起動時へ明示的に設定する。
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
