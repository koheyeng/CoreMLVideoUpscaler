#!/bin/bash
# SwiftPM の実行ファイルを macOS のアプリバンドルに包む。
# バンドルにしないと Dock に出ず、アプリとして扱われない（ドロップ操作も不安定になる）。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="${APP_NAME:-VideoUpscaler}"
APP="$ROOT/build/$APP_NAME.app"

cd "$ROOT"
swift build -c release --product UpscalerApp
BIN_DIR="$(swift build -c release --show-bin-path)"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN_DIR/UpscalerApp" "$APP/Contents/MacOS/$APP_NAME"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>$APP_NAME</string>
    <key>CFBundleDisplayName</key><string>動画アップスケーラー</string>
    <key>CFBundleIdentifier</key><string>dev.kohey.videoupscaler</string>
    <key>CFBundleExecutable</key><string>$APP_NAME</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

# モデルはバンドルの Resources/Models へ。
# .mlpackage のまま入れると codesign が中身をバンドルとみなして失敗するので、
# Xcode と同じくコンパイル済みの .mlmodelc にしてから入れる（起動時のコンパイルも省ける）。
shopt -s nullglob
models=("$ROOT"/Models/*.mlpackage "$ROOT"/Models/*.mlmodel)
if [ ${#models[@]} -gt 0 ]; then
    mkdir -p "$APP/Contents/Resources/Models"
    for model in "${models[@]}"; do
        echo "モデルをコンパイル: $(basename "$model")"
        xcrun coremlcompiler compile "$model" "$APP/Contents/Resources/Models" >/dev/null
    done
fi
for compiled in "$ROOT"/Models/*.mlmodelc; do
    mkdir -p "$APP/Contents/Resources/Models"
    cp -R "$compiled" "$APP/Contents/Resources/Models/"
done
shopt -u nullglob

# 署名なしだと Gatekeeper に弾かれるので ad-hoc 署名する。
codesign --force --sign - "$APP" || echo "警告: ad-hoc 署名に失敗しました"

echo "作成しました: $APP"
echo "起動: open \"$APP\""
