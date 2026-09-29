import Accelerate
import CoreVideo
import Foundation

/// このパッケージ全体で扱うピクセルフォーマット。
/// Core ML の image 入出力が 32BGRA を素直に受け渡しできるため、全段でこれに統一する。
let kUpscalePixelFormat = kCVPixelFormatType_32BGRA

enum UpscaleError: LocalizedError {
    case pixelBufferAllocationFailed(OSStatus)
    case pixelBufferLockFailed
    case unsupportedPixelFormat(OSType)
    case vImageFailed(vImage_Error)
    case modelNotFound(String)
    case modelInterfaceMismatch(String)
    case noVideoTrack
    case readerFailed(String)
    case writerFailed(String)
    case cancelled
    /// フレームレート変換など、OS バージョンやハードウェアの要件を満たさない機能。
    case featureUnavailable(String)

    var errorDescription: String? {
        switch self {
        case .pixelBufferAllocationFailed(let s): "CVPixelBuffer の確保に失敗しました (status: \(s))"
        case .pixelBufferLockFailed: "CVPixelBuffer のロックに失敗しました"
        case .unsupportedPixelFormat(let f): "未対応のピクセルフォーマットです (\(f))"
        case .vImageFailed(let e): "vImage の処理に失敗しました (error: \(e))"
        case .modelNotFound(let p): "Core ML モデルが見つかりません: \(p)"
        case .modelInterfaceMismatch(let m): "Core ML モデルの入出力が想定と異なります: \(m)"
        case .noVideoTrack: "映像トラックが含まれていません"
        case .readerFailed(let m): "動画の読み込みに失敗しました: \(m)"
        case .writerFailed(let m): "動画の書き出しに失敗しました: \(m)"
        case .cancelled: "処理がキャンセルされました"
        case .featureUnavailable(let m): m
        }
    }
}

/// 同一サイズの 32BGRA バッファを繰り返し確保するための薄いプール。
/// 1 フレームごとに CVPixelBufferCreate を呼ぶと確保コストが無視できないため。
final class PixelBufferPool: @unchecked Sendable {
    let width: Int
    let height: Int
    private let pool: CVPixelBufferPool

    init(width: Int, height: Int) throws {
        self.width = width
        self.height = height
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kUpscalePixelFormat,
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

/// ロック済みの 32BGRA バッファを指す軽量ビュー。
struct BGRAView {
    let base: UnsafeMutableRawPointer
    let width: Int
    let height: Int
    let rowBytes: Int

    var vImageBuffer: vImage_Buffer {
        vImage_Buffer(data: base, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: rowBytes)
    }

    /// 部分矩形を指すビュー。ピクセルをコピーせず、開始アドレスをずらすだけ。
    func region(x: Int, y: Int, width w: Int, height h: Int) -> BGRAView {
        BGRAView(base: base.advanced(by: y * rowBytes + x * 4), width: w, height: h, rowBytes: rowBytes)
    }

    func row(_ y: Int) -> UnsafeMutablePointer<UInt8> {
        base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
    }
}

extension CVPixelBuffer {
    var pixelWidth: Int { CVPixelBufferGetWidth(self) }
    var pixelHeight: Int { CVPixelBufferGetHeight(self) }

    /// 32BGRA として base address を触る。read-only なら `readOnly: true` を渡す。
    func withBGRA<T>(readOnly: Bool, _ body: (BGRAView) throws -> T) throws -> T {
        let format = CVPixelBufferGetPixelFormatType(self)
        guard format == kUpscalePixelFormat else {
            throw UpscaleError.unsupportedPixelFormat(format)
        }
        let flags: CVPixelBufferLockFlags = readOnly ? .readOnly : []
        guard CVPixelBufferLockBaseAddress(self, flags) == kCVReturnSuccess else {
            throw UpscaleError.pixelBufferLockFailed
        }
        defer { CVPixelBufferUnlockBaseAddress(self, flags) }
        guard let base = CVPixelBufferGetBaseAddress(self) else {
            throw UpscaleError.pixelBufferLockFailed
        }
        let view = BGRAView(
            base: base,
            width: CVPixelBufferGetWidth(self),
            height: CVPixelBufferGetHeight(self),
            rowBytes: CVPixelBufferGetBytesPerRow(self)
        )
        return try body(view)
    }

    func fillBlack() throws {
        try withBGRA(readOnly: false) { view in
            for y in 0..<view.height {
                memset(view.row(y), 0, view.width * 4)
            }
        }
    }
}
