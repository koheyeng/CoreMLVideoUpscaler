/// Sendable でない参照型を、並行処理の境界を越えて渡すための箱。
///
/// CVPixelBuffer や AVFoundation のオブジェクトは Sendable ではないが、
/// このパッケージ内では「同時に触るのは 1 か所だけ」という使い方に限って渡している。
struct UnsafeSendableBox<Value>: @unchecked Sendable {
    let value: Value

    init(_ value: Value) {
        self.value = value
    }
}
