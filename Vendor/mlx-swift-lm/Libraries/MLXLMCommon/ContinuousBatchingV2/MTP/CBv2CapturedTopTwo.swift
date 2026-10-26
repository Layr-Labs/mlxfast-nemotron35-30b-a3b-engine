// Exact top-two captured-forward capability. Internal Swift model seam only;
// the resident wire protocol and target-authoritative token contract are unchanged.
import MLX

public struct CBv2CapturedTopTwoOutput {
    public let ids: MLXArray
    public let values: MLXArray
    public let lastHidden: MLXArray
    public init(ids: MLXArray, values: MLXArray, lastHidden: MLXArray) {
        self.ids = ids; self.values = values; self.lastHidden = lastHidden
    }
}

public protocol CBv2RecurrentCaptureMTPTopTwoForwardable: CBv2RecurrentCaptureMTPForwardable {
    var cbv2SupportsTopTwoCaptured: Bool { get }
    func cbv2ForwardWithTopTwoCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> CBv2CapturedTopTwoOutput
}

public protocol CBv2RecurrentMTPTopTwoSteppableModel: CBv2RecurrentMTPSteppableModel {
    var supportsTopTwoCaptured: Bool { get }
    func forwardWithTopTwoCaptured(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> CBv2CapturedTopTwoOutput
}
