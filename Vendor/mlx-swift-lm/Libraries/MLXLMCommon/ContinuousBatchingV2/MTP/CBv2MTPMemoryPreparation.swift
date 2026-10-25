import Foundation

/// Memory management only; no request graph may be evaluated here.
public protocol CBv2MTPMemoryPreparing {
    func beginMTPMemoryPreparation() -> (any CBv2MTPMemoryLease)?
}
public protocol CBv2MTPMemoryLease: AnyObject {
    func beginDecode()
    func finish()
}
