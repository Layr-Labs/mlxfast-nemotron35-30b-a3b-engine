// CBv2CriticalPathCuts.swift
//
// Launch cuts on the serial GPU chain of an MTP round. Each cut moves or
// removes data-movement launches only; every arithmetic value, reduction
// order and rounding point is unchanged. Each has its own switch (default
// on, "=0" restores the original path); NEMOTRON_CRITPATH=0 turns all off.

import Foundation

public enum CBv2CriticalPathCuts {
    private static func on(_ key: String) -> Bool {
        let environment = ProcessInfo.processInfo.environment
        return environment["NEMOTRON_CRITPATH"] != "0" && environment[key] != "0"
    }

    /// Contiguous KV append written in place by the producer kernel (the
    /// NemotronH captured-verify Q/K/V launch stores K/V rows straight into
    /// the row's cache buffers): no slice-update copy per array per layer.
    public static let kvInPlaceAppend = on("CBV2_KV_IN_PLACE_APPEND")

    /// Paged segment KV writes read keys/values through their strides, and
    /// `PagedLayerCache.appendTrusted` passes row views instead of `take`
    /// gathers: no contiguous copy or gather launch before a write.
    public static let stridedSegmentWrite = on("CBV2_PAGED_STRIDED_WRITE")

    /// Paged decode merge: the unused placeholder sinks input comes from one
    /// cached host-built zeros array per length instead of a per-call fill.
    public static let cachedZeroSinks = on("CBV2_PAGED_CACHED_ZERO_SINKS")

    /// NemotronH compact verify: a strict-prefix commit replays the Mamba
    /// layers' SSM states in grouped launches (up to six layers each)
    /// instead of one launch per layer. Each layer's replay is the per-layer
    /// kernel's statements on its own operands.
    public static let ssmReplayGroup = on("NEMOTRON_MTP_SSM_REPLAY_GROUP")

    /// MTP device-accepted first draft: the one-row accept walk over the
    /// acceptance packet in one integer launch instead of Equal, AsType,
    /// cumprod and sum (same [1] int32 count; the host still decides).
    public static let acceptWalkKernel = on("CBV2_MTP_ACCEPT_WALK")

    /// NEMOTRON_KERNEL_SELFTEST=1: host-side bitwise checks of the cuts
    /// above against their original builds (process stops on a difference).
    public static let selfTest =
        ProcessInfo.processInfo.environment["NEMOTRON_KERNEL_SELFTEST"] == "1"
}
