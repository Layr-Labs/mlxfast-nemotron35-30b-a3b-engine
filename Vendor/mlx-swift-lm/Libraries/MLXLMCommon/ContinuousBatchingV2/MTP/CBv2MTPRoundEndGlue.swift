// CBv2MTPRoundEndGlue.swift
//
// Per-round integer plumbing of an MTP round with fewer GPU launches. Every
// piece moves or skips integer/index work only; no floating-point value,
// reduction order or rounding point changes. Each piece has its own switch
// (default on, "=0" restores the original path) and
// NEMOTRON_MTP_ROUNDEND=0 turns all of them off.

import Foundation
import MLX

public enum CBv2RoundEndGlue {
    private static func on(_ key: String) -> Bool {
        let environment = ProcessInfo.processInfo.environment
        return environment["NEMOTRON_MTP_ROUNDEND"] != "0" && environment[key] != "0"
    }

    /// Attention caches keep their per-call `+ L` position advance as a host
    /// delta and apply it only when `positionOffsets` is read. Models that do
    /// not read the offsets (NemotronH applies no RoPE) then launch no
    /// per-layer `ss_Add int32`.
    public static let lazyPositionOffsets = on("CBV2_LAZY_POSITION_OFFSETS")

    /// The one-row verify window `[1, 1 + k]` is written by one launch from
    /// the seed and the k draft ids instead of a (1 + k)-input concatenate.
    public static let verifyTokens = on("CBV2_MTP_VERIFY_TOKENS")

    /// The top-two finalize also writes the acceptance packet, so the draft
    /// stack and the packet concatenate launch nothing. Needs the window.
    public static let topTwoPacket = on("CBV2_MTP_TOPTWO_PACKET")

    /// NemotronH captured verify: the window's token embeddings through the
    /// draft step's one-launch row kernel (needs NEMOTRON_MTP_DRAFT_EMBED).
    public static let verifyEmbedding = on("NEMOTRON_MTP_VERIFY_EMBED")

    /// Paged segment KV writes read keys/values through their strides, and
    /// `appendTrusted` passes row views instead of `take` gathers: no copy
    /// or gather launch before the write (prestage first-draft KV append,
    /// post-verify speculation write).
    public static let stridedSegmentWrite = on("CBV2_PAGED_STRIDED_WRITE")

    /// NemotronH MTP prestage: submit the one-token expert tail in its own
    /// asyncEval before the head screen and K/V append, so the first command
    /// buffer commits earlier. Evaluation order only.
    public static let prestageSplitEval = on("NEMOTRON_MTP_PRESTAGE_SPLIT_EVAL")

    /// Engine draft loop: submit each in-round draft step's hidden state in
    /// its own asyncEval before the step's ids (the head screen). Evaluation
    /// order only.
    public static let draftSplitEval = on("CBV2_MTP_DRAFT_SPLIT_EVAL")

    /// NEMOTRON_KERNEL_SELFTEST=1: also build the original window/packet and
    /// compare them on the host at finalize; any difference stops the
    /// process with a message.
    public static let selfTest =
        ProcessInfo.processInfo.environment["NEMOTRON_KERNEL_SELFTEST"] == "1"
}

// MARK: - Verify window tokens

/// One launch: out[0, 0] = seed, out[0, 1 + j] = draft j. int32 copies only.
package enum CBv2MTPVerifyTokens {
    private static func source(drafts: Int) -> String {
        var body = "out[0] = seed[0];\n"
        for j in 0 ..< drafts {
            body += "out[\(j + 1)] = d\(j)[0];\n"
        }
        return body
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [Int: MLXFast.MLXFastKernel] = [:]

    private static func kernel(drafts: Int) -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let cached = kernels[drafts] { return cached }
        let made = MLXFast.metalKernel(
            name: "cbv2_mtp_verify_tokens_k\(drafts)",
            inputNames: ["seed"] + (0 ..< drafts).map { "d\($0)" },
            outputNames: ["out"],
            source: source(drafts: drafts))
        kernels[drafts] = made
        return made
    }

    /// seed (one int32 element) and k drafts (one int32 element each) →
    /// `[1, 1 + k]` int32, or nil when switched off or the inputs do not
    /// qualify.
    package static func assemble(seed: MLXArray, drafts: [MLXArray]) -> MLXArray? {
        guard CBv2RoundEndGlue.verifyTokens, (1 ... 7).contains(drafts.count),
            StreamOrDevice.default == .gpu,
            seed.size == 1, seed.dtype == .int32,
            drafts.allSatisfy({ $0.size == 1 && $0.dtype == .int32 })
        else { return nil }
        return kernel(drafts: drafts.count)(
            [seed.reshaped([1])] + drafts.map { $0.reshaped([1]) },
            grid: (1, 1, 1), threadGroup: (1, 1, 1),
            outputShapes: [[1, drafts.count + 1]], outputDTypes: [.int32])[0]
    }

    /// Load-time JIT of every draft depth the engine accepts (1...7).
    package static func warm() {
        guard CBv2RoundEndGlue.verifyTokens, StreamOrDevice.default == .gpu else { return }
        var outputs: [MLXArray] = []
        for k in 1 ... 7 {
            let drafts = (0 ..< k).map { MLXArray([Int32($0 + 2)]) }
            if let out = assemble(seed: MLXArray([Int32(1)]).reshaped([1, 1]), drafts: drafts) {
                outputs.append(out)
            }
        }
        if !outputs.isEmpty { eval(outputs) }
    }
}
