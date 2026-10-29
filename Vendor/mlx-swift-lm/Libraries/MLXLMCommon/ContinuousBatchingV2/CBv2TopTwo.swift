// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

// MARK: - Hierarchical top-2

// Metal entry-point names remain stable so extraction preserves pipeline identity.
/// Shared exact ordering for the two-stage candidate-only top-2 reduction:
/// value descending, token id ascending on exact ties, and NaNs last.
private let cbv2TopTwoHeader = """
    struct darkbloom_qwen35_mtp_top2_state {
        float first_value;
        float second_value;
        uint first_id;
        uint second_id;
        uint count;
    };

    inline darkbloom_qwen35_mtp_top2_state darkbloom_qwen35_mtp_top2_empty() {
        darkbloom_qwen35_mtp_top2_state state;
        state.first_value = 0.0f;
        state.second_value = 0.0f;
        state.first_id = 0;
        state.second_id = 0;
        state.count = 0;
        return state;
    }

    inline bool darkbloom_qwen35_mtp_top2_better(
        float candidate_value,
        uint candidate_id,
        float current_value,
        uint current_id
    ) {
        bool candidate_nan = isnan(candidate_value);
        bool current_nan = isnan(current_value);
        if (candidate_nan != current_nan) {
            return !candidate_nan;
        }
        if (candidate_value > current_value) {
            return true;
        }
        if (candidate_value < current_value) {
            return false;
        }
        return candidate_id < current_id;
    }

    inline void darkbloom_qwen35_mtp_top2_insert(
        thread darkbloom_qwen35_mtp_top2_state &state,
        float value,
        uint id
    ) {
        if (state.count > 0 && state.first_id == id) {
            return;
        }
        if (state.count > 1 && state.second_id == id) {
            return;
        }
        if (state.count == 0
            || darkbloom_qwen35_mtp_top2_better(
                value, id, state.first_value, state.first_id)) {
            if (state.count > 0) {
                state.second_value = state.first_value;
                state.second_id = state.first_id;
            }
            state.first_value = value;
            state.first_id = id;
            state.count = min(state.count + 1, 2u);
            return;
        }
        if (state.count == 1
            || darkbloom_qwen35_mtp_top2_better(
                value, id, state.second_value, state.second_id)) {
            state.second_value = value;
            state.second_id = id;
            state.count = 2;
        }
    }
"""

/// Stage one: 32 threadgroups per row reduce disjoint vocabulary stripes.
private let cbv2TopTwoPartialKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen35_mtp_top2_partial",
    inputNames: ["logits"],
    outputNames: ["partial_ids", "partial_values"],
    source: """
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint row = group_index / 32;
        uint group = group_index % 32;
        uint vocab = uint(logits_shape[2]);
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();

        for (uint index = group * 256 + lane;
             index < vocab;
             index += 32 * 256) {
            ulong offset = ulong(row) * ulong(logits_strides[1])
                + ulong(index) * ulong(logits_strides[2]);
            darkbloom_qwen35_mtp_top2_insert(local, float(logits[offset]), index);
        }

        threadgroup darkbloom_qwen35_mtp_top2_state scratch[256];
        scratch[lane] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (lane < stride) {
                darkbloom_qwen35_mtp_top2_state merged = scratch[lane];
                darkbloom_qwen35_mtp_top2_state other = scratch[lane + stride];
                if (other.count > 0) {
                    darkbloom_qwen35_mtp_top2_insert(
                        merged, other.first_value, other.first_id);
                }
                if (other.count > 1) {
                    darkbloom_qwen35_mtp_top2_insert(
                        merged, other.second_value, other.second_id);
                }
                scratch[lane] = merged;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            uint base = (row * 32 + group) * 2;
            uint sentinel_id = vocab + group * 2;
            float sentinel_value = as_type<float>(0x7fc00000u);
            partial_ids[base] = scratch[0].count > 0
                ? int(scratch[0].first_id) : int(sentinel_id);
            partial_ids[base + 1] = scratch[0].count > 1
                ? int(scratch[0].second_id) : int(sentinel_id + 1);
            partial_values[base] = scratch[0].count > 0
                ? scratch[0].first_value : sentinel_value;
            partial_values[base + 1] = scratch[0].count > 1
                ? scratch[0].second_value : sentinel_value;
        }
    """,
    header: cbv2TopTwoHeader,
    ensureRowContiguous: false
)

/// Stage two: one 32-lane threadgroup per row merges the partial pairs.
private let cbv2TopTwoFinalizeKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen35_mtp_top2_finalize",
    inputNames: ["partial_ids", "partial_values"],
    outputNames: ["top_ids", "top_values"],
    source: """
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint base = (row * 32 + lane) * 2;
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();
        darkbloom_qwen35_mtp_top2_insert(
            local, partial_values[base], uint(partial_ids[base]));
        darkbloom_qwen35_mtp_top2_insert(
            local, partial_values[base + 1], uint(partial_ids[base + 1]));

        threadgroup darkbloom_qwen35_mtp_top2_state scratch[32];
        scratch[lane] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 16; stride > 0; stride >>= 1) {
            if (lane < stride) {
                darkbloom_qwen35_mtp_top2_state merged = scratch[lane];
                darkbloom_qwen35_mtp_top2_state other = scratch[lane + stride];
                darkbloom_qwen35_mtp_top2_insert(
                    merged, other.first_value, other.first_id);
                darkbloom_qwen35_mtp_top2_insert(
                    merged, other.second_value, other.second_id);
                scratch[lane] = merged;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            uint output_base = row * 2;
            top_ids[output_base] = int(scratch[0].first_id);
            top_ids[output_base + 1] = int(scratch[0].second_id);
            top_values[output_base] = scratch[0].first_value;
            top_values[output_base + 1] = scratch[0].second_value;
        }
    """,
    header: cbv2TopTwoHeader,
    ensureRowContiguous: false
)

/// Exact top-2 token ids and logit values for every row of `[1, rows, vocab]`.
///
/// Returns lazy device arrays shaped `[rows, 2]`: ids are `int32`, values are
/// `float32`. No evaluation or host read occurs here. Results are ordered by
/// value descending, then token id ascending on exact ties, with NaNs last.
package func cbv2TopTwoRows(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
    precondition(logits.ndim == 3 && logits.dim(0) == 1)
    let rows = logits.dim(1)
    let vocabularySize = logits.dim(2)
    precondition(rows > 0 && vocabularySize >= 2)

    let partials = cbv2TopTwoPartialKernel(
        [logits],
        grid: (rows * 32 * 256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[rows, 32, 2], [rows, 32, 2]],
        outputDTypes: [.int32, .float32]
    )
    let outputs = cbv2TopTwoFinalizeKernel(
        partials,
        grid: (rows * 32, 1, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [[rows, 2], [rows, 2]],
        outputDTypes: [.int32, .float32]
    )
    return (outputs[0], outputs[1])
}

// MARK: - Top-2 finalize that also writes the MTP acceptance packet

/// CBV2_MTP_TOP2_PACKET=0 keeps the acceptance packet as a separate
/// concatenate (one copy launch per part) after the top-2 finalize.
public enum CBv2MTPTopTwoPacket {
    public static let enabled =
        ProcessInfo.processInfo.environment["CBV2_MTP_TOP2_PACKET"] != "0"
    /// NEMOTRON_KERNEL_SELFTEST=1: compare with the concatenated packet on
    /// the host when it is built (the process stops on a difference).
    public static let selfTest =
        ProcessInfo.processInfo.environment["NEMOTRON_KERNEL_SELFTEST"] == "1"
}

/// The finalize body above, unchanged, plus the MTP acceptance packet the
/// engine otherwise concatenates: `[B * k]` drafts (window columns 1...k,
/// row-major), the `[B * (1 + k)]` top-1 ids, then the `[B * (1 + k) * 2]`
/// top-2 values as their int32 bit patterns. `window` is the `[B, 1 + k]`
/// int32 verify input; logits row r is its position r. Integer stores only:
/// each element is window[r], or a value this launch already writes.
private let cbv2TopTwoFinalizePacketSource = """
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint base = (row * 32 + lane) * 2;
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();
        darkbloom_qwen35_mtp_top2_insert(
            local, partial_values[base], uint(partial_ids[base]));
        darkbloom_qwen35_mtp_top2_insert(
            local, partial_values[base + 1], uint(partial_ids[base + 1]));

        threadgroup darkbloom_qwen35_mtp_top2_state scratch[32];
        scratch[lane] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 16; stride > 0; stride >>= 1) {
            if (lane < stride) {
                darkbloom_qwen35_mtp_top2_state merged = scratch[lane];
                darkbloom_qwen35_mtp_top2_state other = scratch[lane + stride];
                darkbloom_qwen35_mtp_top2_insert(
                    merged, other.first_value, other.first_id);
                darkbloom_qwen35_mtp_top2_insert(
                    merged, other.second_value, other.second_id);
                scratch[lane] = merged;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            uint output_base = row * 2;
            top_ids[output_base] = int(scratch[0].first_id);
            top_ids[output_base + 1] = int(scratch[0].second_id);
            top_values[output_base] = scratch[0].first_value;
            top_values[output_base + 1] = scratch[0].second_value;

            uint batch = uint(window_shape[0]);
            uint width = uint(window_shape[1]);
            uint drafts = batch * (width - 1);
            uint rows = batch * width;
            packet[drafts + row] = int(scratch[0].first_id);
            packet[drafts + rows + output_base] = as_type<int>(scratch[0].first_value);
            packet[drafts + rows + output_base + 1] = as_type<int>(scratch[0].second_value);
            uint column = row % width;
            if (column > 0) {
                packet[(row / width) * (width - 1) + (column - 1)] = window[row];
            }
        }
    """
private let cbv2TopTwoFinalizePacketKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen35_mtp_top2_finalize_packet",
    inputNames: ["partial_ids", "partial_values", "window"],
    outputNames: ["top_ids", "top_values", "packet"],
    source: cbv2TopTwoFinalizePacketSource,
    header: cbv2TopTwoHeader,
    ensureRowContiguous: true
)

// MARK: - Shuffle-form reductions of the decode round's top-2

/// CBV2_MTP_TOP2_SHUFFLE=0 keeps the threadgroup-tree reductions in the decode
/// round's top-2 (the partial and the packet finalize). Otherwise each SIMD
/// group merges its lanes' states through `simd_shuffle_down` (offsets 16, 8,
/// 4, 2, 1, the lower lane keeping the merge), the partial then merges its
/// eight SIMD-group states after one barrier, and the finalize (one SIMD group)
/// needs no threadgroup memory: one barrier instead of nine, none instead of
/// six. Every merge is the tree's `insert` code. The two best of a set under
/// the shared order (value descending, id ascending on ties, NaNs last; the
/// ids, sentinels included, are distinct) do not depend on the merge order, so
/// the ids and values (moved as raw bits) are the tree's. Used only after
/// `cbv2TopTwoShuffleQualify` found both bitwise equal to the tree launches.
public enum CBv2MTPTopTwoShuffle {
    public static let enabled =
        ProcessInfo.processInfo.environment["CBV2_MTP_TOP2_SHUFFLE"] != "0"
    nonisolated(unsafe) static var verified = false
}

private let cbv2TopTwoShuffleHeader = cbv2TopTwoHeader + """

    inline darkbloom_qwen35_mtp_top2_state darkbloom_qwen35_mtp_top2_shuffle_down(
        darkbloom_qwen35_mtp_top2_state state,
        ushort delta
    ) {
        darkbloom_qwen35_mtp_top2_state other;
        other.first_value = as_type<float>(simd_shuffle_down(as_type<uint>(state.first_value), delta));
        other.second_value = as_type<float>(simd_shuffle_down(as_type<uint>(state.second_value), delta));
        other.first_id = simd_shuffle_down(state.first_id, delta);
        other.second_id = simd_shuffle_down(state.second_id, delta);
        other.count = simd_shuffle_down(state.count, delta);
        return other;
    }
"""

/// The partial with the shuffle-form reduction (see `CBv2MTPTopTwoShuffle`);
/// 256 threads, eight SIMD groups.
private let cbv2TopTwoPartialShuffleSource = """
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint row = group_index / 32;
        uint group = group_index % 32;
        uint vocab = uint(logits_shape[2]);
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();

        for (uint index = group * 256 + lane;
             index < vocab;
             index += 32 * 256) {
            ulong offset = ulong(row) * ulong(logits_strides[1])
                + ulong(index) * ulong(logits_strides[2]);
            darkbloom_qwen35_mtp_top2_insert(local, float(logits[offset]), index);
        }

        const uint simd_lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;
        for (ushort delta = 16; delta > 0; delta >>= 1) {
            darkbloom_qwen35_mtp_top2_state other =
                darkbloom_qwen35_mtp_top2_shuffle_down(local, delta);
            if (simd_lane < delta) {
                if (other.count > 0) {
                    darkbloom_qwen35_mtp_top2_insert(
                        local, other.first_value, other.first_id);
                }
                if (other.count > 1) {
                    darkbloom_qwen35_mtp_top2_insert(
                        local, other.second_value, other.second_id);
                }
            }
        }

        threadgroup darkbloom_qwen35_mtp_top2_state scratch[8];
        if (simd_lane == 0) {
            scratch[simd_group] = local;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            local = simd_lane < 8 ? scratch[simd_lane] : darkbloom_qwen35_mtp_top2_empty();
            for (ushort delta = 4; delta > 0; delta >>= 1) {
                darkbloom_qwen35_mtp_top2_state other =
                    darkbloom_qwen35_mtp_top2_shuffle_down(local, delta);
                if (simd_lane < delta) {
                    if (other.count > 0) {
                        darkbloom_qwen35_mtp_top2_insert(
                            local, other.first_value, other.first_id);
                    }
                    if (other.count > 1) {
                        darkbloom_qwen35_mtp_top2_insert(
                            local, other.second_value, other.second_id);
                    }
                }
            }
            if (simd_lane == 0) {
                uint base = (row * 32 + group) * 2;
                uint sentinel_id = vocab + group * 2;
                float sentinel_value = as_type<float>(0x7fc00000u);
                partial_ids[base] = local.count > 0
                    ? int(local.first_id) : int(sentinel_id);
                partial_ids[base + 1] = local.count > 1
                    ? int(local.second_id) : int(sentinel_id + 1);
                partial_values[base] = local.count > 0
                    ? local.first_value : sentinel_value;
                partial_values[base + 1] = local.count > 1
                    ? local.second_value : sentinel_value;
            }
        }
    """
private let cbv2TopTwoPartialShuffleKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen35_mtp_top2_partial_sh",
    inputNames: ["logits"],
    outputNames: ["partial_ids", "partial_values"],
    source: cbv2TopTwoPartialShuffleSource,
    header: cbv2TopTwoShuffleHeader,
    ensureRowContiguous: false
)

// MARK: - 32-bit element offsets in the shuffle partial

/// CBV2_MTP_TOP2_OFFSET32=0 keeps the shuffle partial's vocabulary walk. Otherwise the decode round's shuffle
/// partial runs as a twin (`cbv2TopTwoOffset32Source`) whose walk reads element `index` of its row at
/// `logits[row * vocab + index]` with the offset in 32 bits, where the record forms `row * logits_strides[1] +
/// index * logits_strides[2]` in 64 bits for every element. The twin is launched with row-contiguous logits
/// (`ensureRowContiguous`; MLX copies any other layout first, the decode round's [1, 3, 131072] logits are the
/// LM head's own row-contiguous output, so no copy) and only when rows * vocab fits in 32 bits (393,216 in the
/// decode round): there the record's offset is row * vocab + index, so every element is read at the record's
/// address. The walk visits the same indices in the same order with the same `insert`, and the shuffle merges
/// and stores are the record's, so the ids and values are bitwise the record's. Used only after
/// `cbv2TopTwoOffset32Qualify` found it bitwise equal to the shuffle partial.
public enum CBv2MTPTopTwoOffset32 {
    public static let enabled =
        ProcessInfo.processInfo.environment["CBV2_MTP_TOP2_OFFSET32"] != "0"
    nonisolated(unsafe) static var verified = false
}

private func cbv2TopTwoOffset32Source(_ s: String) -> String? {
    let walk = """
            for (uint index = group * 256 + lane;
                 index < vocab;
                 index += 32 * 256) {
                ulong offset = ulong(row) * ulong(logits_strides[1])
                    + ulong(index) * ulong(logits_strides[2]);
                darkbloom_qwen35_mtp_top2_insert(local, float(logits[offset]), index);
            }

        """
    guard s.components(separatedBy: walk).count == 2 else { return nil }
    return s.replacingOccurrences(of: walk, with: """
            const auto t2_row = logits + row * vocab;
            for (uint index = group * 256 + lane;
                 index < vocab;
                 index += 32 * 256) {
                darkbloom_qwen35_mtp_top2_insert(local, float(t2_row[index]), index);
            }

        """)
}

private let cbv2TopTwoPartialShuffleOffset32Kernel = cbv2TopTwoOffset32Source(cbv2TopTwoPartialShuffleSource).map {
    MLXFast.metalKernel(
        name: "darkbloom_qwen35_mtp_top2_partial_sh_o32",
        inputNames: ["logits"],
        outputNames: ["partial_ids", "partial_values"],
        source: $0,
        header: cbv2TopTwoShuffleHeader,
        ensureRowContiguous: true)
}

/// The packet finalize with the shuffle-form reduction: one SIMD group per row.
private let cbv2TopTwoFinalizePacketShuffleSource = """
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint base = (row * 32 + lane) * 2;
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();
        darkbloom_qwen35_mtp_top2_insert(
            local, partial_values[base], uint(partial_ids[base]));
        darkbloom_qwen35_mtp_top2_insert(
            local, partial_values[base + 1], uint(partial_ids[base + 1]));

        for (ushort delta = 16; delta > 0; delta >>= 1) {
            darkbloom_qwen35_mtp_top2_state other =
                darkbloom_qwen35_mtp_top2_shuffle_down(local, delta);
            if (lane < delta) {
                darkbloom_qwen35_mtp_top2_insert(
                    local, other.first_value, other.first_id);
                darkbloom_qwen35_mtp_top2_insert(
                    local, other.second_value, other.second_id);
            }
        }

        if (lane == 0) {
            uint output_base = row * 2;
            top_ids[output_base] = int(local.first_id);
            top_ids[output_base + 1] = int(local.second_id);
            top_values[output_base] = local.first_value;
            top_values[output_base + 1] = local.second_value;

            uint batch = uint(window_shape[0]);
            uint width = uint(window_shape[1]);
            uint drafts = batch * (width - 1);
            uint rows = batch * width;
            packet[drafts + row] = int(local.first_id);
            packet[drafts + rows + output_base] = as_type<int>(local.first_value);
            packet[drafts + rows + output_base + 1] = as_type<int>(local.second_value);
            uint column = row % width;
            if (column > 0) {
                packet[(row / width) * (width - 1) + (column - 1)] = window[row];
            }
        }
    """
private let cbv2TopTwoFinalizePacketShuffleKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen35_mtp_top2_finalize_packet_sh",
    inputNames: ["partial_ids", "partial_values", "window"],
    outputNames: ["top_ids", "top_values", "packet"],
    source: cbv2TopTwoFinalizePacketShuffleSource,
    header: cbv2TopTwoShuffleHeader,
    ensureRowContiguous: true
)

// MARK: - Packet finalize reading the verify columns

/// CBV2_MTP_PACKET_COLUMNS=0 keeps the packet finalize reading the draft ids from the verify window. Otherwise,
/// while the engine publishes the window's columns (`CBv2MTPVerifyWindowColumns`, batch 1, width 3), a twin of
/// each finalize form (tree, shuffle) takes the two draft columns ([1, 1] int32 each) and stores row r's draft
/// id from column r (`row == 1 ? w1[0] : w2[0]`, the rows `column > 0` admits) where the record stores
/// `window[row]`, with the window's batch 1 and width 3 as constants. Every other statement is the record's.
/// With the verify entry reading the columns too, nothing reads the window and its concatenation is never
/// launched. Used only after `cbv2TopTwoPacketColumnsQualify` found the twins bitwise equal to the window reads.
public enum CBv2MTPTopTwoPacketColumns {
    public static let enabled = ProcessInfo.processInfo.environment["CBV2_MTP_PACKET_COLUMNS"] != "0"
    nonisolated(unsafe) static var verified = false
}

private func cbv2PacketColumnsSource(_ s: String) -> String {
    let batch = "uint batch = uint(window_shape[0]);"
    let width = "uint width = uint(window_shape[1]);"
    let store = "packet[(row / width) * (width - 1) + (column - 1)] = window[row];"
    for needle in [batch, width, store] {
        precondition(s.components(separatedBy: needle).count == 2, "packet columns: \(needle)")
    }
    return s.replacingOccurrences(of: batch, with: "uint batch = 1;")
        .replacingOccurrences(of: width, with: "uint width = 3;")
        .replacingOccurrences(
            of: store, with: "packet[(row / width) * (width - 1) + (column - 1)] = row == 1 ? w1[0] : w2[0];")
}

private let cbv2TopTwoFinalizePacketColumnsKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen35_mtp_top2_finalize_packet_cols",
    inputNames: ["partial_ids", "partial_values", "w1", "w2"],
    outputNames: ["top_ids", "top_values", "packet"],
    source: cbv2PacketColumnsSource(cbv2TopTwoFinalizePacketSource),
    header: cbv2TopTwoHeader,
    ensureRowContiguous: true
)

private let cbv2TopTwoFinalizePacketColumnsShuffleKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen35_mtp_top2_finalize_packet_sh_cols",
    inputNames: ["partial_ids", "partial_values", "w1", "w2"],
    outputNames: ["top_ids", "top_values", "packet"],
    source: cbv2PacketColumnsSource(cbv2TopTwoFinalizePacketShuffleSource),
    header: cbv2TopTwoShuffleHeader,
    ensureRowContiguous: true
)

/// The draft columns (window columns 1 and 2) the packet finalize reads in place of `window`, while `window` is
/// the published verify window: batch 1, width 3, three [1, 1] int32 columns.
private func cbv2PacketDraftColumns(_ window: MLXArray) -> [MLXArray]? {
    guard CBv2MTPTopTwoPacketColumns.verified, window.shape == [1, 3], window.dtype == .int32,
          let columns = CBv2MTPVerifyWindowColumns.columns(of: window), columns.count == 3,
          columns.allSatisfy({ $0.shape == [1, 1] && $0.dtype == .int32 })
    else { return nil }
    return [columns[1].reshaped(1), columns[2].reshaped(1)]
}

// MARK: - Top-2 on 64-bit keys

/// CBV2_MTP_TOP2_KEYS=0 keeps the record partial and packet finalize. Otherwise the decode round's top-2 (BF16
/// logits, launched row-contiguous, rows x vocab below 2^32) runs as two twin launches on the record's grids
/// (32 threadgroups of 256 threads per row, then one SIMD group per row) that order the elements by one 64-bit
/// key each where the record runs the state `insert`: key = rank << 48 | ~id << 16 | bits, with bits the
/// element's BF16 pattern, id its vocabulary index and rank 0 for a NaN, else the pattern with the sign bit set
/// (positive) or every bit inverted (negative), a value the comparator finds equal to zero (`v == 0.0f`: -0, +0)
/// taking +0's pattern first. The record's `better` orders by NaN last, then value descending (`>`, `<` on the
/// floats), then id ascending; the rank is 0 for every NaN and above 0 otherwise, rises with the value and is
/// equal exactly for values the comparator finds equal, and ~id falls with the id. So for distinct ids a key is
/// larger exactly when its element is better, and the row's two largest keys are its record top two. 0 is below
/// every element's key (ids below 2^32 - 1), so a thread or threadgroup without elements holds 0 and no sentinel.
/// Each thread walks the record's elements (step 8192), loading four before it keys them, and keeps its two
/// largest keys (min / max); a SIMD group takes the largest over its lanes and then the largest over the other
/// lanes' largest and the winner's second (two 32-bit `simd_max` per 64-bit maximum); the threadgroup merges its
/// eight SIMD groups' pairs the same way after one barrier and stores its pair; the finalize merges a row's 32
/// pairs the same way, decodes id = ~(key >> 16) and value = float(bits as bfloat16_t), the record's own
/// conversion of that element, and stores through the record finalize's text.
/// Used only after `cbv2TopTwoKeysQualify` found ids, values (bits) and packet bitwise equal to the record.
public enum CBv2MTPTopTwoKeys {
    public static let enabled = ProcessInfo.processInfo.environment["CBV2_MTP_TOP2_KEYS"] != "0"
    nonisolated(unsafe) static var verified = false
}

public enum CBv2MTPTopTwoColumnsEntry {
    public static let enabled = ProcessInfo.processInfo.environment["CBV2_MTP_TOP2_COLUMNS_ENTRY"] != "0"
    nonisolated(unsafe) static var verified = false
}

private func cbv2PacketColumnsEntrySource(_ s: String) -> String? {
    let read = "row == 1 ? w1[0] : w2[0]"
    guard s.components(separatedBy: read).count == 2 else { return nil }
    return "        const int cbe_w1 = w1[0];\n        const int cbe_w2 = w2[0];\n"
        + s.replacingOccurrences(of: read, with: "row == 1 ? cbe_w1 : cbe_w2")
}

private let cbv2TopTwoKeysHeader = cbv2TopTwoHeader + """

    inline ulong darkbloom_qwen35_mtp_top2_key(bfloat16_t x, uint id) {
        const float value = float(x);
        const uint bits = uint(as_type<ushort>(x));
        const uint ordered = value == 0.0f ? 0u : bits;
        const uint rank = (ordered & 0x8000u) != 0u ? (~ordered & 0xffffu) : (ordered | 0x8000u);
        return (ulong(isnan(value) ? 0u : rank) << 48) | (ulong(~id) << 16) | ulong(bits);
    }

    inline void darkbloom_qwen35_mtp_top2_key_push(thread ulong &first, thread ulong &second, ulong key) {
        const ulong lower = min(first, key);
        first = max(first, key);
        second = max(second, lower);
    }

    inline ulong darkbloom_qwen35_mtp_top2_key_max(ulong key) {
        const uint high = simd_max(uint(key >> 32));
        const uint low = simd_max(uint(key >> 32) == high ? uint(key) : 0u);
        return (ulong(high) << 32) | ulong(low);
    }

    inline darkbloom_qwen35_mtp_top2_state darkbloom_qwen35_mtp_top2_key_state(ulong first, ulong second) {
        darkbloom_qwen35_mtp_top2_state state;
        state.first_value = float(as_type<bfloat16_t>(ushort(first)));
        state.second_value = float(as_type<bfloat16_t>(ushort(second)));
        state.first_id = ~uint(first >> 16);
        state.second_id = ~uint(second >> 16);
        state.count = 2;
        return state;
    }
"""

private let cbv2TopTwoKeysPartialSource = """
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint row = group_index / 32;
        uint group = group_index % 32;
        uint vocab = uint(logits_shape[2]);
        const auto t2_row = logits + row * vocab;
        ulong first = 0;
        ulong second = 0;
        uint index = group * 256 + lane;
        for (; index < vocab && vocab - index > 3 * (32 * 256); index += 4 * (32 * 256)) {
            const bfloat16_t x0 = t2_row[index];
            const bfloat16_t x1 = t2_row[index + 32 * 256];
            const bfloat16_t x2 = t2_row[index + 2 * (32 * 256)];
            const bfloat16_t x3 = t2_row[index + 3 * (32 * 256)];
            darkbloom_qwen35_mtp_top2_key_push(first, second, darkbloom_qwen35_mtp_top2_key(x0, index));
            darkbloom_qwen35_mtp_top2_key_push(
                first, second, darkbloom_qwen35_mtp_top2_key(x1, index + 32 * 256));
            darkbloom_qwen35_mtp_top2_key_push(
                first, second, darkbloom_qwen35_mtp_top2_key(x2, index + 2 * (32 * 256)));
            darkbloom_qwen35_mtp_top2_key_push(
                first, second, darkbloom_qwen35_mtp_top2_key(x3, index + 3 * (32 * 256)));
        }
        for (; index < vocab; index += 32 * 256) {
            darkbloom_qwen35_mtp_top2_key_push(
                first, second, darkbloom_qwen35_mtp_top2_key(t2_row[index], index));
        }

        const uint simd_lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;
        const ulong simd_first = darkbloom_qwen35_mtp_top2_key_max(first);
        const ulong simd_second = darkbloom_qwen35_mtp_top2_key_max(first == simd_first ? second : first);

        threadgroup ulong scratch[16];
        if (simd_lane == 0) {
            scratch[2 * simd_group] = simd_first;
            scratch[2 * simd_group + 1] = simd_second;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            const ulong key = simd_lane < 16 ? scratch[simd_lane] : 0ul;
            const ulong group_first = darkbloom_qwen35_mtp_top2_key_max(key);
            const ulong group_second = darkbloom_qwen35_mtp_top2_key_max(key == group_first ? 0ul : key);
            if (simd_lane == 0) {
                partial_keys[(row * 32 + group) * 2] = group_first;
                partial_keys[(row * 32 + group) * 2 + 1] = group_second;
            }
        }
    """

private let cbv2TopTwoKeysPartialKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen35_mtp_top2_partial_keys",
    inputNames: ["logits"],
    outputNames: ["partial_keys"],
    source: cbv2TopTwoKeysPartialSource,
    header: cbv2TopTwoKeysHeader,
    ensureRowContiguous: true)

/// The shuffle-form packet finalize with its loads and shuffle merges swapped for the key merge of the row's 32
/// threadgroup pairs (one counted swap; nil if the text moved). The stores are the record's text.
private func cbv2TopTwoKeysFinalizeSource(_ s: String) -> String? {
    let merge = """
            uint base = (row * 32 + lane) * 2;
            darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();
            darkbloom_qwen35_mtp_top2_insert(
                local, partial_values[base], uint(partial_ids[base]));
            darkbloom_qwen35_mtp_top2_insert(
                local, partial_values[base + 1], uint(partial_ids[base + 1]));

            for (ushort delta = 16; delta > 0; delta >>= 1) {
                darkbloom_qwen35_mtp_top2_state other =
                    darkbloom_qwen35_mtp_top2_shuffle_down(local, delta);
                if (lane < delta) {
                    darkbloom_qwen35_mtp_top2_insert(
                        local, other.first_value, other.first_id);
                    darkbloom_qwen35_mtp_top2_insert(
                        local, other.second_value, other.second_id);
                }
            }

        """
    guard s.components(separatedBy: merge).count == 2 else { return nil }
    return s.replacingOccurrences(of: merge, with: """
            uint base = (row * 32 + lane) * 2;
            const ulong first = partial_keys[base];
            const ulong second = partial_keys[base + 1];
            const ulong row_first = darkbloom_qwen35_mtp_top2_key_max(first);
            const ulong row_second = darkbloom_qwen35_mtp_top2_key_max(first == row_first ? second : first);
            darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_key_state(row_first, row_second);

        """)
}

/// The key finalize reading the window, and its column form (`cbv2PacketColumnsSource`); nil if the text moved.
private let cbv2TopTwoKeysFinalizeKernels: (window: MLXFast.MLXFastKernel, columns: MLXFast.MLXFastKernel)? =
    cbv2TopTwoKeysFinalizeSource(cbv2TopTwoFinalizePacketShuffleSource).map {
        (MLXFast.metalKernel(
            name: "darkbloom_qwen35_mtp_top2_finalize_packet_keys",
            inputNames: ["partial_keys", "window"],
            outputNames: ["top_ids", "top_values", "packet"],
            source: $0, header: cbv2TopTwoKeysHeader, ensureRowContiguous: true),
         MLXFast.metalKernel(
            name: "darkbloom_qwen35_mtp_top2_finalize_packet_keys_cols",
            inputNames: ["partial_keys", "w1", "w2"],
            outputNames: ["top_ids", "top_values", "packet"],
            source: cbv2PacketColumnsSource($0), header: cbv2TopTwoKeysHeader, ensureRowContiguous: true))
    }

private let cbv2TopTwoKeysColumnsEntryKernel: MLXFast.MLXFastKernel? =
    cbv2TopTwoKeysFinalizeSource(cbv2TopTwoFinalizePacketShuffleSource)
        .flatMap { cbv2PacketColumnsEntrySource(cbv2PacketColumnsSource($0)) }.map {
            MLXFast.metalKernel(
                name: "darkbloom_qwen35_mtp_top2_finalize_packet_keys_cols_e",
                inputNames: ["partial_keys", "w1", "w2"],
                outputNames: ["top_ids", "top_values", "packet"],
                source: $0, header: cbv2TopTwoKeysHeader, ensureRowContiguous: true)
        }

/// The packet finalize of either form on the partials, reading the draft ids from `draftColumns` ([1] int32
/// each, window columns 1 and 2) or from `window`.
private func cbv2TopTwoPacketFinalize(
    _ partials: [MLXArray], window: MLXArray?, draftColumns: [MLXArray]?, rows: Int, drafts: Int, shuffled: Bool
) -> [MLXArray] {
    let inputs: [MLXArray]
    let kernel: MLXFast.MLXFastKernel
    if let draftColumns {
        inputs = [partials[0], partials[1], draftColumns[0], draftColumns[1]]
        kernel = shuffled ? cbv2TopTwoFinalizePacketColumnsShuffleKernel : cbv2TopTwoFinalizePacketColumnsKernel
    } else {
        inputs = [partials[0], partials[1], window!]
        kernel = shuffled ? cbv2TopTwoFinalizePacketShuffleKernel : cbv2TopTwoFinalizePacketKernel
    }
    return kernel(
        inputs,
        grid: (rows * 32, 1, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [[rows, 2], [rows, 2], [drafts + rows * 3]],
        outputDTypes: [.int32, .float32, .int32]
    )
}

/// `cbv2TopTwoRows` plus the MTP acceptance packet from the same finalize
/// launch, for logits `[1, B * (1 + k), vocab]` whose rows are the positions
/// of the int32 verify window `[B, 1 + k]`. `ids` and `values` are those of
/// `cbv2TopTwoRows` (same partial kernel and merge code). nil when the
/// shapes do not qualify or `CBV2_MTP_TOP2_PACKET=0`.
package func cbv2TopTwoRowsPacket(
    _ logits: MLXArray, window: MLXArray, shuffle: Bool? = nil
) -> (ids: MLXArray, values: MLXArray, packet: MLXArray)? {
    cbv2TopTwoRowsPacket(logits, window: window, draftColumns: cbv2PacketDraftColumns(window), shuffle: shuffle)
}

/// `cbv2TopTwoRowsPacket` with the packet's draft ids read from `draftColumns` (window columns 1 and 2, [1]
/// int32 each; batch 1, width 3) when given, else from `window`.
private func cbv2TopTwoRowsPacket(
    _ logits: MLXArray, window: MLXArray, draftColumns: [MLXArray]?, shuffle: Bool?, offset32: Bool? = nil,
    keys: Bool? = nil, columnsEntry: Bool? = nil
) -> (ids: MLXArray, values: MLXArray, packet: MLXArray)? {
    guard CBv2MTPTopTwoPacket.enabled,
        logits.ndim == 3, logits.dim(0) == 1, logits.dim(2) >= 2,
        window.ndim == 2, window.dtype == .int32, window.dim(1) >= 2,
        window.dim(0) * window.dim(1) == logits.dim(1)
    else { return nil }
    let rows = logits.dim(1)
    let drafts = window.dim(0) * (window.dim(1) - 1)
    if keys ?? CBv2MTPTopTwoKeys.verified, logits.dtype == .bfloat16, rows * logits.dim(2) <= Int(UInt32.max),
        let finalize = cbv2TopTwoKeysFinalizeKernels
    {
        let partial = cbv2TopTwoKeysPartialKernel(
            [logits],
            grid: (rows * 32 * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[rows, 32, 2]],
            outputDTypes: [.uint64])[0]
        let columns = window.shape == [1, 3] ? draftColumns : nil
        let entry = (columnsEntry ?? CBv2MTPTopTwoColumnsEntry.verified) ? cbv2TopTwoKeysColumnsEntryKernel : nil
        let outputs = (columns == nil ? finalize.window : entry ?? finalize.columns)(
            columns.map { [partial, $0[0], $0[1]] } ?? [partial, window],
            grid: (rows * 32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[rows, 2], [rows, 2], [drafts + rows * 3]],
            outputDTypes: [.int32, .float32, .int32])
        return (outputs[0], outputs[1], outputs[2])
    }
    let shuffled = shuffle ?? CBv2MTPTopTwoShuffle.verified
    let shufflePartial = (offset32 ?? CBv2MTPTopTwoOffset32.verified) && rows * logits.dim(2) <= Int(UInt32.max)
        ? cbv2TopTwoPartialShuffleOffset32Kernel ?? cbv2TopTwoPartialShuffleKernel : cbv2TopTwoPartialShuffleKernel
    let partials = (shuffled ? shufflePartial : cbv2TopTwoPartialKernel)(
        [logits],
        grid: (rows * 32 * 256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[rows, 32, 2], [rows, 32, 2]],
        outputDTypes: [.int32, .float32]
    )
    let outputs = cbv2TopTwoPacketFinalize(
        partials, window: window, draftColumns: window.shape == [1, 3] ? draftColumns : nil,
        rows: rows, drafts: drafts, shuffled: shuffled)
    return (outputs[0], outputs[1], outputs[2])
}

/// Load time, after `cbv2TopTwoShuffleQualify`: the column-reading finalize of each form that can run (tree;
/// shuffle when it qualified) against the same form reading the concatenated window, bit for bit in the ids, the
/// values (as bits) and the packet. Seeded logits at width 3 over the full vocabulary and a short one: Gaussian,
/// exact ties, NaNs, infinities, signed zeros and constant rows; draft columns of seeded ids, the int32 extremes
/// and negative values. Engages only if every case matches (`CBv2MTPTopTwoPacketColumns`).
package func cbv2TopTwoPacketColumnsQualify(logitsDType: DType, vocabulary: Int) {
    CBv2MTPTopTwoPacketColumns.verified = false
    func report(_ message: String) {
        FileHandle.standardError.write(Data("cbv2 mtp top2 packet columns: \(message)\n".utf8))
    }
    guard CBv2MTPTopTwoPacket.enabled else { return report("off (packet finalize off)") }
    guard CBv2MTPTopTwoPacketColumns.enabled else { return report("off (setting)") }
    let forms = CBv2MTPTopTwoShuffle.verified ? [false, true] : [false]
    var cases = 0
    do {
        let matches = try withError { () -> Bool in
            var ok = MLXArray(true)
            for vocab in [vocabulary, 5003] {
                for kind in 0 ..< 6 {
                    let key = MLXRandom.key(UInt64(0x7c00_0000 + vocab * 8 + kind))
                    var logits = MLXRandom.normal([1, 3, vocab], key: key) * 3
                    switch kind {
                    case 1: logits = round(logits)
                    case 2: logits = which(abs(logits) .> MLXArray(Float(5.5)), MLXArray(Float.nan), logits)
                    case 3:
                        logits = which(logits .> MLXArray(Float(6)), MLXArray(Float.infinity),
                            which(logits .< MLXArray(Float(-6)), MLXArray(-Float.infinity), logits))
                    case 4: logits = which(abs(logits) .< MLXArray(Float(1)), MLXArray(Float(-0.0)), round(logits))
                    case 5: logits = MLXArray.zeros([1, 3, vocab]) + MLXArray(Float(0.5))
                    default: break
                    }
                    let typed = logits.asType(logitsDType)
                    let ids: [[Int32]] = [
                        MLXRandom.randInt(0 ..< Int32(vocab), [3], key: MLXRandom.key(UInt64(0x7c01_0000 + cases)))
                            .asType(.int32).asArray(Int32.self),
                        [Int32(kind), Int32.max, Int32.min], [-1, 0, Int32(vocab - 1)],
                    ]
                    for column in ids {
                        let parts = column.map { MLXArray([$0]).reshaped(1, 1) }
                        let window = concatenated(parts, axis: 1)
                        for shuffled in forms {
                            guard let a = cbv2TopTwoRowsPacket(
                                    typed, window: window, draftColumns: nil, shuffle: shuffled),
                                  let b = cbv2TopTwoRowsPacket(
                                    typed, window: window,
                                    draftColumns: [parts[1].reshaped(1), parts[2].reshaped(1)], shuffle: shuffled)
                            else { return false }
                            ok = ok .&& (a.ids .== b.ids).all()
                                .&& (a.values.view(dtype: .uint32) .== b.values.view(dtype: .uint32)).all()
                                .&& (a.packet .== b.packet).all()
                            cases += 1
                        }
                    }
                }
                if !ok.item(Bool.self) { return false }
            }
            return true
        }
        CBv2MTPTopTwoPacketColumns.verified = matches
    } catch {
        return report("off (\(error))")
    }
    report(CBv2MTPTopTwoPacketColumns.verified
        ? "on (\(cases) cases bitwise equal to the window reads, forms \(forms.count))"
        : "off (differs from the window reads)")
}

/// Load time: the shuffle-form partial + packet finalize against the tree
/// launches, bit for bit in the ids, the values (as bits) and the packet, on
/// seeded logits: full and short vocabularies, R = 1 ... 4, exact ties (rounded
/// values), NaNs, infinities, signed zeros and constant rows. Engages only if
/// every case matches (`CBv2MTPTopTwoShuffle`).
package func cbv2TopTwoShuffleQualify(logitsDType: DType, vocabulary: Int) {
    CBv2MTPTopTwoShuffle.verified = false
    func report(_ message: String) {
        FileHandle.standardError.write(Data("cbv2 mtp top2 shuffle: \(message)\n".utf8))
    }
    guard CBv2MTPTopTwoPacket.enabled else { return report("off (packet finalize off)") }
    guard CBv2MTPTopTwoShuffle.enabled else { return report("off (setting)") }
    var cases = 0
    do {
        let matches = try withError { () -> Bool in
            for vocab in [vocabulary, 64, 5003] {
                for rows in 1 ... 4 {
                    for kind in 0 ..< 6 {
                        let key = MLXRandom.key(UInt64(0x7a90_0000 + vocab * 64 + rows * 8 + kind))
                        var logits = MLXRandom.normal([1, rows, vocab], key: key) * 3
                        switch kind {
                        case 1: logits = round(logits)  // many exact ties
                        case 2: logits = which(abs(logits) .> MLXArray(Float(5.5)), MLXArray(Float.nan), logits)
                        case 3:
                            logits = which(logits .> MLXArray(Float(6)), MLXArray(Float.infinity),
                                which(logits .< MLXArray(Float(-6)), MLXArray(-Float.infinity), logits))
                        case 4: logits = which(abs(logits) .< MLXArray(Float(1)), MLXArray(Float(-0.0)), round(logits))
                        case 5: logits = MLXArray.zeros([1, rows, vocab]) + MLXArray(Float(0.5))
                        default: break
                        }
                        let typed = logits.asType(logitsDType)
                        let window = MLXRandom.randInt(
                            0 ..< Int32(vocab), [1, rows], key: MLXRandom.key(UInt64(0x7a91_0000 + cases)))
                            .asType(.int32)
                        let width = rows >= 2 ? rows : 2
                        let w = rows >= 2 ? window : concatenated([window, window], axis: 1)
                        let l = rows >= 2 ? typed : concatenated([typed, typed], axis: 1)
                        guard w.dim(1) == width,
                              let a = cbv2TopTwoRowsPacket(l, window: w, shuffle: false),
                              let b = cbv2TopTwoRowsPacket(l, window: w, shuffle: true)
                        else { return false }
                        let same = (a.ids .== b.ids).all()
                            .&& (a.values.view(dtype: .uint32) .== b.values.view(dtype: .uint32)).all()
                            .&& (a.packet .== b.packet).all()
                        if !same.item(Bool.self) { return false }
                        cases += 1
                    }
                }
            }
            return true
        }
        CBv2MTPTopTwoShuffle.verified = matches
    } catch {
        return report("off (\(error))")
    }
    report(CBv2MTPTopTwoShuffle.verified
        ? "on (\(cases) cases bitwise equal to the tree reductions)"
        : "off (differs from the tree reductions after \(cases) cases)")
}

/// Load time, after `cbv2TopTwoShuffleQualify`: the shuffle partial with 32-bit element offsets
/// (`CBv2MTPTopTwoOffset32`) and the packet finalize against the shuffle partial and the same finalize, bit for bit
/// in the ids, the values (as bits) and the packet, on seeded logits at R = 1 ... 4 over the full vocabulary, 64,
/// 5003, 8193 and the full vocabulary plus 8195 (walks of 0 to 17 elements per thread): Gaussian, exact ties,
/// NaNs, infinities, signed zeros and constant rows; and on two strided views (a column window of wider rows, a
/// transposed layout with element stride 3), which the twin reads through MLX's row-contiguous copy. Engages only
/// if every case matches.
package func cbv2TopTwoOffset32Qualify(logitsDType: DType, vocabulary: Int) {
    CBv2MTPTopTwoOffset32.verified = false
    func report(_ message: String) {
        FileHandle.standardError.write(Data("cbv2 mtp top2 offset32: \(message)\n".utf8))
    }
    guard CBv2MTPTopTwoPacket.enabled else { return report("off (packet finalize off)") }
    guard CBv2MTPTopTwoShuffle.verified else { return report("off (shuffle partial off)") }
    guard CBv2MTPTopTwoOffset32.enabled else { return report("off (setting)") }
    guard cbv2TopTwoPartialShuffleOffset32Kernel != nil else { return report("off (partial text moved)") }
    var cases = 0
    do {
        let matches = try withError { () -> Bool in
            func same(_ logits: MLXArray, _ window: MLXArray) -> MLXArray? {
                guard let a = cbv2TopTwoRowsPacket(
                        logits, window: window, draftColumns: nil, shuffle: true, offset32: false),
                      let b = cbv2TopTwoRowsPacket(
                        logits, window: window, draftColumns: nil, shuffle: true, offset32: true)
                else { return nil }
                cases += 1
                return (a.ids .== b.ids).all()
                    .&& (a.values.view(dtype: .uint32) .== b.values.view(dtype: .uint32)).all()
                    .&& (a.packet .== b.packet).all()
            }
            for vocab in [vocabulary, 64, 5003, 8193, vocabulary + 8195] {
                var ok = MLXArray(true)
                for rows in 1 ... 4 {
                    for kind in 0 ..< 6 {
                        let key = MLXRandom.key(UInt64(0x7a93_0000 + vocab * 64 + rows * 8 + kind))
                        var logits = MLXRandom.normal([1, rows, vocab], key: key) * 3
                        switch kind {
                        case 1: logits = round(logits)
                        case 2: logits = which(abs(logits) .> MLXArray(Float(5.5)), MLXArray(Float.nan), logits)
                        case 3:
                            logits = which(logits .> MLXArray(Float(6)), MLXArray(Float.infinity),
                                which(logits .< MLXArray(Float(-6)), MLXArray(-Float.infinity), logits))
                        case 4: logits = which(abs(logits) .< MLXArray(Float(1)), MLXArray(Float(-0.0)), round(logits))
                        case 5: logits = MLXArray.zeros([1, rows, vocab]) + MLXArray(Float(0.5))
                        default: break
                        }
                        let typed = logits.asType(logitsDType)
                        let window = MLXRandom.randInt(
                            0 ..< Int32(vocab), [1, rows], key: MLXRandom.key(UInt64(0x7a94_0000 + cases)))
                            .asType(.int32)
                        let w = rows >= 2 ? window : concatenated([window, window], axis: 1)
                        let l = rows >= 2 ? typed : concatenated([typed, typed], axis: 1)
                        guard let same = same(l, w) else { return false }
                        ok = ok .&& same
                    }
                }
                if !ok.item(Bool.self) { return false }
            }
            let window = MLXRandom.randInt(0 ..< Int32(5003), [1, 3], key: MLXRandom.key(0x7a95_0000))
                .asType(.int32)
            let wide = (MLXRandom.normal([1, 3, vocabulary + 64], key: MLXRandom.key(0x7a95_0001)) * 3)
                .asType(logitsDType)
            let tall = (MLXRandom.normal([1, 5003, 3], key: MLXRandom.key(0x7a95_0002)) * 3).asType(logitsDType)
            guard let a = same(wide[0..., 0..., 32 ..< (32 + vocabulary)], window),
                  let b = same(tall.transposed(0, 2, 1), window) else { return false }
            return (a .&& b).item(Bool.self)
        }
        CBv2MTPTopTwoOffset32.verified = matches
    } catch {
        return report("off (\(error))")
    }
    report(CBv2MTPTopTwoOffset32.verified
        ? "on (\(cases) cases bitwise equal to the shuffle partial)"
        : "off (differs from the shuffle partial after \(cases) cases)")
}

/// Load time, after `cbv2TopTwoPacketColumnsQualify`: the key launches (`CBv2MTPTopTwoKeys`) against the record
/// launches the round takes without them (partial and packet finalize of the qualified form), bit for bit in the
/// ids, the values (as bits) and the packet, reading the window and (at width 3) the draft columns. Seeded BF16
/// logits at R = 1 ... 4 over the full vocabulary, 2, 3, 64, 5003, 8193 and the full vocabulary plus 8195:
/// Gaussian, exact ties, NaNs, infinities, signed zeros, constant rows, random bit patterns (every NaN payload of
/// both signs, infinities, zeros, subnormals), a pool of special patterns (heavy ties across zeros, subnormals,
/// NaNs and infinities), rows of NaNs only, zeros and subnormals only, -inf rows with a few finite values, and
/// rows whose largest value sits at the first and last index; then a row-contiguous view at an element offset of
/// 5, a column window of wider rows and a transposed layout (the last two through MLX's row-contiguous copy).
/// Engages only if every case matches.
package func cbv2TopTwoKeysQualify(logitsDType: DType, vocabulary: Int) {
    CBv2MTPTopTwoKeys.verified = false
    func report(_ message: String) {
        FileHandle.standardError.write(Data("cbv2 mtp top2 keys: \(message)\n".utf8))
    }
    guard CBv2MTPTopTwoPacket.enabled else { return report("off (packet finalize off)") }
    guard CBv2MTPTopTwoKeys.enabled else { return report("off (setting)") }
    guard logitsDType == .bfloat16 else { return report("off (logits \(logitsDType))") }
    guard cbv2TopTwoKeysFinalizeKernels != nil else { return report("off (finalize text moved)") }
    var cases = 0
    do {
        let matches = try withError { () -> Bool in
            func same(_ logits: MLXArray, _ window: MLXArray, columns: Bool) -> MLXArray? {
                let draftColumns = columns ? [window[0..., 1].reshaped(1), window[0..., 2].reshaped(1)] : nil
                guard let a = cbv2TopTwoRowsPacket(
                        logits, window: window, draftColumns: draftColumns, shuffle: nil, keys: false),
                      let b = cbv2TopTwoRowsPacket(
                        logits, window: window, draftColumns: draftColumns, shuffle: nil, keys: true)
                else { return nil }
                cases += 1
                return (a.ids .== b.ids).all()
                    .&& (a.values.view(dtype: .uint32) .== b.values.view(dtype: .uint32)).all()
                    .&& (a.packet .== b.packet).all()
            }
            func bits(_ range: Range<Int32>, _ shape: [Int], _ seed: UInt64) -> MLXArray {
                MLXRandom.randInt(range, shape, key: MLXRandom.key(seed)).asType(.uint16)
            }
            func patterns(_ pool: [UInt16], _ shape: [Int], _ seed: UInt64) -> MLXArray {
                let pick = MLXRandom.randInt(0 ..< Int32(pool.count), shape, key: MLXRandom.key(seed))
                return take(MLXArray(pool), pick).view(dtype: .bfloat16)
            }
            for vocab in [vocabulary, 2, 3, 64, 5003, 8193, vocabulary + 8195] {
                var ok = MLXArray(true)
                for rows in 1 ... 4 {
                    for kind in 0 ..< 12 {
                        let seed = UInt64(0x7a96_0000 + vocab * 64 + rows * 16 + kind)
                        let shape = [1, rows, vocab]
                        var logits = MLXRandom.normal(shape, key: MLXRandom.key(seed)) * 3
                        switch kind {
                        case 1: logits = round(logits)
                        case 2: logits = which(abs(logits) .> MLXArray(Float(5.5)), MLXArray(Float.nan), logits)
                        case 3:
                            logits = which(logits .> MLXArray(Float(6)), MLXArray(Float.infinity),
                                which(logits .< MLXArray(Float(-6)), MLXArray(-Float.infinity), logits))
                        case 4: logits = which(abs(logits) .< MLXArray(Float(1)), MLXArray(Float(-0.0)), round(logits))
                        case 5: logits = MLXArray.zeros(shape) + MLXArray(Float(0.5))
                        case 10:
                            let draw = MLXRandom.uniform(Float(0) ..< Float(1), shape, key: MLXRandom.key(seed + 1))
                            logits = which(draw .< MLXArray(Float(0.002)), logits, MLXArray(-Float.infinity))
                        case 11:
                            let edge = MLXArray.zeros([1, rows, 1]) + MLXArray(Float(9))
                            logits = vocab == 2 ? concatenated([edge, edge], axis: 2)
                                : concatenated([edge, logits[0..., 0..., 1 ..< (vocab - 1)], edge], axis: 2)
                        default: break
                        }
                        var typed = logits.asType(.bfloat16)
                        switch kind {
                        case 6: typed = bits(0 ..< 65536, shape, seed + 2).view(dtype: .bfloat16)
                        case 7:
                            typed = patterns([0x0000, 0x8000, 0x0001, 0x8001, 0x7fc0, 0xffc0, 0x7f81, 0xffff,
                                0x7f80, 0xff80, 0x3f80, 0xbf80], shape, seed + 3)
                        case 8:
                            typed = which(bits(0 ..< 2, shape, seed + 4) .== MLXArray(UInt16(0)),
                                bits(0x7f81 ..< 0x8000, shape, seed + 5), bits(0xff81 ..< 0x10000, shape, seed + 6))
                                .view(dtype: .bfloat16)
                        case 9:
                            typed = patterns([0x0000, 0x8000, 0x0001, 0x8001, 0x0002, 0x8002, 0x0040, 0x8040,
                                0x007f, 0x807f, 0x0080, 0x8080], shape, seed + 7)
                        default: break
                        }
                        let window = MLXRandom.randInt(
                            0 ..< Int32(vocab), [1, rows], key: MLXRandom.key(UInt64(0x7a97_0000 + cases)))
                            .asType(.int32)
                        let w = rows >= 2 ? window : concatenated([window, window], axis: 1)
                        let l = rows >= 2 ? typed : concatenated([typed, typed], axis: 1)
                        guard let s = same(l, w, columns: false) else { return false }
                        ok = ok .&& s
                        if rows == 3 {
                            guard let c = same(l, w, columns: true) else { return false }
                            ok = ok .&& c
                        }
                    }
                }
                if !ok.item(Bool.self) { return false }
            }
            let window = MLXRandom.randInt(0 ..< Int32(5003), [1, 3], key: MLXRandom.key(0x7a98_0000))
                .asType(.int32)
            let flat = (MLXRandom.normal([3 * vocabulary + 5], key: MLXRandom.key(0x7a98_0001)) * 3)
                .asType(.bfloat16)
            let wide = (MLXRandom.normal([1, 3, vocabulary + 64], key: MLXRandom.key(0x7a98_0002)) * 3)
                .asType(.bfloat16)
            let tall = (MLXRandom.normal([1, 5003, 3], key: MLXRandom.key(0x7a98_0003)) * 3).asType(.bfloat16)
            var ok = MLXArray(true)
            for logits in [flat[5...].reshaped(1, 3, vocabulary), wide[0..., 0..., 32 ..< (32 + vocabulary)],
                           tall.transposed(0, 2, 1)] {
                for columns in [false, true] {
                    guard let s = same(logits, window, columns: columns) else { return false }
                    ok = ok .&& s
                }
            }
            return ok.item(Bool.self)
        }
        CBv2MTPTopTwoKeys.verified = matches
    } catch {
        return report("off (\(error))")
    }
    report(CBv2MTPTopTwoKeys.verified
        ? "on (\(cases) cases bitwise equal to the record partial and finalize)"
        : "off (differs from the record partial and finalize after \(cases) cases)")
}

package func cbv2TopTwoColumnsEntryQualify(logitsDType: DType, vocabulary: Int) {
    CBv2MTPTopTwoColumnsEntry.verified = false
    func report(_ message: String) {
        FileHandle.standardError.write(Data("cbv2 mtp top2 columns entry: \(message)\n".utf8))
    }
    guard CBv2MTPTopTwoColumnsEntry.enabled else { return report("off (setting)") }
    guard CBv2MTPTopTwoKeys.verified, logitsDType == .bfloat16, cbv2TopTwoKeysColumnsEntryKernel != nil else {
        return report("off (key finalize off or its text moved)")
    }
    var cases = 0
    do {
        let matches = try withError { () -> Bool in
            var ok = MLXArray(true)
            let extremes: [(Int32, Int32)] = [(Int32.max, Int32.min), (0, -1), (-7, 123_456), (Int32.min, Int32.max)]
            for vocab in [vocabulary, 5003] {
                for kind in 0 ..< 8 {
                    let seed = UInt64(0x7a99_0000 + vocab * 16 + kind)
                    var logits = MLXRandom.normal([1, 3, vocab], key: MLXRandom.key(seed)) * 3
                    switch kind {
                    case 1: logits = round(logits)
                    case 2: logits = which(abs(logits) .> MLXArray(Float(5.5)), MLXArray(Float.nan), logits)
                    case 3:
                        logits = which(logits .> MLXArray(Float(6)), MLXArray(Float.infinity),
                            which(logits .< MLXArray(Float(-6)), MLXArray(-Float.infinity), logits))
                    case 4: logits = which(abs(logits) .< MLXArray(Float(1)), MLXArray(Float(-0.0)), round(logits))
                    case 5: logits = MLXArray.zeros([1, 3, vocab]) + MLXArray(Float(0.5))
                    default: break
                    }
                    var typed = logits.asType(.bfloat16)
                    if kind == 6 {
                        typed = MLXRandom.randInt(0 ..< 65536, [1, 3, vocab], key: MLXRandom.key(seed + 1))
                            .asType(.uint16).view(dtype: .bfloat16)
                    }
                    var ids = MLXRandom.randInt(0 ..< Int32(vocab), [1, 3], key: MLXRandom.key(seed + 2))
                        .asType(.int32)
                    if kind == 7 {
                        let e = extremes[vocab % extremes.count]
                        ids = MLXArray([Int32(1), e.0, e.1]).reshaped(1, 3)
                    }
                    let columns = [ids[0..., 1].reshaped(1), ids[0..., 2].reshaped(1)]
                    guard let a = cbv2TopTwoRowsPacket(
                            typed, window: ids, draftColumns: columns, shuffle: nil, keys: true, columnsEntry: false),
                          let b = cbv2TopTwoRowsPacket(
                            typed, window: ids, draftColumns: columns, shuffle: nil, keys: true, columnsEntry: true)
                    else { return false }
                    ok = ok .&& (a.ids .== b.ids).all()
                        .&& (a.values.view(dtype: .uint32) .== b.values.view(dtype: .uint32)).all()
                        .&& (a.packet .== b.packet).all()
                    cases += 1
                }
            }
            return ok.item(Bool.self)
        }
        CBv2MTPTopTwoColumnsEntry.verified = matches && cases > 0
    } catch {
        return report("off (\(error))")
    }
    report(CBv2MTPTopTwoColumnsEntry.verified
        ? "on (\(cases) cases bitwise equal to the column form: ids, values and packet)"
        : "off (differs from the column form)")
}

/// Load-time JIT of the packet finalize (its pipeline depends only on the
/// int32/float32/int32 input types, not on the window shape).
package func cbv2TopTwoRowsPacketWarm(logitsDType: DType) {
    let logits = MLXArray.zeros([1, 3, 64], dtype: logitsDType)
    let window = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
    guard let warmed = cbv2TopTwoRowsPacket(logits, window: window) else { return }
    eval(warmed.ids, warmed.values, warmed.packet)
}
