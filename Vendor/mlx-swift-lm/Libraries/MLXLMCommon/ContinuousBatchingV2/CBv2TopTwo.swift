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
    keys: Bool? = nil
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
        // A geometry the load-time tune adopted replaces the record's stage one (its finalize merges the adopted
        // pair count). The unread `dep` input binds the verify window, which the round publishes before the launch.
        if let tuned = NemotronTopTwoTune.choice,
           let out = NemotronTopTwoTune.launch(
            tuned, logits, window: window, draftColumns: draftColumns, dep: window)
        {
            return out
        }
        let partial = cbv2TopTwoKeysPartialKernel(
            [logits],
            grid: (rows * 32 * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[rows, 32, 2]],
            outputDTypes: [.uint64])[0]
        let columns = window.shape == [1, 3] ? draftColumns : nil
        let outputs = (columns == nil ? finalize.window : finalize.columns)(
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

// MARK: - Load-time geometry tune of the top-2's stage one (NEMOTRON_TOP2_TUNE)
//
// The decode round's top-2 runs `cbv2TopTwoKeysPartialSource` once the keys form qualified: 32 threadgroups of 256
// threads per row, each thread walking every 8192nd element of its row, a SIMD-group maximum over the lanes' pairs,
// one barrier, then the threadgroup's eight SIMD-group pairs merged by its first group. That geometry came with the
// tree partial and describes the M4-class host it was written on; the ranked host is an M5 with far more GPU cores.
// `NemotronTopTwoTune` rebuilds the same body under other launch geometries. A variant moves only
// - groups: threadgroups per row (the divisor of the row in `group_index`, and the 32 of the walk's stride);
// - threads: threads per threadgroup (the lane bound, the `scratch` size, the SIMD-group count);
// - unroll: loads per step of the main walk (1: the record's single-load loop only).
// None of them touches the key, the ids, the per-thread walk itself (a different stride visits a disjoint set of
// threads but the same elements of the row), the per-thread pair, the merge tree's comparisons or the stores. A
// maximum over keys is a maximum over a set and keys are distinct within a row, so which thread or threadgroup
// holds an element cannot change the row's two largest keys: every variant's ids, values and packet are the
// record's, which is what `tune` proves bit for bit before it times anything.
package enum NemotronTopTwoTune {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_TOP2_TUNE"] != "0"
    /// NEMOTRON_TOP2_FORCE=<n> adopts variant n (1-based, `variants`) when it qualified; 0 keeps the incumbent.
    static let forced = Int(ProcessInfo.processInfo.environment["NEMOTRON_TOP2_FORCE"] ?? "")
    /// An adopted variant's median is at most this fraction of the incumbent's (the shared tune bar).
    static let medianBar = Double(ProcessInfo.processInfo.environment["NEMOTRON_TOP2_BAR"] ?? "") ?? 0.97
    /// Wall time of the timed rounds, sized from one warm-up round.
    static let timedMilliseconds = 600.0

    struct Variant: Sendable, Equatable {
        let name: String
        /// Threadgroups per row.
        let groups: Int
        /// Threads per threadgroup.
        let threads: Int
        /// Loads per step of the main walk.
        let unroll: Int
    }

    /// The incumbent's geometry is `variants[0]` (32 x 256, four loads per step), the record's own.
    static let variants: [Variant] = [
        Variant(name: "g32t256u4", groups: 32, threads: 256, unroll: 4),
        Variant(name: "g16t256u4", groups: 16, threads: 256, unroll: 4),
        Variant(name: "g64t256u4", groups: 64, threads: 256, unroll: 4),
        Variant(name: "g32t128u4", groups: 32, threads: 128, unroll: 4),
        Variant(name: "g32t512u4", groups: 32, threads: 512, unroll: 4),
        Variant(name: "g64t128u4", groups: 64, threads: 128, unroll: 4),
        Variant(name: "g32t256u2", groups: 32, threads: 256, unroll: 2),
        Variant(name: "g32t256u1", groups: 32, threads: 256, unroll: 1),
    ]

    /// The incumbent geometry through the tune's pipeline: the baseline every candidate is judged against.
    static var incumbentTwin: Variant { variants[0] }

    /// The adopted geometry (nil: the record's 32 x 256 x 4 launch). Set once at load, before any verify graph is
    /// traced or warmed; `cbv2TopTwoRowsPacket` then takes the keys partial and finalize of this geometry.
    nonisolated(unsafe) private(set) static var choice: Variant? = nil

    /// Whether the kernel can launch the variant: whole SIMD groups, at most 512 threads (the merge reads at most
    /// two pairs per lane), and a walk step the text supports.
    static func eligible(_ v: Variant) -> Bool {
        v.groups > 0 && v.threads >= 32 && v.threads <= 512 && v.threads % 32 == 0
            && 2 * (v.threads / 32) <= 32 && [1, 2, 4].contains(v.unroll)
    }

    struct Kernels {
        /// Stage one of this geometry; the unread `dep` input (int32) only orders the launch after its producer.
        let partial: MLXFast.MLXFastKernel
        let finalize: MLXFast.MLXFastKernel
        /// The same finalize reading the verify columns (built on first use: the round takes this form only while
        /// `CBv2MTPTopTwoPacketColumns` is verified).
        let columns: @Sendable () -> MLXFast.MLXFastKernel
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var cache: [String: Kernels?] = [:]

    /// The record's key partial with the variant's geometry and walk step; every other statement is its text.
    static func partialSource(_ v: Variant) -> String {
        let stride = v.groups * v.threads
        let entries = 2 * (v.threads / 32)
        var main = ""
        switch v.unroll {
        case 4:
            main = """
                        for (; index < vocab && vocab - index > 3 * (\(stride)); index += 4 * (\(stride))) {
                            const bfloat16_t x0 = t2_row[index];
                            const bfloat16_t x1 = t2_row[index + \(stride)];
                            const bfloat16_t x2 = t2_row[index + 2 * (\(stride))];
                            const bfloat16_t x3 = t2_row[index + 3 * (\(stride))];
                            darkbloom_qwen35_mtp_top2_key_push(
                                first, second, darkbloom_qwen35_mtp_top2_key(x0, index));
                            darkbloom_qwen35_mtp_top2_key_push(
                                first, second, darkbloom_qwen35_mtp_top2_key(x1, index + \(stride)));
                            darkbloom_qwen35_mtp_top2_key_push(
                                first, second, darkbloom_qwen35_mtp_top2_key(x2, index + 2 * (\(stride))));
                            darkbloom_qwen35_mtp_top2_key_push(
                                first, second, darkbloom_qwen35_mtp_top2_key(x3, index + 3 * (\(stride))));
                        }
                """
        case 2:
            main = """
                        for (; index < vocab && vocab - index > \(stride); index += 2 * (\(stride))) {
                            const bfloat16_t x0 = t2_row[index];
                            const bfloat16_t x1 = t2_row[index + \(stride)];
                            darkbloom_qwen35_mtp_top2_key_push(
                                first, second, darkbloom_qwen35_mtp_top2_key(x0, index));
                            darkbloom_qwen35_mtp_top2_key_push(
                                first, second, darkbloom_qwen35_mtp_top2_key(x1, index + \(stride)));
                        }
                """
        default:
            break
        }
        return """
                uint lane = thread_position_in_threadgroup.x;
                uint group_index = threadgroup_position_in_grid.x;
                uint row = group_index / \(v.groups);
                uint group = group_index % \(v.groups);
                uint vocab = uint(logits_shape[2]);
                const auto t2_row = logits + row * vocab;
                ulong first = 0;
                ulong second = 0;
                uint index = group * \(v.threads) + lane;
            \(main)        for (; index < vocab; index += \(stride)) {
                    darkbloom_qwen35_mtp_top2_key_push(
                        first, second, darkbloom_qwen35_mtp_top2_key(t2_row[index], index));
                }

                const uint simd_lane = thread_index_in_simdgroup;
                const uint simd_group = simdgroup_index_in_threadgroup;
                const ulong simd_first = darkbloom_qwen35_mtp_top2_key_max(first);
                const ulong simd_second = darkbloom_qwen35_mtp_top2_key_max(first == simd_first ? second : first);

                threadgroup ulong scratch[\(entries)];
                if (simd_lane == 0) {
                    scratch[2 * simd_group] = simd_first;
                    scratch[2 * simd_group + 1] = simd_second;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (simd_group == 0) {
                    const ulong key = simd_lane < \(entries) ? scratch[simd_lane] : 0ul;
                    const ulong group_first = darkbloom_qwen35_mtp_top2_key_max(key);
                    const ulong group_second = darkbloom_qwen35_mtp_top2_key_max(key == group_first ? 0ul : key);
                    if (simd_lane == 0) {
                        partial_keys[(row * \(v.groups) + group) * 2] = group_first;
                        partial_keys[(row * \(v.groups) + group) * 2 + 1] = group_second;
                    }
                }
            """
    }

    /// The record's key finalize with the row's `groups` pairs taken as one SIMD group: each lane folds the pairs it
    /// owns with the same `key_push` (a maximum over the row's keys does not depend on how they are dealt out), then
    /// the row's two largest keys come from the record's two `key_max` passes and its stores write them.
    static func finalizeSource(_ v: Variant) -> String {
        """
                uint lane = thread_position_in_threadgroup.x;
                uint row = threadgroup_position_in_grid.x;
                ulong first = 0;
                ulong second = 0;
                for (uint part = lane; part < \(v.groups); part += 32) {
                    darkbloom_qwen35_mtp_top2_key_push(
                        first, second, partial_keys[(row * \(v.groups) + part) * 2]);
                    darkbloom_qwen35_mtp_top2_key_push(
                        first, second, partial_keys[(row * \(v.groups) + part) * 2 + 1]);
                }
                const ulong row_first = darkbloom_qwen35_mtp_top2_key_max(first);
                const ulong row_second = darkbloom_qwen35_mtp_top2_key_max(first == row_first ? second : first);
                darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_key_state(row_first, row_second);

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
    }

    static func kernels(_ v: Variant) -> Kernels? {
        lock.withLock { () -> Kernels? in
            if let made = cache[v.name] { return made }
            var made: Kernels? = nil
            guard eligible(v) else {
                cache[v.name] = made
                return made
            }
            let partial = MLXFast.metalKernel(
                name: "darkbloom_qwen35_mtp_top2_partial_keys_" + v.name,
                inputNames: ["logits", "dep"],
                outputNames: ["partial_keys"],
                source: partialSource(v),
                header: cbv2TopTwoKeysHeader,
                ensureRowContiguous: true)
            let body = finalizeSource(v)
            made = Kernels(
                partial: partial,
                finalize: MLXFast.metalKernel(
                    name: "darkbloom_qwen35_mtp_top2_finalize_packet_keys_" + v.name,
                    inputNames: ["partial_keys", "window"],
                    outputNames: ["top_ids", "top_values", "packet"],
                    source: body, header: cbv2TopTwoKeysHeader, ensureRowContiguous: true),
                columns: {
                    MLXFast.metalKernel(
                        name: "darkbloom_qwen35_mtp_top2_finalize_packet_keys_cols_" + v.name,
                        inputNames: ["partial_keys", "w1", "w2"],
                        outputNames: ["top_ids", "top_values", "packet"],
                        source: cbv2PacketColumnsSource(body), header: cbv2TopTwoKeysHeader,
                        ensureRowContiguous: true)
                })
            cache[v.name] = made
            return made
        }
    }

    /// One variant launch of the packet path on `logits`: nil when the kernel is ineligible. `dep` (int32, unread by
    /// the partial) only orders the launch after its producer; production binds the verify window there.
    static func launch(
        _ v: Variant, _ logits: MLXArray, window: MLXArray, draftColumns: [MLXArray]?, dep: MLXArray
    ) -> (ids: MLXArray, values: MLXArray, packet: MLXArray)? {
        guard let k = kernels(v) else { return nil }
        let rows = logits.dim(1)
        let drafts = window.dim(0) * (window.dim(1) - 1)
        let partial = k.partial(
            [logits, dep],
            grid: (rows * v.groups * v.threads, 1, 1),
            threadGroup: (v.threads, 1, 1),
            outputShapes: [[rows, v.groups, 2]],
            outputDTypes: [.uint64])[0]
        let columns = window.shape == [1, 3] ? draftColumns : nil
        let outputs = (columns == nil ? k.finalize : k.columns())(
            columns.map { [partial, $0[0], $0[1]] } ?? [partial, window],
            grid: (rows * 32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[rows, 2], [rows, 2], [drafts + rows * 3]],
            outputDTypes: [.int32, .float32, .int32])
        return (outputs[0], outputs[1], outputs[2])
    }

    /// The shared adoption rule, replicated from `NemotronTuneAdoption` (MLXLLM), which this module cannot see:
    /// each cycle takes one sample of every candidate in `order`; the candidate with the lowest median is adopted
    /// only when its median is at most `medianBar` of the incumbent's and it was faster than the incumbent in at
    /// least `winBar` of the cycles. Otherwise the incumbent stays.
    enum Adoption {
        static let minCycles = 16
        static let winBar = 0.75

        static func order(_ count: Int, cycle: Int) -> [Int] {
            let rotated = (0 ..< count).map { ($0 + cycle / 2) % count }
            return cycle % 2 == 0 ? rotated : Array(rotated.reversed())
        }

        static func cycles(target: Double, wall: Double, cap: Int) -> Int {
            let n = max(minCycles, min(cap, Int(target / max(wall, 0.01))))
            return n - n % 2
        }

        struct Verdict {
            let medians: [Double]
            let wins: [Int]
            let cycles: Int
            let best: Int?
            let adopted: Int?

            func won(_ i: Int) -> String { i < wins.count ? "\(wins[i])/\(cycles)" : "n/a" }

            func rule(_ name: (Int) -> String) -> String {
                guard let b = best, medians[0] > 0 else { return "no candidate timed" }
                return "best \(name(b)) " + String(format: "%+.1f%%", 100 * (medians[b] / medians[0] - 1))
                    + " wins \(won(b))"
            }
        }

        static func decide(_ samples: [[Double]], medianBar: Double) -> Verdict {
            let cycles = samples.first?.count ?? 0
            let medians = samples.map { $0.isEmpty ? Double.nan : $0.sorted()[$0.count / 2] }
            let wins = samples.map { s in zip(s, samples[0]).filter { $0.0 < $0.1 }.count }
            let best = medians.indices.dropFirst().min(by: { medians[$0] < medians[$1] })
            var adopted: Int? = nil
            if let b = best, cycles > 0, samples.allSatisfy({ $0.count == cycles }),
               medians[b] <= medianBar * medians[0], Double(wins[b]) >= winBar * Double(cycles) {
                adopted = b
            }
            return Verdict(medians: medians, wins: wins, cycles: cycles, best: best, adopted: adopted)
        }
    }

    private static func millis(_ d: Duration) -> Double {
        let c = d.components
        return Double(c.seconds) * 1000 + Double(c.attoseconds) / 1e15
    }

    /// Load time, after `cbv2TopTwoKeysQualify` and before any verify graph is traced or warmed: every variant
    /// against the production packet launch bit for bit in the ids, the values (as bits) and the packet on seeded
    /// BF16 logits (Gaussian, exact ties, NaNs, infinities, signed zeros, constant rows, rows of one repeated
    /// pattern, a short vocabulary and the verify width), then the survivors timed round robin as chains of
    /// partial+finalize pairs chained through the unread `dep` input, and the winner adopted only by the rule.
    /// NEMOTRON_TOP2_TUNE=0 keeps the record geometry; NEMOTRON_TOP2_FORCE=<n> adopts variant n when it qualified.
    package static func tune(vocabulary: Int) {
        choice = nil
        let clock = CBv2Clock.continuous
        let started = clock.now()
        func report(_ message: String) {
            let s = millis(clock.now() - started) / 1000
            FileHandle.standardError.write(Data(
                "nemotron top2 tune: \(message) (\(String(format: "%.2f", s)) s)\n".utf8))
        }
        guard enabled else { return report("off (setting)") }
        guard CBv2MTPTopTwoPacket.enabled else { return report("off (packet finalize off)") }
        guard CBv2MTPTopTwoKeys.verified else { return report("off (keys form not in use)") }
        guard vocabulary >= 2 else { return report("off (vocabulary \(vocabulary))") }

        struct Case {
            let logits: MLXArray
            let window: MLXArray
        }
        var cases: [Case] = []
        var timedLogits: MLXArray? = nil
        var timedWindow: MLXArray? = nil
        do {
            cases = try withError { () -> [Case] in
                var made: [Case] = []
                for (vocab, rows) in [(vocabulary, 3), (vocabulary, 1), (5003, 3), (64, 2)] {
                    for kind in 0 ..< 5 {
                        let key = MLXRandom.key(UInt64(0x7090_0000 + vocab * 64 + rows * 8 + kind))
                        var logits = MLXRandom.normal([1, rows, vocab], key: key) * 3
                        switch kind {
                        case 1: logits = round(logits)
                        case 2: logits = which(abs(logits) .> MLXArray(Float(5.5)), MLXArray(Float.nan), logits)
                        case 3:
                            logits = which(logits .> MLXArray(Float(6)), MLXArray(Float.infinity),
                                which(logits .< MLXArray(Float(-6)), MLXArray(-Float.infinity), logits))
                        case 4: logits = MLXArray.zeros([1, rows, vocab]) + MLXArray(Float(0.5))
                        default: break
                        }
                        let typed = logits.asType(.bfloat16)
                        var window = MLXRandom.randInt(
                            0 ..< Int32(min(vocab, Int(Int32.max))), [1, rows],
                            key: MLXRandom.key(UInt64(0x7091_0000 + vocab * 64 + rows * 8 + kind)))
                            .asType(.int32)
                        if rows < 2 {
                            window = concatenated([window, window], axis: 1)
                        }
                        let l = rows < 2 ? concatenated([typed, typed], axis: 1) : typed
                        eval([l, window])
                        if vocab == vocabulary, rows == 3, kind == 0 {
                            timedLogits = l
                            timedWindow = window
                        }
                        made.append(Case(logits: l, window: window))
                    }
                }
                return made
            }
        } catch {
            return report("off (inputs: \(error))")
        }
        guard let timedL = timedLogits, let timedW = timedWindow, !cases.isEmpty else {
            return report("off (no timed case)")
        }

        // The production launch the round takes today (the record's 32 x 256 x 4 keys partial + its finalize). While
        // `CBv2MTPTopTwoPacketColumns` is verified the round takes the column-reading finalize instead, so that form
        // is a production form too and is checked on every case as well; one reference evaluation per form.
        func incumbent(_ c: Case, _ columns: [MLXArray]?) -> (ids: MLXArray, values: MLXArray, packet: MLXArray)? {
            cbv2TopTwoRowsPacket(c.logits, window: c.window, draftColumns: columns, shuffle: nil, keys: nil)
        }
        func draftColumns(_ c: Case) -> [MLXArray]? {
            guard CBv2MTPTopTwoPacketColumns.verified, c.window.shape == [1, 3] else { return nil }
            return [c.window[0..., 1].reshaped(1), c.window[0..., 2].reshaped(1)]
        }
        var forms: [[MLXArray]?] = [nil]
        if let columns = draftColumns(cases[0]) { forms.append(columns) }

        // Qualify: bit for bit against the production launch; a mismatch, a missing kernel or a Metal error drops
        // the variant. The incumbent geometry runs through the tune's own pipeline too (the timing twin).
        var alive = variants.filter(eligible)
        var dropped: [String] = []
        var checks = 0
        for (ci, c) in cases.enumerated() {
            // One reference evaluation per production form of this case.
            let references: [[Int32]]
            do {
                references = try withError { () -> [[Int32]] in
                    var made: [[Int32]] = []
                    for columns in forms {
                        guard let reference = incumbent(c, columns) else { return [] }
                        eval(reference.ids, reference.values, reference.packet)
                        made.append(reference.ids.asArray(Int32.self) + reference.packet.asArray(Int32.self)
                            + reference.values.view(dtype: .uint32).asArray(UInt32.self)
                            .map { Int32(bitPattern: $0) })
                    }
                    return made
                }
            } catch {
                return report("off (production unavailable in case \(ci))")
            }
            guard references.count == forms.count else {
                return report("off (production unavailable in case \(ci))")
            }
            alive = alive.filter { v in
                do {
                    let same = try withError { () -> Bool in
                        for (columns, referenceValues) in zip(forms, references) {
                            guard let out = launch(v, c.logits, window: c.window, draftColumns: columns,
                                                   dep: c.window) else { return false }
                            eval(out.ids, out.values, out.packet)
                            let got = out.ids.asArray(Int32.self) + out.packet.asArray(Int32.self)
                                + out.values.view(dtype: .uint32).asArray(UInt32.self).map { Int32(bitPattern: $0) }
                            if got != referenceValues { return false }
                        }
                        return true
                    }
                    if !same { dropped.append("\(v.name) differs@\(ci)") }
                    return same
                } catch {
                    dropped.append("\(v.name) error")
                    return false
                }
            }
            checks += 1
            if alive.isEmpty { break }
        }
        guard !alive.isEmpty else { return report("off (no variant bitwise equal: \(dropped.prefix(6).joined(separator: ", ")))") }
        let twin = incumbentTwin
        guard alive.contains(twin) else { return report("off (incumbent twin not bitwise equal)") }
        let candidates: [Variant] = [twin] + alive.filter { $0 != twin }

        // Time the twin and the survivors round robin: each sample is a chain of partial+finalize pairs at the
        // record's verify width, every finalize's packet bound as the next partial's `dep` so one launch starts only
        // after the one before it finished, as in the round. The chains are built before the clock starts.
        let chain = 8
        var medians: [Double] = []
        var verdict: Adoption.Verdict? = nil
        var reps = 0
        var samples = Array(repeating: [Double](), count: candidates.count)
        do {
            try withError {
                func build(_ i: Int) -> MLXArray? {
                    var prev = timedW
                    for _ in 0 ..< chain {
                        guard let out = launch(candidates[i], timedL, window: timedW, draftColumns: nil, dep: prev)
                        else { return nil }
                        prev = out.packet
                    }
                    return prev
                }
                func round(_ order: [Int]) -> [Double?] {
                    let chains = order.map { build($0) }
                    return chains.map { last -> Double? in
                        guard let last else { return nil }
                        let start = clock.now()
                        eval(last)
                        Stream.gpu.synchronize()
                        return millis(clock.now() - start) / Double(chain)
                    }
                }
                let warm = clock.now()
                _ = round(Array(candidates.indices))
                let wall = millis(clock.now() - warm)
                reps = Adoption.cycles(target: timedMilliseconds, wall: wall, cap: 41)
                for rep in 0 ..< reps {
                    let order = Adoption.order(candidates.count, cycle: rep)
                    for (i, ms) in zip(order, round(order)) {
                        if let ms { samples[i].append(ms) }
                    }
                }
            }
        } catch {
            dropped.append("timing error")
            samples = Array(repeating: [], count: candidates.count)
        }
        if samples.allSatisfy({ $0.count == reps }), reps > 0 {
            verdict = Adoption.decide(samples, medianBar: medianBar)
            medians = verdict?.medians ?? []
        } else {
            dropped.append("timing incomplete")
        }

        var picked: Variant? = nil
        var why = "kept"
        if !medians.isEmpty {
            if let f = forced {
                if f >= 1, f <= variants.count, let i = candidates.firstIndex(of: variants[f - 1]) {
                    picked = candidates[i]
                    why = "forced"
                } else {
                    why = f == 0 ? "forced V0" : "forced \(f) not qualified, kept"
                }
            } else if let best = verdict?.adopted {
                picked = candidates[best]
                why = "adopted"
            }
        } else if let f = forced, f == 0 {
            why = "forced V0"
        }
        choice = picked

        func ms(_ v: Variant) -> String {
            guard let i = candidates.firstIndex(of: v), i < medians.count else { return "n/a" }
            return String(format: "%.4f", medians[i])
        }
        let times = candidates.indices.dropFirst().map {
            "\(candidates[$0].name) \(ms(candidates[$0])) \(verdict?.won($0) ?? "n/a")"
        }
        let rule = verdict?.rule({ candidates[$0].name }) ?? "no candidate timed"
        report("\(why) \(picked?.name ?? "record 32x256u4") \(ms(picked ?? twin)) ms vs V0 (g32t256u4)"
            + " \(ms(twin)) ms per launch (\(rule); median of \(reps), \(chain) chained partial+finalize pairs"
            + " per sample at \(timedL.dim(1)) rows x \(timedL.dim(2)); \(candidates.count)/\(variants.count)"
            + " bitwise over \(checks) cases against the production launch"
            + (times.isEmpty ? "" : "; [" + times.joined(separator: ", ") + "]")
            + (dropped.isEmpty ? "" : "; dropped: " + dropped.prefix(8).joined(separator: ", ")))
    }
}

/// Load-time JIT of the packet finalize (its pipeline depends only on the
/// int32/float32/int32 input types, not on the window shape).
package func cbv2TopTwoRowsPacketWarm(logitsDType: DType) {
    let logits = MLXArray.zeros([1, 3, 64], dtype: logitsDType)
    let window = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
    guard let warmed = cbv2TopTwoRowsPacket(logits, window: window) else { return }
    eval(warmed.ids, warmed.values, warmed.packet)
}
