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
private let cbv2TopTwoPartialShuffleKernel = MLXFast.metalKernel(
    name: CBv2MTPTopTwoLoadsFirst.enabled
        ? "darkbloom_qwen35_mtp_top2_partial_sh_lf" : "darkbloom_qwen35_mtp_top2_partial_sh",
    inputNames: ["logits"],
    outputNames: ["partial_ids", "partial_values"],
    source: """
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint row = group_index / 32;
        uint group = group_index % 32;
        uint vocab = uint(logits_shape[2]);
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();

        \(CBv2MTPTopTwoLoadsFirst.prefix)for (uint index = group * 256 + lane;
             index < vocab;
             index += 32 * 256) {
            ulong offset = ulong(row) * ulong(logits_strides[1])
                + ulong(index) * ulong(logits_strides[2]);
            darkbloom_qwen35_mtp_top2_insert(local, float(logits[offset]), index);
        }\(CBv2MTPTopTwoLoadsFirst.suffix)

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
    """,
    header: cbv2TopTwoShuffleHeader,
    ensureRowContiguous: false
)

/// CBV2_MTP_TOP2_LOADS_FIRST=0 keeps the shuffle partial's scan as is. Otherwise,
/// for a 131072-entry row with unit stride, a thread loads its sixteen logits into
/// registers first and then runs the same sixteen inserts in the same order (same
/// values, same ids); any other shape takes the unchanged loop.
enum CBv2MTPTopTwoLoadsFirst {
    static let enabled = ProcessInfo.processInfo.environment["CBV2_MTP_TOP2_LOADS_FIRST"] != "0"
    static var prefix: String {
        guard enabled else { return "" }
        return """
            if (vocab == 131072u && logits_strides[2] == 1) {
                const ulong first = ulong(row) * ulong(logits_strides[1]) + ulong(group * 256 + lane);
                float values[16];
                for (uint i = 0; i < 16; i++) {
                    values[i] = float(logits[first + ulong(i) * 8192ul]);
                }
                for (uint i = 0; i < 16; i++) {
                    darkbloom_qwen35_mtp_top2_insert(local, values[i], group * 256 + lane + i * 8192u);
                }
            } else
            """ + " "
    }
    static var suffix: String { "" }
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
    _ logits: MLXArray, window: MLXArray, draftColumns: [MLXArray]?, shuffle: Bool?
) -> (ids: MLXArray, values: MLXArray, packet: MLXArray)? {
    guard CBv2MTPTopTwoPacket.enabled,
        logits.ndim == 3, logits.dim(0) == 1, logits.dim(2) >= 2,
        window.ndim == 2, window.dtype == .int32, window.dim(1) >= 2,
        window.dim(0) * window.dim(1) == logits.dim(1)
    else { return nil }
    let rows = logits.dim(1)
    let drafts = window.dim(0) * (window.dim(1) - 1)
    let shuffled = shuffle ?? CBv2MTPTopTwoShuffle.verified
    let partials = (shuffled ? cbv2TopTwoPartialShuffleKernel : cbv2TopTwoPartialKernel)(
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

/// Load-time JIT of the packet finalize (its pipeline depends only on the
/// int32/float32/int32 input types, not on the window shape).
package func cbv2TopTwoRowsPacketWarm(logitsDType: DType) {
    let logits = MLXArray.zeros([1, 3, 64], dtype: logitsDType)
    let window = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
    guard let warmed = cbv2TopTwoRowsPacket(logits, window: window) else { return }
    eval(warmed.ids, warmed.values, warmed.packet)
}
