//
//  NemotronH35.swift
//  mlx-swift-lm
//
//  Additive configuration boundary for Nemotron 3.5 Lightning checkpoints.
//  The target equations remain in NemotronHModel; this type keeps the newer
//  checkpoint contract distinct from legacy Nemotron Nano selection.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct NemotronH35Configuration: Codable, Sendable {
    public let target: NemotronHConfiguration
    let hasExplicitExecutionLimits: Bool

    private enum CodingKeys: String, CodingKey {
        case layersBlockType = "layers_block_type"
        case timeStepLimit = "time_step_limit"
        case timeStepLimitMin = "time_step_limit_min"
        case timeStepLimitMax = "time_step_limit_max"
        case timeStepMin = "time_step_min"
        case timeStepMax = "time_step_max"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        guard container.contains(.layersBlockType) else {
            throw DecodingError.keyNotFound(
                CodingKeys.layersBlockType,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription:
                        "Nemotron 3.5 requires explicit layers_block_type"))
        }

        let target = try NemotronHConfiguration(from: decoder)
        guard target.mambaSSMCacheDType.lowercased() == "float32" else {
            throw DecodingError.dataCorruptedError(
                forKey: .layersBlockType, in: container,
                debugDescription:
                    "unsupported mamba_ssm_cache_dtype "
                    + target.mambaSSMCacheDType)
        }
        self.target = target
        hasExplicitExecutionLimits = try [CodingKeys.timeStepLimit, .timeStepLimitMin, .timeStepLimitMax]
            .contains { key in
                guard container.contains(key) else { return false }
                return !(try container.decodeNil(forKey: key))
            }
    }

    public func encode(to encoder: Encoder) throws {
        // JSON has no infinity. Encode native defaults as null, and retain
        // the distinction between training metadata and execution limits.
        var encoded = target
        if !encoded.timeStepLimitMax.isFinite { encoded.timeStepLimitMax = Float.greatestFiniteMagnitude }
        try encoded.encode(to: encoder)
        var container = encoder.container(keyedBy: CodingKeys.self)
        if hasExplicitExecutionLimits {
            try container.encode(target.timeStepLimitMin, forKey: .timeStepLimitMin)
            if target.timeStepLimitMax.isFinite {
                try container.encode(target.timeStepLimitMax, forKey: .timeStepLimitMax)
            } else { try container.encodeNil(forKey: .timeStepLimitMax) }
        } else {
            try container.encodeNil(forKey: .timeStepLimitMin)
            try container.encodeNil(forKey: .timeStepLimitMax)
            try container.encode(target.timeStepLimitMin, forKey: .timeStepMin)
            if target.timeStepLimitMax.isFinite {
                try container.encode(target.timeStepLimitMax, forKey: .timeStepMax)
            } else { try container.encodeNil(forKey: .timeStepMax) }
        }
        let blockTypes = target.hybridOverridePattern.map { blockType in
            switch blockType {
            case "M": return "mamba"
            case "*": return "attention"
            case "-": return "mlp"
            case "E": return "moe"
            default: preconditionFailure("validated Nemotron block type changed")
            }
        }
        try container.encode(blockTypes, forKey: .layersBlockType)
    }
}

public final class NemotronH35Model: NemotronHModel {
    public let lightningConfiguration: NemotronH35Configuration

    public override var cbv2Capabilities: CBv2ModelCapabilities {
        var capabilities = super.cbv2Capabilities
        capabilities.supportsRecurrentCheckpointReuse = true
        capabilities.supportsPagedKV = true
        capabilities.requiresNativePagedKV = true
        capabilities.supportsMTP = configuration.numNextnPredictLayers == 1
            && configuration.mtpLayersBlockType == ["attention", "moe"]
        return capabilities
    }

    public init(_ args: NemotronH35Configuration) {
        self.lightningConfiguration = args
        // mlx-lm 0.31.3 parses time_step_min/time_step_max as metadata but
        // initializes the NemotronH mixer from the absent time_step_limit
        // tuple, whose reference default is (0, +infinity). Applying the
        // copied HF min/max here changes greedy output ("NEMOTRON..." becomes
        // "The count: 1" on the audited checkpoint). Preserve the decoded
        // values above for provenance, but execute the selected MLX artifact
        // with its independent serial oracle's actual contract.
        var executionTarget = args.target
        if !args.hasExplicitExecutionLimits {
            executionTarget.timeStepLimitMin = 0
            executionTarget.timeStepLimitMax = .infinity
        }
        super.init(executionTarget)
    }
}

extension NemotronH35Model: CBv2CompleteCheckpointKVTypeProviding {
    public var cbv2CompleteCheckpointKVDTypes: [DType]? {
        observedAttentionKVDTypes()
    }
}

// MARK: - Seed retile (R4)

/// The run gather with free tiling, appended to `nemotronRunGatherHeader`. For
/// every output fragment pair the same `BaseNAXFrag::mma` (matmul2d(16, 32, 16),
/// multiply_accumulate) runs over k chunks 0, 16, ..., K - 16 in order, on A
/// fragments loaded by the incumbent's statements and B fragments holding the
/// incumbent's BF16 dequantized value at each (n, k); the typed store is the
/// incumbent's. Free: BN, BK, the SIMD-group grid WM x WN, and whether B passes
/// through threadgroup memory (DIRECT_B false) or is dequantized straight into
/// the fragment registers at `BaseNAXFrag::get_coord` (no Ws, no barrier). TN
/// stays even, so each mma pairs the incumbent's two 16-column fragments
/// (columns 32c ... 32c + 31), and rows keep their 16-row fragments from the
/// tile's first row.
let nemotronSeedRetileSource = #"""
// Nemotron seed retile (R4): the incumbent's arithmetic for one bank byte
// (dequantize<T, N, 4>, the pair dictionary loader, the decode-copy loader):
// k and k + 1 from float scale s and bias b.
template <typename T>
METAL_FUNC void nemotron_seed_deq2(
    const uint8_t byte, const float s, const float b, thread T& v0, thread T& v1) {
  float sc[2] = {s, s / 16.0f};
  v0 = static_cast<T>(sc[0] * (byte & 0x0f) + b);
  v1 = static_cast<T>(sc[1] * (byte & 0xf0) + b);
}

// B source: the bank (codes [E, N, K / 8] uint32, BF16 scales and biases
// [E, N, K / 64]); scale and bias widen as in dequantize.
template <typename T>
struct NemotronSeedBank {
  const device uint8_t* w;
  const device T* scales;
  const device T* biases;
  int K;

  METAL_FUNC void expert(const uint index, const int N) thread {
    w += size_t(index) * size_t(N) * size_t(K / 2);
    scales += size_t(index) * size_t(N) * size_t(K / 64);
    biases += size_t(index) * size_t(N) * size_t(K / 64);
  }
  METAL_FUNC float2 sb(const int n, const int g) const thread {
    T scale = scales[size_t(n) * size_t(K / 64) + size_t(g)];
    T bias = biases[size_t(n) * size_t(K / 64) + size_t(g)];
    return float2(float(scale), float(bias));
  }
  METAL_FUNC uint8_t byte(const int n, const int k) const thread {
    return w[size_t(n) * size_t(K / 2) + size_t(k >> 1)];
  }
};

// B source: the bank's codes with (scale, bias) through the pair dictionary
// (uint16 index in the scales' layout, FP32 table of exact BF16 widenings), as
// QuantizedBlockLoaderDict.
template <typename T>
struct NemotronSeedDict {
  const device uint8_t* w;
  const device uint16_t* sidx;
  const device float* table;
  int K;

  METAL_FUNC void expert(const uint index, const int N) thread {
    w += size_t(index) * size_t(N) * size_t(K / 2);
    sidx += size_t(index) * size_t(N) * size_t(K / 64);
  }
  // The table holds exact BF16 widenings, so these are float(T(table)) bit for
  // bit (the fc2 copy loader reads it the same way); the T round trip is not
  // spelled out because one Metal compiler folds it wrongly when unconditional.
  METAL_FUNC float2 sb(const int n, const int g) const thread {
    const uint pair = uint(sidx[size_t(n) * size_t(K / 64) + size_t(g)]);
    const float s = table[2 * pair];
    const float b = table[2 * pair + 1];
    return float2(s, b);
  }
  METAL_FUNC uint8_t byte(const int n, const int k) const thread {
    return w[size_t(n) * size_t(K / 2) + size_t(k >> 1)];
  }
};

// B source: the tile-major zero-skip decode copy (codes [E, N / 128, K, 32]
// uint16, four outputs per word; pair index [E, N / 128, K / 64, 128]; FP32
// pair table): the bank byte is rebuilt from the two nibbles as the fc2 copy
// loader does. With expert groups, `codes` is the tile's group and `first` its
// first expert: the codes advance by the local expert, the pair index by index.
template <typename T>
struct NemotronSeedCopy {
  const device uint16_t* codes;
  const device uint16_t* sidx;
  const device float* table;
  int K;
  uint first;

  METAL_FUNC void expert(const uint index, const int N) thread {
    codes += size_t(index - first) * size_t(N / 128) * size_t(K) * 32;
    sidx += size_t(index) * size_t(N / 128) * size_t(K / 64) * 128;
  }
  METAL_FUNC float2 sb(const int n, const int g) const thread {
    const uint pair = uint(sidx[(size_t(n / 128) * size_t(K / 64) + size_t(g)) * 128 + size_t(n % 128)]);
    const float s = table[2 * pair];
    const float b = table[2 * pair + 1];
    return float2(s, b);
  }
  METAL_FUNC uint8_t byte(const int n, const int k) const thread {
    const device uint16_t* c =
        codes + (size_t(n / 128) * size_t(K) + size_t(k)) * 32 + size_t((n % 128) / 4);
    const short o = short(n % 4);
    const uint lo = (uint(c[0]) >> (4 * o)) & 0xfu;
    const uint hi = (uint(c[32]) >> (4 * o)) & 0xfu;
    return uint8_t(lo | (hi << 4));
  }
};

// The direct fc1's row-indexed A load (nemotron_load_routed), unchanged.
template <typename Tile, typename T>
METAL_FUNC void nemotron_seed_load_routed(
    thread Tile& tile, const device T* const thread* rows, const int offset,
    const short valid_rows) {
  using Frag = typename Tile::NAXFrag_t;
  const short2 sc = Frag::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short mr = 0; mr < Tile::kTileRows; ++mr) {
    STEEL_PRAGMA_UNROLL
    for (short mc = 0; mc < Tile::kTileCols; ++mc) {
      thread typename Tile::frag_type& values = tile.frag_at(mr, mc);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < Frag::kElemRows; ++i) {
        const short row = mr * Frag::kFragRows + sc.y + i * Frag::kElemRowsJump;
        if (row < valid_rows) {
          const device T* source_row = rows[mr * Frag::kElemRows + i] + offset;
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < Frag::kElemCols; ++j) {
            const short col = mc * Frag::kFragCols + sc.x + j;
            values[i * Frag::kElemCols + j] = static_cast<T>(source_row[col]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < Frag::kElemCols; ++j) {
            values[i * Frag::kElemCols + j] = T(0);
          }
        }
      }
    }
  }
}

// The A fragments at absolute k: routed rows, or the sorted rows with the
// run gather's aligned / safe loads.
template <bool ROUTED, bool ALIGNED_M, short SK, typename ATile, typename T>
METAL_FUNC void nemotron_seed_load_a(
    thread ATile& Atile,
    const device T* xn,
    const device T* const thread* routed,
    const int k,
    const int K,
    const short sgp_sm) {
  if constexpr (ROUTED) {
    nemotron_seed_load_routed(Atile, routed, k, sgp_sm);
  } else if constexpr (ALIGNED_M) {
    Atile.load(xn + k, K);
  } else {
    Atile.load_safe(xn + k, K, short2(SK, sgp_sm));
  }
}

// tiles[t] = (first sorted row, rows <= BM, expert), as the run gather.
template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool DIRECT_B,
    bool ALIGN_N,
    bool RELU2,
    bool ROUTED,
    class BSrc>
METAL_FUNC void nemotron_seed_gather_impl(
    const device T* x,
    const device uint32_t* routed_rows,
    BSrc bsrc,
    const device int* tiles,
    device T* y,
    const int N,
    const int K,
    threadgroup T* Ws,
    uint3 tid,
    uint simd_group_id,
    uint simd_lane_id) {
  const int tile_rows = tiles[tid.y * 3 + 1];
  if (tile_rows <= 0) {
    return;
  }
  const int row0 = tiles[tid.y * 3 + 0];
  const uint32_t index = uint32_t(tiles[tid.y * 3 + 2]);
  bsrc.expert(index, N);

  constexpr bool transpose = true;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;
  static_assert(SM % 16 == 0 && TN % 2 == 0, "the run gather's fragment pairs");
  static_assert(BK % 64 == 0, "B blocks start on a quantization group");

  const int y_col = tid.x * BN;
  const short tgp_bm = short(min(BM, tile_rows));
  const short tgp_bn = ALIGN_N ? BN : short(min(BN, N - y_col));
  y += size_t(row0) * N + size_t(y_col);

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  const short sgp_sm = min(SM, short(max(0, int(tgp_bm) - tm)));
  const short sgp_sn = ALIGN_N ? SN : min(SN, short(max(0, (N - (y_col + tn)))));
  const bool is_unaligned_sm = (sgp_sm != SM);
  const bool is_unaligned_bn = ALIGN_N ? false : (tgp_bn != BN);

  const short m_lo_lim = 0;
  const short m_hi_lim = min(int(sgp_sm), max(0, int(tgp_bm) - tm));
  const bool sg_active = m_hi_lim > m_lo_lim;

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  const device T* xn = ROUTED ? x : x + (size_t(row0) + size_t(tm)) * size_t(K);
  const device T* routed[TM * BaseNAXFrag::kElemRows];
  if constexpr (ROUTED) {
    const short2 row_coord = BaseNAXFrag::get_coord();
    STEEL_PRAGMA_UNROLL
    for (short mr = 0; mr < TM; ++mr) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BaseNAXFrag::kElemRows; ++i) {
        const short row = mr * BaseNAXFrag::kFragRows + row_coord.y
            + i * BaseNAXFrag::kElemRowsJump;
        routed[mr * BaseNAXFrag::kElemRows + i] = row < sgp_sm
            ? x + size_t(routed_rows[row0 + tm + row]) * K : x;
      }
    }
  }

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    constexpr bool aligned_m = decltype(kAlignedM)::value;
    if constexpr (DIRECT_B) {
      // Each thread dequantizes its own B fragment elements: rows n =
      // tn + 16 nn + coord.y + 8 i, k = 16-chunk + coord.x + j (the
      // positions BaseNAXFrag::load reads from Ws).
      const short2 bc = BaseNAXFrag::get_coord();
      for (int kg = 0; kg < K; kg += 64) {
        float2 sbv[TN * 2];
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < TN; ++nn) {
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < 2; ++i) {
            const int n = y_col + tn + nn * 16 + bc.y + i * 8;
            sbv[nn * 2 + i] = (ALIGN_N || n < N) ? bsrc.sb(n, kg / 64) : float2(0.0f);
          }
        }

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < 64; kk1 += SK) {
          if (sg_active) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, TN, TK> Btile;

            volatile int compiler_barrier;

            nemotron_seed_load_a<ROUTED, aligned_m, SK>(Atile, xn, routed, kg + kk1, K, sgp_sm);

            STEEL_PRAGMA_UNROLL
            for (short nn = 0; nn < TN; ++nn) {
              STEEL_PRAGMA_UNROLL
              for (short kk = 0; kk < TK; ++kk) {
                thread typename NAXTile<T, TN, TK>::frag_type& frag = Btile.frag_at(nn, kk);
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; ++i) {
                  const int n = y_col + tn + nn * 16 + bc.y + i * 8;
                  const int k = kg + kk1 + kk * 16 + bc.x;
                  T v0 = T(0), v1 = T(0), v2 = T(0), v3 = T(0);
                  if (ALIGN_N || n < N) {
                    const float2 p = sbv[nn * 2 + i];
                    nemotron_seed_deq2(bsrc.byte(n, k), p.x, p.y, v0, v1);
                    nemotron_seed_deq2(bsrc.byte(n, k + 2), p.x, p.y, v2, v3);
                  }
                  frag[i * 4 + 0] = v0;
                  frag[i * 4 + 1] = v1;
                  frag[i * 4 + 2] = v2;
                  frag[i * 4 + 3] = v3;
                }
              }
            }

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }
        }
      }
    } else {
      // Threadgroup weight block BN x BK: thread t dequantizes KPT
      // consecutive k (inside one group) of one row, then the run gather's
      // fragment loads from Ws.
      constexpr int NT = WM * WN * 32;
      constexpr int KPT = BN * BK / NT;
      static_assert(KPT % 4 == 0 && KPT <= 64 && 64 % KPT == 0 && BK % KPT == 0, "loader split");
      const int t = int(simd_group_id) * 32 + int(simd_lane_id);
      const short lr = short(t / (BK / KPT));
      const short lk = short((t % (BK / KPT)) * KPT);
      threadgroup T* wdst = Ws + lr * BK_padded + lk;
      const int ln = y_col + lr;
      auto fill = [&](const int kb, const int klim) {
        if ((ALIGN_N || lr < tgp_bn) && lk < klim) {
          const float2 p = bsrc.sb(ln, (kb + lk) / 64);
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < KPT; j += 2) {
            T v0, v1;
            nemotron_seed_deq2(bsrc.byte(ln, kb + lk + j), p.x, p.y, v0, v1);
            wdst[j] = v0;
            wdst[j + 1] = v1;
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < KPT; ++j) {
            wdst[j] = T(0);
          }
        }
      };
      auto block = [&](const int kb, const int klim) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fill(kb, klim);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < klim; kk1 += SK) {
          if (sg_active) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, TN, TK> Btile;

            volatile int compiler_barrier;

            nemotron_seed_load_a<ROUTED, aligned_m, SK>(Atile, xn, routed, kb + kk1, K, sgp_sm);

            Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }
        }
      };
      const int K_it = K / BK;
      for (int kb_i = 0; kb_i < K_it; kb_i++) {
        block(kb_i * BK, BK);
      }
      // K % 64 == 0, so the tail is whole groups and whole 32-k steps.
      const int k_rem = K - K_it * BK;
      if (k_rem > 0) {
        block(K_it * BK, k_rem);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  });

  if (ALIGN_N || !is_unaligned_bn) {
    if (m_lo_lim == 0 && m_hi_lim == SM) {
      Dtile.template store<RELU2>(y + tm * N + tn, N);
    } else {
      Dtile.template store_slice<RELU2>(
          y + tm * N + tn, N, short2(0, m_lo_lim), short2(SN, m_hi_lim));
    }
  } else {
    Dtile.template store_slice<RELU2>(
        y + tm * N + tn,
        N,
        short2(0, m_lo_lim),
        short2(sgp_sn, m_hi_lim));
  }
}
"""#

/// Seed-only (request prefill, `NemotronHPrefillX5.active`) routed MoE fc1 and
/// fc2 through a retiled run gather (`nemotron_seed_gather_impl`) chosen once at
/// load. Every variant keeps each output's `BaseNAXFrag::mma` sequence over k
/// chunks 0..K/16 in order with the incumbent's BF16 A and B operands and FP32
/// accumulators, and the incumbent's typed store (`nemotronSeedRetileSource`).
/// `tune` (load time, after every other qualify) compares each variant with the
/// launch the seed uses today (direct fc1 with or without its pair dictionary,
/// or the plain run gather; the fc2 decode-copy gather, or the plain run
/// gather) bit for bit on seeded routes and inputs (every MoE layer in the first
/// case), drops any mismatch or Metal error, times the survivors and the
/// incumbent round robin on the 488-token gathers of the first four MoE layers
/// (5 warm-up and 25 timed repetitions, median), and adopts the fastest only if
/// it is at least 0.5% faster than the incumbent. NEMOTRON_SEED_RETILE=0 keeps
/// the incumbents; NEMOTRON_SEED_RETILE_FORCE=<variant> adopts that variant
/// for both roles when it qualifies.
enum NemotronSeedRetile {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE"] != "0"
    static let forced = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE_FORCE"]

    struct Variant: Sendable, Equatable {
        let name: String
        let bn: Int, bk: Int, wm: Int, wn: Int
        let direct: Bool
        var threads: Int { wm * wn * 32 }
    }

    /// ws: B through threadgroup memory; d: B dequantized into the fragments.
    /// Name = path, BN (x BK), then WM WN. ws64x64w22 is the incumbent's geometry.
    static let variants: [Variant] = [
        Variant(name: "ws64x64w22", bn: 64, bk: 64, wm: 2, wn: 2, direct: false),
        Variant(name: "ws64x128w22", bn: 64, bk: 128, wm: 2, wn: 2, direct: false),
        Variant(name: "ws128x64w24", bn: 128, bk: 64, wm: 2, wn: 4, direct: false),
        Variant(name: "ws128x64w14", bn: 128, bk: 64, wm: 1, wn: 4, direct: false),
        Variant(name: "d64w22", bn: 64, bk: 64, wm: 2, wn: 2, direct: true),
        Variant(name: "d64w12", bn: 64, bk: 64, wm: 1, wn: 2, direct: true),
        Variant(name: "d128w14", bn: 128, bk: 64, wm: 1, wn: 4, direct: true),
        Variant(name: "d128w24", bn: 128, bk: 64, wm: 2, wn: 4, direct: true),
        Variant(name: "ws64x64w42", bn: 64, bk: 64, wm: 4, wn: 2, direct: false),
        Variant(name: "d64w42", bn: 64, bk: 64, wm: 4, wn: 2, direct: true),
        Variant(name: "ws128x64w42", bn: 128, bk: 64, wm: 4, wn: 2, direct: false),
    ]

    enum Source: String, Sendable { case bank, dict, copy }

    /// The launch a role's incumbent uses at the seed.
    struct Role: Sendable, Equatable {
        let routed: Bool
        let source: Source
        let relu2: Bool
        var label: String { (routed ? "routed-" : "") + source.rawValue }
    }

    nonisolated(unsafe) static var fc1Choice: (role: Role, variant: Variant)? = nil
    nonisolated(unsafe) static var fc2Choice: (role: Role, variant: Variant)? = nil

    /// `pieces`: the copy's codes as the target's expert groups w0 ... (`NemotronZSkip.copyPieces`, EP experts
    /// each); the tile's expert picks its group as the verify slot kernels do.
    private static func body(routed: Bool, source: Source, pieces: Bool = false) -> String {
        let make: String
        switch source {
        case .bank: make = "NemotronSeedBank<T> b{(const device uint8_t*)w, scales, biases, K};"
        case .dict: make = "NemotronSeedDict<T> b{(const device uint8_t*)w, scales, biases, K};"
        case .copy: make = !pieces ? "NemotronSeedCopy<T> b{w, scales, biases, K, 0};"
            : "const uint nmz_piece = uint(tiles[threadgroup_position_in_grid.y * 3 + 2]) / EP;\n"
                + "NemotronSeedCopy<T> b{\(NemotronZSkip.pieceSelect("w", "nmz_piece")), scales, biases, K, nmz_piece * EP};"
        }
        let rows = routed ? "routed_rows" : "(const device uint32_t*)tiles"
        func call(_ bm: String) -> String {
            """
                    nemotron_seed_gather_impl<T, \(bm), BN, BK, WM, WN, DIRECT_B, ALIGN_N, RELU2, \(routed)>(
                        x, \(rows), b, tiles, y, N, K, Ws,
                        threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup,
                        thread_index_in_simdgroup);
            """
        }
        return """
                threadgroup T Ws[DIRECT_B ? 1 : BN * (BK + 16 / sizeof(T))];
                \(make)
                if constexpr (32 / WM >= 16) {
                if (BM == 64 && tiles[threadgroup_position_in_grid.y * 3 + 1] <= 32) {
            \(call("32"))
                } else {
            \(call("BM"))
                }
                } else {
            \(call("BM"))
                }
            """
    }

    private static func makeKernel(routed: Bool, source: Source, pieces: Bool = false) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: "nemotron_seed_gather_\(routed ? "routed_" : "")\(source.rawValue)"
                + (pieces ? "_p\(NemotronZSkip.copyPieces)" : ""),
            inputNames: ["x"] + (pieces ? NemotronZSkip.pieceInputs("w") : ["w"]) + ["scales", "biases", "tiles"]
                + (routed ? ["routed_rows"] : []) + ["N", "K"],
            outputNames: ["y"],
            source: body(routed: routed, source: source, pieces: pieces),
            header: nemotronRunGatherHeader + "\n" + nemotronSeedRetileSource)
    }

    private static let routedBank = makeKernel(routed: true, source: .bank)
    private static let routedDict = makeKernel(routed: true, source: .dict)
    private static let plainBank = makeKernel(routed: false, source: .bank)
    private static let plainCopy = makeKernel(routed: false, source: .copy)
    private static let plainCopyPieces = makeKernel(routed: false, source: .copy, pieces: true)

    /// One retiled launch; nil wherever the incumbent's guards would not admit it.
    static func launch(_ v: Variant, role: Role, x: MLXArray, lin: QuantizedSwitchLinear,
                       rows: MLXArray?, tiles: NemotronHRunTiles) -> MLXArray? {
        guard NemotronHPrefillX5.runGather, lin.mode == .affine, lin.bits == 4, lin.groupSize == 64,
              x.ndim == 3, x.dim(1) == 1, x.dtype == .bfloat16,
              !role.relu2 || MLX.maximum(x, MLXArray(0)).dtype == x.dtype
        else { return nil }
        let ops = lin.nemotronGatherOperands(for: x.dtype)
        guard ops.weight.dtype == .uint32, ops.weight.ndim == 3 else { return nil }
        let n = tiles.rows, k = x.dim(-1), experts = ops.weight.dim(0), nOut = ops.weight.dim(1)
        guard n >= 16, n / experts >= 4, tiles.bm == nemotronGatherTileRows(rows: n, experts: experts),
              ops.weight.dim(2) * 8 == k, k % 64 == 0, nOut % 32 == 0
        else { return nil }
        if role.routed {
            guard let rows, rows.ndim == 1, rows.dtype == .uint32, rows.size == n else { return nil }
        } else {
            guard x.dim(0) == n else { return nil }
        }
        let kernel: MLXFast.MLXFastKernel
        let operands: [MLXArray]
        var ep: [(String, any KernelTemplateArg)] = []
        switch role.source {
        case .bank, .dict:
            guard !ops.hasLinearBias, let biases = ops.biases, ops.scales.dtype == x.dtype,
                  biases.dtype == x.dtype, ops.scales.shape == [experts, nOut, k / 64],
                  biases.shape == ops.scales.shape
            else { return nil }
            if role.source == .dict {
                guard role.routed, let d = NemotronSBDict.dict(lin), d.index.shape == ops.scales.shape,
                      d.index.dtype == .uint16, d.table.dtype == .float32
                else { return nil }
                (kernel, operands) = (routedDict, [ops.weight, d.index, d.table])
            } else {
                (kernel, operands) = (role.routed ? routedBank : plainBank, [ops.weight, ops.scales, biases])
            }
        case .copy:
            guard !role.routed, !role.relu2, nOut % 128 == 0, let copy = NemotronZSkip.copy(for: lin), copy.dict,
                  case let p = copy.codes.count, p == 1 || p == NemotronZSkip.copyPieces, experts % p == 0,
                  copy.codes.allSatisfy({ $0.shape == [experts / p, nOut / 128, k, 32] && $0.dtype == .uint16 }),
                  copy.scales.shape == [experts, nOut / 128, k / 64, 128], copy.scales.dtype == .uint16,
                  copy.biases.dtype == .float32
            else { return nil }
            (kernel, operands) = (p > 1 ? plainCopyPieces : plainCopy, copy.arrays)
            ep = p > 1 ? [("EP", experts / p)] : []
        }
        var inputs: [any ScalarOrArray] = [x] + operands + [tiles.tiles]
        if role.routed, let rows { inputs.append(rows) }
        inputs += [nOut, k]
        return kernel(
            inputs,
            template: [("T", x.dtype), ("BM", tiles.bm), ("BN", v.bn), ("BK", v.bk), ("WM", v.wm), ("WN", v.wn),
                       ("DIRECT_B", v.direct), ("ALIGN_N", nOut % v.bn == 0), ("RELU2", role.relu2)] + ep,
            grid: (((nOut + v.bn - 1) / v.bn) * v.threads, tiles.count, 1), threadGroup: (v.threads, 1, 1),
            outputShapes: [[n, 1, nOut]], outputDTypes: [x.dtype])[0]
    }

    // Dispatch: the adopted variant, only for the incumbent launch it qualified against.

    /// From `NemotronPrefillDirectFC1.apply` (already request prefill only).
    static func fc1Routed(_ x: MLXArray, _ lin: QuantizedSwitchLinear, rows: MLXArray,
                          tiles: NemotronHRunTiles, dict: Bool) -> MLXArray? {
        if let e = NemotronSeedRetileExt.apply(0, routed: true, source: dict ? .dict : .bank, x: x, lin: lin,
                                               rows: rows, tiles: tiles) { return e }
        guard let c = fc1Choice, c.role.routed, c.role.source == (dict ? .dict : .bank) else { return nil }
        return launch(c.variant, role: c.role, x: x, lin: lin, rows: rows, tiles: tiles)
    }

    /// The fc1 run gather on the expanded rows (`nemotronRunGather(xs, relu2: true)`).
    static func fc1Plain(_ xs: MLXArray, _ lin: QuantizedSwitchLinear, tiles: NemotronHRunTiles) -> MLXArray? {
        guard NemotronHPrefillX5.active else { return nil }
        if let e = NemotronSeedRetileExt.apply(0, routed: false, source: .bank, x: xs, lin: lin,
                                               rows: nil, tiles: tiles) { return e }
        guard let c = fc1Choice, !c.role.routed, c.role.source == .bank else { return nil }
        return launch(c.variant, role: c.role, x: xs, lin: lin, rows: nil, tiles: tiles)
    }

    /// From `NemotronPrefillFC2Copy.apply` (already request prefill only).
    static func fc2Copy(_ x: MLXArray, _ lin: QuantizedSwitchLinear, tiles: NemotronHRunTiles) -> MLXArray? {
        if let e = NemotronSeedRetileExt.apply(1, routed: false, source: .copy, x: x, lin: lin,
                                               rows: nil, tiles: tiles) { return e }
        guard let c = fc2Choice, c.role.source == .copy else { return nil }
        return launch(c.variant, role: c.role, x: x, lin: lin, rows: nil, tiles: tiles)
    }

    /// The fc2 run gather on the bank (`nemotronRunGather(y)`).
    static func fc2Plain(_ x: MLXArray, _ lin: QuantizedSwitchLinear, tiles: NemotronHRunTiles) -> MLXArray? {
        guard NemotronHPrefillX5.active else { return nil }
        if let e = NemotronSeedRetileExt.apply(1, routed: false, source: .bank, x: x, lin: lin,
                                               rows: nil, tiles: tiles) { return e }
        guard let c = fc2Choice, c.role.source == .bank else { return nil }
        return launch(c.variant, role: c.role, x: x, lin: lin, rows: nil, tiles: tiles)
    }

    /// The launch the seed makes today for `role`.
    static func incumbent(_ role: Role, x: MLXArray, lin: QuantizedSwitchLinear, rows: MLXArray?,
                                  tiles: NemotronHRunTiles) -> MLXArray? {
        switch role.source {
        case .copy:
            return NemotronPrefillFC2Copy.launch(x, lin, tiles: tiles)
        case .bank, .dict:
            if role.routed, let rows {
                return NemotronPrefillDirectFC1.launch(x, lin, rows: rows, tiles: tiles, useDict: role.source == .dict)
            }
            return nemotronRunGather(x, lin, tiles: tiles, relu2: role.relu2)
        }
    }

    private struct Case {
        let tiles: NemotronHRunTiles
        let rows: MLXArray
        let x: MLXArray
        let xs: MLXArray
        let act: MLXArray
    }

    static func tune(_ target: Module) {
        fc1Choice = nil
        fc2Choice = nil
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron seed retile: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.runGather, NemotronPrefillCountingSort.verified else {
            return report("off (setting)")
        }
        let moes = target.modules().compactMap { $0 as? NemotronHMoE }
        guard let moe = moes.first, moes.allSatisfy({
            $0.switchMLP.fc1 is QuantizedSwitchLinear && $0.switchMLP.fc2 is QuantizedSwitchLinear
        }), let fc2First = moe.switchMLP.fc2 as? QuantizedSwitchLinear
        else { return report("off (no quantized MoE)") }
        let experts = moe.switchMLP.numExperts, topK = moe.numExpertsPerTok
        let width = moe.switchMLP.inputDims
        let hidden = fc2First.nemotronGatherOperands(for: .bfloat16).weight.dim(2) * 8
        let roles = [
            NemotronPrefillDirectFC1.verified
                ? Role(routed: true, source: NemotronPrefillDirectFC1.dictVerified ? .dict : .bank, relu2: true)
                : Role(routed: false, source: .bank, relu2: true),
            NemotronPrefillFC2Copy.verified
                ? Role(routed: false, source: .copy, relu2: false)
                : Role(routed: false, source: .bank, relu2: false),
        ]
        func linear(_ m: NemotronHMoE, _ r: Int) -> QuantizedSwitchLinear {
            (r == 0 ? m.switchMLP.fc1 : m.switchMLP.fc2) as! QuantizedSwitchLinear
        }
        func input(_ r: Int, _ c: Case) -> (MLXArray, MLXArray?) {
            r == 0 ? (roles[0].routed ? (c.x, c.rows) : (c.xs, nil)) : (c.act, nil)
        }

        // Seeded routes and inputs (the fc2 copy qualify's cases; case 2 has
        // ReLU2-like rows, mostly exact zeros).
        var cases: [Case] = []
        do {
            cases = try withError { () -> [Case] in
                var made: [Case] = []
                for (index, (tokens, skew)) in [(488, Float(2)), (488, 0), (488, 9), (300, 4), (97, 1)].enumerated() {
                    let seed = UInt64(0x5eed_0000 + index)
                    let logits = MLXRandom.normal([tokens, experts], key: MLXRandom.key(seed))
                        + skew * MLXRandom.normal([1, experts], key: MLXRandom.key(seed + 1))
                    let flat = argSort(-logits, axis: -1)[0..., ..<topK].asType(.uint32).flattened()
                    guard let sorted = nemotronCountingSortChunked(flat, experts: experts, k: topK) else { return [] }
                    var x = MLXRandom.normal([tokens, 1, width], scale: 2, key: MLXRandom.key(seed + 2))
                    var act = MLXRandom.normal([tokens * topK, 1, hidden], scale: 2, key: MLXRandom.key(seed + 3))
                    if index == 2 {
                        x = MLX.maximum(x - 1.5, MLXArray(Float(0)))
                        x = x * x
                        act = MLX.maximum(act - 1.5, MLXArray(Float(0)))
                        act = act * act
                    }
                    let xb = x.asType(.bfloat16)
                    let c = Case(tiles: sorted.tiles, rows: sorted.orderK, x: xb, xs: xb[sorted.orderK],
                                 act: act.asType(.bfloat16))
                    eval([c.tiles.tiles, c.rows, c.x, c.xs, c.act])
                    made.append(c)
                }
                return made
            }
        } catch {
            return report("off (inputs: \(error))")
        }
        guard cases.count == 5 else { return report("off (counting sort unavailable)") }

        // Qualify: bit for bit against the incumbent; any mismatch or error drops a variant.
        var alive: [[Variant]] = [variants, variants]
        var dropped: [[String]] = [[], []]
        for r in 0 ..< 2 {
            caseLoop: for (ci, c) in cases.enumerated() {
                for m in (ci == 0 ? moes : [moe]) {
                    let lin = linear(m, r)
                    let (x, rows) = input(r, c)
                    let reference: MLXArray?
                    do {
                        reference = try withError { () -> MLXArray? in
                            guard let out = incumbent(roles[r], x: x, lin: lin, rows: rows, tiles: c.tiles)
                            else { return nil }
                            eval(out)
                            return out
                        }
                    } catch {
                        reference = nil
                    }
                    guard let reference else {
                        dropped[r].append("incumbent unavailable")
                        alive[r] = []
                        break caseLoop
                    }
                    alive[r] = alive[r].filter { v in
                        do {
                            let same = try withError { () -> Bool in
                                guard let out = launch(v, role: roles[r], x: x, lin: lin, rows: rows, tiles: c.tiles)
                                else { return false }
                                return (out.view(dtype: .uint16) .== reference.view(dtype: .uint16)).all()
                                    .item(Bool.self)
                            }
                            if !same { dropped[r].append("\(v.name) differs@\(ci)") }
                            return same
                        } catch {
                            dropped[r].append("\(v.name) error")
                            return false
                        }
                    }
                    if alive[r].isEmpty { break caseLoop }
                }
            }
        }

        // Time the incumbent and the survivors round robin on the 488-token
        // case, first four MoE layers per repetition; median ms per layer.
        let layers = Array(moes.prefix(4))
        var medians: [[(Variant?, Double)]] = [[], []]
        var choice: [Variant?] = [nil, nil]
        for r in 0 ..< 2 where !alive[r].isEmpty {
            let candidates: [Variant?] = [nil] + alive[r].map { Optional($0) }
            let (x, rows) = input(r, cases[0])
            var samples = Array(repeating: [Double](), count: candidates.count)
            do {
                try withError {
                    for rep in 0 ..< 30 {
                        for (i, candidate) in candidates.enumerated() {
                            let start = DispatchTime.now().uptimeNanoseconds
                            let outs = layers.compactMap { m -> MLXArray? in
                                let lin = linear(m, r)
                                if let candidate {
                                    return launch(candidate, role: roles[r], x: x, lin: lin, rows: rows,
                                                  tiles: cases[0].tiles)
                                }
                                return incumbent(roles[r], x: x, lin: lin, rows: rows, tiles: cases[0].tiles)
                            }
                            eval(outs)
                            Stream.gpu.synchronize()
                            let elapsed = DispatchTime.now().uptimeNanoseconds - start
                            if rep >= 5, outs.count == layers.count {
                                samples[i].append(Double(elapsed) / 1e6 / Double(layers.count))
                            }
                        }
                    }
                }
            } catch {
                dropped[r].append("timing error")
                continue
            }
            guard samples.allSatisfy({ $0.count == 25 }) else { continue }
            medians[r] = zip(candidates, samples).map { candidate, s in
                (candidate, s.sorted()[s.count / 2])
            }
            let base = medians[r][0].1
            if let forced, let f = alive[r].first(where: { $0.name == forced }) {
                choice[r] = f
            } else if let best = medians[r].dropFirst().min(by: { $0.1 < $1.1 }), best.1 <= 0.995 * base {
                choice[r] = best.0
            }
        }
        fc1Choice = choice[0].map { (roles[0], $0) }
        fc2Choice = choice[1].map { (roles[1], $0) }

        func ms(_ r: Int, _ v: Variant?) -> String {
            guard let m = medians[r].first(where: { $0.0 == v }) else { return "n/a" }
            return String(format: "%.2f", m.1)
        }
        func detail(_ r: Int) -> String {
            let times = medians[r].dropFirst().map { "\($0.0!.name) \(String(format: "%.2f", $0.1))" }
            return "\(roles[r].label) \(alive[r].count)/\(variants.count) bitwise"
                + (times.isEmpty ? "" : " [" + times.joined(separator: ", ") + "]")
                + (dropped[r].isEmpty ? "" : " dropped: " + dropped[r].prefix(8).joined(separator: ", "))
        }
        report("\(choice[0]?.name ?? "V0")/\(choice[1]?.name ?? "V0") fc1 \(ms(0, choice[0])) ms fc2 \(ms(1, choice[1])) ms"
            + " vs V0 fc1 \(ms(0, nil)) ms fc2 \(ms(1, nil)) ms (per layer, 488 tokens, median of 25;"
            + " fc1 \(detail(0)); fc2 \(detail(1)))")
    }
}

// MARK: - Seed dense retile

/// The seed's dense run gathers (one "expert", `tiles[t]` = (first row, rows <=
/// BM, 0) over the request's rows) with free tiling, appended to
/// `nemotronRunGatherHeader` and `nemotronSeedRetileSource`. For every output
/// fragment pair the same `BaseNAXFrag::mma` (matmul2d(16, 32, 16),
/// multiply_accumulate) runs over k chunks 0, 16, ..., K - 16 in order, on A
/// fragments loaded by the incumbents' statements and B fragments read from a
/// threadgroup block holding the incumbents' BF16 dequantized value at each
/// (n, k); the typed store is the incumbents'. Free: BM, BN, BK, the SIMD-group
/// grid WM x WN (SM, SN multiples of 16 and 32, so rows keep their 16-row
/// fragments from the tile's first row and each mma pairs columns 32c ...
/// 32c + 31), which thread dequantizes which B value, and DB: two threadgroup
/// stages with one barrier per K block instead of one stage with two.
let nemotronSeedDenseSource = #"""
// B source: the bank's codes with (scale, bias) through the pair dictionary,
// as QuantizedBlockLoaderDict (the FP32 table holds exact BF16 widenings, so
// s and b are float(T(table)) bit for bit). Thread t of NT dequantizes KPT
// consecutive k of one output row (the stock loader's split), reading the
// codes as 32-bit words (little endian: byte j = bits 8j ... 8j + 7).
template <typename T>
struct NemotronSeedDenseDict {
  const device uint8_t* w;
  const device uint16_t* sidx;
  const device float* table;
  int K;

  template <int BN, int BK, int NT, bool ALIGN_N>
  METAL_FUNC void fill(threadgroup T* ws, const int t, const int y_col, const int N,
                       const int kb, const int klim) const thread {
    constexpr int BK_padded = BK + 16 / sizeof(T);
    constexpr int KPT = BN * BK / NT;
    static_assert(KPT % 8 == 0 && KPT <= 64 && 64 % KPT == 0, "dict fill split");
    const int lr = t / (BK / KPT);
    const int lk = (t % (BK / KPT)) * KPT;
    const int n = y_col + lr;
    threadgroup T* dst = ws + lr * BK_padded + lk;
    if ((ALIGN_N || n < N) && lk < klim) {
      const uint pair = uint(sidx[size_t(n) * size_t(K / 64) + size_t((kb + lk) / 64)]);
      const float s = table[2 * pair];
      const float b = table[2 * pair + 1];
      const device uint32_t* src =
          (const device uint32_t*)(w + size_t(n) * size_t(K / 2) + size_t((kb + lk) / 2));
      STEEL_PRAGMA_UNROLL
      for (short q = 0; q < KPT / 8; ++q) {
        const uint word = src[q];
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; ++j) {
          T v0, v1;
          nemotron_seed_deq2(uint8_t((word >> (8 * j)) & 0xffu), s, b, v0, v1);
          dst[q * 8 + 2 * j] = v0;
          dst[q * 8 + 2 * j + 1] = v1;
        }
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < KPT; ++j) {
        dst[j] = T(0);
      }
    }
  }
};

// B source: the shared down's tile-major decode copy (codes [N / 128, K, 32]
// uint16, four outputs per word; BF16 scales and biases [N / 128, K / 64, 128]),
// as the shared down copy loader: thread t of NT holds four consecutive outputs
// (one code word lane) and CK consecutive k, puts each k pair back into the
// bank's byte and widens the group's scale and bias to float.
template <typename T>
struct NemotronSeedDenseCopy {
  const device uint16_t* codes;
  const device T* sk;
  const device T* bk;
  int K;

  template <int BN, int BK, int NT, bool ALIGN_N>
  METAL_FUNC void fill(threadgroup T* ws, const int t, const int y_col, const int N,
                       const int kb, const int klim) const thread {
    constexpr int BK_padded = BK + 16 / sizeof(T);
    constexpr int CK = BK * (BN / 4) / NT;
    static_assert(CK % 2 == 0 && CK <= 64 && 64 % CK == 0 && NT % (BN / 4) == 0, "copy fill split");
    const int cm = t % (BN / 4);
    const int cc = t / (BN / 4);
    const int n = y_col + 4 * cm;
    const int k0 = kb + cc * CK;
    threadgroup T* dst = ws + (4 * cm) * BK_padded + cc * CK;
    if ((ALIGN_N || n < N) && cc * CK < klim) {
      const device uint16_t* c =
          codes + (size_t(n / 128) * size_t(K) + size_t(k0)) * 32 + size_t((n % 128) / 4);
      const size_t g = (size_t(n / 128) * size_t(K / 64) + size_t(k0 / 64)) * 128 + size_t(n % 128);
      STEEL_PRAGMA_UNROLL
      for (short o = 0; o < 4; o++) {
        const float s = float(sk[g + o]);
        const float b = float(bk[g + o]);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < CK; j += 2) {
          const uint lo = (uint(c[j * 32]) >> (4 * o)) & 0xfu;
          const uint hi = (uint(c[(j + 1) * 32]) >> (4 * o)) & 0xfu;
          T v0, v1;
          nemotron_seed_deq2(uint8_t(lo | (hi << 4)), s, b, v0, v1);
          dst[o * BK_padded + j] = v0;
          dst[o * BK_padded + j + 1] = v1;
        }
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (short o = 0; o < 4; o++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < CK; ++j) {
          dst[o * BK_padded + j] = T(0);
        }
      }
    }
  }
};

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool DB,
    bool ALIGN_N,
    bool RELU2,
    class BSrc>
METAL_FUNC void nemotron_seed_dense_impl(
    const device T* x,
    BSrc bsrc,
    const device int* tiles,
    device T* y,
    const int N,
    const int K,
    threadgroup T* Ws,
    uint3 tid,
    uint simd_group_id,
    uint simd_lane_id) {
  const int tile_rows = tiles[tid.y * 3 + 1];
  if (tile_rows <= 0) {
    return;
  }
  const int row0 = tiles[tid.y * 3 + 0];

  constexpr bool transpose = true;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int STAGE = BN * BK_padded;
  constexpr int NT = WM * WN * 32;
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;
  static_assert(SM % 16 == 0 && TN % 2 == 0, "the run gather's fragment pairs");
  static_assert(BK % 64 == 0, "B blocks start on a quantization group");

  const int y_col = tid.x * BN;
  const short tgp_bm = short(min(BM, tile_rows));
  const short tgp_bn = ALIGN_N ? BN : short(min(BN, N - y_col));
  y += size_t(row0) * N + size_t(y_col);

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  const short sgp_sm = min(SM, short(max(0, int(tgp_bm) - tm)));
  const short sgp_sn = ALIGN_N ? SN : min(SN, short(max(0, (N - (y_col + tn)))));
  const bool is_unaligned_sm = (sgp_sm != SM);
  const bool is_unaligned_bn = ALIGN_N ? false : (tgp_bn != BN);

  const short m_lo_lim = 0;
  const short m_hi_lim = min(int(sgp_sm), max(0, int(tgp_bm) - tm));
  const bool sg_active = m_hi_lim > m_lo_lim;

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  const device T* xn = x + (size_t(row0) + size_t(tm)) * size_t(K);
  const int t = int(simd_group_id) * 32 + int(simd_lane_id);
  // K % 64 == 0: every block is whole groups and whole 32-k steps.
  const int nb = (K + BK - 1) / BK;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    constexpr bool aligned_m = decltype(kAlignedM)::value;
    auto fill = [&](threadgroup T* ws, const int b) {
      bsrc.template fill<BN, BK, NT, ALIGN_N>(ws, t, y_col, N, b * BK, min(BK, K - b * BK));
    };
    auto compute = [&](threadgroup T* ws, const int b) {
      const int kb = b * BK;
      const int klim = min(BK, K - kb);
      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < klim; kk1 += SK) {
        if (sg_active) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<T, TN, TK> Btile;

          volatile int compiler_barrier;

          if constexpr (aligned_m) {
            Atile.load(xn + kb + kk1, K);
          } else {
            Atile.load_safe(xn + kb + kk1, K, short2(SK, sgp_sm));
          }

          Btile.template load<T, BK_padded, 1>(ws + tn * BK_padded + kk1);

          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<false>{},
              Btile,
              metal::bool_constant<transpose>{});

          (void)compiler_barrier;
        }
      }
    };
    if constexpr (DB) {
      // Stage b & 1 holds block b. Block b + 1 is written while block b is
      // read; the barrier closing iteration b - 1 orders both (its stage was
      // last read in iteration b - 1, and block b was written there).
      fill(Ws, 0);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int b = 0; b < nb; b++) {
        if (b + 1 < nb) {
          fill(Ws + ((b + 1) & 1) * STAGE, b + 1);
        }
        compute(Ws + (b & 1) * STAGE, b);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    } else {
      for (int b = 0; b < nb; b++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fill(Ws, b);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        compute(Ws, b);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  });

  if (ALIGN_N || !is_unaligned_bn) {
    if (m_lo_lim == 0 && m_hi_lim == SM) {
      Dtile.template store<RELU2>(y + tm * N + tn, N);
    } else {
      Dtile.template store_slice<RELU2>(
          y + tm * N + tn, N, short2(0, m_lo_lim), short2(SN, m_hi_lim));
    }
  } else {
    Dtile.template store_slice<RELU2>(
        y + tm * N + tn,
        N,
        short2(0, m_lo_lim),
        short2(sgp_sn, m_hi_lim));
  }
}
"""#

/// Seed-only (request prefill, `NemotronHPrefillX5.active`) dense GEMMs through a
/// retiled run gather (`nemotron_seed_dense_impl`) chosen once at load, per kind:
/// Mamba `in_proj` / `out_proj` and the shared `up_proj` (typed ReLU2) through
/// their pair dictionaries (incumbent `NemotronPrefillDenseDict.launch`), and the
/// shared down through its decode copy (incumbent
/// `NemotronPrefillSharedDownCopy.launch`). Every variant keeps each output's
/// `BaseNAXFrag::mma` sequence (`nemotronSeedDenseSource`). `tune` (load time,
/// after the routed retile) compares each variant with the incumbent bit for bit
/// on seeded inputs (488 rows on every layer of the kind; 488 rows x 40, 300, 97
/// and 17 rows on the first; ReLU2-like rows where the kind sees them), drops any
/// mismatch or Metal error, times the survivors and the incumbent round robin on
/// every layer of the kind per repetition (3 warm-up and 12 timed repetitions,
/// median), and adopts the fastest only if it is at least 3% faster. Direct-B
/// variants are left out: forced on every seed role they cost ~40 ms (331295a3).
/// NEMOTRON_SEED_DENSE_RETILE=0 keeps the incumbents;
/// NEMOTRON_SEED_DENSE_RETILE_FORCE=<variant> adopts that variant wherever it
/// qualifies.
enum NemotronSeedDenseRetile {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_SEED_DENSE_RETILE"] != "0"
    static let forced = ProcessInfo.processInfo.environment["NEMOTRON_SEED_DENSE_RETILE_FORCE"]

    struct Variant: Sendable, Equatable {
        let name: String
        let bm: Int, bn: Int, bk: Int, wm: Int, wn: Int
        let db: Bool
        var threads: Int { wm * wn * 32 }
        var threadgroupBytes: Int { (db ? 2 : 1) * bn * (bk + 8) * 2 }

        init(_ bm: Int, _ bn: Int, _ bk: Int, wm: Int, wn: Int, db: Bool = false) {
            (self.bm, self.bn, self.bk, self.wm, self.wn, self.db) = (bm, bn, bk, wm, wn, db)
            name = "\(bm)x\(bn)x\(bk)w\(wm)\(wn)" + (db ? "db" : "")
        }
    }

    /// Name = BM x BN x BK, w WM WN, db for two stages. 64x64x64w22 is the
    /// incumbents' geometry (control); 64x128x64w24 is MLX's NAX bf16 GEMM layout
    /// for Max-class GPUs (32 x 32 per SIMD group); the 128-row tiles share each
    /// dequantized B block between twice the rows. Each fits 32 KB of threadgroup memory.
    static let variants: [Variant] = [
        Variant(64, 64, 64, wm: 2, wn: 2),
        Variant(64, 64, 64, wm: 2, wn: 2, db: true),
        Variant(64, 64, 128, wm: 2, wn: 2),
        Variant(64, 128, 64, wm: 2, wn: 4),
        Variant(128, 64, 64, wm: 4, wn: 2),
        Variant(128, 64, 64, wm: 4, wn: 2, db: true),
        Variant(128, 128, 64, wm: 4, wn: 4),
    ]

    enum Source: String, Sendable { case dict, copy }

    /// Kinds: "in_proj", "out_proj", "up_proj" (pair dictionaries), "shared_down" (decode copy).
    nonisolated(unsafe) static var choice: [String: Variant] = [:]

    private static func makeKernel(_ source: Source) -> MLXFast.MLXFastKernel {
        let make = source == .dict
            ? "NemotronSeedDenseDict<T> b{(const device uint8_t*)w, scales, biases, K};"
            : "NemotronSeedDenseCopy<T> b{w, scales, biases, K};"
        func call(_ bm: String) -> String {
            """
                    nemotron_seed_dense_impl<T, \(bm), BN, BK, WM, WN, DB, ALIGN_N, RELU2>(
                        x, b, tiles, y, N, K, Ws,
                        threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup,
                        thread_index_in_simdgroup);
            """
        }
        // A tile of at most BM / 2 rows runs the BM / 2 body (for BM 64 the
        // incumbents' 32-row bound); either way rows keep their 16-row fragments.
        return MLXFast.metalKernel(
            name: "nemotron_seed_dense_\(source.rawValue)",
            inputNames: ["x", "w", "scales", "biases", "tiles", "N", "K"],
            outputNames: ["y"],
            source: """
                threadgroup T Ws[(DB ? 2 : 1) * BN * (BK + 16 / sizeof(T))];
                \(make)
                if (tiles[threadgroup_position_in_grid.y * 3 + 1] <= BM / 2) {
            \(call("BM / 2"))
                } else {
            \(call("BM"))
                }
            """,
            header: nemotronRunGatherHeader + "\n" + nemotronSeedRetileSource + "\n" + nemotronSeedDenseSource)
    }

    private static let dictKernel = makeKernel(.dict)
    private static let copyKernel = makeKernel(.copy)

    /// One retiled launch over the request's rows (`x`: [1, rows, K]) in tiles of
    /// `v.bm` rows (at least three descriptors, as the incumbents); nil wherever
    /// the incumbent's guards (`NemotronPrefillDenseDict.launch` for .dict,
    /// `NemotronPrefillSharedDownCopy.launch` for .copy) would not admit it.
    static func launch(_ v: Variant, source: Source, x: MLXArray, linear: Linear, relu2: Bool) -> MLXArray? {
        guard NemotronHPrefillX5.runGather, v.threadgroupBytes <= 32768, let q = linear as? QuantizedLinear,
              type(of: linear) == QuantizedLinear.self, q.mode == .affine, q.bits == 4, q.groupSize == 64,
              q.bias == nil, x.ndim == 3, x.dim(0) == 1, x.dim(1) > 8, x.dtype == .bfloat16
        else { return nil }
        let rows = x.dim(1), k = x.dim(2), nOut = q.weight.dim(0)
        guard q.weight.dim(1) * 8 == k, k % 64 == 0, nOut % 32 == 0 else { return nil }
        let kernel: MLXFast.MLXFastKernel
        let operands: [MLXArray]
        switch source {
        case .dict:
            guard !relu2 || MLX.maximum(x, MLXArray(0)).dtype == x.dtype,
                  q.weight.dtype == .uint32, q.weight.ndim == 2, q.scales.shape == [nOut, k / 64],
                  let d = NemotronSBDict.dict(q), d.index.shape == q.scales.shape, d.index.dtype == .uint16,
                  d.table.dtype == .float32
            else { return nil }
            (kernel, operands) = (dictKernel, [q.weight, d.index, d.table])
        case .copy:
            guard !relu2, nOut % 128 == 0, let copy = NemotronSharedDownZSkip.copyArrays(for: linear),
                  copy.codes.shape == [1, nOut / 128, k, 32], copy.codes.dtype == .uint16,
                  copy.scales.shape == [1, nOut / 128, k / 64, 128], copy.scales.dtype == x.dtype,
                  copy.biases.shape == copy.scales.shape, copy.biases.dtype == x.dtype
            else { return nil }
            (kernel, operands) = (copyKernel, [copy.codes, copy.scales, copy.biases])
        }
        let used = (rows + v.bm - 1) / v.bm
        let count = max(used, 3)
        var desc = [Int32]()
        desc.reserveCapacity(count * 3)
        for t in 0 ..< count {
            desc += t < used ? [Int32(t * v.bm), Int32(min(v.bm, rows - t * v.bm)), 0] : [0, 0, 0]
        }
        let tiles = MLXArray(desc).reshaped(count, 3)
        let y = kernel(
            [x.reshaped(rows, 1, k)] + operands + [tiles, nOut, k],
            template: [("T", x.dtype), ("BM", v.bm), ("BN", v.bn), ("BK", v.bk), ("WM", v.wm), ("WN", v.wn),
                       ("DB", v.db), ("ALIGN_N", nOut % v.bn == 0), ("RELU2", relu2)],
            grid: (((nOut + v.bn - 1) / v.bn) * v.threads, count, 1), threadGroup: (v.threads, 1, 1),
            outputShapes: [[rows, 1, nOut]], outputDTypes: [x.dtype])[0]
        return y.reshaped(1, rows, nOut)
    }

    /// From `NemotronPrefillDenseDict.apply` (request prefill, admitted kinds only).
    static func dense(_ x: MLXArray, _ linear: Linear, kind: String, relu2: Bool) -> MLXArray? {
        guard let v = choice[kind], relu2 == (kind == "up_proj") else { return nil }
        return launch(v, source: .dict, x: x, linear: linear, relu2: relu2)
    }

    /// From `NemotronPrefillSharedDownCopy.apply` (request prefill, verified only).
    static func sharedDown(_ x: MLXArray, _ down: Linear) -> MLXArray? {
        guard let v = choice["shared_down"] else { return nil }
        return launch(v, source: .copy, x: x, linear: down, relu2: false)
    }

    private struct Outcome {
        let alive: [Variant]
        let dropped: [String]
        let medians: [(Variant?, Double)]
        let choice: Variant?
    }

    /// Qualifies `variants` against the incumbent (`run(nil, case, layer)`) bit for
    /// bit (every layer in case 0, the first layer in the others; any mismatch or
    /// error drops a variant), then times the incumbent and the survivors round
    /// robin on case 0, every layer per repetition: 3 warm-up and 12 timed
    /// repetitions, median ms per launch.
    private static func evaluate(cases: Int, layers: Int, run: (Variant?, Int, Int) -> MLXArray?) -> Outcome {
        var alive = variants.filter { $0.threadgroupBytes <= 32768 }
        var dropped: [String] = []
        caseLoop: for c in 0 ..< cases {
            for layer in 0 ..< (c == 0 ? layers : 1) {
                let reference = try? withError { () -> MLXArray? in
                    guard let out = run(nil, c, layer) else { return nil }
                    eval(out)
                    return out
                }
                guard let reference else {
                    dropped.append("incumbent unavailable")
                    alive = []
                    break caseLoop
                }
                alive = alive.filter { v in
                    do {
                        let same = try withError { () -> Bool in
                            guard let out = run(v, c, layer) else { return false }
                            return (out.view(dtype: .uint16) .== reference.view(dtype: .uint16)).all()
                                .item(Bool.self)
                        }
                        if !same { dropped.append("\(v.name) differs@\(c)") }
                        return same
                    } catch {
                        dropped.append("\(v.name) error")
                        return false
                    }
                }
                if alive.isEmpty { break caseLoop }
            }
        }
        guard !alive.isEmpty else { return Outcome(alive: alive, dropped: dropped, medians: [], choice: nil) }
        let candidates: [Variant?] = [nil] + alive.map { Optional($0) }
        var samples = Array(repeating: [Double](), count: candidates.count)
        do {
            try withError {
                for rep in 0 ..< 15 {
                    for (i, candidate) in candidates.enumerated() {
                        let start = DispatchTime.now().uptimeNanoseconds
                        let outs = (0 ..< layers).compactMap { run(candidate, 0, $0) }
                        eval(outs)
                        Stream.gpu.synchronize()
                        let elapsed = DispatchTime.now().uptimeNanoseconds - start
                        if rep >= 3, outs.count == layers {
                            samples[i].append(Double(elapsed) / 1e6 / Double(layers))
                        }
                    }
                }
            }
        } catch {
            dropped.append("timing error")
        }
        guard samples.allSatisfy({ $0.count == 12 }) else {
            return Outcome(alive: alive, dropped: dropped, medians: [], choice: nil)
        }
        let medians = zip(candidates, samples).map { candidate, s in (candidate, s.sorted()[s.count / 2]) }
        var pick: Variant? = nil
        if let forced, let f = alive.first(where: { $0.name == forced }) {
            pick = f
        } else if let best = medians.dropFirst().min(by: { $0.1 < $1.1 }), best.1 <= 0.97 * medians[0].1 {
            pick = best.0
        }
        return Outcome(alive: alive, dropped: dropped, medians: medians, choice: pick)
    }

    static func tune(_ target: Module) {
        choice = [:]
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron seed dense retile: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.runGather else { return report("off (setting)") }
        var head: [String] = []
        var detail: [String] = []
        func note(_ kind: String, _ o: Outcome) {
            func ms(_ v: Variant?) -> String {
                o.medians.first(where: { $0.0 == v }).map { String(format: "%.3f", $0.1) } ?? "n/a"
            }
            head.append("\(kind) \(o.choice?.name ?? "V0") \(ms(o.choice)) ms vs V0 \(ms(nil))")
            let times = o.medians.dropFirst().map { "\($0.0!.name) \(String(format: "%.3f", $0.1))" }
            detail.append("\(kind) \(o.alive.count)/\(variants.count) bitwise"
                + (times.isEmpty ? "" : " [" + times.joined(separator: ", ") + "]")
                + (o.dropped.isEmpty ? "" : " dropped: " + o.dropped.prefix(6).joined(separator: ", ")))
        }
        // Rows: 488, 488 x 40, 300, 97, 17; `zeros(case)`: ReLU2-like rows, mostly exact zeros.
        let rowCounts = [488, 488, 300, 97, 17]
        func inputs(_ k: Int, seed: UInt64, zeros: (Int) -> Bool) -> [MLXArray] {
            (try? withError { () -> [MLXArray] in
                rowCounts.enumerated().map { index, rows in
                    var a = MLXRandom.normal([1, rows, k], scale: 2, key: MLXRandom.key(seed + UInt64(index)))
                    if index == 1 { a = a * 40 }
                    if zeros(index) {
                        a = MLX.maximum(a - 1.5, MLXArray(Float(0)))
                        a = a * a
                    }
                    let x = a.asType(.bfloat16)
                    eval(x)
                    return x
                }
            }) ?? []
        }
        for (kindIndex, kind) in ["in_proj", "out_proj", "up_proj"].enumerated() {
            let all = NemotronPrefillDenseDict.linears(target, kind: kind)
            guard NemotronPrefillDenseDict.kinds.contains(kind), let first = all.first else {
                detail.append("\(kind) off (not admitted)")
                continue
            }
            let relu2 = kind == "up_proj"
            let xs = inputs(first.weight.dim(1) * 8, seed: UInt64(0x5eed_de00 + 16 * kindIndex), zeros: { $0 == 3 })
            guard xs.count == rowCounts.count else {
                detail.append("\(kind) off (inputs)")
                continue
            }
            let o = evaluate(cases: xs.count, layers: all.count) { v, c, layer in
                if let v { return launch(v, source: .dict, x: xs[c], linear: all[layer], relu2: relu2) }
                return NemotronPrefillDenseDict.launch(xs[c], all[layer], relu2: relu2)
            }
            if let pick = o.choice { choice[kind] = pick }
            note(kind, o)
        }
        let downs = target.modules().compactMap { ($0 as? NemotronHMoE)?.sharedExperts?.downProj }
        if NemotronPrefillSharedDownCopy.verified, let first = downs.first {
            // The shared up's ReLU2 output, except the scaled case.
            let xs = inputs(first.weight.dim(1) * 8, seed: 0x5eed_de80, zeros: { $0 != 1 })
            if xs.count == rowCounts.count {
                let o = evaluate(cases: xs.count, layers: downs.count) { v, c, layer in
                    if let v { return launch(v, source: .copy, x: xs[c], linear: downs[layer], relu2: false) }
                    return NemotronPrefillSharedDownCopy.launch(xs[c], downs[layer])
                }
                if let pick = o.choice { choice["shared_down"] = pick }
                note("shared_down", o)
            } else {
                detail.append("shared_down off (inputs)")
            }
        } else {
            detail.append("shared_down off (not verified)")
        }
        report(head.joined(separator: ", ") + " (per launch, 488 rows, median of 12 over every layer; "
            + detail.joined(separator: "; ") + ")")
    }
}

// MARK: - Seed routed retile, extended forms

/// Extended forms of the routed run gather (`nemotron_seed_gather_impl`'s
/// arithmetic, appended to `nemotronRunGatherHeader` and
/// `nemotronSeedRetileSource`): the same `BaseNAXFrag::mma` per output over k
/// chunks 0, 16, ..., K - 16 in order, on the incumbents' A fragments (routed
/// rows or sorted rows) and threadgroup B blocks holding the incumbents' BF16
/// dequantized values, and the incumbents' typed store. New here: B loaders that
/// read the codes as words (bank, pair dictionary) or as the decode copy
/// loader's four-outputs-per-word lanes, and the K-block schedule MODE:
///   0: one stage, load and dequantize between two barriers (the incumbents');
///   1: two stages, block b + 1 dequantized before block b's mma, one barrier;
///   2: two stages, block b + 1's codes and (scale, bias) loaded into registers
///      before block b's mma and dequantized after it, one barrier;
///   3: one stage, as 2 but with two barriers (fits BK 128 or BN 128).
let nemotronSeedExtSource = #"""
// The codes and (scale, bias) one thread dequantizes for one K block.
template <int W>
struct NemotronSeedExtRaw {
  uint w[W];
  float2 sb[4];
};

// B source: the bank's codes [E, N, K / 8] with BF16 scales and biases
// [E, N, K / 64] (DICT false: S = B = T, widened as in dequantize), or with the
// pair dictionary (DICT true: S = uint16 index in the scales' layout, B = the
// FP32 table of exact BF16 widenings, as QuantizedBlockLoaderDict). Thread t of
// NT holds KPT consecutive k of one output row (the stock loader's split).
template <typename T, typename S, typename B, bool DICT>
struct NemotronSeedExtRows {
  const device uint8_t* w;
  const device S* scales;
  const device B* biases;
  int K;

  template <int BN, int BK, int NT>
  using Raw = NemotronSeedExtRaw<BN * BK / NT / 8>;

  METAL_FUNC void expert(const uint index, const int N) thread {
    w += size_t(index) * size_t(N) * size_t(K / 2);
    scales += size_t(index) * size_t(N) * size_t(K / 64);
    if (!DICT) {
      biases += size_t(index) * size_t(N) * size_t(K / 64);
    }
  }

  template <int BN, int BK, int NT, bool ALIGN_N>
  METAL_FUNC void load(thread Raw<BN, BK, NT>& r, const int t, const int y_col, const int N,
                       const int kb, const int klim) const thread {
    constexpr int KPT = BN * BK / NT;
    static_assert(KPT % 8 == 0 && KPT <= 64 && 64 % KPT == 0, "rows fill split");
    const int lr = t / (BK / KPT);
    const int lk = (t % (BK / KPT)) * KPT;
    const int n = y_col + lr;
    if ((ALIGN_N || n < N) && lk < klim) {
      const size_t gi = size_t(n) * size_t(K / 64) + size_t((kb + lk) / 64);
      if (DICT) {
        const uint pair = uint(scales[gi]);
        r.sb[0] = float2(float(biases[2 * pair]), float(biases[2 * pair + 1]));
      } else {
        r.sb[0] = float2(float(scales[gi]), float(biases[gi]));
      }
      const device uint32_t* src =
          (const device uint32_t*)(w + size_t(n) * size_t(K / 2) + size_t((kb + lk) / 2));
      STEEL_PRAGMA_UNROLL
      for (short q = 0; q < KPT / 8; ++q) {
        r.w[q] = src[q];
      }
    }
  }

  template <int BN, int BK, int NT, bool ALIGN_N>
  METAL_FUNC void store(threadgroup T* ws, const thread Raw<BN, BK, NT>& r, const int t, const int y_col,
                        const int N, const int kb, const int klim) const thread {
    constexpr int BK_padded = BK + 16 / sizeof(T);
    constexpr int KPT = BN * BK / NT;
    const int lr = t / (BK / KPT);
    const int lk = (t % (BK / KPT)) * KPT;
    const int n = y_col + lr;
    threadgroup T* dst = ws + lr * BK_padded + lk;
    if ((ALIGN_N || n < N) && lk < klim) {
      STEEL_PRAGMA_UNROLL
      for (short q = 0; q < KPT / 8; ++q) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; ++j) {
          T v0, v1;
          nemotron_seed_deq2(uint8_t((r.w[q] >> (8 * j)) & 0xffu), r.sb[0].x, r.sb[0].y, v0, v1);
          dst[q * 8 + 2 * j] = v0;
          dst[q * 8 + 2 * j + 1] = v1;
        }
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < KPT; ++j) {
        dst[j] = T(0);
      }
    }
  }
};

// B source: the routed fc2's tile-major zero-skip decode copy (codes
// [E, N / 128, K, 32] uint16, four outputs per word; pair index
// [E, N / 128, K / 64, 128]; FP32 pair table), as the fc2 copy loader: thread t
// of NT holds four consecutive outputs (one code word lane) and CK consecutive k.
template <typename T>
struct NemotronSeedExtCopy {
  const device uint16_t* codes;
  const device uint16_t* sidx;
  const device float* table;
  int K;

  template <int BN, int BK, int NT>
  using Raw = NemotronSeedExtRaw<BK * (BN / 4) / NT>;

  METAL_FUNC void expert(const uint index, const int N) thread {
    codes += size_t(index) * size_t(N / 128) * size_t(K) * 32;
    sidx += size_t(index) * size_t(N / 128) * size_t(K / 64) * 128;
  }

  template <int BN, int BK, int NT, bool ALIGN_N>
  METAL_FUNC void load(thread Raw<BN, BK, NT>& r, const int t, const int y_col, const int N,
                       const int kb, const int klim) const thread {
    constexpr int CK = BK * (BN / 4) / NT;
    static_assert(CK % 2 == 0 && CK <= 64 && 64 % CK == 0 && NT % (BN / 4) == 0, "copy fill split");
    const int cm = t % (BN / 4);
    const int cc = t / (BN / 4);
    const int n = y_col + 4 * cm;
    const int k0 = kb + cc * CK;
    if ((ALIGN_N || n < N) && cc * CK < klim) {
      const device uint16_t* c =
          codes + (size_t(n / 128) * size_t(K) + size_t(k0)) * 32 + size_t((n % 128) / 4);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < CK; ++j) {
        r.w[j] = uint(c[j * 32]);
      }
      const size_t g = (size_t(n / 128) * size_t(K / 64) + size_t(k0 / 64)) * 128 + size_t(n % 128);
      STEEL_PRAGMA_UNROLL
      for (short o = 0; o < 4; o++) {
        const uint pair = uint(sidx[g + o]);
        r.sb[o] = float2(table[2 * pair], table[2 * pair + 1]);
      }
    }
  }

  template <int BN, int BK, int NT, bool ALIGN_N>
  METAL_FUNC void store(threadgroup T* ws, const thread Raw<BN, BK, NT>& r, const int t, const int y_col,
                        const int N, const int kb, const int klim) const thread {
    constexpr int BK_padded = BK + 16 / sizeof(T);
    constexpr int CK = BK * (BN / 4) / NT;
    const int cm = t % (BN / 4);
    const int cc = t / (BN / 4);
    const int n = y_col + 4 * cm;
    threadgroup T* dst = ws + (4 * cm) * BK_padded + cc * CK;
    if ((ALIGN_N || n < N) && cc * CK < klim) {
      STEEL_PRAGMA_UNROLL
      for (short o = 0; o < 4; o++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < CK; j += 2) {
          const uint lo = (r.w[j] >> (4 * o)) & 0xfu;
          const uint hi = (r.w[j + 1] >> (4 * o)) & 0xfu;
          T v0, v1;
          nemotron_seed_deq2(uint8_t(lo | (hi << 4)), r.sb[o].x, r.sb[o].y, v0, v1);
          dst[o * BK_padded + j] = v0;
          dst[o * BK_padded + j + 1] = v1;
        }
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (short o = 0; o < 4; o++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < CK; ++j) {
          dst[o * BK_padded + j] = T(0);
        }
      }
    }
  }
};

// tiles[t] = (first sorted row, rows <= BM, expert), as the run gather.
template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    int MODE,
    bool ALIGN_N,
    bool RELU2,
    bool ROUTED,
    class BSrc>
METAL_FUNC void nemotron_seed_ext_impl(
    const device T* x,
    const device uint32_t* routed_rows,
    BSrc bsrc,
    const device int* tiles,
    device T* y,
    const int N,
    const int K,
    threadgroup T* Ws,
    uint3 tid,
    uint simd_group_id,
    uint simd_lane_id) {
  const int tile_rows = tiles[tid.y * 3 + 1];
  if (tile_rows <= 0) {
    return;
  }
  const int row0 = tiles[tid.y * 3 + 0];
  const uint32_t index = uint32_t(tiles[tid.y * 3 + 2]);
  bsrc.expert(index, N);

  constexpr bool transpose = true;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int STAGE = BN * BK_padded;
  constexpr int NT = WM * WN * 32;
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;
  static_assert(SM % 16 == 0 && TN % 2 == 0, "the run gather's fragment pairs");
  static_assert(BK % 64 == 0, "B blocks start on a quantization group");
  static_assert(MODE >= 0 && MODE <= 3, "K-block schedule");

  const int y_col = tid.x * BN;
  const short tgp_bm = short(min(BM, tile_rows));
  const short tgp_bn = ALIGN_N ? BN : short(min(BN, N - y_col));
  y += size_t(row0) * N + size_t(y_col);

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  const short sgp_sm = min(SM, short(max(0, int(tgp_bm) - tm)));
  const short sgp_sn = ALIGN_N ? SN : min(SN, short(max(0, (N - (y_col + tn)))));
  const bool is_unaligned_sm = (sgp_sm != SM);
  const bool is_unaligned_bn = ALIGN_N ? false : (tgp_bn != BN);

  const short m_lo_lim = 0;
  const short m_hi_lim = min(int(sgp_sm), max(0, int(tgp_bm) - tm));
  const bool sg_active = m_hi_lim > m_lo_lim;

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  const device T* xn = ROUTED ? x : x + (size_t(row0) + size_t(tm)) * size_t(K);
  const device T* routed[TM * BaseNAXFrag::kElemRows];
  if constexpr (ROUTED) {
    const short2 row_coord = BaseNAXFrag::get_coord();
    STEEL_PRAGMA_UNROLL
    for (short mr = 0; mr < TM; ++mr) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BaseNAXFrag::kElemRows; ++i) {
        const short row = mr * BaseNAXFrag::kFragRows + row_coord.y
            + i * BaseNAXFrag::kElemRowsJump;
        routed[mr * BaseNAXFrag::kElemRows + i] = row < sgp_sm
            ? x + size_t(routed_rows[row0 + tm + row]) * K : x;
      }
    }
  }

  const int t = int(simd_group_id) * 32 + int(simd_lane_id);
  // K % 64 == 0: every block is whole groups and whole 32-k steps.
  const int nb = (K + BK - 1) / BK;
  using Raw = typename BSrc::template Raw<BN, BK, NT>;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    constexpr bool aligned_m = decltype(kAlignedM)::value;
    auto load = [&](thread Raw& r, const int b) {
      bsrc.template load<BN, BK, NT, ALIGN_N>(r, t, y_col, N, b * BK, min(BK, K - b * BK));
    };
    auto store = [&](threadgroup T* ws, thread Raw& r, const int b) {
      bsrc.template store<BN, BK, NT, ALIGN_N>(ws, r, t, y_col, N, b * BK, min(BK, K - b * BK));
    };
    auto compute = [&](threadgroup T* ws, const int b) {
      const int kb = b * BK;
      const int klim = min(BK, K - kb);
      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < klim; kk1 += SK) {
        if (sg_active) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<T, TN, TK> Btile;

          volatile int compiler_barrier;

          nemotron_seed_load_a<ROUTED, aligned_m, SK>(Atile, xn, routed, kb + kk1, K, sgp_sm);

          Btile.template load<T, BK_padded, 1>(ws + tn * BK_padded + kk1);

          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<false>{},
              Btile,
              metal::bool_constant<transpose>{});

          (void)compiler_barrier;
        }
      }
    };
    Raw r;
    if constexpr (MODE == 0) {
      for (int b = 0; b < nb; b++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load(r, b);
        store(Ws, r, b);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        compute(Ws, b);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      // Stage s holds block b (s = b & 1 for MODE 1 and 2, s = 0 for MODE 3).
      // A stage is written only after the barrier that follows its last read,
      // and read only after the barrier that follows its write.
      load(r, 0);
      store(Ws, r, 0);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int b = 0; b < nb; b++) {
        const bool next = b + 1 < nb;
        if constexpr (MODE == 1) {
          if (next) {
            load(r, b + 1);
            store(Ws + ((b + 1) & 1) * STAGE, r, b + 1);
          }
          compute(Ws + (b & 1) * STAGE, b);
          threadgroup_barrier(mem_flags::mem_threadgroup);
        } else if constexpr (MODE == 2) {
          if (next) {
            load(r, b + 1);
          }
          compute(Ws + (b & 1) * STAGE, b);
          if (next) {
            store(Ws + ((b + 1) & 1) * STAGE, r, b + 1);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
          if (next) {
            load(r, b + 1);
          }
          compute(Ws, b);
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (next) {
            store(Ws, r, b + 1);
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }
        }
      }
    }
  });

  if (ALIGN_N || !is_unaligned_bn) {
    if (m_lo_lim == 0 && m_hi_lim == SM) {
      Dtile.template store<RELU2>(y + tm * N + tn, N);
    } else {
      Dtile.template store_slice<RELU2>(
          y + tm * N + tn, N, short2(0, m_lo_lim), short2(SN, m_hi_lim));
    }
  } else {
    Dtile.template store_slice<RELU2>(
        y + tm * N + tn,
        N,
        short2(0, m_lo_lim),
        short2(sgp_sn, m_hi_lim));
  }
}
"""#

/// Seed-only routed MoE fc1 / fc2 through the extended forms
/// (`nemotron_seed_ext_impl`), on top of `NemotronSeedRetile`: `tune` (load time,
/// after `NemotronSeedRetile.tune`) qualifies each form bit for bit against the
/// launch the seed makes today for the role (every MoE layer at 488 tokens, the
/// first layer at the other four seeded routings; any mismatch or Metal error
/// drops the form), times the survivors against the role's current pick
/// (`NemotronSeedRetile`'s adopted variant, or the incumbent) round robin on
/// every MoE layer per repetition (3 warm-up and 12 timed repetitions, median),
/// and adopts the fastest only if it is at least 3% faster than that pick.
/// Otherwise `NemotronSeedRetile` behaves as before. 128-row tiles are left out:
/// the seed's tiles are expert runs of at most 64 rows (about 23 rows per
/// expert at 488 tokens) from the counting sort. NEMOTRON_SEED_RETILE_EXT=0
/// turns this off; NEMOTRON_SEED_RETILE_EXT_FORCE=<form> adopts that form
/// wherever it qualifies.
enum NemotronSeedRetileExt {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE_EXT"] != "0"
    static let forced = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE_EXT_FORCE"]

    typealias Role = NemotronSeedRetile.Role

    struct Variant: Sendable, Equatable {
        let name: String
        let bn: Int, bk: Int, wm: Int, wn: Int
        /// K-block schedule (`nemotronSeedExtSource`).
        let mode: Int
        var threads: Int { wm * wn * 32 }
        var threadgroupBytes: Int { (mode == 1 || mode == 2 ? 2 : 1) * bn * (bk + 8) * 2 }

        init(_ bn: Int, _ bk: Int, wm: Int, wn: Int, mode: Int) {
            (self.bn, self.bk, self.wm, self.wn, self.mode) = (bn, bk, wm, wn, mode)
            name = "x\(bn)x\(bk)w\(wm)\(wn)" + ["", "db", "pf", "pf1"][mode]
        }
    }

    /// Name = x BN x BK, w WM WN, then the schedule: db (MODE 1), pf (MODE 2),
    /// pf1 (MODE 3). BM is the tiles' 64 (32 for runs of at most 32 rows).
    static let variants: [Variant] = [
        Variant(64, 64, wm: 2, wn: 2, mode: 0),
        Variant(64, 64, wm: 2, wn: 2, mode: 1),
        Variant(64, 64, wm: 2, wn: 2, mode: 2),
        Variant(64, 128, wm: 2, wn: 2, mode: 3),
        Variant(128, 64, wm: 2, wn: 4, mode: 0),
        Variant(128, 64, wm: 2, wn: 4, mode: 3),
    ]

    nonisolated(unsafe) static var fc1Choice: (role: Role, variant: Variant)? = nil
    nonisolated(unsafe) static var fc2Choice: (role: Role, variant: Variant)? = nil

    private static func makeKernel(routed: Bool, source: NemotronSeedRetile.Source) -> MLXFast.MLXFastKernel {
        let make: String
        switch source {
        case .bank: make = "NemotronSeedExtRows<T, T, T, false> b{(const device uint8_t*)w, scales, biases, K};"
        case .dict: make = "NemotronSeedExtRows<T, uint16_t, float, true> b{(const device uint8_t*)w, scales, biases, K};"
        case .copy: make = "NemotronSeedExtCopy<T> b{w, scales, biases, K};"
        }
        let rows = routed ? "routed_rows" : "(const device uint32_t*)tiles"
        func call(_ bm: String) -> String {
            """
                    nemotron_seed_ext_impl<T, \(bm), BN, BK, WM, WN, MODE, ALIGN_N, RELU2, \(routed)>(
                        x, \(rows), b, tiles, y, N, K, Ws,
                        threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup,
                        thread_index_in_simdgroup);
            """
        }
        return MLXFast.metalKernel(
            name: "nemotron_seed_ext_\(routed ? "routed_" : "")\(source.rawValue)",
            inputNames: ["x", "w", "scales", "biases", "tiles"] + (routed ? ["routed_rows"] : []) + ["N", "K"],
            outputNames: ["y"],
            source: """
                threadgroup T Ws[(MODE == 1 || MODE == 2 ? 2 : 1) * BN * (BK + 16 / sizeof(T))];
                \(make)
                if (BM == 64 && tiles[threadgroup_position_in_grid.y * 3 + 1] <= 32) {
            \(call("32"))
                } else {
            \(call("BM"))
                }
            """,
            header: nemotronRunGatherHeader + "\n" + nemotronSeedRetileSource + "\n" + nemotronSeedExtSource)
    }

    private static let routedBank = makeKernel(routed: true, source: .bank)
    private static let routedDict = makeKernel(routed: true, source: .dict)
    private static let plainBank = makeKernel(routed: false, source: .bank)
    private static let plainCopy = makeKernel(routed: false, source: .copy)

    /// One launch of form `v`; nil wherever the incumbent's guards (as
    /// `NemotronSeedRetile.launch`) would not admit it.
    static func launch(_ v: Variant, role: Role, x: MLXArray, lin: QuantizedSwitchLinear,
                       rows: MLXArray?, tiles: NemotronHRunTiles) -> MLXArray? {
        guard NemotronHPrefillX5.runGather, v.threadgroupBytes <= 32768,
              lin.mode == .affine, lin.bits == 4, lin.groupSize == 64,
              x.ndim == 3, x.dim(1) == 1, x.dtype == .bfloat16,
              !role.relu2 || MLX.maximum(x, MLXArray(0)).dtype == x.dtype
        else { return nil }
        let ops = lin.nemotronGatherOperands(for: x.dtype)
        guard ops.weight.dtype == .uint32, ops.weight.ndim == 3 else { return nil }
        let n = tiles.rows, k = x.dim(-1), experts = ops.weight.dim(0), nOut = ops.weight.dim(1)
        guard n >= 16, n / experts >= 4, tiles.bm == nemotronGatherTileRows(rows: n, experts: experts),
              ops.weight.dim(2) * 8 == k, k % 64 == 0, nOut % 32 == 0
        else { return nil }
        if role.routed {
            guard let rows, rows.ndim == 1, rows.dtype == .uint32, rows.size == n else { return nil }
        } else {
            guard x.dim(0) == n else { return nil }
        }
        let kernel: MLXFast.MLXFastKernel
        let operands: [MLXArray]
        switch role.source {
        case .bank, .dict:
            guard !ops.hasLinearBias, let biases = ops.biases, ops.scales.dtype == x.dtype,
                  biases.dtype == x.dtype, ops.scales.shape == [experts, nOut, k / 64],
                  biases.shape == ops.scales.shape
            else { return nil }
            if role.source == .dict {
                guard role.routed, let d = NemotronSBDict.dict(lin), d.index.shape == ops.scales.shape,
                      d.index.dtype == .uint16, d.table.dtype == .float32
                else { return nil }
                (kernel, operands) = (routedDict, [ops.weight, d.index, d.table])
            } else {
                (kernel, operands) = (role.routed ? routedBank : plainBank, [ops.weight, ops.scales, biases])
            }
        case .copy:
            // A copy held as expert groups keeps the routed retile's grouped copy launch (no form here).
            guard !role.routed, !role.relu2, nOut % 128 == 0, let copy = NemotronZSkip.copy(for: lin), copy.dict,
                  copy.codes.count == 1, let codes = copy.codes.first,
                  codes.shape == [experts, nOut / 128, k, 32], codes.dtype == .uint16,
                  copy.scales.shape == [experts, nOut / 128, k / 64, 128], copy.scales.dtype == .uint16,
                  copy.biases.dtype == .float32
            else { return nil }
            (kernel, operands) = (plainCopy, [codes, copy.scales, copy.biases])
        }
        var inputs: [any ScalarOrArray] = [x] + operands + [tiles.tiles]
        if role.routed, let rows { inputs.append(rows) }
        inputs += [nOut, k]
        return kernel(
            inputs,
            template: [("T", x.dtype), ("BM", tiles.bm), ("BN", v.bn), ("BK", v.bk), ("WM", v.wm), ("WN", v.wn),
                       ("MODE", v.mode), ("ALIGN_N", nOut % v.bn == 0), ("RELU2", role.relu2)],
            grid: (((nOut + v.bn - 1) / v.bn) * v.threads, tiles.count, 1), threadGroup: (v.threads, 1, 1),
            outputShapes: [[n, 1, nOut]], outputDTypes: [x.dtype])[0]
    }

    /// From `NemotronSeedRetile`'s dispatch (request prefill only): the adopted
    /// form for role `r` (0 fc1, 1 fc2) when the call is the launch it qualified against.
    static func apply(_ r: Int, routed: Bool, source: NemotronSeedRetile.Source, x: MLXArray,
                      lin: QuantizedSwitchLinear, rows: MLXArray?, tiles: NemotronHRunTiles) -> MLXArray? {
        guard let c = r == 0 ? fc1Choice : fc2Choice, c.role.routed == routed, c.role.source == source
        else { return nil }
        return launch(c.variant, role: c.role, x: x, lin: lin, rows: rows, tiles: tiles)
    }

    static func tune(_ target: Module) {
        fc1Choice = nil
        fc2Choice = nil
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron seed retile ext: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.runGather, NemotronPrefillCountingSort.verified else {
            return report("off (setting)")
        }
        let moes = target.modules().compactMap { $0 as? NemotronHMoE }
        guard let moe = moes.first, moes.allSatisfy({
            $0.switchMLP.fc1 is QuantizedSwitchLinear && $0.switchMLP.fc2 is QuantizedSwitchLinear
        }), let fc2First = moe.switchMLP.fc2 as? QuantizedSwitchLinear
        else { return report("off (no quantized MoE)") }
        let experts = moe.switchMLP.numExperts, topK = moe.numExpertsPerTok
        let width = moe.switchMLP.inputDims
        let hidden = fc2First.nemotronGatherOperands(for: .bfloat16).weight.dim(2) * 8
        // The launches the seed makes today (as `NemotronSeedRetile.tune`).
        let roles = [
            NemotronPrefillDirectFC1.verified
                ? Role(routed: true, source: NemotronPrefillDirectFC1.dictVerified ? .dict : .bank, relu2: true)
                : Role(routed: false, source: .bank, relu2: true),
            NemotronPrefillFC2Copy.verified
                ? Role(routed: false, source: .copy, relu2: false)
                : Role(routed: false, source: .bank, relu2: false),
        ]
        let picks = [NemotronSeedRetile.fc1Choice, NemotronSeedRetile.fc2Choice]
        func linear(_ m: NemotronHMoE, _ r: Int) -> QuantizedSwitchLinear {
            (r == 0 ? m.switchMLP.fc1 : m.switchMLP.fc2) as! QuantizedSwitchLinear
        }
        typealias Case = (tiles: NemotronHRunTiles, rows: MLXArray, x: MLXArray, xs: MLXArray, act: MLXArray)
        func input(_ r: Int, _ c: Case) -> (MLXArray, MLXArray?) {
            r == 0 ? (roles[0].routed ? (c.x, c.rows) : (c.xs, nil)) : (c.act, nil)
        }
        // Seeded routes and inputs (case 2: ReLU2-like rows, mostly exact zeros).
        let cases = (try? withError { () -> [Case] in
            var made: [Case] = []
            for (index, (tokens, skew)) in [(488, Float(2)), (488, 0), (488, 9), (300, 4), (97, 1)].enumerated() {
                let seed = UInt64(0x5eed_e000 + 16 * index)
                let logits = MLXRandom.normal([tokens, experts], key: MLXRandom.key(seed))
                    + skew * MLXRandom.normal([1, experts], key: MLXRandom.key(seed + 1))
                let flat = argSort(-logits, axis: -1)[0..., ..<topK].asType(.uint32).flattened()
                guard let sorted = nemotronCountingSortChunked(flat, experts: experts, k: topK) else { return [] }
                var x = MLXRandom.normal([tokens, 1, width], scale: 2, key: MLXRandom.key(seed + 2))
                var act = MLXRandom.normal([tokens * topK, 1, hidden], scale: 2, key: MLXRandom.key(seed + 3))
                if index == 2 {
                    x = MLX.maximum(x - 1.5, MLXArray(Float(0)))
                    x = x * x
                    act = MLX.maximum(act - 1.5, MLXArray(Float(0)))
                    act = act * act
                }
                let xb = x.asType(.bfloat16)
                let c: Case = (sorted.tiles, sorted.orderK, xb, xb[sorted.orderK], act.asType(.bfloat16))
                eval([c.tiles.tiles, c.rows, c.x, c.xs, c.act])
                made.append(c)
            }
            return made
        }) ?? []
        guard cases.count == 5 else { return report("off (inputs)") }

        var notes: [String] = []
        var chosen: [Variant?] = [nil, nil]
        for r in 0 ..< 2 {
            let role = roles[r]
            let label = (r == 0 ? "fc1 " : "fc2 ") + role.label
            // Qualify against the incumbent launch.
            var alive = variants.filter { $0.threadgroupBytes <= 32768 }
            var dropped: [String] = []
            caseLoop: for (ci, c) in cases.enumerated() {
                for m in (ci == 0 ? moes : [moe]) {
                    let lin = linear(m, r)
                    let (x, rows) = input(r, c)
                    let reference = try? withError { () -> MLXArray? in
                        guard let out = NemotronSeedRetile.incumbent(role, x: x, lin: lin, rows: rows, tiles: c.tiles)
                        else { return nil }
                        eval(out)
                        return out
                    }
                    guard let reference else {
                        dropped.append("incumbent unavailable")
                        alive = []
                        break caseLoop
                    }
                    alive = alive.filter { v in
                        do {
                            let same = try withError { () -> Bool in
                                guard let out = launch(v, role: role, x: x, lin: lin, rows: rows, tiles: c.tiles)
                                else { return false }
                                return (out.view(dtype: .uint16) .== reference.view(dtype: .uint16)).all()
                                    .item(Bool.self)
                            }
                            if !same { dropped.append("\(v.name) differs@\(ci)") }
                            return same
                        } catch {
                            dropped.append("\(v.name) error")
                            return false
                        }
                    }
                    if alive.isEmpty { break caseLoop }
                }
            }
            // The current pick: NemotronSeedRetile's adopted variant for this role, else the incumbent.
            let pick = picks[r].flatMap { $0.role == role ? $0.variant : nil }
            let base = pick?.name ?? "V0"
            guard !alive.isEmpty else {
                notes.append("\(label) keep \(base) (0/\(variants.count) bitwise"
                    + (dropped.isEmpty ? "" : "; dropped: " + dropped.prefix(6).joined(separator: ", ")) + ")")
                continue
            }
            // Time the current pick and the survivors round robin, every MoE layer per repetition.
            let candidates: [Variant?] = [nil] + alive.map { Optional($0) }
            let (x, rows) = input(r, cases[0])
            var samples = Array(repeating: [Double](), count: candidates.count)
            do {
                try withError {
                    for rep in 0 ..< 15 {
                        for (i, candidate) in candidates.enumerated() {
                            let start = DispatchTime.now().uptimeNanoseconds
                            let outs = moes.compactMap { m -> MLXArray? in
                                let lin = linear(m, r)
                                if let candidate {
                                    return launch(candidate, role: role, x: x, lin: lin, rows: rows, tiles: cases[0].tiles)
                                }
                                if let pick {
                                    return NemotronSeedRetile.launch(pick, role: role, x: x, lin: lin, rows: rows,
                                                                     tiles: cases[0].tiles)
                                }
                                return NemotronSeedRetile.incumbent(role, x: x, lin: lin, rows: rows, tiles: cases[0].tiles)
                            }
                            eval(outs)
                            Stream.gpu.synchronize()
                            let elapsed = DispatchTime.now().uptimeNanoseconds - start
                            if rep >= 3, outs.count == moes.count {
                                samples[i].append(Double(elapsed) / 1e6 / Double(moes.count))
                            }
                        }
                    }
                }
            } catch {
                dropped.append("timing error")
            }
            guard samples.allSatisfy({ $0.count == 12 }) else {
                notes.append("\(label) keep \(base) (timing incomplete"
                    + (dropped.isEmpty ? "" : "; dropped: " + dropped.prefix(6).joined(separator: ", ")) + ")")
                continue
            }
            let medians = zip(candidates, samples).map { candidate, s in (candidate, s.sorted()[s.count / 2]) }
            if let forced, let f = alive.first(where: { $0.name == forced }) {
                chosen[r] = f
            } else if let best = medians.dropFirst().min(by: { $0.1 < $1.1 }), best.1 <= 0.97 * medians[0].1 {
                chosen[r] = best.0
            }
            let times = medians.dropFirst().map { "\($0.0!.name) \(String(format: "%.3f", $0.1))" }
            notes.append("\(label) \(chosen[r]?.name ?? "keep " + base) vs \(base) "
                + String(format: "%.3f", medians[0].1) + " ms (\(alive.count)/\(variants.count) bitwise ["
                + times.joined(separator: ", ") + "]"
                + (dropped.isEmpty ? "" : "; dropped: " + dropped.prefix(6).joined(separator: ", ")) + ")")
        }
        fc1Choice = chosen[0].map { (roles[0], $0) }
        fc2Choice = chosen[1].map { (roles[1], $0) }
        report(notes.joined(separator: "; ") + " (per layer, 488 tokens, median of 12 over every MoE layer)")
    }
}
