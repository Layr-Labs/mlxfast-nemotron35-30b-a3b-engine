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
// loader does.
template <typename T>
struct NemotronSeedCopy {
  const device uint16_t* codes;
  const device uint16_t* sidx;
  const device float* table;
  int K;

  METAL_FUNC void expert(const uint index, const int N) thread {
    codes += size_t(index) * size_t(N / 128) * size_t(K) * 32;
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
/// it is at least 3% faster than the incumbent. NEMOTRON_SEED_RETILE=0 keeps
/// the incumbents; NEMOTRON_SEED_RETILE_FORCE=<variant> adopts that variant
/// for both roles when it qualifies.
enum NemotronSeedRetile {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE"] != "0"
    static let forced = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE_FORCE"]

    struct Variant: Sendable, Equatable {
        let name: String
        let bn: Int, bk: Int, wm: Int, wn: Int
        let direct: Bool
        /// Set: the incumbent's own launch through its A-staged clone (`NemotronSeedAStage`).
        var stage: NemotronSeedAStage.Mode? = nil
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
    ] + NemotronSeedAStage.variants

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

    private static func body(routed: Bool, source: Source) -> String {
        let make: String
        switch source {
        case .bank: make = "NemotronSeedBank<T> b{(const device uint8_t*)w, scales, biases, K};"
        case .dict: make = "NemotronSeedDict<T> b{(const device uint8_t*)w, scales, biases, K};"
        case .copy: make = "NemotronSeedCopy<T> b{w, scales, biases, K};"
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
                if (BM == 64 && tiles[threadgroup_position_in_grid.y * 3 + 1] <= 32) {
            \(call("32"))
                } else {
            \(call("BM"))
                }
            """
    }

    private static func makeKernel(routed: Bool, source: Source) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: "nemotron_seed_gather_\(routed ? "routed_" : "")\(source.rawValue)",
            inputNames: ["x", "w", "scales", "biases", "tiles"] + (routed ? ["routed_rows"] : []) + ["N", "K"],
            outputNames: ["y"],
            source: body(routed: routed, source: source),
            header: nemotronRunGatherHeader + "\n" + nemotronSeedRetileSource)
    }

    private static let routedBank = makeKernel(routed: true, source: .bank)
    private static let routedDict = makeKernel(routed: true, source: .dict)
    private static let plainBank = makeKernel(routed: false, source: .bank)
    private static let plainCopy = makeKernel(routed: false, source: .copy)

    /// One retiled launch; nil wherever the incumbent's guards would not admit it.
    static func launch(_ v: Variant, role: Role, x: MLXArray, lin: QuantizedSwitchLinear,
                       rows: MLXArray?, tiles: NemotronHRunTiles) -> MLXArray? {
        if let mode = v.stage {
            return NemotronSeedAStage.launch(mode, role: role, x: x, lin: lin, rows: rows, tiles: tiles)
        }
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
                  copy.codes.shape == [experts, nOut / 128, k, 32], copy.codes.dtype == .uint16,
                  copy.scales.shape == [experts, nOut / 128, k / 64, 128], copy.scales.dtype == .uint16,
                  copy.biases.dtype == .float32
            else { return nil }
            (kernel, operands) = (plainCopy, [copy.codes, copy.scales, copy.biases])
        }
        var inputs: [any ScalarOrArray] = [x] + operands + [tiles.tiles]
        if role.routed, let rows { inputs.append(rows) }
        inputs += [nOut, k]
        return kernel(
            inputs,
            template: [("T", x.dtype), ("BM", tiles.bm), ("BN", v.bn), ("BK", v.bk), ("WM", v.wm), ("WN", v.wn),
                       ("DIRECT_B", v.direct), ("ALIGN_N", nOut % v.bn == 0), ("RELU2", role.relu2)],
            grid: (((nOut + v.bn - 1) / v.bn) * v.threads, tiles.count, 1), threadGroup: (v.threads, 1, 1),
            outputShapes: [[n, 1, nOut]], outputDTypes: [x.dtype])[0]
    }

    // Dispatch: the adopted variant, only for the incumbent launch it qualified against.

    /// From `NemotronPrefillDirectFC1.apply` (already request prefill only).
    static func fc1Routed(_ x: MLXArray, _ lin: QuantizedSwitchLinear, rows: MLXArray,
                          tiles: NemotronHRunTiles, dict: Bool) -> MLXArray? {
        guard let c = fc1Choice, c.role.routed, c.role.source == (dict ? .dict : .bank) else { return nil }
        return launch(c.variant, role: c.role, x: x, lin: lin, rows: rows, tiles: tiles)
    }

    /// The fc1 run gather on the expanded rows (`nemotronRunGather(xs, relu2: true)`).
    static func fc1Plain(_ xs: MLXArray, _ lin: QuantizedSwitchLinear, tiles: NemotronHRunTiles) -> MLXArray? {
        guard NemotronHPrefillX5.active, let c = fc1Choice, !c.role.routed, c.role.source == .bank
        else { return nil }
        return launch(c.variant, role: c.role, x: xs, lin: lin, rows: nil, tiles: tiles)
    }

    /// From `NemotronPrefillFC2Copy.apply` (already request prefill only).
    static func fc2Copy(_ x: MLXArray, _ lin: QuantizedSwitchLinear, tiles: NemotronHRunTiles) -> MLXArray? {
        guard let c = fc2Choice, c.role.source == .copy else { return nil }
        return launch(c.variant, role: c.role, x: x, lin: lin, rows: nil, tiles: tiles)
    }

    /// The fc2 run gather on the bank (`nemotronRunGather(y)`).
    static func fc2Plain(_ x: MLXArray, _ lin: QuantizedSwitchLinear, tiles: NemotronHRunTiles) -> MLXArray? {
        guard NemotronHPrefillX5.active, let c = fc2Choice, c.role.source == .bank else { return nil }
        return launch(c.variant, role: c.role, x: x, lin: lin, rows: nil, tiles: tiles)
    }

    /// The launch the seed makes today for `role`.
    private static func incumbent(_ role: Role, x: MLXArray, lin: QuantizedSwitchLinear, rows: MLXArray?,
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
        NemotronSeedAStage.report()
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
            } else if let best = medians[r].dropFirst().min(by: { $0.1 < $1.1 }), best.1 <= 0.97 * base {
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

// MARK: - Seed A staging (S1)

/// After `xn` is set: the staging geometry. `AS_K` (BK, or SK for the slice
/// mode) is prepended by the edit.
private let nemotronSeedAStageGeometry = #"""
  constexpr short AS_LD = AS_K + 16 / sizeof(T);
  // A staging: lane (fn, fm) of simdgroup (tm, tn) copies its own fragment
  // rows (mr * 16 + fm + 8 * i) of the A block, AS_COLS consecutive k from
  // as_col; the 4 * WN lanes that share a row cover its AS_K k once each.
  const short2 as_coord = BaseNAXFrag::get_coord();
  constexpr short AS_COPIERS = (BaseNAXFrag::kFragCols / BaseNAXFrag::kElemCols) * WN;
  constexpr short AS_COLS = AS_K / AS_COPIERS;
  static_assert(AS_K % AS_COPIERS == 0, "A staging columns");
  const short as_col = short((as_coord.x / BaseNAXFrag::kElemCols
      + short(simd_group_id % WN) * (BaseNAXFrag::kFragCols / BaseNAXFrag::kElemCols)) * AS_COLS);

"""#

/// One staging pass: every thread writes its rows' AS_COLS values of the A
/// block (rows past the tile as T(0), as `load_safe` writes them). The edit
/// substitutes AS_SRC_ROW (the device row of a valid row) and AS_KOFS.
private let nemotronSeedAStageCopy = #"""
        STEEL_PRAGMA_UNROLL
        for (short mr = 0; mr < TM; ++mr) {
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < BaseNAXFrag::kElemRows; ++i) {
            const short row = mr * BaseNAXFrag::kFragRows + as_coord.y
                + i * BaseNAXFrag::kElemRowsJump;
            threadgroup T* as_dst = As + (tm + row) * AS_LD + as_col;
            if (row < sgp_sm) {
              const device T* as_src = AS_SRC_ROW + AS_KOFS + as_col;
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < AS_COLS; ++j) {
                as_dst[j] = as_src[j];
              }
            } else {
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < AS_COLS; ++j) {
                as_dst[j] = T(0);
              }
            }
          }
        }

"""#

/// The head of the main loop's inner SK loop, up to the A-fragment load.
private let nemotronSeedAStageInnerHead = #"""
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          if (sg_active) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, BR, BC> Btile;

            volatile int compiler_barrier;


"""#

/// Seed-only A staging for the incumbent routed fc1 (direct rows, pair
/// dictionary) and fc2 (tile-major copy) run gathers: clones that copy each K
/// step's A block (BM x BK BF16 activations) once per threadgroup into
/// threadgroup memory, inside the barrier pair that already brackets the weight
/// block, and load the A fragments from there with the B tile's call. Grid,
/// threads, weight loader, fragment positions and NAX MMA sequence are the
/// incumbent's; rows past the tile are staged as T(0), as `load_safe` writes
/// them. The clones are offered to `NemotronSeedRetile.tune` as extra variants
/// of the incumbent's own launch, so they are qualified bit for bit against it
/// and adopted only by the retile's timing rule; nothing dispatches them
/// otherwise. NEMOTRON_SEED_A_STAGE=0 offers none; =full stages the BM x BK
/// block per K step; =slice stages one SK slice per inner step (smaller
/// threadgroup memory, two more barriers per K step); unset offers both.
enum NemotronSeedAStage {
    enum Mode: String, Sendable { case full, slice }
    static let modes: [Mode] = {
        switch ProcessInfo.processInfo.environment["NEMOTRON_SEED_A_STAGE"] {
        case "0": return []
        case "full": return [.full]
        case "slice": return [.slice]
        default: return [.full, .slice]
        }
    }()

    /// `source` with each (old, new) applied; nil if any `old` is not found exactly once.
    private static func edited(_ source: String, _ edits: [(String, String)]) -> String? {
        var s = source
        for (old, new) in edits {
            guard s.components(separatedBy: old).count == 2 else { return nil }
            s = s.replacingOccurrences(of: old, with: new)
        }
        return s
    }

    /// The staging edits of one run-gather impl. `afterLoader`: the main loop's
    /// weight-block load (the copy follows it, before the second barrier);
    /// `loadBlock`: the main loop's A-fragment load; `rowSource`: the device row
    /// of a valid A row at this K step.
    private static func edits(_ mode: Mode, impl: String, xnLine: String, afterLoader: String,
                              loadBlock: String, rowSource: String) -> [(String, String)] {
        func copy(_ kofs: String) -> String {
            nemotronSeedAStageCopy.replacingOccurrences(of: "AS_SRC_ROW", with: rowSource)
                .replacingOccurrences(of: "AS_KOFS", with: kofs)
        }
        let load: (String, String)
        switch mode {
        case .full:
            load = (loadBlock, "            Atile.template load<T, AS_LD, 1>(As + tm * AS_LD + kk1);\n")
        case .slice:
            load = (nemotronSeedAStageInnerHead + loadBlock,
                    "        for (int kk1 = 0; kk1 < BK; kk1 += SK) {\n"
                    + "          if (kk1 > 0) {\n"
                    + "            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                    + copy("kk1")
                    + "            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                    + "          }\n"
                    + "          if (sg_active) {\n"
                    + "            NAXTile<T, TM, TK> Atile;\n"
                    + "            NAXTile<T, BR, BC> Btile;\n\n"
                    + "            volatile int compiler_barrier;\n\n"
                    + "            Atile.template load<T, AS_LD, 1>(As + tm * AS_LD);\n")
        }
        return [
            ("METAL_FUNC void \(impl)(", "METAL_FUNC void \(impl)_astage("),
            ("    threadgroup T* Ws,\n    uint3 tid,", "    threadgroup T* Ws,\n    threadgroup T* As,\n    uint3 tid,"),
            (xnLine, xnLine + "  constexpr short AS_K = \(mode == .full ? "BK" : "SK");\n" + nemotronSeedAStageGeometry),
            (afterLoader, afterLoader + copy("0")),
            load,
        ]
    }

    /// The fc1 dictionary header with the A staging; nil if an edit point is missed.
    static func fc1Header(_ mode: Mode) -> String? {
        NemotronPrefillDirectFC1.dictHeader.flatMap {
            edited($0, edits(
                mode, impl: "nemotron_gather_qmm_runs_direct_dict_impl",
                xnLine: "  const device T* xn = x;\n",
                afterLoader: "          loader_w.load_safe(short2(BK, tgp_bn));\n        }\n",
                loadBlock: "            if constexpr (kAlignedM.value) {\n"
                    + "              nemotron_load_routed(Atile, routed, int(xn - x) + kk1, sgp_sm);\n"
                    + "            } else {\n"
                    + "              nemotron_load_routed(Atile, routed, int(xn - x) + kk1, sgp_sm);\n"
                    + "            }\n",
                rowSource: "(routed[mr * BaseNAXFrag::kElemRows + i] + int(xn - x))"))
        }
    }

    /// The fc2 copy header with the A staging (edits on its copy impl only); nil if an edit point is missed.
    static func fc2Header(_ mode: Mode) -> String? {
        let prefix = nemotronRunGatherHeader + "\n"
        guard NemotronPrefillFC2Copy.header.hasPrefix(prefix) else { return nil }
        return edited(String(NemotronPrefillFC2Copy.header.dropFirst(prefix.count)), edits(
            mode, impl: "nemotron_gather_qmm_runs_copy_impl",
            xnLine: "  const device T* xn = x + tm * K;\n",
            afterLoader: "        cl_load();\n\n",
            loadBlock: "            if constexpr (kAlignedM.value) {\n"
                + "              Atile.load(xn + kk1, K);\n"
                + "            } else {\n"
                + "              Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));\n"
                + "            }\n",
            rowSource: "(xn + row * K)")).map { prefix + $0 }
    }

    /// The kernel body: the incumbent's adaptive 32/64-row dispatch with `As` beside `Ws`.
    private static func body(_ mode: Mode, impl: String, operands: String) -> String {
        let call = "\(impl)_astage<T, GS, BITS, %@, BN, BK, 2, 2, ALIGN_N, ALIGN_K, RELU2>(\n"
            + "        \(operands), Ws, As,\n"
            + "        threadgroup_position_in_grid,\n"
            + "        simdgroup_index_in_threadgroup,\n"
            + "        thread_index_in_simdgroup);\n"
        return "threadgroup T Ws[BN * (BK + 16 / sizeof(T))];\n"
            + "threadgroup T As[BM * (\(mode == .full ? "BK" : "32") + 16 / sizeof(T))];\n"
            + "if (BM == 64 && tiles[threadgroup_position_in_grid.y * 3 + 1] <= 32) {\n    "
            + call.replacingOccurrences(of: "%@", with: "32")
            + "} else {\n    "
            + call.replacingOccurrences(of: "%@", with: "BM")
            + "}\n"
    }

    private static func makeFC1(_ mode: Mode) -> MLXFast.MLXFastKernel? {
        fc1Header(mode).map {
            MLXFast.metalKernel(
                name: "nemotron_gather_qmm_runs_direct_input_dict_astage" + (mode == .full ? "" : "_slice"),
                inputNames: ["x", "w", "scales", "biases", "tiles", "routed_rows", "N", "K"],
                outputNames: ["y"],
                source: body(mode, impl: "nemotron_gather_qmm_runs_direct_dict_impl",
                             operands: "x, w, scales, biases, tiles, routed_rows, y, N, K"),
                header: $0)
        }
    }

    private static func makeFC2(_ mode: Mode) -> MLXFast.MLXFastKernel? {
        fc2Header(mode).map {
            MLXFast.metalKernel(
                name: "nemotron_gather_qmm_runs_fc2_copy_astage" + (mode == .full ? "" : "_slice"),
                inputNames: ["x", "wk", "sidx", "table", "tiles", "N", "K"],
                outputNames: ["y"],
                source: body(mode, impl: "nemotron_gather_qmm_runs_copy_impl",
                             operands: "x, wk, sidx, table, tiles, y, N, K"),
                header: $0)
        }
    }

    private static let fc1Full = makeFC1(.full)
    private static let fc1Slice = makeFC1(.slice)
    private static let fc2Full = makeFC2(.full)
    private static let fc2Slice = makeFC2(.slice)

    static func fc1Kernel(_ mode: Mode) -> MLXFast.MLXFastKernel? { mode == .full ? fc1Full : fc1Slice }
    static func fc2Kernel(_ mode: Mode) -> MLXFast.MLXFastKernel? { mode == .full ? fc2Full : fc2Slice }

    /// The retile variants: the incumbent geometry through the staged clone.
    static let variants: [NemotronSeedRetile.Variant] = modes.map {
        NemotronSeedRetile.Variant(name: $0 == .full ? "v0-astage" : "v0-astage-slice",
                                   bn: 64, bk: 64, wm: 2, wn: 2, direct: false, stage: $0)
    }

    /// The staged clone of the incumbent launch for `role`; nil for any other role.
    static func launch(_ mode: Mode, role: NemotronSeedRetile.Role, x: MLXArray, lin: QuantizedSwitchLinear,
                       rows: MLXArray?, tiles: NemotronHRunTiles) -> MLXArray? {
        switch (role.source, role.routed, rows) {
        case (.dict, true, let rows?):
            return NemotronPrefillDirectFC1.launch(x, lin, rows: rows, tiles: tiles, useDict: true, staged: mode)
        case (.copy, false, _):
            return NemotronPrefillFC2Copy.launch(x, lin, tiles: tiles, staged: mode)
        default:
            return nil
        }
    }

    static func report() {
        let line = modes.isEmpty ? "off (setting)"
            : "offered to the seed retile: " + modes.map {
                $0.rawValue + (fc1Kernel($0) != nil && fc2Kernel($0) != nil ? "" : " (source not as expected)")
            }.joined(separator: ", ")
        FileHandle.standardError.write(Data("nemotron seed a-stage: \(line)\n".utf8))
    }
}

// MARK: - Seed dense tune (S2)

/// B source for the shared down's tile-major decode copy (codes as
/// `NemotronSeedCopy`) with its tile-major BF16 scales and biases, widened to
/// FP32 as the shared-down copy loader widens them.
private let nemotronSeedDenseSource = #"""
template <typename T>
struct NemotronSeedCopySB {
  const device uint16_t* codes;
  const device T* scales;
  const device T* biases;
  int K;

  METAL_FUNC void expert(const uint index, const int N) thread {
    codes += size_t(index) * size_t(N / 128) * size_t(K) * 32;
    scales += size_t(index) * size_t(N / 128) * size_t(K / 64) * 128;
    biases += size_t(index) * size_t(N / 128) * size_t(K / 64) * 128;
  }
  METAL_FUNC float2 sb(const int n, const int g) const thread {
    const size_t o = (size_t(n / 128) * size_t(K / 64) + size_t(g)) * 128 + size_t(n % 128);
    return float2(float(scales[o]), float(biases[o]));
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
"""#

/// Seed-only (request prefill) dense projections through the retiled run
/// gather (`nemotron_seed_gather_impl`) with a tile chosen once at load: each
/// Mamba layer's input and output projections and each MoE layer's shared
/// expert up (typed ReLU2 store) and down projections, wherever the seed runs
/// them through their run-gather clones today (`NemotronPrefillDenseDict`,
/// `NemotronPrefillSharedDownCopy`). A variant reads the clone's weights (the
/// pair dictionary, or the shared down's tile-major decode copy) and keeps each
/// output's NAX MMA sequence over k chunks 0..K/16 in order; it changes only
/// the row-tile height BM (the launch's row descriptors), BN, BK, the
/// SIMD-group grid and whether B passes through threadgroup memory. `tune`
/// (load time, after the routed retile) compares each variant the role offers
/// (`Role.offered`) with the clone bit for bit on seeded rows (every layer of
/// the role in the first case), drops any mismatch or Metal error, times the survivors and the clone round
/// robin as a dependent chain over every layer of the role (each launch waits
/// for the previous one through a small elementwise launch, and the weights
/// cycle through all layers), and adopts the fastest only if it is at least 6%
/// faster than the clone. NEMOTRON_SEED_DENSE_TUNE=0 keeps the clones;
/// NEMOTRON_SEED_DENSE_TUNE_FORCE=<variant> (every role) or
/// <role>=<variant>,... adopts a variant that qualifies.
enum NemotronSeedDenseTune {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_SEED_DENSE_TUNE"] != "0"
    static let forced = ProcessInfo.processInfo.environment["NEMOTRON_SEED_DENSE_TUNE_FORCE"]

    struct Variant: Sendable, Equatable {
        let name: String
        let bm: Int, bn: Int, bk: Int, wm: Int, wn: Int
        let direct: Bool
        var threads: Int { wm * wn * 32 }
    }

    /// The pool. Name = m + BM, then the retile's naming (ws: B through
    /// threadgroup memory, d: B dequantized into the fragments; BN (x BK);
    /// WM WN). The clones run m64ws64x64w22's tiling with their own loader.
    /// Each role offers the tuner only `Role.offered`; the force switch can
    /// name any pool variant.
    static let variants: [Variant] = [
        Variant(name: "m64ws64x64w22", bm: 64, bn: 64, bk: 64, wm: 2, wn: 2, direct: false),
        Variant(name: "m64ws64x128w22", bm: 64, bn: 64, bk: 128, wm: 2, wn: 2, direct: false),
        Variant(name: "m64ws128x64w24", bm: 64, bn: 128, bk: 64, wm: 2, wn: 4, direct: false),
        Variant(name: "m64d64w22", bm: 64, bn: 64, bk: 64, wm: 2, wn: 2, direct: true),
        Variant(name: "m64d128w24", bm: 64, bn: 128, bk: 64, wm: 2, wn: 4, direct: true),
        Variant(name: "m128ws64x64w42", bm: 128, bn: 64, bk: 64, wm: 4, wn: 2, direct: false),
        Variant(name: "m128ws64x64w22", bm: 128, bn: 64, bk: 64, wm: 2, wn: 2, direct: false),
        Variant(name: "m128ws128x64w44", bm: 128, bn: 128, bk: 64, wm: 4, wn: 4, direct: false),
        Variant(name: "m128ws64x128w42", bm: 128, bn: 64, bk: 128, wm: 4, wn: 2, direct: false),
        Variant(name: "m32ws64x64w12", bm: 32, bn: 64, bk: 64, wm: 1, wn: 2, direct: false),
        Variant(name: "m32ws128x64w14", bm: 32, bn: 128, bk: 64, wm: 1, wn: 4, direct: false),
    ]

    /// The four dense projections, named as the clones' kinds.
    enum Role: String, CaseIterable, Sendable {
        case inProj = "in_proj", outProj = "out_proj", up = "up_proj", down = "down_proj"
        var relu2: Bool { self == .up }
        /// The pool variants this role offers to the tuner: the clone's
        /// tiling through the retile's loader, and the same with BK 128. The
        /// shared down offers none.
        var offered: [String] { self == .down ? [] : ["m64ws64x64w22", "m64ws64x128w22"] }
    }

    /// Fewest rows the qualify covers.
    static let minRows = 17
    nonisolated(unsafe) static var choices: [Role: Variant] = [:]
    /// The target projections the choices qualified on.
    nonisolated(unsafe) private static var owners: Set<ObjectIdentifier> = []

    private static func makeKernel(copy: Bool) -> MLXFast.MLXFastKernel {
        let make = copy ? "NemotronSeedCopySB<T> b{w, scales, biases, K};"
            : "NemotronSeedDict<T> b{(const device uint8_t*)w, scales, biases, K};"
        return MLXFast.metalKernel(
            name: copy ? "nemotron_seed_dense_copy_sb" : "nemotron_seed_dense_dict",
            inputNames: ["x", "w", "scales", "biases", "tiles", "N", "K"],
            outputNames: ["y"],
            source: """
                threadgroup T Ws[DIRECT_B ? 1 : BN * (BK + 16 / sizeof(T))];
                \(make)
                nemotron_seed_gather_impl<T, BM, BN, BK, WM, WN, DIRECT_B, ALIGN_N, RELU2, false>(
                    x, (const device uint32_t*)tiles, b, tiles, y, N, K, Ws,
                    threadgroup_position_in_grid,
                    simdgroup_index_in_threadgroup,
                    thread_index_in_simdgroup);
            """,
            header: nemotronRunGatherHeader + "\n" + nemotronSeedRetileSource + "\n" + nemotronSeedDenseSource)
    }

    private static let dictKernel = makeKernel(copy: false)
    private static let copyKernel = makeKernel(copy: true)

    /// One variant on the clone's weights; nil wherever the clone would not run.
    static func launch(_ v: Variant, role: Role, x: MLXArray, linear: Linear) -> MLXArray? {
        guard NemotronHPrefillX5.runGather, let q = linear as? QuantizedLinear,
              type(of: linear) == QuantizedLinear.self, q.mode == .affine, q.bits == 4, q.groupSize == 64,
              q.bias == nil, x.ndim == 3, x.dim(0) == 1, x.dim(1) > 8, x.dtype == .bfloat16,
              !role.relu2 || MLX.maximum(x, MLXArray(0)).dtype == x.dtype,
              q.weight.dtype == .uint32, q.weight.ndim == 2
        else { return nil }
        let rows = x.dim(1), k = x.dim(2), nOut = q.weight.dim(0)
        guard q.weight.dim(1) * 8 == k, k % 64 == 0, nOut % 32 == 0 else { return nil }
        let kernel: MLXFast.MLXFastKernel
        let operands: [MLXArray]
        if role == .down {
            guard nOut % 128 == 0, let copy = NemotronSharedDownZSkip.copyArrays(for: linear),
                  copy.codes.shape == [1, nOut / 128, k, 32], copy.codes.dtype == .uint16,
                  copy.scales.shape == [1, nOut / 128, k / 64, 128], copy.scales.dtype == x.dtype,
                  copy.biases.shape == copy.scales.shape, copy.biases.dtype == x.dtype
            else { return nil }
            (kernel, operands) = (copyKernel, [copy.codes, copy.scales, copy.biases])
        } else {
            guard let d = NemotronSBDict.dict(q), d.index.shape == [nOut, k / 64], d.index.dtype == .uint16,
                  d.table.dtype == .float32
            else { return nil }
            (kernel, operands) = (dictKernel, [q.weight.expandedDimensions(axis: 0),
                                               d.index.expandedDimensions(axis: 0), d.table])
        }
        // Row tiles of BM over the tokens, at least three descriptors (as the
        // clones); a zero-row descriptor is skipped.
        let used = (rows + v.bm - 1) / v.bm
        let count = max(used, 3)
        var desc = [Int32]()
        desc.reserveCapacity(count * 3)
        for t in 0 ..< count {
            desc += t < used ? [Int32(t * v.bm), Int32(min(v.bm, rows - t * v.bm)), 0] : [0, 0, 0]
        }
        var inputs: [any ScalarOrArray] = [x.reshaped(rows, 1, k)] + operands
        inputs += [MLXArray(desc).reshaped(count, 3), nOut, k]
        let y = kernel(
            inputs,
            template: [("T", x.dtype), ("BM", v.bm), ("BN", v.bn), ("BK", v.bk), ("WM", v.wm), ("WN", v.wn),
                       ("DIRECT_B", v.direct), ("ALIGN_N", nOut % v.bn == 0), ("RELU2", role.relu2)],
            grid: (((nOut + v.bn - 1) / v.bn) * v.threads, count, 1), threadGroup: (v.threads, 1, 1),
            outputShapes: [[rows, 1, nOut]], outputDTypes: [x.dtype])[0]
        return y.reshaped(1, rows, nOut)
    }

    /// From the clones' `apply` (already request prefill only): the adopted
    /// variant, only on a projection it qualified on.
    static func apply(_ x: MLXArray, _ linear: Linear, kind: String) -> MLXArray? {
        guard !choices.isEmpty, let role = Role(rawValue: kind), let v = choices[role],
              owners.contains(ObjectIdentifier(linear)), x.ndim == 3, x.dim(1) >= minRows
        else { return nil }
        return launch(v, role: role, x: x, linear: linear)
    }

    /// The clone the seed launches today for `role`.
    private static func incumbent(_ role: Role, x: MLXArray, linear: Linear) -> MLXArray? {
        role == .down ? NemotronPrefillSharedDownCopy.launch(x, linear)
            : NemotronPrefillDenseDict.launch(x, linear, relu2: role.relu2)
    }

    private static func forcedName(_ role: Role) -> String? {
        var every: String? = nil
        for item in (forced ?? "").split(separator: ",") {
            let parts = item.split(separator: "=", maxSplits: 1).map(String.init)
            if parts.count == 2, parts[0] == role.rawValue { return parts[1] }
            if parts.count == 1 { every = parts[0] }
        }
        return every
    }

    static func tune(_ target: Module) {
        choices = [:]
        owners = []
        let started = DispatchTime.now().uptimeNanoseconds
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron seed dense tune: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.runGather else { return report("off (setting)") }
        var picks: [String] = []
        var details: [String] = []
        for (roleIndex, role) in Role.allCases.enumerated() {
            let admitted = role == .down ? NemotronPrefillSharedDownCopy.verified
                : NemotronPrefillDenseDict.kinds.contains(role.rawValue)
            let all = role == .down
                ? target.modules().compactMap { ($0 as? NemotronHMoE)?.sharedExperts?.downProj }
                : NemotronPrefillDenseDict.linears(target, kind: role.rawValue)
            let pool = variants.filter { role.offered.contains($0.name) || forcedName(role) == $0.name }
            guard admitted, !pool.isEmpty, let first = all.first as? QuantizedLinear else {
                picks.append("\(role.rawValue) V0")
                details.append("\(role.rawValue) off (\(admitted ? "no variant offered" : "no clone"))")
                continue
            }
            let k = first.weight.dim(1) * 8

            // Seeded rows (case 1 at large magnitudes); the shared down's
            // input is ReLU2-like rows, mostly exact zeros, as its clone's qualify.
            let cases: [MLXArray]
            do {
                cases = try withError { () -> [MLXArray] in
                    let made = [488, 488, 300, 97, minRows].enumerated().map { index, rows -> MLXArray in
                        var a = MLXRandom.normal([1, rows, k], scale: 2,
                                                 key: MLXRandom.key(UInt64(0xde5e_0000 + 16 * roleIndex + index)))
                        if index == 1 { a = a * 40 }
                        if role == .down, index != 1 {
                            a = MLX.maximum(a - 1.5, MLXArray(Float(0)))
                            a = a * a
                        }
                        return a.asType(.bfloat16)
                    }
                    eval(made)
                    return made
                }
            } catch {
                picks.append("\(role.rawValue) V0")
                details.append("\(role.rawValue) off (inputs: \(error))")
                continue
            }

            // Qualify: bit for bit against the clone; any mismatch or error drops a variant.
            var alive = pool
            var dropped: [String] = []
            caseLoop: for (ci, x) in cases.enumerated() {
                for linear in (ci == 0 ? all : [first]) {
                    let reference = try? withError { () -> MLXArray? in
                        guard let out = incumbent(role, x: x, linear: linear) else { return nil }
                        eval(out)
                        return out
                    }
                    guard let reference = reference ?? nil else {
                        dropped.append("clone unavailable")
                        alive = []
                        break caseLoop
                    }
                    alive = alive.filter { v in
                        do {
                            let same = try withError { () -> Bool in
                                guard let out = launch(v, role: role, x: x, linear: linear) else { return false }
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

            // Time the clone and the survivors round robin on the 488-row case
            // as one dependent chain over every layer of the role per
            // repetition (cold weights; each launch waits for the previous one,
            // as in the seed); median ms per layer of 10 after 5 warm-up.
            var choice: Variant? = nil
            var times: [String] = []
            if let name = forcedName(role) {
                choice = alive.first { $0.name == name }
            } else if !alive.isEmpty {
                let candidates: [Variant?] = [nil] + alive.map { Optional($0) }
                var samples = Array(repeating: [Double](), count: candidates.count)
                do {
                    try withError {
                        for rep in 0 ..< 15 {
                            for (i, candidate) in candidates.enumerated() {
                                let start = DispatchTime.now().uptimeNanoseconds
                                var x = cases[0]
                                var complete = true
                                for linear in all {
                                    let out = candidate.map { launch($0, role: role, x: x, linear: linear) }
                                        ?? incumbent(role, x: x, linear: linear)
                                    guard let y = out else {
                                        complete = false
                                        break
                                    }
                                    x = x + y[0..., 0..., ..<1] * 0
                                }
                                eval(x)
                                Stream.gpu.synchronize()
                                let elapsed = DispatchTime.now().uptimeNanoseconds - start
                                if rep >= 5, complete {
                                    samples[i].append(Double(elapsed) / 1e6 / Double(all.count))
                                }
                            }
                        }
                    }
                } catch {
                    dropped.append("timing error")
                }
                if samples.allSatisfy({ $0.count == 10 }) {
                    let medians = samples.map { $0.sorted()[$0.count / 2] }
                    times = zip(candidates, medians).map {
                        "\($0?.name ?? "V0") \(String(format: "%.3f", $1))"
                    }
                    if let best = medians.indices.dropFirst().min(by: { medians[$0] < medians[$1] }),
                       medians[best] <= 0.94 * medians[0] {
                        choice = candidates[best]
                    }
                }
            }
            if let choice {
                choices[role] = choice
                owners.formUnion(all.map { ObjectIdentifier($0) })
            }
            picks.append("\(role.rawValue) \(choice?.name ?? "V0")")
            details.append("\(role.rawValue) \(alive.count)/\(pool.count) bitwise"
                + (times.isEmpty ? "" : " [" + times.joined(separator: ", ") + "]")
                + (dropped.isEmpty ? "" : " dropped: " + dropped.prefix(8).joined(separator: ", ")))
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
        report(picks.joined(separator: ", ")
            + " (ms per layer, 488 rows, dependent chain over every layer, median of 10; "
            + details.joined(separator: "; ") + String(format: "; %.1f s)", seconds))
    }
}
