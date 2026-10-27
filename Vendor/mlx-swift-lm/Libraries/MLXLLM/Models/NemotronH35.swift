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

// B source: the shared down's zero-skip decode copy (codes as NemotronSeedCopy,
// tile-major BF16 scales and biases [E, N / 128, K / 64, 128]), as the shared
// down copy loader: each group's scale and bias widen to float.
template <typename T>
struct NemotronSeedCopySB {
  const device uint16_t* codes;
  const device T* sk;
  const device T* bk;
  int K;

  METAL_FUNC void expert(const uint index, const int N) thread {
    codes += size_t(index) * size_t(N / 128) * size_t(K) * 32;
    sk += size_t(index) * size_t(N / 128) * size_t(K / 64) * 128;
    bk += size_t(index) * size_t(N / 128) * size_t(K / 64) * 128;
  }
  METAL_FUNC float2 sb(const int n, const int g) const thread {
    const size_t i = (size_t(n / 128) * size_t(K / 64) + size_t(g)) * 128 + size_t(n % 128);
    const float s = float(sk[i]);
    const float b = float(bk[i]);
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
      // Rows past N (unaligned BN only) load row N - 1 and then store zeros,
      // the value load_safe gives them; the loads stay branch free.
      const short2 bc = BaseNAXFrag::get_coord();
      for (int kg = 0; kg < K; kg += 64) {
        float2 sbv[TN * 2];
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < TN; ++nn) {
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < 2; ++i) {
            const int n = y_col + tn + nn * 16 + bc.y + i * 8;
            sbv[nn * 2 + i] = bsrc.sb(ALIGN_N ? n : min(n, N - 1), kg / 64);
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
                  const int nl = ALIGN_N ? n : min(n, N - 1);
                  const int k = kg + kk1 + kk * 16 + bc.x;
                  const float2 p = sbv[nn * 2 + i];
                  T v0, v1, v2, v3;
                  nemotron_seed_deq2(bsrc.byte(nl, k), p.x, p.y, v0, v1);
                  nemotron_seed_deq2(bsrc.byte(nl, k + 2), p.x, p.y, v2, v3);
                  if (!ALIGN_N && n >= N) {
                    v0 = v1 = v2 = v3 = T(0);
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

/// Seed-only (request prefill, `NemotronHPrefillX5.active`) run gathers through a
/// retiled kernel (`nemotron_seed_gather_impl`) chosen once at load, per role:
/// the routed MoE fc1 and fc2, and the dense one-"expert" gathers (Mamba in_proj
/// and out_proj and the shared up through pair dictionaries,
/// `NemotronPrefillDenseDict`; the shared down through its decode copy,
/// `NemotronPrefillSharedDownCopy`). Every variant keeps each output's
/// `BaseNAXFrag::mma` sequence over k chunks 0..K/16 in order with the
/// incumbent's BF16 A and B operands and FP32 accumulators, and the incumbent's
/// typed store (`nemotronSeedRetileSource`); dense roles may also use 128-row
/// tiles (rows keep their 16-row fragments from the first row). `tune` (load
/// time, after every other qualify) compares each variant with the launch the
/// seed makes today bit for bit on seeded inputs (every layer in the first
/// case), drops any mismatch or Metal error, times the survivors and the
/// incumbent round robin on 488-row launches of the first layers (5 warm-up and
/// 25 timed repetitions, median), and per role adopts the fastest only if it is
/// at least 3% faster than the incumbent. NEMOTRON_SEED_RETILE=0 keeps every
/// incumbent, NEMOTRON_SEED_RETILE_DENSE=0 the dense ones;
/// NEMOTRON_SEED_RETILE_FORCE=<variant> adopts that variant wherever it qualifies.
enum NemotronSeedRetile {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE"] != "0"
    static let denseEnabled = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE_DENSE"] != "0"
    /// Default "d128w14": the direct-B variant (BN 128, one SIMD group per 32 output columns) is adopted
    /// for every role (routed and dense) whenever it qualifies bitwise, regardless of timing.
    /// "none" restores the timing-based choice.
    static let forced: String? = {
        let v = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE_FORCE"] ?? "d128w14"
        return v == "none" ? nil : v
    }()

    struct Variant: Sendable, Equatable {
        let name: String
        let bm: Int, bn: Int, bk: Int, wm: Int, wn: Int
        let direct: Bool
        var threads: Int { wm * wn * 32 }

        init(_ name: String, bm: Int = 64, bn: Int, bk: Int = 64, wm: Int, wn: Int, direct: Bool) {
            (self.name, self.bm, self.bn, self.bk, self.wm, self.wn, self.direct) = (name, bm, bn, bk, wm, wn, direct)
        }
    }

    /// ws: B through threadgroup memory; d: B dequantized into the fragments.
    /// Name = path, BN (x BK), WM WN, then m128 for 128-row tiles (dense only).
    /// ws64x64w22 is the incumbents' geometry.
    static let variants: [Variant] = [
        Variant("ws64x64w22", bn: 64, wm: 2, wn: 2, direct: false),
        Variant("ws64x128w22", bn: 64, bk: 128, wm: 2, wn: 2, direct: false),
        Variant("ws128x64w24", bn: 128, wm: 2, wn: 4, direct: false),
        Variant("ws128x64w14", bn: 128, wm: 1, wn: 4, direct: false),
        Variant("d64w22", bn: 64, wm: 2, wn: 2, direct: true),
        Variant("d64w12", bn: 64, wm: 1, wn: 2, direct: true),
        Variant("d128w14", bn: 128, wm: 1, wn: 4, direct: true),
        Variant("d128w24", bn: 128, wm: 2, wn: 4, direct: true),
    ]

    /// The dense gathers are compute bound at the seed: larger tiles.
    static let denseVariants: [Variant] = [
        Variant("ws64x64w22", bn: 64, wm: 2, wn: 2, direct: false),
        Variant("ws128x64w24", bn: 128, wm: 2, wn: 4, direct: false),
        Variant("ws128x64w24m128", bm: 128, bn: 128, wm: 2, wn: 4, direct: false),
        Variant("d64w22", bn: 64, wm: 2, wn: 2, direct: true),
        Variant("d128w14", bn: 128, wm: 1, wn: 4, direct: true),
        Variant("d128w24", bn: 128, wm: 2, wn: 4, direct: true),
        Variant("d128w14m128", bm: 128, bn: 128, wm: 1, wn: 4, direct: true),
        Variant("d128w24m128", bm: 128, bn: 128, wm: 2, wn: 4, direct: true),
        Variant("d128w22m128", bm: 128, bn: 128, wm: 2, wn: 2, direct: true),
    ]

    enum Source: String, Sendable { case bank, dict, copy, copySB }

    /// The launch a routed role's incumbent uses at the seed.
    struct Role: Sendable, Equatable {
        let routed: Bool
        let source: Source
        let relu2: Bool
        var label: String { (routed ? "routed-" : "") + source.rawValue }
    }

    nonisolated(unsafe) static var fc1Choice: (role: Role, variant: Variant)? = nil
    nonisolated(unsafe) static var fc2Choice: (role: Role, variant: Variant)? = nil
    /// Dense kinds: "in_proj", "out_proj", "up_proj" (pair dictionaries), "shared_down" (decode copy).
    nonisolated(unsafe) static var denseChoice: [String: Variant] = [:]

    private static func body(routed: Bool, source: Source) -> String {
        let make: String
        switch source {
        case .bank: make = "NemotronSeedBank<T> b{(const device uint8_t*)w, scales, biases, K};"
        case .dict: make = "NemotronSeedDict<T> b{(const device uint8_t*)w, scales, biases, K};"
        case .copy: make = "NemotronSeedCopy<T> b{w, scales, biases, K};"
        case .copySB: make = "NemotronSeedCopySB<T> b{w, scales, biases, K};"
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
        // A tile of at most BM / 2 rows runs the BM / 2 body (for BM 64 the run
        // gather's 32-row switch); either way each row keeps its 16-row fragment.
        return """
                threadgroup T Ws[DIRECT_B ? 1 : BN * (BK + 16 / sizeof(T))];
                \(make)
                if (BM > 32 && tiles[threadgroup_position_in_grid.y * 3 + 1] <= BM / 2) {
            \(call("BM / 2"))
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
    private static let plainDict = makeKernel(routed: false, source: .dict)
    private static let plainCopySB = makeKernel(routed: false, source: .copySB)

    /// One retiled routed MoE launch; nil wherever the incumbent's guards would not admit it.
    static func launch(_ v: Variant, role: Role, x: MLXArray, lin: QuantizedSwitchLinear,
                       rows: MLXArray?, tiles: NemotronHRunTiles) -> MLXArray? {
        guard NemotronHPrefillX5.runGather, lin.mode == .affine, lin.bits == 4, lin.groupSize == 64,
              x.ndim == 3, x.dim(1) == 1, x.dtype == .bfloat16,
              !role.relu2 || MLX.maximum(x, MLXArray(0)).dtype == x.dtype
        else { return nil }
        let ops = lin.nemotronGatherOperands(for: x.dtype)
        guard ops.weight.dtype == .uint32, ops.weight.ndim == 3 else { return nil }
        let n = tiles.rows, k = x.dim(-1), experts = ops.weight.dim(0), nOut = ops.weight.dim(1)
        guard v.bm == tiles.bm, n >= 16, n / experts >= 4,
              tiles.bm == nemotronGatherTileRows(rows: n, experts: experts),
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
        case .copySB:
            return nil
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

    /// One retiled dense launch over the request's rows (`x`: [1, rows, K]) in
    /// tiles of `v.bm` rows (at least three descriptors, as the incumbents); the
    /// guards are `NemotronPrefillDenseDict.launch`'s (.dict) or
    /// `NemotronPrefillSharedDownCopy.launch`'s (.copySB).
    static func denseLaunch(_ v: Variant, source: Source, x: MLXArray, linear: Linear, relu2: Bool) -> MLXArray? {
        guard NemotronHPrefillX5.runGather, let q = linear as? QuantizedLinear,
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
                  q.weight.dtype == .uint32, q.weight.ndim == 2,
                  let d = NemotronSBDict.dict(q), d.index.shape == q.scales.shape, d.index.dtype == .uint16,
                  d.table.dtype == .float32, q.scales.shape == [nOut, k / 64]
            else { return nil }
            (kernel, operands) = (plainDict, [q.weight.expandedDimensions(axis: 0),
                                              d.index.expandedDimensions(axis: 0), d.table])
        case .copySB:
            guard !relu2, nOut % 128 == 0, let copy = NemotronSharedDownZSkip.copyArrays(for: linear),
                  copy.codes.shape == [1, nOut / 128, k, 32], copy.codes.dtype == .uint16,
                  copy.scales.shape == [1, nOut / 128, k / 64, 128], copy.scales.dtype == x.dtype,
                  copy.biases.shape == copy.scales.shape, copy.biases.dtype == x.dtype
            else { return nil }
            (kernel, operands) = (plainCopySB, [copy.codes, copy.scales, copy.biases])
        default:
            return nil
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
                       ("DIRECT_B", v.direct), ("ALIGN_N", nOut % v.bn == 0), ("RELU2", relu2)],
            grid: (((nOut + v.bn - 1) / v.bn) * v.threads, count, 1), threadGroup: (v.threads, 1, 1),
            outputShapes: [[rows, 1, nOut]], outputDTypes: [x.dtype])[0]
        return y.reshaped(1, rows, nOut)
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

    /// From `NemotronPrefillDenseDict.apply` (request prefill, admitted kinds only).
    static func dense(_ x: MLXArray, _ linear: Linear, kind: String, relu2: Bool) -> MLXArray? {
        guard let v = denseChoice[kind], relu2 == (kind == "up_proj") else { return nil }
        return denseLaunch(v, source: .dict, x: x, linear: linear, relu2: relu2)
    }

    /// From `NemotronPrefillSharedDownCopy.apply` (already request prefill only).
    static func sharedDown(_ x: MLXArray, _ down: Linear) -> MLXArray? {
        guard let v = denseChoice["shared_down"] else { return nil }
        return denseLaunch(v, source: .copySB, x: x, linear: down, relu2: false)
    }

    /// The launch the seed makes today for a routed `role`.
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
        case .copySB:
            return nil
        }
    }

    private struct Outcome {
        let alive: [Variant]
        let dropped: [String]
        let medians: [(Variant?, Double)]
        let choice: Variant?
    }

    /// Qualifies `variants` against the incumbent (`run(nil, case, layer)`) bit for
    /// bit on every case (every layer in case 0; any mismatch or error drops a
    /// variant), then times the incumbent and the survivors round robin on case 0
    /// over the first `timed` layers: 5 warm-up and 25 timed repetitions, median
    /// ms per launch. The choice is the fastest survivor if at least 3% faster.
    private static func evaluate(_ variants: [Variant], cases: Int, layers: Int, timed: Int,
                                 run: (Variant?, Int, Int) -> MLXArray?) -> Outcome {
        var alive = variants
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
        let n = max(1, min(timed, layers))
        var samples = Array(repeating: [Double](), count: candidates.count)
        do {
            try withError {
                for rep in 0 ..< 30 {
                    for (i, candidate) in candidates.enumerated() {
                        let start = DispatchTime.now().uptimeNanoseconds
                        let outs = (0 ..< n).compactMap { run(candidate, 0, $0) }
                        eval(outs)
                        Stream.gpu.synchronize()
                        let elapsed = DispatchTime.now().uptimeNanoseconds - start
                        if rep >= 5, outs.count == n {
                            samples[i].append(Double(elapsed) / 1e6 / Double(n))
                        }
                    }
                }
            }
        } catch {
            dropped.append("timing error")
        }
        guard samples.allSatisfy({ $0.count == 25 }) else {
            return Outcome(alive: alive, dropped: dropped, medians: [], choice: nil)
        }
        let medians = zip(candidates, samples).map { candidate, s in (candidate, s.sorted()[s.count / 2]) }
        var choice: Variant? = nil
        if let forced, let f = alive.first(where: { $0.name == forced }) {
            choice = f
        } else if let best = medians.dropFirst().min(by: { $0.1 < $1.1 }), best.1 <= 0.97 * medians[0].1 {
            choice = best.0
        }
        return Outcome(alive: alive, dropped: dropped, medians: medians, choice: choice)
    }

    static func tune(_ target: Module) {
        fc1Choice = nil
        fc2Choice = nil
        denseChoice = [:]
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron seed retile: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.runGather else { return report("off (setting)") }
        var head: [String] = []
        var detail: [String] = []
        func note(_ name: String, _ label: String, _ total: Int, _ o: Outcome) {
            func ms(_ v: Variant?) -> String {
                o.medians.first(where: { $0.0 == v }).map { String(format: "%.2f", $0.1) } ?? "n/a"
            }
            head.append("\(name) \(o.choice?.name ?? "V0") \(ms(o.choice)) ms vs V0 \(ms(nil))")
            let times = o.medians.dropFirst().map { "\($0.0!.name) \(String(format: "%.2f", $0.1))" }
            detail.append("\(name) \(label) \(o.alive.count)/\(total) bitwise"
                + (times.isEmpty ? "" : " [" + times.joined(separator: ", ") + "]")
                + (o.dropped.isEmpty ? "" : " dropped: " + o.dropped.prefix(6).joined(separator: ", ")))
        }

        // Routed MoE fc1 / fc2 (the launches `fusedSortedCombine` makes at the seed).
        let moes = target.modules().compactMap { $0 as? NemotronHMoE }
        if NemotronPrefillCountingSort.verified, let moe = moes.first, moes.allSatisfy({
            $0.switchMLP.fc1 is QuantizedSwitchLinear && $0.switchMLP.fc2 is QuantizedSwitchLinear
        }), let fc2First = moe.switchMLP.fc2 as? QuantizedSwitchLinear {
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
            // Seeded routes and inputs (case 2: ReLU2-like rows, mostly exact zeros).
            typealias Case = (tiles: NemotronHRunTiles, rows: MLXArray, x: MLXArray, xs: MLXArray, act: MLXArray)
            let cases = (try? withError { () -> [Case] in
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
                    let c: Case = (sorted.tiles, sorted.orderK, xb, xb[sorted.orderK], act.asType(.bfloat16))
                    eval([c.tiles.tiles, c.rows, c.x, c.xs, c.act])
                    made.append(c)
                }
                return made
            }) ?? []
            if cases.count == 5 {
                for r in 0 ..< 2 {
                    let role = roles[r]
                    let o = evaluate(variants, cases: cases.count, layers: moes.count, timed: 4) { v, c, layer in
                        let lin = (r == 0 ? moes[layer].switchMLP.fc1 : moes[layer].switchMLP.fc2) as! QuantizedSwitchLinear
                        let x: MLXArray = r == 0 ? (role.routed ? cases[c].x : cases[c].xs) : cases[c].act
                        let rows: MLXArray? = r == 0 && role.routed ? cases[c].rows : nil
                        if let v { return launch(v, role: role, x: x, lin: lin, rows: rows, tiles: cases[c].tiles) }
                        return incumbent(role, x: x, lin: lin, rows: rows, tiles: cases[c].tiles)
                    }
                    if r == 0 { fc1Choice = o.choice.map { (role, $0) } } else { fc2Choice = o.choice.map { (role, $0) } }
                    note(r == 0 ? "fc1" : "fc2", role.label, variants.count, o)
                }
            } else {
                detail.append("moe off (inputs)")
            }
        } else {
            detail.append("moe off (setting)")
        }

        // Dense one-expert gathers (the launches the seed makes for the admitted kinds).
        guard denseEnabled else {
            detail.append("dense off (setting)")
            return report(head.joined(separator: ", ") + " (per launch, 488 rows, median of 25; "
                + detail.joined(separator: "; ") + ")")
        }
        let rowCounts = [488, 488, 300, 97, 17]
        func denseInputs(_ k: Int, seed: UInt64, zeros: (Int) -> Bool) -> [MLXArray] {
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
            let xs = denseInputs(first.weight.dim(1) * 8, seed: UInt64(0x5eed_d000 + 16 * kindIndex),
                                 zeros: { $0 == 3 })
            guard xs.count == rowCounts.count else {
                detail.append("\(kind) off (inputs)")
                continue
            }
            let o = evaluate(denseVariants, cases: xs.count, layers: all.count, timed: 2) { v, c, layer in
                if let v { return denseLaunch(v, source: .dict, x: xs[c], linear: all[layer], relu2: relu2) }
                return NemotronPrefillDenseDict.launch(xs[c], all[layer], relu2: relu2)
            }
            if let choice = o.choice { denseChoice[kind] = choice }
            note(kind, "dict", denseVariants.count, o)
        }
        let downs = moes.compactMap { $0.sharedExperts?.downProj }
        if NemotronPrefillSharedDownCopy.verified, let first = downs.first {
            // ReLU2-like rows (the shared up's output) except the scaled case.
            let xs = denseInputs(first.weight.dim(1) * 8, seed: 0x5eed_d500, zeros: { $0 != 1 })
            if xs.count == rowCounts.count {
                let o = evaluate(denseVariants, cases: xs.count, layers: downs.count, timed: 2) { v, c, layer in
                    if let v { return denseLaunch(v, source: .copySB, x: xs[c], linear: downs[layer], relu2: false) }
                    return NemotronPrefillSharedDownCopy.launch(xs[c], downs[layer])
                }
                if let choice = o.choice { denseChoice["shared_down"] = choice }
                note("shared_down", "copy", denseVariants.count, o)
            } else {
                detail.append("shared_down off (inputs)")
            }
        } else {
            detail.append("shared_down off (not verified)")
        }
        report(head.joined(separator: ", ") + " (per launch, 488 rows, median of 25; "
            + detail.joined(separator: "; ") + ")")
    }
}
