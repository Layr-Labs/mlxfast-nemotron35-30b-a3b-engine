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

// WF (wide fill) of a bank-layout source (the bank, its pair dictionary):
// thread t of NT takes the per-row fill's KPT consecutive k of one output row
// and reads their codes as 32-bit words (little endian: byte j = bits 8j ...
// 8j + 7) from the source's `words` (the address `byte` reads for the first
// of them) instead of one byte per k pair, with the per-row fill's scale and
// bias (`sb`). `load` reads them into registers under the per-row fill's
// predicate; `store` writes the per-row fill's values from them with its
// statements (zeros outside the predicate).
template <int BN, int BK, int NT>
struct NemotronSeedWordRegs {
  uint word[(BN * BK / NT) / 8];
  float2 p;
};

template <int BN, int BK, int NT, bool ALIGN_N, class S>
METAL_FUNC void nemotron_seed_word_load(
    thread const S& s, thread NemotronSeedWordRegs<BN, BK, NT>& r, const int t,
    const int y_col, const int N, const int kb, const int klim) {
  constexpr int KPT = BN * BK / NT;
  static_assert(KPT % 8 == 0 && KPT <= 64 && 64 % KPT == 0, "word fill split");
  const int lr = t / (BK / KPT);
  const int lk = (t % (BK / KPT)) * KPT;
  const int n = y_col + lr;
  if ((ALIGN_N || n < N) && lk < klim) {
    r.p = s.sb(n, (kb + lk) / 64);
    const device uint32_t* src = s.words(n, kb + lk);
    STEEL_PRAGMA_UNROLL
    for (short q = 0; q < KPT / 8; ++q) {
      r.word[q] = src[q];
    }
  }
}

template <typename T, int BN, int BK, int NT, bool ALIGN_N>
METAL_FUNC void nemotron_seed_word_store(
    threadgroup T* ws, thread const NemotronSeedWordRegs<BN, BK, NT>& r, const int t,
    const int y_col, const int N, const int klim) {
  constexpr int BK_padded = BK + 16 / sizeof(T);
  constexpr int KPT = BN * BK / NT;
  const int lr = t / (BK / KPT);
  const int lk = (t % (BK / KPT)) * KPT;
  const int n = y_col + lr;
  threadgroup T* dst = ws + lr * BK_padded + lk;
  if ((ALIGN_N || n < N) && lk < klim) {
    STEEL_PRAGMA_UNROLL
    for (short q = 0; q < KPT / 8; ++q) {
      const uint word = r.word[q];
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < 4; ++j) {
        T v0, v1;
        nemotron_seed_deq2(uint8_t((word >> (8 * j)) & 0xffu), r.p.x, r.p.y, v0, v1);
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

// tile_matmad_nax (A not transposed, TN even: its fragment-pair branch), except
// that a 16-row fragment after the first is skipped when none of its rows is
// below valid_rows (uniform across the SIMD group). A skipped fragment is never
// stored; every other fragment gets tile_matmad_nax's mma sequence. (Here, ahead
// of the B sources, so the lean texts derived below do not repeat it.)
template <class CTile, class ATile, class BTile, bool transpose_b>
METAL_FUNC void nemotron_seed_matmad_rows(
    thread CTile& C,
    thread ATile& A,
    thread BTile& B,
    metal::bool_constant<transpose_b>,
    const short valid_rows) {
  constexpr short TM = CTile::kTileRows;
  constexpr short TN = CTile::kTileCols;
  constexpr short TK = transpose_b ? BTile::kTileCols : BTile::kTileRows;
  static_assert(ATile::kTileRows == TM && ATile::kTileCols == TK, "A tile");
  static_assert((transpose_b ? BTile::kTileRows : BTile::kTileCols) == TN && TN % 2 == 0, "B tile");
  constexpr auto tb = metal::bool_constant<transpose_b>{};
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < TM; ++mm) {
    if (mm == 0 || mm * CTile::kFragRows < valid_rows) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; nn += 2) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn),
              C.frag_at(mm, nn + 1),
              A.frag_at(mm, kk),
              metal::bool_constant<false>{},
              B.frag_at(kk, nn, tb),
              B.frag_at(kk, nn + 1, tb),
              metal::bool_constant<transpose_b>{});
        }
      }
    }
  }
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
  // HB: byte(n, k) | byte(n, k + 2) << 8 as one 16-bit load (k % 4 == 0 and
  // K / 2 even, so the address is 2-byte aligned).
  METAL_FUNC uint16_t code16(const int n, const int k) const thread {
    return *(const device uint16_t*)(w + size_t(n) * size_t(K / 2) + size_t(k >> 1));
  }

  // WF: the word fill (nemotron_seed_word_load / _store). `words`: the
  // address `byte` reads for k (a multiple of 8), as 32-bit code words.
  METAL_FUNC const device uint32_t* words(const int n, const int k) const thread {
    return (const device uint32_t*)(w + size_t(n) * size_t(K / 2) + size_t(k >> 1));
  }
  template <int BN, int BK, int NT>
  using Regs = NemotronSeedWordRegs<BN, BK, NT>;
  template <int BN, int BK, int NT, bool ALIGN_N>
  METAL_FUNC void load(thread Regs<BN, BK, NT>& r, const int t, const int y_col, const int N,
                       const int kb, const int klim) const thread {
    nemotron_seed_word_load<BN, BK, NT, ALIGN_N>(*this, r, t, y_col, N, kb, klim);
  }
  template <int BN, int BK, int NT, bool ALIGN_N>
  METAL_FUNC void store(threadgroup T* ws, thread const Regs<BN, BK, NT>& r, const int t, const int y_col,
                        const int N, const int klim) const thread {
    nemotron_seed_word_store<T, BN, BK, NT, ALIGN_N>(ws, r, t, y_col, N, klim);
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
  METAL_FUNC uint16_t code16(const int n, const int k) const thread {
    return *(const device uint16_t*)(w + size_t(n) * size_t(K / 2) + size_t(k >> 1));
  }

  // WF: the word fill (nemotron_seed_word_load / _store). `words`: the
  // address `byte` reads for k (a multiple of 8), as 32-bit code words.
  METAL_FUNC const device uint32_t* words(const int n, const int k) const thread {
    return (const device uint32_t*)(w + size_t(n) * size_t(K / 2) + size_t(k >> 1));
  }
  template <int BN, int BK, int NT>
  using Regs = NemotronSeedWordRegs<BN, BK, NT>;
  template <int BN, int BK, int NT, bool ALIGN_N>
  METAL_FUNC void load(thread Regs<BN, BK, NT>& r, const int t, const int y_col, const int N,
                       const int kb, const int klim) const thread {
    nemotron_seed_word_load<BN, BK, NT, ALIGN_N>(*this, r, t, y_col, N, kb, klim);
  }
  template <int BN, int BK, int NT, bool ALIGN_N>
  METAL_FUNC void store(threadgroup T* ws, thread const Regs<BN, BK, NT>& r, const int t, const int y_col,
                        const int N, const int klim) const thread {
    nemotron_seed_word_store<T, BN, BK, NT, ALIGN_N>(ws, r, t, y_col, N, klim);
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
  // CN: the code word of n ... n + 3 at k (n % 4 == 0): nibble j is n + j's code.
  METAL_FUNC uint16_t word(const int n, const int k) const thread {
    return codes[(size_t(n / 128) * size_t(K) + size_t(k)) * 32 + size_t((n % 128) / 4)];
  }

  // WF: thread t of NT holds four consecutive outputs n ... n + 3 (one code
  // word lane, n a multiple of 4) and CK consecutive k, as the fc2 copy
  // loader, so each code word is read once for its four outputs. `load` reads
  // the CK code words and the four outputs' `sb` under the per-row fill's
  // predicate (N % 4 == 0: the four outputs are in range together); `store`
  // rebuilds each bank byte as `byte` does and writes the per-row fill's values
  // at the per-row fill's positions (zeros outside the predicate).
  template <int BN, int BK, int NT>
  struct Regs {
    uint16_t c[BK * (BN / 4) / NT];
    float s[4];
    float b[4];
  };

  template <int BN, int BK, int NT, bool ALIGN_N>
  METAL_FUNC void load(thread Regs<BN, BK, NT>& r, const int t, const int y_col, const int N,
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
      // sb(n + o, k0 / 64)'s pair index for o = 0 ... 3: n % 4 == 0, so
      // (n + o) / 128 == n / 128 and (n + o) % 128 == n % 128 + o.
      const device uint16_t* si =
          sidx + (size_t(n / 128) * size_t(K / 64) + size_t(k0 / 64)) * 128 + size_t(n % 128);
      STEEL_PRAGMA_UNROLL
      for (short o = 0; o < 4; o++) {
        const uint pair = uint(si[o]);
        r.s[o] = table[2 * pair];
        r.b[o] = table[2 * pair + 1];
      }
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < CK; ++j) {
        r.c[j] = c[j * 32];
      }
    }
  }

  template <int BN, int BK, int NT, bool ALIGN_N>
  METAL_FUNC void store(threadgroup T* ws, thread const Regs<BN, BK, NT>& r, const int t, const int y_col,
                        const int N, const int klim) const thread {
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
          const uint lo = (uint(r.c[j]) >> (4 * o)) & 0xfu;
          const uint hi = (uint(r.c[j + 1]) >> (4 * o)) & 0xfu;
          T v0, v1;
          nemotron_seed_deq2(uint8_t(lo | (hi << 4)), r.s[o], r.b[o], v0, v1);
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
    bool PF,
    bool DB,
    bool WF,
    bool HB,
    bool CN,
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
    if constexpr (CN) {
      // CN (decode copy source): B enters the mma as K x N (transpose_b
      // false), so lane (fn, fm) = get_coord() holds k = 16-chunk + fm + 8 i
      // and n = 16-column block + fn + j, j = 0 ... 3: nibble j of one copy
      // word (word(n, k) for n = block + fn). Every k a lane holds has fm's
      // parity, so the lane applies one of the copy loader's two statements
      // to the same FP32 operands: s * lo + b (even k) or
      // (s / 16) * (hi << 4) + b (odd k). Same mma pairs and k order as the
      // other paths, rows guarded as HB. PF: step k + 32's words (and, at a
      // group start, its scales and biases) are loaded after step k's B
      // fragments are dequantized and before step k's mma.
      static_assert(ALIGN_N && BN % 4 == 0, "copy words cover four columns");
      const short2 bc = BaseNAXFrag::get_coord();
      const uint sh = uint(bc.y & 1) * 4;
      const uint msk = 0xfu << sh;
      float scv[TN * 4];
      float bsv[TN * 4];
      uint16_t raw[TN * TK * 2];
      auto load_sb = [&](const int kg) {
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < TN; ++nn) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < 4; ++j) {
            const float2 p = bsrc.sb(y_col + tn + nn * 16 + bc.x + j, kg / 64);
            scv[nn * 4 + j] = sh != 0 ? p.x / 16.0f : p.x;
            bsv[nn * 4 + j] = p.y;
          }
        }
      };
      auto load_raw = [&](const int ks) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          STEEL_PRAGMA_UNROLL
          for (short nn = 0; nn < TN; ++nn) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; ++i) {
              raw[(kk * TN + nn) * 2 + i] =
                  bsrc.word(y_col + tn + nn * 16 + bc.x, ks + kk * 16 + bc.y + i * 8);
            }
          }
        }
      };
      if constexpr (PF) {
        if (sg_active) {
          load_sb(0);
          load_raw(0);
        }
      }
      STEEL_PRAGMA_NO_UNROLL
      for (int ks = 0; ks < K; ks += SK) {
        if (sg_active) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<T, TK, TN> Btile;

          volatile int compiler_barrier;

          if constexpr (!PF) {
            if (ks % 64 == 0) {
              load_sb(ks);
            }
            load_raw(ks);
          }

          nemotron_seed_load_a<ROUTED, aligned_m, SK>(Atile, xn, routed, ks, K, sgp_sm);

          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < TK; ++kk) {
            STEEL_PRAGMA_UNROLL
            for (short nn = 0; nn < TN; ++nn) {
              thread typename NAXTile<T, TK, TN>::frag_type& frag = Btile.frag_at(kk, nn);
              STEEL_PRAGMA_UNROLL
              for (short i = 0; i < 2; ++i) {
                const uint w = uint(raw[(kk * TN + nn) * 2 + i]) << sh;
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; ++j) {
                  frag[i * 4 + j] = static_cast<T>(
                      scv[nn * 4 + j] * float((w >> (4 * j)) & msk) + bsv[nn * 4 + j]);
                }
              }
            }
          }

          if constexpr (PF) {
            const int kn = ks + SK;
            if (kn < K) {
              if (kn % 64 == 0) {
                load_sb(kn);
              }
              load_raw(kn);
            }
          }

          nemotron_seed_matmad_rows(Dtile, Atile, Btile, metal::bool_constant<false>{}, sgp_sm);

          (void)compiler_barrier;
        }
      }
    } else if constexpr (DIRECT_B && PF) {
      // PF: the direct path's per-step loop over k = 0, 32, ... with step
      // k + 32's code bytes (and, at a group start, its scales and biases)
      // loaded into registers after step k's B fragments are dequantized and
      // before step k's mma. Same bytes, same dequantize statements, same mma
      // order as the loop below; only the loads move. HB: as there.
      const short2 bc = BaseNAXFrag::get_coord();
      float2 sbv[TN * 2];
      uint8_t raw[HB ? 1 : TN * TK * 2 * 2];
      uint16_t raw16[HB ? TN * TK * 2 : 1];
      auto load_sb = [&](const int kg) {
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < TN; ++nn) {
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < 2; ++i) {
            const int n = y_col + tn + nn * 16 + bc.y + i * 8;
            sbv[nn * 2 + i] = (ALIGN_N || n < N) ? bsrc.sb(n, kg / 64) : float2(0.0f);
          }
        }
      };
      auto load_raw = [&](const int ks) {
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < TN; ++nn) {
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < TK; ++kk) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; ++i) {
              const int n = y_col + tn + nn * 16 + bc.y + i * 8;
              const int k = ks + kk * 16 + bc.x;
              const short q = ((nn * TK + kk) * 2 + i) * 2;
              if (ALIGN_N || n < N) {
                if constexpr (HB) {
                  raw16[q / 2] = bsrc.code16(n, k);
                } else {
                  raw[q] = bsrc.byte(n, k);
                  raw[q + 1] = bsrc.byte(n, k + 2);
                }
              }
            }
          }
        }
      };
      if (sg_active) {
        load_sb(0);
        load_raw(0);
      }
      STEEL_PRAGMA_NO_UNROLL
      for (int ks = 0; ks < K; ks += SK) {
        if (sg_active) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<T, TN, TK> Btile;

          volatile int compiler_barrier;

          nemotron_seed_load_a<ROUTED, aligned_m, SK>(Atile, xn, routed, ks, K, sgp_sm);

          STEEL_PRAGMA_UNROLL
          for (short nn = 0; nn < TN; ++nn) {
            STEEL_PRAGMA_UNROLL
            for (short kk = 0; kk < TK; ++kk) {
              thread typename NAXTile<T, TN, TK>::frag_type& frag = Btile.frag_at(nn, kk);
              STEEL_PRAGMA_UNROLL
              for (short i = 0; i < 2; ++i) {
                const int n = y_col + tn + nn * 16 + bc.y + i * 8;
                const short q = ((nn * TK + kk) * 2 + i) * 2;
                T v0 = T(0), v1 = T(0), v2 = T(0), v3 = T(0);
                if (ALIGN_N || n < N) {
                  const float2 p = sbv[nn * 2 + i];
                  if constexpr (HB) {
                    const uint16_t h = raw16[q / 2];
                    nemotron_seed_deq2(uint8_t(h & 0xff), p.x, p.y, v0, v1);
                    nemotron_seed_deq2(uint8_t(h >> 8), p.x, p.y, v2, v3);
                  } else {
                    nemotron_seed_deq2(raw[q], p.x, p.y, v0, v1);
                    nemotron_seed_deq2(raw[q + 1], p.x, p.y, v2, v3);
                  }
                }
                frag[i * 4 + 0] = v0;
                frag[i * 4 + 1] = v1;
                frag[i * 4 + 2] = v2;
                frag[i * 4 + 3] = v3;
              }
            }
          }

          const int kn = ks + SK;
          if (kn < K) {
            if (kn % 64 == 0) {
              load_sb(kn);
            }
            load_raw(kn);
          }

          if constexpr (HB) {
            nemotron_seed_matmad_rows(Dtile, Atile, Btile, metal::bool_constant<transpose>{}, sgp_sm);
          } else {
            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});
          }

          (void)compiler_barrier;
        }
      }
    } else if constexpr (DIRECT_B) {
      // Each thread dequantizes its own B fragment elements: rows n =
      // tn + 16 nn + coord.y + 8 i, k = 16-chunk + coord.x + j (the
      // positions BaseNAXFrag::load reads from Ws). HB: the two code bytes
      // of a quad (k and k + 2) come from one 16-bit load and feed the same
      // dequantize statements; the mma skips 16-row fragments without a
      // valid row (nemotron_seed_matmad_rows).
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
                    if constexpr (HB) {
                      const uint16_t h = bsrc.code16(n, k);
                      nemotron_seed_deq2(uint8_t(h & 0xff), p.x, p.y, v0, v1);
                      nemotron_seed_deq2(uint8_t(h >> 8), p.x, p.y, v2, v3);
                    } else {
                      nemotron_seed_deq2(bsrc.byte(n, k), p.x, p.y, v0, v1);
                      nemotron_seed_deq2(bsrc.byte(n, k + 2), p.x, p.y, v2, v3);
                    }
                  }
                  frag[i * 4 + 0] = v0;
                  frag[i * 4 + 1] = v1;
                  frag[i * 4 + 2] = v2;
                  frag[i * 4 + 3] = v3;
                }
              }
            }

            if constexpr (HB) {
              nemotron_seed_matmad_rows(Dtile, Atile, Btile, metal::bool_constant<transpose>{}, sgp_sm);
            } else {
              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<false>{},
                  Btile,
                  metal::bool_constant<transpose>{});
            }

            (void)compiler_barrier;
          }
        }
      }
    } else if constexpr (DB || WF) {
      // The threadgroup-staged path below over blocks b = 0, 1, ... of BK k
      // (the last one K - b * BK k when BK does not divide K: the same blocks
      // and the same compute statements). WF: the B source's wide fill
      // (`load` then `store`) writes each block; otherwise the per-row fill
      // below. Both write the same value to every Ws slot the compute reads.
      // DB: two stages, block b in stage b & 1, one barrier per block (block
      // b + 1 is written while block b is read; the barrier closing iteration
      // b - 1 orders both). PF: each block's B loads are issued into registers
      // one block ahead (two with DB), before the current block's mma.
      constexpr int NT = WM * WN * 32;
      constexpr int STAGE = BN * BK_padded;
      constexpr int KPT = BN * BK / NT;
      static_assert(WF || (KPT % 4 == 0 && KPT <= 64 && 64 % KPT == 0 && BK % KPT == 0), "loader split");
      // The per-row fill's split (unused with WF).
      constexpr int KPR = WF ? 1 : BK / KPT;
      const int t = int(simd_group_id) * 32 + int(simd_lane_id);
      const int nb = (K + BK - 1) / BK;
      const short lr = short(t / KPR);
      const short lk = short((t % KPR) * KPT);
      const int ln = y_col + lr;
      typename BSrc::template Regs<BN, BK, NT> wregs;
      uint8_t pf_bytes[WF ? 1 : KPT / 2];
      float2 pf_sb = float2(0.0f);
      auto pf_load = [&](const int b) {
        const int kb = b * BK;
        const int klim = min(BK, K - kb);
        if constexpr (WF) {
          bsrc.template load<BN, BK, NT, ALIGN_N>(wregs, t, y_col, N, kb, klim);
        } else {
          if ((ALIGN_N || lr < tgp_bn) && lk < klim) {
            pf_sb = bsrc.sb(ln, (kb + lk) / 64);
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < KPT; j += 2) {
              pf_bytes[j / 2] = bsrc.byte(ln, kb + lk + j);
            }
          }
        }
      };
      auto pf_store = [&](threadgroup T* ws, const int b) {
        const int klim = min(BK, K - b * BK);
        if constexpr (WF) {
          bsrc.template store<BN, BK, NT, ALIGN_N>(ws, wregs, t, y_col, N, klim);
        } else {
          threadgroup T* wdst = ws + lr * BK_padded + lk;
          if ((ALIGN_N || lr < tgp_bn) && lk < klim) {
            const float2 p = pf_sb;
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < KPT; j += 2) {
              T v0, v1;
              nemotron_seed_deq2(pf_bytes[j / 2], p.x, p.y, v0, v1);
              wdst[j] = v0;
              wdst[j + 1] = v1;
            }
          } else {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < KPT; ++j) {
              wdst[j] = T(0);
            }
          }
        }
      };
      auto fill = [&](threadgroup T* ws, const int b) {
        pf_load(b);
        pf_store(ws, b);
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
      if constexpr (DB && PF) {
        pf_load(0);
        pf_store(Ws, 0);
        if (1 < nb) {
          pf_load(1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int b = 0; b < nb; b++) {
          if (b + 1 < nb) {
            pf_store(Ws + ((b + 1) & 1) * STAGE, b + 1);
            if (b + 2 < nb) {
              pf_load(b + 2);
            }
          }
          compute(Ws + (b & 1) * STAGE, b);
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      } else if constexpr (DB) {
        fill(Ws, 0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int b = 0; b < nb; b++) {
          if (b + 1 < nb) {
            fill(Ws + ((b + 1) & 1) * STAGE, b + 1);
          }
          compute(Ws + (b & 1) * STAGE, b);
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      } else if constexpr (PF) {
        pf_load(0);
        for (int b = 0; b < nb; b++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          pf_store(Ws, b);
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (b + 1 < nb) {
            pf_load(b + 1);
          }
          compute(Ws, b);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      } else {
        for (int b = 0; b < nb; b++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          fill(Ws, b);
          threadgroup_barrier(mem_flags::mem_threadgroup);
          compute(Ws, b);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
      // PF: block kb + BK's code bytes, scale and bias are loaded into
      // registers after block kb's Ws write and its barrier, before block kb's
      // mma; the next fill dequantizes them with fill's statements.
      uint8_t pf_bytes[PF ? KPT / 2 : 1];
      float2 pf_sb = float2(0.0f);
      auto prefetch = [&](const int kb, const int klim) {
        if ((ALIGN_N || lr < tgp_bn) && lk < klim) {
          pf_sb = bsrc.sb(ln, (kb + lk) / 64);
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < KPT; j += 2) {
            pf_bytes[j / 2] = bsrc.byte(ln, kb + lk + j);
          }
        }
      };
      auto fill_pf = [&](const int klim) {
        if ((ALIGN_N || lr < tgp_bn) && lk < klim) {
          const float2 p = pf_sb;
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < KPT; j += 2) {
            T v0, v1;
            nemotron_seed_deq2(pf_bytes[j / 2], p.x, p.y, v0, v1);
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
      const int K_it = K / BK;
      // K % 64 == 0, so the tail is whole groups and whole 32-k steps.
      const int k_rem = K - K_it * BK;
      auto block = [&](const int kb, const int klim) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (PF) {
          fill_pf(klim);
        } else {
          fill(kb, klim);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (PF) {
          const int kn = kb + klim;
          const int kn_lim = kn + BK <= K ? BK : K - kn;
          if (kn_lim > 0) {
            prefetch(kn, kn_lim);
          }
        }

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
      if constexpr (PF) {
        prefetch(0, K_it > 0 ? BK : k_rem);
      }
      for (int kb_i = 0; kb_i < K_it; kb_i++) {
        block(kb_i * BK, BK);
      }
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

/// NEMOTRON_SEED_LEAN: lean twins of the seed retile kernels, offered as extra rows to `NemotronSeedRetile`,
/// `NemotronSeedDenseRetile` and `NemotronSeedQMMTune` (each row passes its tuner's bit check and adoption rule
/// like any other). Every text is derived from the record text by exact swaps; nil (no rows) when one misses.
/// - B sources (`...Ln`): the record structs with each lane's code, scale, bias and pair-index element offsets in
///   32-bit `uint` where the record forms `size_t` (every offset lies inside one bank, decode copy or projection:
///   under 128 x 2688 x 1344 bytes, below 2^32).
/// - The routed gather keeps one 32-bit row offset per fragment row (`routed_rows[r] * K`, below 2^32 for any
///   seed) read from the `x` base, where the record keeps a 64-bit row pointer per fragment row.
/// - The direct (non-prefetch) gather walks k with one counter (k = 0, 32, ..., K - 32, reloading the group's
///   scale and bias at k % 64 == 0), where the record nests a group counter and a step counter: the same steps, the
///   same loads and the same mma order.
/// - The staged paths' template switches (PF, DB, WF) come with the gather text, so a lean row can take them; the
///   wide fill reads its code words through the lean B source's `words` (32-bit offsets, like `byte`). The direct
///   paths' HB and CN switches come with it too (the A-load swap counts the CN path's call: 5 calls); no lean row
///   takes them.
/// Every load reads the record's address and every product, mma, accumulator and store is the record's.
/// NEMOTRON_SEED_LEAN=0 removes the rows.
enum NemotronSeedLean {
    static let offered = ProcessInfo.processInfo.environment["NEMOTRON_SEED_LEAN"] != "0"

    /// `text` from `from` (inclusive) to `to` (exclusive, or the end); nil unless each occurs once.
    static func slice(_ text: String, from: String, to: String?) -> String? {
        guard text.components(separatedBy: from).count == 2, let a = text.range(of: from) else { return nil }
        guard let to else { return String(text[a.lowerBound...]) }
        guard text.components(separatedBy: to).count == 2, let b = text.range(of: to), a.lowerBound < b.lowerBound
        else { return nil }
        return String(text[a.lowerBound ..< b.lowerBound])
    }

    /// `text` through `swaps` (old, new, occurrences); nil unless each occurs exactly as often.
    static func swapped(_ text: String, _ swaps: [(String, String, Int)]) -> String? {
        var s = text
        for (old, new, count) in swaps {
            guard s.components(separatedBy: old).count == count + 1 else { return nil }
            s = s.replacingOccurrences(of: old, with: new)
        }
        return s
    }

    /// The B-source structs `names` in `text` renamed `...Ln`, with 32-bit element offsets.
    static func sources(_ text: String, _ names: [String]) -> String? {
        guard var s = swapped(text, names.map { ("struct \($0) {", "struct \($0)Ln {", 1) }),
              s.contains("size_t(") else { return nil }
        s = s.replacingOccurrences(of: "const size_t ", with: "const uint ")
            .replacingOccurrences(of: "size_t(", with: "uint(")
        return s.contains("size_t") ? nil : s
    }

    /// Appended to the routed retile header: the lean B sources, A loads and gather (`nemotron_seed_gather_impl_ln`).
    static let retileHeader: String? = {
        let src = nemotronSeedRetileSource
        guard let structs = slice(src, from: "template <typename T>\nstruct NemotronSeedBank {",
                                  to: "// The direct fc1's row-indexed A load"),
              let b = sources(structs, ["NemotronSeedBank", "NemotronSeedDict", "NemotronSeedCopy"]),
              let loads = slice(src, from: "template <typename Tile, typename T>\nMETAL_FUNC void nemotron_seed_load_routed(",
                                to: "// tiles[t] = (first sorted row"),
              let a = swapped(loads, [
                ("METAL_FUNC void nemotron_seed_load_routed(", "METAL_FUNC void nemotron_seed_load_routed_ln(", 1),
                ("    thread Tile& tile, const device T* const thread* rows, const int offset,",
                 "    thread Tile& tile, const device T* x, const thread uint* rows, const int offset,", 1),
                ("const device T* source_row = rows[mr * Frag::kElemRows + i] + offset;",
                 "const device T* source_row = x + (rows[mr * Frag::kElemRows + i] + uint(offset));", 1),
                ("METAL_FUNC void nemotron_seed_load_a(", "METAL_FUNC void nemotron_seed_load_a_ln(", 1),
                ("    const device T* const thread* routed,", "    const thread uint* routed,", 1),
                ("nemotron_seed_load_routed(Atile, routed, k, sgp_sm);",
                 "nemotron_seed_load_routed_ln(Atile, xn, routed, k, sgp_sm);", 1),
              ]),
              let impl = slice(src, from: "template <\n    typename T,\n    int BM,", to: nil),
              let g = swapped(impl, [
                ("METAL_FUNC void nemotron_seed_gather_impl(", "METAL_FUNC void nemotron_seed_gather_impl_ln(", 1),
                ("  const device T* routed[TM * BaseNAXFrag::kElemRows];",
                 "  uint routed[TM * BaseNAXFrag::kElemRows];", 1),
                ("? x + size_t(routed_rows[row0 + tm + row]) * K : x;", "? routed_rows[row0 + tm + row] * uint(K) : 0u;", 1),
                ("nemotron_seed_load_a<ROUTED, aligned_m, SK>(", "nemotron_seed_load_a_ln<ROUTED, aligned_m, SK>(", 5),
                ("      for (int kg = 0; kg < K; kg += 64) {\n        float2 sbv[TN * 2];\n        STEEL_PRAGMA_UNROLL\n"
                    + "        for (short nn = 0; nn < TN; ++nn) {\n          STEEL_PRAGMA_UNROLL\n"
                    + "          for (short i = 0; i < 2; ++i) {\n"
                    + "            const int n = y_col + tn + nn * 16 + bc.y + i * 8;\n"
                    + "            sbv[nn * 2 + i] = (ALIGN_N || n < N) ? bsrc.sb(n, kg / 64) : float2(0.0f);\n"
                    + "          }\n        }\n\n        STEEL_PRAGMA_NO_UNROLL\n"
                    + "        for (int kk1 = 0; kk1 < 64; kk1 += SK) {\n",
                 "      float2 sbv[TN * 2];\n      STEEL_PRAGMA_NO_UNROLL\n      for (int ks = 0; ks < K; ks += SK) {\n"
                    + "        if (ks % 64 == 0) {\n        STEEL_PRAGMA_UNROLL\n"
                    + "        for (short nn = 0; nn < TN; ++nn) {\n          STEEL_PRAGMA_UNROLL\n"
                    + "          for (short i = 0; i < 2; ++i) {\n"
                    + "            const int n = y_col + tn + nn * 16 + bc.y + i * 8;\n"
                    + "            sbv[nn * 2 + i] = (ALIGN_N || n < N) ? bsrc.sb(n, ks / 64) : float2(0.0f);\n"
                    + "          }\n        }\n        }\n\n        {\n", 1),
                ("(Atile, xn, routed, kg + kk1, K, sgp_sm);", "(Atile, xn, routed, ks, K, sgp_sm);", 1),
                ("const int k = kg + kk1 + kk * 16 + bc.x;", "const int k = ks + kk * 16 + bc.x;", 1),
              ]),
              !g.contains("kg + kk1"), !g.contains("routed_rows[row0 + tm + row]) * K")
        else { return nil }
        return b + a + g
    }()

    /// Appended to the dense retile header: `NemotronSeedDenseDictLn` and `NemotronSeedDenseCopyLn`.
    static let denseHeader: String? = slice(
        nemotronSeedDenseSource, from: "template <typename T>\nstruct NemotronSeedDenseDict {",
        to: "template <\n    typename T,\n    int BM,").flatMap {
            sources($0, ["NemotronSeedDenseDict", "NemotronSeedDenseCopy"])
        }
}

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
/// (5 warm-up and 25 timed cycles, `NemotronTuneAdoption.order`, median), and
/// adopts the fastest only when `NemotronTuneAdoption.decide` admits it (at
/// least 3% faster than the incumbent by median and faster in at least 75% of
/// the cycles). NEMOTRON_SEED_RETILE=0 keeps the incumbents;
/// NEMOTRON_SEED_RETILE_FORCE=<variant> adopts that variant for both roles when
/// it qualifies.
/// The pf rows are the same geometries with the next K step's B loads issued
/// before this step's mma (same values, same mma order);
/// NEMOTRON_SEED_RETILE_PF=0 leaves them out. NEMOTRON_SEED_RETILE_X=0 leaves out
/// `extraVariants` (BN 128 on a 2 x 2 grid);
/// NEMOTRON_SEED_RETILE_DB=0 `stagedVariants` (two threadgroup stages);
/// NEMOTRON_SEED_RETILE_WF=0 `wideVariants` (the B source's wide fill); each also
/// removes its rows' lean twins (`stagedWideLeanVariants`, NEMOTRON_SEED_LEAN).
/// NEMOTRON_SEED_RETILE_HB=0 leaves out `halfVariants` (direct B, one 16-bit
/// code load per quad, row-guarded mma; bank and dictionary roles) and
/// NEMOTRON_SEED_RETILE_CN=0 `copyNativeVariants` (direct B from the decode
/// copy's words; copy role), both appended after every other row.
enum NemotronSeedRetile {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE"] != "0"
    static let forced = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE_FORCE"]

    struct Variant: Sendable, Equatable {
        let name: String
        let bn: Int, bk: Int, wm: Int, wn: Int
        let direct: Bool
        var pf: Bool = false
        /// `NemotronSeedLean`'s twin of the geometry.
        var lean: Bool = false
        /// Two threadgroup stages, one barrier per K block (`nemotron_seed_gather_impl`'s DB).
        var db: Bool = false
        /// The B source's wide fill (`nemotron_seed_gather_impl`'s WF).
        var wf: Bool = false
        /// HB: each pair of code bytes (k, k + 2) of a direct B quad comes from
        /// one 16-bit load, and the mma skips 16-row fragments without a valid
        /// row (`nemotron_seed_matmad_rows`). Bank and dictionary sources.
        var half: Bool = false
        /// CN: B read from the decode copy's words straight into K x N
        /// fragments (transpose_b false). Copy source only; not bit-exact by
        /// construction (the mma's other operand layout), so only the bitwise
        /// qualify admits it.
        var cn: Bool = false
        var threads: Int { wm * wn * 32 }
        var threadgroupBytes: Int { direct ? 0 : (db ? 2 : 1) * bn * (bk + 8) * 2 }
    }

    /// NEMOTRON_SEED_RETILE_PF=0 stops offering the PF rows (the K loop with
    /// the next step's B loads issued before this step's mma).
    static let offerPF = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE_PF"] != "0"

    /// ws: B through threadgroup memory; d: B dequantized into the fragments.
    /// Name = path, BN (x BK), then WM WN (pf: prefetching K loop). ws64x64w22
    /// is the incumbent's geometry.
    static let variants: [Variant] = [
        Variant(name: "ws64x64w22", bn: 64, bk: 64, wm: 2, wn: 2, direct: false),
        Variant(name: "ws64x128w22", bn: 64, bk: 128, wm: 2, wn: 2, direct: false),
        Variant(name: "ws128x64w24", bn: 128, bk: 64, wm: 2, wn: 4, direct: false),
        Variant(name: "ws128x64w14", bn: 128, bk: 64, wm: 1, wn: 4, direct: false),
        Variant(name: "d64w22", bn: 64, bk: 64, wm: 2, wn: 2, direct: true),
        Variant(name: "d64w12", bn: 64, bk: 64, wm: 1, wn: 2, direct: true),
        Variant(name: "d128w14", bn: 128, bk: 64, wm: 1, wn: 4, direct: true),
        Variant(name: "d128w24", bn: 128, bk: 64, wm: 2, wn: 4, direct: true),
    ] + (offerPF ? [
        Variant(name: "ws64x64w22pf", bn: 64, bk: 64, wm: 2, wn: 2, direct: false, pf: true),
        Variant(name: "d64w22pf", bn: 64, bk: 64, wm: 2, wn: 2, direct: true, pf: true),
        // The prefetching K loop at the other base geometries.
        Variant(name: "ws64x128w22pf", bn: 64, bk: 128, wm: 2, wn: 2, direct: false, pf: true),
        Variant(name: "ws128x64w24pf", bn: 128, bk: 64, wm: 2, wn: 4, direct: false, pf: true),
        Variant(name: "d128w14pf", bn: 128, bk: 64, wm: 1, wn: 4, direct: true, pf: true),
        Variant(name: "d128w24pf", bn: 128, bk: 64, wm: 2, wn: 4, direct: true, pf: true),
    ] : []) + (NemotronSeedLean.offered && NemotronSeedLean.retileHeader != nil ? [
        Variant(name: "ws64x64w22ln", bn: 64, bk: 64, wm: 2, wn: 2, direct: false, lean: true),
        Variant(name: "d64w22ln", bn: 64, bk: 64, wm: 2, wn: 2, direct: true, lean: true),
        Variant(name: "ws64x64w22pfln", bn: 64, bk: 64, wm: 2, wn: 2, direct: false, pf: true, lean: true),
        Variant(name: "d64w22pfln", bn: 64, bk: 64, wm: 2, wn: 2, direct: true, pf: true, lean: true),
    ] : []) + (offerX ? extraVariants : []) + (offerDB ? stagedVariants : []) + (offerWF ? wideVariants : [])
        + stagedWideLeanVariants

    /// NEMOTRON_SEED_RETILE_X=0 stops offering `extraVariants`.
    static let offerX = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE_X"] != "0"
    /// More threadgroup-staged rows through the same kernel text (template values only): BN 128 with a 2 x 2
    /// SIMD-group grid (each SIMD group 64 columns: four 16-column fragments, two mma pairs per 16-row fragment and
    /// 16-k chunk; 128 threads, KPT 64, 18,432 B of threadgroup memory). It keeps the run gather's fragment pairs (SN
    /// a multiple of 32 from the tile's first column) and 16-row fragments (WM 2, so the 32-row body keeps SM a
    /// multiple of 16), and the k chunks in order. (The prefetching K loop on the BK 128 and BN 128 geometries,
    /// ws64x128w22pf and ws128x64w24pf, is among the pf rows above.) Appended after the rows above.
    static let extraVariants: [Variant] = [
        Variant(name: "ws128x64w22", bn: 128, bk: 64, wm: 2, wn: 2, direct: false),
    ]

    /// NEMOTRON_SEED_RETILE_DB=0 stops offering `stagedVariants`.
    static let offerDB = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE_DB"] != "0"
    /// The incumbent geometry with two threadgroup stages (db: block b in stage b & 1, block b + 1 written while
    /// block b is read, one barrier per K block instead of two; 18,432 B), without and with the prefetching K loop
    /// (block b + 2's loads issued before block b's mma). Every stage slot holds the per-row fill's value; the mma
    /// sequence, accumulators and store are the single-stage path's. Appended.
    static let stagedVariants: [Variant] = [
        Variant(name: "ws64x64w22db", bn: 64, bk: 64, wm: 2, wn: 2, direct: false, db: true),
        Variant(name: "ws64x64w22dbpf", bn: 64, bk: 64, wm: 2, wn: 2, direct: false, pf: true, db: true),
    ]

    /// NEMOTRON_SEED_RETILE_WF=0 stops offering `wideVariants`.
    static let offerWF = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE_WF"] != "0"
    /// Threadgroup-staged rows whose blocks are written by the B source's wide fill (wf): the bank and the pair
    /// dictionary read each thread's KPT codes as 32-bit words (KPT / 8 loads instead of KPT / 2 byte loads); the
    /// fc2 decode copy gives each thread four consecutive outputs and CK consecutive k, as the incumbent fc2 copy
    /// loader, so each 16-bit code word is read once for its four outputs (the per-row fill reads it once per
    /// output). Every Ws slot gets the per-row fill's value (the same byte, scale and bias, the same dequantize
    /// statements); the K blocks, mma sequence, accumulators and store are unchanged. On the incumbent geometry
    /// alone, with pf, with db and with both, and with pf on the BK 128 and BN 128 geometries. Appended.
    static let wideVariants: [Variant] = [
        Variant(name: "ws64x64w22wf", bn: 64, bk: 64, wm: 2, wn: 2, direct: false, wf: true),
        Variant(name: "ws64x64w22wfpf", bn: 64, bk: 64, wm: 2, wn: 2, direct: false, pf: true, wf: true),
        Variant(name: "ws64x64w22wfdb", bn: 64, bk: 64, wm: 2, wn: 2, direct: false, db: true, wf: true),
        Variant(name: "ws64x64w22wfdbpf", bn: 64, bk: 64, wm: 2, wn: 2, direct: false, pf: true, db: true, wf: true),
        Variant(name: "ws64x128w22wfpf", bn: 64, bk: 128, wm: 2, wn: 2, direct: false, pf: true, wf: true),
        Variant(name: "ws128x64w24wfpf", bn: 128, bk: 64, wm: 2, wn: 4, direct: false, pf: true, wf: true),
    ]

    /// `NemotronSeedLean`'s twins of the staged and wide-fill rows above: the same geometry and template switches
    /// (PF, DB, WF) on the lean text (the lean B sources, whose wide fill reads its code words through `words` with
    /// 32-bit offsets, and the gather's 32-bit row offsets), appended last. Each is offered while NEMOTRON_SEED_LEAN
    /// and its row's own switch (NEMOTRON_SEED_RETILE_DB or _WF) are on, and qualified and timed like every row.
    static let stagedWideLeanVariants: [Variant] = NemotronSeedLean.offered && NemotronSeedLean.retileHeader != nil
        ? ((offerDB ? stagedVariants : []) + (offerWF ? wideVariants : [])).map {
            Variant(name: $0.name + "ln", bn: $0.bn, bk: $0.bk, wm: $0.wm, wn: $0.wn, direct: $0.direct, pf: $0.pf,
                    lean: true, db: $0.db, wf: $0.wf)
        } : []

    /// NEMOTRON_SEED_RETILE_HB=0 stops offering `halfVariants`.
    static let offerHalf = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE_HB"] != "0"
    /// NEMOTRON_SEED_RETILE_CN=0 stops offering `copyNativeVariants`.
    static let offerCopyNative = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE_CN"] != "0"

    /// h: direct B with HB (name ...h), one SIMD group across M (WM 1: no B value is dequantized twice) and BN a
    /// divisor of 1856 and 2688 (ALIGN_N); pf: the prefetching K loop. The plain direct path (no DB, WF or lean
    /// text). Offered to the bank and dictionary roles, after every row above.
    static let halfVariants: [Variant] = offerHalf ? [
        Variant(name: "d64w12h", bn: 64, bk: 64, wm: 1, wn: 2, direct: true, half: true),
        Variant(name: "d32w11h", bn: 32, bk: 64, wm: 1, wn: 1, direct: true, half: true),
    ] + (offerPF ? [
        Variant(name: "d64w12hpf", bn: 64, bk: 64, wm: 1, wn: 2, direct: true, pf: true, half: true),
        Variant(name: "d32w11hpf", bn: 32, bk: 64, wm: 1, wn: 1, direct: true, pf: true, half: true),
    ] : []) : []

    /// cn: CN with WM 1, BN 64 (copy source; rows guarded as HB); pf: the prefetching K loop. Offered to the copy
    /// role, after every row above.
    static let copyNativeVariants: [Variant] = offerCopyNative ? [
        Variant(name: "cn64w12h", bn: 64, bk: 64, wm: 1, wn: 2, direct: true, cn: true),
    ] + (offerPF ? [
        Variant(name: "cn64w12hpf", bn: 64, bk: 64, wm: 1, wn: 2, direct: true, pf: true, cn: true),
    ] : []) : []

    /// The rows a role is offered: every row above, then the h rows (bank and dictionary sources: they read the
    /// bank's code bytes) or the cn rows (the decode copy's words).
    static func offered(_ role: Role) -> [Variant] {
        variants + (role.source == .copy ? copyNativeVariants : halfVariants)
    }

    enum Source: String, Sendable { case bank, dict, copy }

    /// The timing runs the four layers as one dependent chain (each layer's input a view of the previous layer's
    /// output, where it is large enough), so a geometry's tail is timed as the seed meets it.
    /// NEMOTRON_SEED_RETILE_CHAIN=0 keeps the independent launches on one input.
    static let chainTiming = ProcessInfo.processInfo.environment["NEMOTRON_SEED_RETILE_CHAIN"] != "0"

    /// The launch a role's incumbent uses at the seed.
    struct Role: Sendable, Equatable {
        let routed: Bool
        let source: Source
        let relu2: Bool
        var label: String { (routed ? "routed-" : "") + source.rawValue }
    }

    nonisolated(unsafe) static var fc1Choice: (role: Role, variant: Variant)? = nil
    nonisolated(unsafe) static var fc2Choice: (role: Role, variant: Variant)? = nil

    private static func body(routed: Bool, source: Source, lean: Bool = false) -> String {
        let make: String
        let ln = lean ? "Ln" : ""
        switch source {
        case .bank: make = "NemotronSeedBank\(ln)<T> b{(const device uint8_t*)w, scales, biases, K};"
        case .dict: make = "NemotronSeedDict\(ln)<T> b{(const device uint8_t*)w, scales, biases, K};"
        case .copy: make = "NemotronSeedCopy\(ln)<T> b{w, scales, biases, K};"
        }
        let rows = routed ? "routed_rows" : "(const device uint32_t*)tiles"
        func call(_ bm: String) -> String {
            """
                    nemotron_seed_gather_impl\(lean ? "_ln" : "")<T, \(bm), BN, BK, WM, WN, DIRECT_B, ALIGN_N, RELU2, \(routed), PF, DB, WF, HB, CN>(
                        x, \(rows), b, tiles, y, N, K, Ws,
                        threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup,
                        thread_index_in_simdgroup);
            """
        }
        return """
                threadgroup T Ws[DIRECT_B ? 1 : (DB ? 2 : 1) * BN * (BK + 16 / sizeof(T))];
                \(make)
                if (BM == 64 && tiles[threadgroup_position_in_grid.y * 3 + 1] <= 32) {
            \(call("32"))
                } else {
            \(call("BM"))
                }
            """
    }

    private static func makeKernel(routed: Bool, source: Source, lean: Bool = false) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: "nemotron_seed_gather_\(routed ? "routed_" : "")\(source.rawValue)" + (lean ? "_ln" : ""),
            inputNames: ["x", "w", "scales", "biases", "tiles"] + (routed ? ["routed_rows"] : []) + ["N", "K"],
            outputNames: ["y"],
            source: body(routed: routed, source: source, lean: lean),
            header: nemotronRunGatherHeader + "\n" + nemotronSeedRetileSource
                + (lean ? "\n" + (NemotronSeedLean.retileHeader ?? "") : ""))
    }

    private static let routedBank = makeKernel(routed: true, source: .bank)
    private static let routedDict = makeKernel(routed: true, source: .dict)
    private static let plainBank = makeKernel(routed: false, source: .bank)
    private static let plainCopy = makeKernel(routed: false, source: .copy)
    private static let routedBankLn = makeKernel(routed: true, source: .bank, lean: true)
    private static let routedDictLn = makeKernel(routed: true, source: .dict, lean: true)
    private static let plainBankLn = makeKernel(routed: false, source: .bank, lean: true)
    private static let plainCopyLn = makeKernel(routed: false, source: .copy, lean: true)

    /// One retiled launch; nil wherever the incumbent's guards would not admit it.
    static func launch(_ v: Variant, role: Role, x: MLXArray, lin: QuantizedSwitchLinear,
                       rows: MLXArray?, tiles: NemotronHRunTiles) -> MLXArray? {
        guard NemotronHPrefillX5.runGather, v.threadgroupBytes <= 32768, lin.mode == .affine, lin.bits == 4,
              lin.groupSize == 64, x.ndim == 3, x.dim(1) == 1, x.dtype == .bfloat16,
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
            guard !v.cn, !ops.hasLinearBias, let biases = ops.biases, ops.scales.dtype == x.dtype,
                  biases.dtype == x.dtype, ops.scales.shape == [experts, nOut, k / 64],
                  biases.shape == ops.scales.shape
            else { return nil }
            if role.source == .dict {
                guard role.routed, let d = NemotronSBDict.dict(lin), d.index.shape == ops.scales.shape,
                      d.index.dtype == .uint16, d.table.dtype == .float32
                else { return nil }
                (kernel, operands) = (v.lean ? routedDictLn : routedDict, [ops.weight, d.index, d.table])
            } else {
                (kernel, operands) = (role.routed ? (v.lean ? routedBankLn : routedBank) : (v.lean ? plainBankLn : plainBank),
                                      [ops.weight, ops.scales, biases])
            }
        case .copy:
            guard !v.half, !role.routed, !role.relu2, nOut % 128 == 0, let copy = NemotronZSkip.copy(for: lin), copy.dict,
                  copy.codes.shape == [experts, nOut / 128, k, 32], copy.codes.dtype == .uint16,
                  copy.scales.shape == [experts, nOut / 128, k / 64, 128], copy.scales.dtype == .uint16,
                  copy.biases.dtype == .float32
            else { return nil }
            (kernel, operands) = (v.lean ? plainCopyLn : plainCopy, [copy.codes, copy.scales, copy.biases])
        }
        var inputs: [any ScalarOrArray] = [x] + operands + [tiles.tiles]
        if role.routed, let rows { inputs.append(rows) }
        inputs += [nOut, k]
        return kernel(
            inputs,
            template: [("T", x.dtype), ("BM", tiles.bm), ("BN", v.bn), ("BK", v.bk), ("WM", v.wm), ("WN", v.wn),
                       ("DIRECT_B", v.direct), ("ALIGN_N", nOut % v.bn == 0), ("RELU2", role.relu2), ("PF", v.pf),
                       ("DB", v.db), ("WF", v.wf), ("HB", v.half), ("CN", v.cn)],
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
        let tuneStart = DispatchTime.now().uptimeNanoseconds
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
        let offers = roles.map { offered($0) }
        var alive: [[Variant]] = offers
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
        // 25 timed cycles, scaled by `nemotronTuneTimingScale` (even).
        let timedCycles = max(25, Int((25 * nemotronTuneTimingScale).rounded()) / 2 * 2)
        var medians: [[(Variant?, Double)]] = [[], []]
        var verdicts: [NemotronTuneAdoption.Verdict?] = [nil, nil]
        var choice: [Variant?] = [nil, nil]
        for r in 0 ..< 2 where !alive[r].isEmpty {
            let candidates: [Variant?] = [nil] + alive[r].map { Optional($0) }
            let (x, rows) = input(r, cases[0])
            var samples = Array(repeating: [Double](), count: candidates.count)
            do {
                try withError {
                    for rep in 0 ..< 5 + timedCycles {
                        for i in NemotronTuneAdoption.order(candidates.count, cycle: max(0, rep - 5)) {
                            let candidate = candidates[i]
                            let start = DispatchTime.now().uptimeNanoseconds
                            // `chainTiming`: each layer's input is a view of the previous layer's output (same
                            // shape and dtype as `x`), so the launches run back to back as in the seed.
                            var current = x
                            var outs: [MLXArray] = []
                            var chained = chainTiming
                            for m in layers {
                                let lin = linear(m, r)
                                let out = candidate.map {
                                    launch($0, role: roles[r], x: current, lin: lin, rows: rows, tiles: cases[0].tiles)
                                } ?? incumbent(roles[r], x: current, lin: lin, rows: rows, tiles: cases[0].tiles)
                                guard let out else { break }
                                outs.append(out)
                                if chained, out.dtype == x.dtype, out.size >= x.size {
                                    current = out.reshaped(-1)[0 ..< x.size].reshaped(x.shape)
                                } else {
                                    chained = false
                                }
                            }
                            // A chain is complete once its last launch is; otherwise every launch is evaluated.
                            if chained, let last = outs.last { eval(last) } else { eval(outs) }
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
            guard samples.allSatisfy({ $0.count == timedCycles }) else { continue }
            let verdict = NemotronTuneAdoption.decide(samples, seed: true)
            verdicts[r] = verdict
            medians[r] = Array(zip(candidates, verdict.medians))
            if let forced, let f = alive[r].first(where: { $0.name == forced }) {
                choice[r] = f
            } else if let best = verdict.adopted {
                choice[r] = candidates[best]
            }
        }
        fc1Choice = choice[0].map { (roles[0], $0) }
        fc2Choice = choice[1].map { (roles[1], $0) }

        func ms(_ r: Int, _ v: Variant?) -> String {
            guard let m = medians[r].first(where: { $0.0 == v }) else { return "n/a" }
            return String(format: "%.2f", m.1)
        }
        func detail(_ r: Int) -> String {
            let times = medians[r].indices.dropFirst().map {
                "\(medians[r][$0].0!.name) \(String(format: "%.2f", medians[r][$0].1)) \(verdicts[r]?.won($0) ?? "n/a")"
            }
            return "\(roles[r].label) \(alive[r].count)/\(offers[r].count) bitwise"
                + (verdicts[r].map { v in ", " + v.rule({ medians[r][$0].0?.name ?? "V0" }) } ?? "")
                + (times.isEmpty ? "" : " [" + times.joined(separator: ", ") + "]")
                + (dropped[r].isEmpty ? "" : " dropped: " + dropped[r].prefix(16).joined(separator: ", "))
        }
        report("\(choice[0]?.name ?? "V0")/\(choice[1]?.name ?? "V0") fc1 \(ms(0, choice[0])) ms fc2 \(ms(1, choice[1])) ms"
            + " vs V0 fc1 \(ms(0, nil)) ms fc2 \(ms(1, nil)) ms (per layer, 488 tokens, median of 25;"
            + " fc1 \(detail(0)); fc2 \(detail(1)))"
            + String(format: " tune %.2f s", Double(DispatchTime.now().uptimeNanoseconds - tuneStart) / 1e9))
    }
}

// MARK: - Seed dense retile

/// The seed's dense run gathers (one "expert", `tiles[t]` = (first row, rows <=
/// BM, 0) over the request's rows) with free tiling, appended to
/// `nemotronRunGatherHeader` and `nemotronSeedRetileSource`. For every output
/// fragment pair the same `BaseNAXFrag::mma` (matmul2d(16, 32, 16),
/// multiply_accumulate) runs over k chunks 0, 16, ..., K - 16 in order, on A
/// fragments loaded by the incumbents' statements and B fragments holding the
/// incumbents' BF16 dequantized value at each (n, k), from a threadgroup block
/// or (HB) dequantized straight into the registers; the typed store is the
/// incumbents'. Free: BM, BN, BK, the SIMD-group grid WM x WN (SM, SN multiples
/// of 16 and 32, so rows keep their 16-row fragments from the tile's first row
/// and each mma pairs columns 32c ... 32c + 31), which thread dequantizes which
/// B value, DB (two threadgroup stages, one barrier per K block) and HB.
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

  // PF: `fill` in two halves. `load` reads the block's code words, scale and
  // bias into registers under fill's predicate; `store` writes fill's values
  // from them with fill's statements (the same bytes, the same s and b).
  template <int BN, int BK, int NT>
  struct Regs {
    uint word[(BN * BK / NT) / 8];
    float s;
    float b;
  };

  template <int BN, int BK, int NT, bool ALIGN_N>
  METAL_FUNC void load(thread Regs<BN, BK, NT>& r, const int t, const int y_col, const int N,
                       const int kb, const int klim) const thread {
    constexpr int KPT = BN * BK / NT;
    const int lr = t / (BK / KPT);
    const int lk = (t % (BK / KPT)) * KPT;
    const int n = y_col + lr;
    if ((ALIGN_N || n < N) && lk < klim) {
      const uint pair = uint(sidx[size_t(n) * size_t(K / 64) + size_t((kb + lk) / 64)]);
      r.s = table[2 * pair];
      r.b = table[2 * pair + 1];
      const device uint32_t* src =
          (const device uint32_t*)(w + size_t(n) * size_t(K / 2) + size_t((kb + lk) / 2));
      STEEL_PRAGMA_UNROLL
      for (short q = 0; q < KPT / 8; ++q) {
        r.word[q] = src[q];
      }
    }
  }

  template <int BN, int BK, int NT, bool ALIGN_N>
  METAL_FUNC void store(threadgroup T* ws, thread const Regs<BN, BK, NT>& r, const int t, const int y_col,
                        const int N, const int klim) const thread {
    constexpr int BK_padded = BK + 16 / sizeof(T);
    constexpr int KPT = BN * BK / NT;
    const int lr = t / (BK / KPT);
    const int lk = (t % (BK / KPT)) * KPT;
    const int n = y_col + lr;
    threadgroup T* dst = ws + lr * BK_padded + lk;
    if ((ALIGN_N || n < N) && lk < klim) {
      const float s = r.s;
      const float b = r.b;
      STEEL_PRAGMA_UNROLL
      for (short q = 0; q < KPT / 8; ++q) {
        const uint word = r.word[q];
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

  // HB: fill's reads for one output n. sb: the (scale, bias) of group g, from
  // fill's pair index and table entries. code16: byte(n, k) | byte(n, k + 2) << 8
  // (the bytes fill takes from its code words) as one 16-bit load (k % 4 == 0
  // and K / 2 even, so the address is 2-byte aligned).
  METAL_FUNC float2 sb(const int n, const int g) const thread {
    const uint pair = uint(sidx[size_t(n) * size_t(K / 64) + size_t(g)]);
    const float s = table[2 * pair];
    const float b = table[2 * pair + 1];
    return float2(s, b);
  }
  METAL_FUNC uint16_t code16(const int n, const int k) const thread {
    return *(const device uint16_t*)(w + size_t(n) * size_t(K / 2) + size_t(k >> 1));
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

  // PF: `fill` in two halves, as NemotronSeedDenseDict's: `load` reads the
  // block's CK code words and the four outputs' scales and biases (widened to
  // float as fill does) under fill's predicate; `store` runs fill's statements
  // on them.
  template <int BN, int BK, int NT>
  struct Regs {
    uint16_t c[BK * (BN / 4) / NT];
    float s[4];
    float b[4];
  };

  template <int BN, int BK, int NT, bool ALIGN_N>
  METAL_FUNC void load(thread Regs<BN, BK, NT>& r, const int t, const int y_col, const int N,
                       const int kb, const int klim) const thread {
    constexpr int CK = BK * (BN / 4) / NT;
    const int cm = t % (BN / 4);
    const int cc = t / (BN / 4);
    const int n = y_col + 4 * cm;
    const int k0 = kb + cc * CK;
    if ((ALIGN_N || n < N) && cc * CK < klim) {
      const device uint16_t* c =
          codes + (size_t(n / 128) * size_t(K) + size_t(k0)) * 32 + size_t((n % 128) / 4);
      const size_t g = (size_t(n / 128) * size_t(K / 64) + size_t(k0 / 64)) * 128 + size_t(n % 128);
      STEEL_PRAGMA_UNROLL
      for (short o = 0; o < 4; o++) {
        r.s[o] = float(sk[g + o]);
        r.b[o] = float(bk[g + o]);
      }
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < CK; ++j) {
        r.c[j] = c[j * 32];
      }
    }
  }

  template <int BN, int BK, int NT, bool ALIGN_N>
  METAL_FUNC void store(threadgroup T* ws, thread const Regs<BN, BK, NT>& r, const int t, const int y_col,
                        const int N, const int klim) const thread {
    constexpr int BK_padded = BK + 16 / sizeof(T);
    constexpr int CK = BK * (BN / 4) / NT;
    const int cm = t % (BN / 4);
    const int cc = t / (BN / 4);
    const int n = y_col + 4 * cm;
    threadgroup T* dst = ws + (4 * cm) * BK_padded + cc * CK;
    if ((ALIGN_N || n < N) && cc * CK < klim) {
      STEEL_PRAGMA_UNROLL
      for (short o = 0; o < 4; o++) {
        const float s = r.s[o];
        const float b = r.b[o];
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < CK; j += 2) {
          const uint lo = (uint(r.c[j]) >> (4 * o)) & 0xfu;
          const uint hi = (uint(r.c[j + 1]) >> (4 * o)) & 0xfu;
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
    bool PF,
    bool ALIGN_N,
    bool RELU2,
    bool HB,
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
    if constexpr (HB) {
      // HB: no threadgroup stage and no barrier. Each thread dequantizes its
      // own B fragment elements at the positions BaseNAXFrag::load reads from
      // the stage (n = y_col + tn + 16 nn + coord.y + 8 i, k = 32-k step +
      // 16 kk + coord.x + j): the code bytes of k and k + 2 come from one
      // 16-bit load (code16) and feed fill's dequantize statements with
      // fill's (scale, bias) (sb). The A loads, the mma pairs and their k
      // order are compute's; a 16-row fragment without a valid row skips its
      // mma (nemotron_seed_matmad_rows; it is never stored). PF: step k + 32's
      // code bytes (and, at a group start, its scales and biases) are loaded
      // after step k's B fragments are dequantized and before step k's mma.
      static_assert(ALIGN_N, "HB: whole column tiles");
      const short2 bc = BaseNAXFrag::get_coord();
      float2 sbv[TN * 2];
      uint16_t raw[TN * TK * 2];
      auto load_sb = [&](const int kg) {
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < TN; ++nn) {
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < 2; ++i) {
            sbv[nn * 2 + i] = bsrc.sb(y_col + tn + nn * 16 + bc.y + i * 8, kg / 64);
          }
        }
      };
      auto load_raw = [&](const int ks) {
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < TN; ++nn) {
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < TK; ++kk) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; ++i) {
              raw[(nn * TK + kk) * 2 + i] =
                  bsrc.code16(y_col + tn + nn * 16 + bc.y + i * 8, ks + kk * 16 + bc.x);
            }
          }
        }
      };
      if constexpr (PF) {
        if (sg_active) {
          load_sb(0);
          load_raw(0);
        }
      }
      STEEL_PRAGMA_NO_UNROLL
      for (int ks = 0; ks < K; ks += SK) {
        if (sg_active) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<T, TN, TK> Btile;

          volatile int compiler_barrier;

          if constexpr (!PF) {
            if (ks % 64 == 0) {
              load_sb(ks);
            }
            load_raw(ks);
          }

          if constexpr (aligned_m) {
            Atile.load(xn + ks, K);
          } else {
            Atile.load_safe(xn + ks, K, short2(SK, sgp_sm));
          }

          STEEL_PRAGMA_UNROLL
          for (short nn = 0; nn < TN; ++nn) {
            STEEL_PRAGMA_UNROLL
            for (short kk = 0; kk < TK; ++kk) {
              thread typename NAXTile<T, TN, TK>::frag_type& frag = Btile.frag_at(nn, kk);
              STEEL_PRAGMA_UNROLL
              for (short i = 0; i < 2; ++i) {
                const uint16_t h = raw[(nn * TK + kk) * 2 + i];
                const float2 p = sbv[nn * 2 + i];
                T v0, v1, v2, v3;
                nemotron_seed_deq2(uint8_t(h & 0xff), p.x, p.y, v0, v1);
                nemotron_seed_deq2(uint8_t(h >> 8), p.x, p.y, v2, v3);
                frag[i * 4 + 0] = v0;
                frag[i * 4 + 1] = v1;
                frag[i * 4 + 2] = v2;
                frag[i * 4 + 3] = v3;
              }
            }
          }

          if constexpr (PF) {
            const int kn = ks + SK;
            if (kn < K) {
              if (kn % 64 == 0) {
                load_sb(kn);
              }
              load_raw(kn);
            }
          }

          nemotron_seed_matmad_rows(Dtile, Atile, Btile, metal::bool_constant<transpose>{}, sgp_sm);

          (void)compiler_barrier;
        }
      }
    } else {
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
      // PF: each block's B loads are issued into registers (`load`) one block
      // ahead, before the current block's mma, and `store` writes fill's values
      // from them; the stages, barriers and compute are the non-PF form's.
      typename BSrc::template Regs<BN, BK, NT> pf_regs;
      auto pf_load = [&](const int b) {
        bsrc.template load<BN, BK, NT, ALIGN_N>(pf_regs, t, y_col, N, b * BK, min(BK, K - b * BK));
      };
      auto pf_store = [&](threadgroup T* ws, const int b) {
        bsrc.template store<BN, BK, NT, ALIGN_N>(ws, pf_regs, t, y_col, N, min(BK, K - b * BK));
      };
      if constexpr (DB && PF) {
        // As the DB form below, with block b + 2's loads issued after block
        // b + 1's stage write and before block b's mma.
        pf_load(0);
        pf_store(Ws, 0);
        if (1 < nb) {
          pf_load(1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int b = 0; b < nb; b++) {
          if (b + 1 < nb) {
            pf_store(Ws + ((b + 1) & 1) * STAGE, b + 1);
            if (b + 2 < nb) {
              pf_load(b + 2);
            }
          }
          compute(Ws + (b & 1) * STAGE, b);
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      } else if constexpr (PF) {
        // As the single-stage form below, with block b + 1's loads issued after
        // block b's stage write and barrier, before block b's mma.
        pf_load(0);
        for (int b = 0; b < nb; b++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          pf_store(Ws, b);
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (b + 1 < nb) {
            pf_load(b + 1);
          }
          compute(Ws, b);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      } else if constexpr (DB) {
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
/// every layer of the kind per cycle (1 warm-up and 16 timed cycles,
/// `NemotronTuneAdoption.order`, median), and adopts the fastest only when
/// `NemotronTuneAdoption.decide` admits it (at least 3% faster by median and
/// faster in at least 75% of the cycles). The direct-B rows with two byte loads
/// per code pair are left out: forced on every seed role they cost ~40 ms
/// (331295a3); the h rows (`halfVariants`) are offered to the pair-dictionary kinds.
/// NEMOTRON_SEED_DENSE_RETILE=0 keeps the incumbents; NEMOTRON_SEED_DENSE_RETILE_X=0 leaves the pf row out;
/// NEMOTRON_SEED_DENSE_RETILE_X2=0 leaves `moreVariants` out; NEMOTRON_SEED_DENSE_HB=0 the h rows (both tunes);
/// NEMOTRON_SEED_DENSE_RETILE_FORCE=<variant> adopts that variant wherever it
/// qualifies.
enum NemotronSeedDenseRetile {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_SEED_DENSE_RETILE"] != "0"
    static let forced = ProcessInfo.processInfo.environment["NEMOTRON_SEED_DENSE_RETILE_FORCE"]

    struct Variant: Sendable, Equatable {
        let name: String
        let bm: Int, bn: Int, bk: Int, wm: Int, wn: Int
        let db: Bool
        /// K-loop prefetch: each block's B loads issued one block ahead (`nemotron_seed_dense_impl`'s PF).
        let pf: Bool
        /// `NemotronSeedLean`'s B sources (`NemotronSeedDenseDictLn` / `NemotronSeedDenseCopyLn`).
        let lean: Bool
        /// HB: B dequantized straight into the fragment registers (`nemotron_seed_dense_impl`'s HB; no threadgroup
        /// stage); pair dictionary source only.
        let half: Bool
        var threads: Int { wm * wn * 32 }
        var threadgroupBytes: Int { half ? 0 : (db ? 2 : 1) * bn * (bk + 8) * 2 }

        init(_ bm: Int, _ bn: Int, _ bk: Int, wm: Int, wn: Int, db: Bool = false, pf: Bool = false,
             lean: Bool = false, half: Bool = false) {
            (self.bm, self.bn, self.bk, self.wm, self.wn, self.db, self.pf) = (bm, bn, bk, wm, wn, db, pf)
            (self.lean, self.half) = (lean, half)
            name = "\(bm)x\(bn)x\(bk)w\(wm)\(wn)" + (db ? "db" : "") + (half ? "h" : "") + (pf ? "pf" : "")
                + (lean ? "ln" : "")
        }
    }

    /// Name = BM x BN x BK, w WM WN, db for two stages. 64x64x64w22 is the
    /// incumbents' geometry (control); 64x128x64w24 is MLX's NAX bf16 GEMM layout
    /// for Max-class GPUs (32 x 32 per SIMD group); the 128-row tiles share each
    /// dequantized B block between twice the rows. Each fits 32 KB of threadgroup memory.
    static let baseVariants: [Variant] = [
        Variant(64, 64, 64, wm: 2, wn: 2),
        Variant(64, 64, 64, wm: 2, wn: 2, db: true),
        Variant(64, 64, 128, wm: 2, wn: 2),
        Variant(64, 128, 64, wm: 2, wn: 4),
        Variant(128, 64, 64, wm: 4, wn: 2),
        Variant(128, 64, 64, wm: 4, wn: 2, db: true),
        Variant(128, 128, 64, wm: 4, wn: 4),
    ]
    /// The incumbent geometry's K-loop prefetch twin (pf): the B loads (code words, scale and bias) of block
    /// b + 1 are issued into registers before block b's mma, and the stage write dequantizes them with the
    /// fill's statements. The same values reach the same threadgroup slots, and the mma sequence, accumulators
    /// and store are the base geometry's. (The kernel also takes pf with two stages: block b + 2's loads
    /// before block b's mma.) Appended after the base rows, so the base rows keep their indices. One row: the
    /// dense tune has no load-time cap, and each row adds a qualify and a timing pass over every layer.
    static let extraVariants: [Variant] = [
        Variant(64, 64, 64, wm: 2, wn: 2, pf: true),
        // The other base geometries with the same K-loop prefetch (single stage; each fits 32 KB).
        Variant(64, 128, 64, wm: 2, wn: 4, pf: true),
        Variant(128, 64, 64, wm: 4, wn: 2, pf: true),
        Variant(128, 128, 64, wm: 4, wn: 4, pf: true),
        Variant(64, 64, 128, wm: 2, wn: 2, pf: true),
    ]
    /// NEMOTRON_SEED_DENSE_RETILE_X=0 leaves the pf row out of this tune (the base rows only).
    static let offerX = ProcessInfo.processInfo.environment["NEMOTRON_SEED_DENSE_RETILE_X"] != "0"
    static let variants: [Variant] = baseVariants + (offerX ? extraVariants : [])
        + leanVariants + (offerX2 ? moreVariants : []) + halfVariants
    /// The incumbent geometry and its prefetch twin through `NemotronSeedLean`'s B sources (this tune and
    /// `NemotronSeedQMMTune`), appended after the pf rows so every earlier row keeps its index (`moreVariants`
    /// follow them); NEMOTRON_SEED_LEAN=0 leaves them out.
    /// They are qualified and (chain-)timed like every other row.
    static let leanVariants: [Variant] = NemotronSeedLean.offered && NemotronSeedLean.denseHeader != nil
        ? [Variant(64, 64, 64, wm: 2, wn: 2, lean: true), Variant(64, 64, 64, wm: 2, wn: 2, pf: true, lean: true)] : []
    /// More rows through the same kernel text (template values only), appended after the lean rows so every earlier
    /// row keeps its index: pf with two stages on the incumbent geometry (block b + 2's loads before block b's
    /// mma) and on the 128-row 4 x 2 geometry (pf alone there is among the pf rows above); 128-row tiles on a 2 x 2
    /// grid (each SIMD group 64 rows: four 16-row fragments per 32-column pair; 128 threads); 256-row tiles on an
    /// 8 x 2 grid (each dequantized B block shared by twice the rows of the 128-row tiles; 512 threads, KPT 8 for the
    /// dictionary fill, CK 2 for the copy fill; a tile of at most 128 rows runs the 128-row body), alone and with
    /// two stages. Each keeps SM a multiple of 16 in both bodies and SN 32 (the incumbents' 32-column pairs and
    /// 16-row fragments from the tile's first row) and fits 32 KB. Each row adds a qualify and a timing pass over
    /// every layer of the kind (the tune has no load cap).
    static let moreVariants: [Variant] = [
        Variant(64, 64, 64, wm: 2, wn: 2, db: true, pf: true),
        Variant(128, 64, 64, wm: 4, wn: 2, db: true, pf: true),
        Variant(128, 64, 64, wm: 2, wn: 2),
        Variant(256, 64, 64, wm: 8, wn: 2),
        Variant(256, 64, 64, wm: 8, wn: 2, db: true),
    ]
    /// NEMOTRON_SEED_DENSE_RETILE_X2=0 leaves `moreVariants` out of this tune.
    static let offerX2 = ProcessInfo.processInfo.environment["NEMOTRON_SEED_DENSE_RETILE_X2"] != "0"
    /// h (HB): no threadgroup stage and no barrier; each thread dequantizes its own B fragment elements, the code
    /// bytes of k and k + 2 from one 16-bit load, with the pair dictionary fill's statements, scale and bias, and a
    /// 16-row fragment without a valid row skips its mma (never stored). The A loads, the mma pairs and their k
    /// order, the accumulators and the store are the base rows'. BM 64 with one SIMD group across M (WM 1: no B
    /// value dequantized twice) at BN 64 and 32, the incumbent grid (WM 2), and the WM 1 rows' K-loop prefetch twins
    /// (pf: the next 32-k step's code bytes, and at a group start its scales and biases, loaded before this step's
    /// mma). Pair dictionary kinds only. Appended after every other row (here and in `NemotronSeedQMMTune`), so
    /// those keep their names and indices. NEMOTRON_SEED_DENSE_HB=0 leaves them out of both tunes.
    static let offerHB = ProcessInfo.processInfo.environment["NEMOTRON_SEED_DENSE_HB"] != "0"
    static let halfVariants: [Variant] = offerHB ? [
        Variant(64, 64, 64, wm: 1, wn: 2, half: true),
        Variant(64, 32, 64, wm: 1, wn: 1, half: true),
        Variant(64, 64, 64, wm: 2, wn: 2, half: true),
        Variant(64, 64, 64, wm: 1, wn: 2, pf: true, half: true),
        Variant(64, 32, 64, wm: 1, wn: 1, pf: true, half: true),
    ] : []

    /// The rows a source is offered: the h rows read the pair dictionary's code bytes, so the copy source skips
    /// them (they trail every list, so every other row keeps its index).
    static func offered(_ all: [Variant], _ source: Source) -> [Variant] {
        source == .dict ? all : all.filter { !$0.half }
    }
    /// The timing runs every layer's launch as one dependent chain (each reads its pair table, or the copy's biases,
    /// from a view of the previous launch's output), so a geometry's tail is timed as the seed meets it, not hidden
    /// by the next layer's independent launch. NEMOTRON_SEED_DENSE_CHAIN=0 keeps the independent launches.
    static let chainTiming = ProcessInfo.processInfo.environment["NEMOTRON_SEED_DENSE_CHAIN"] != "0"

    enum Source: String, Sendable { case dict, copy }

    /// Kinds: "in_proj", "out_proj", "up_proj" (pair dictionaries), "shared_down" (decode copy).
    nonisolated(unsafe) static var choice: [String: Variant] = [:]

    private static func makeKernel(_ source: Source, lean: Bool = false) -> MLXFast.MLXFastKernel {
        let ln = lean ? "Ln" : ""
        let make = source == .dict
            ? "NemotronSeedDenseDict\(ln)<T> b{(const device uint8_t*)w, scales, biases, K};"
            : "NemotronSeedDenseCopy\(ln)<T> b{w, scales, biases, K};"
        func call(_ bm: String) -> String {
            """
                    nemotron_seed_dense_impl<T, \(bm), BN, BK, WM, WN, DB, PF, ALIGN_N, RELU2, HB>(
                        x, b, tiles, y, N, K, Ws,
                        threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup,
                        thread_index_in_simdgroup);
            """
        }
        // A tile of at most BM / 2 rows runs the BM / 2 body (for BM 64 the
        // incumbents' 32-row bound); either way rows keep their 16-row fragments.
        return MLXFast.metalKernel(
            name: "nemotron_seed_dense_\(source.rawValue)" + (lean ? "_ln" : ""),
            inputNames: ["x", "w", "scales", "biases", "tiles", "N", "K"],
            outputNames: ["y"],
            source: """
                threadgroup T Ws[HB ? 1 : (DB ? 2 : 1) * BN * (BK + 16 / sizeof(T))];
                \(make)
                if (tiles[threadgroup_position_in_grid.y * 3 + 1] <= BM / 2) {
            \(call("BM / 2"))
                } else {
            \(call("BM"))
                }
            """,
            header: nemotronRunGatherHeader + "\n" + nemotronSeedRetileSource + "\n" + nemotronSeedDenseSource
                + (lean ? "\n" + (NemotronSeedLean.denseHeader ?? "") : ""))
    }

    private static let dictKernel = makeKernel(.dict)
    private static let copyKernel = makeKernel(.copy)
    private static let dictKernelLn = makeKernel(.dict, lean: true)
    private static let copyKernelLn = makeKernel(.copy, lean: true)

    /// One retiled launch over the request's rows (`x`: [1, rows, K]) in tiles of
    /// `v.bm` rows (at least three descriptors, as the incumbents); nil wherever
    /// the incumbent's guards (`NemotronPrefillDenseDict.launch` for .dict,
    /// `NemotronPrefillSharedDownCopy.launch` for .copy) would not admit it.
    /// `dep` (timing chains only, `NemotronSeedQMMTune`): the pair table (.dict) or
    /// the BF16 biases (.copy) are read from a view of `dep`'s leading bytes, so the
    /// launch waits for `dep`'s producer.
    static func launch(_ v: Variant, source: Source, x: MLXArray, linear: Linear, relu2: Bool,
                       dep: MLXArray? = nil) -> MLXArray? {
        guard NemotronHPrefillX5.runGather, v.threadgroupBytes <= 32768, let q = linear as? QuantizedLinear,
              type(of: linear) == QuantizedLinear.self, q.mode == .affine, q.bits == 4, q.groupSize == 64,
              q.bias == nil, x.ndim == 3, x.dim(0) == 1, x.dim(1) > 8, x.dtype == .bfloat16
        else { return nil }
        let rows = x.dim(1), k = x.dim(2), nOut = q.weight.dim(0)
        guard q.weight.dim(1) * 8 == k, k % 64 == 0, nOut % 32 == 0, !v.half || nOut % v.bn == 0 else { return nil }
        let kernel: MLXFast.MLXFastKernel
        let operands: [MLXArray]
        switch source {
        case .dict:
            guard !relu2 || MLX.maximum(x, MLXArray(0)).dtype == x.dtype,
                  q.weight.dtype == .uint32, q.weight.ndim == 2, q.scales.shape == [nOut, k / 64],
                  let d = NemotronSBDict.dict(q), d.index.shape == q.scales.shape, d.index.dtype == .uint16,
                  d.table.dtype == .float32
            else { return nil }
            let table = dep.map { NemotronSeedQMMTune.operandView($0, like: d.table) } ?? d.table
            guard let table else { return nil }
            (kernel, operands) = (v.lean ? dictKernelLn : dictKernel, [q.weight, d.index, table])
        case .copy:
            guard !v.half, !relu2, nOut % 128 == 0, let copy = NemotronSharedDownZSkip.copyArrays(for: linear),
                  copy.codes.shape == [1, nOut / 128, k, 32], copy.codes.dtype == .uint16,
                  copy.scales.shape == [1, nOut / 128, k / 64, 128], copy.scales.dtype == x.dtype,
                  copy.biases.shape == copy.scales.shape, copy.biases.dtype == x.dtype
            else { return nil }
            let biases = dep.map { NemotronSeedQMMTune.operandView($0, like: copy.biases) } ?? copy.biases
            guard let biases else { return nil }
            (kernel, operands) = (v.lean ? copyKernelLn : copyKernel, [copy.codes, copy.scales, biases])
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
                       ("DB", v.db), ("PF", v.pf), ("ALIGN_N", nOut % v.bn == 0), ("RELU2", relu2),
                       ("HB", v.half)],
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
        let offered: Int
        let alive: [Variant]
        let dropped: [String]
        let medians: [(Variant?, Double)]
        let choice: Variant?
        var verdict: NemotronTuneAdoption.Verdict? = nil
    }

    /// Qualifies the variants `source` is offered against the incumbent (`run(nil, case, layer)`) bit for
    /// bit (every layer in case 0, the first layer in the others; any mismatch or
    /// error drops a variant), then times the incumbent and the survivors round
    /// robin on case 0, every layer per cycle: 1 warm-up and
    /// `NemotronTuneAdoption.minCycles` timed cycles (`NemotronTuneAdoption.order`),
    /// median ms per launch; `NemotronTuneAdoption.decide` picks.
    /// `chain(v, layer, dep)` (when given): the case-0 launch of `layer` reading one operand from a view of `dep` (the
    /// previous layer's output in the timing chain), so each timed launch waits for the one before it, as the seed's
    /// projections run; nil `chain` keeps the independent per-layer launches.
    private static func evaluate(_ source: Source, cases: Int, layers: Int, run: (Variant?, Int, Int) -> MLXArray?,
                                 chain: ((Variant?, Int, MLXArray?) -> MLXArray?)? = nil) -> Outcome {
        let offers = offered(variants, source)
        var alive = offers.filter { $0.threadgroupBytes <= 32768 }
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
        guard !alive.isEmpty else {
            return Outcome(offered: offers.count, alive: alive, dropped: dropped, medians: [], choice: nil)
        }
        let candidates: [Variant?] = [nil] + alive.map { Optional($0) }
        var samples = Array(repeating: [Double](), count: candidates.count)
        let timed = max(NemotronTuneAdoption.minCycles,
                        Int((Double(NemotronTuneAdoption.minCycles) * nemotronTuneTimingScale).rounded()) / 2 * 2)
        do {
            try withError {
                for rep in 0 ..< 1 + timed {
                    for i in NemotronTuneAdoption.order(candidates.count, cycle: max(0, rep - 1)) {
                        let start = DispatchTime.now().uptimeNanoseconds
                        var outs: [MLXArray] = []
                        if let chain {
                            var previous: MLXArray? = nil
                            for layer in 0 ..< layers {
                                guard let out = chain(candidates[i], layer, previous) else { break }
                                previous = out
                                outs.append(out)
                            }
                            if outs.count == layers, let last = outs.last {
                                eval(last)
                            } else {
                                // A link the chain cannot build: this sample times the independent launches.
                                outs = (0 ..< layers).compactMap { run(candidates[i], 0, $0) }
                                eval(outs)
                            }
                        } else {
                            outs = (0 ..< layers).compactMap { run(candidates[i], 0, $0) }
                            eval(outs)
                        }
                        Stream.gpu.synchronize()
                        let elapsed = DispatchTime.now().uptimeNanoseconds - start
                        if rep >= 1, outs.count == layers {
                            samples[i].append(Double(elapsed) / 1e6 / Double(layers))
                        }
                    }
                }
            }
        } catch {
            dropped.append("timing error")
        }
        guard samples.allSatisfy({ $0.count == timed }) else {
            return Outcome(offered: offers.count, alive: alive, dropped: dropped, medians: [], choice: nil)
        }
        let verdict = NemotronTuneAdoption.decide(samples, seed: true)
        let medians = Array(zip(candidates, verdict.medians))
        var pick: Variant? = nil
        if let forced, let f = alive.first(where: { $0.name == forced }) {
            pick = f
        } else if let best = verdict.adopted {
            pick = candidates[best]
        }
        return Outcome(offered: offers.count, alive: alive, dropped: dropped, medians: medians, choice: pick,
                       verdict: verdict)
    }

    static func tune(_ target: Module) {
        choice = [:]
        let t0 = DispatchTime.now().uptimeNanoseconds
        func report(_ message: String) {
            let s = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
            FileHandle.standardError.write(Data(
                "nemotron seed dense retile: \(message) (\(String(format: "%.2f", s)) s)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.runGather else { return report("off (setting)") }
        var head: [String] = []
        var detail: [String] = []
        func note(_ kind: String, _ o: Outcome) {
            func ms(_ v: Variant?) -> String {
                o.medians.first(where: { $0.0 == v }).map { String(format: "%.3f", $0.1) } ?? "n/a"
            }
            head.append("\(kind) \(o.choice?.name ?? "V0") \(ms(o.choice)) ms vs V0 \(ms(nil))")
            let times = o.medians.indices.dropFirst().map {
                "\(o.medians[$0].0!.name) \(String(format: "%.3f", o.medians[$0].1)) \(o.verdict?.won($0) ?? "n/a")"
            }
            detail.append("\(kind) \(o.alive.count)/\(o.offered) bitwise"
                + (o.verdict.map { v in ", " + v.rule({ o.medians[$0].0?.name ?? "V0" }) } ?? "")
                + (times.isEmpty ? "" : " [" + times.joined(separator: ", ") + "]")
                + (o.dropped.isEmpty ? "" : " dropped: " + o.dropped.prefix(16).joined(separator: ", ")))
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
            let o = evaluate(.dict, cases: xs.count, layers: all.count, run: { v, c, layer in
                if let v { return launch(v, source: .dict, x: xs[c], linear: all[layer], relu2: relu2) }
                return NemotronPrefillDenseDict.launch(xs[c], all[layer], relu2: relu2)
            }, chain: chainTiming ? { v, layer, dep in
                if let v { return launch(v, source: .dict, x: xs[0], linear: all[layer], relu2: relu2, dep: dep) }
                return NemotronPrefillDenseDict.launch(xs[0], all[layer], relu2: relu2, dep: dep)
            } : nil)
            if let pick = o.choice { choice[kind] = pick }
            note(kind, o)
        }
        let downs = target.modules().compactMap { ($0 as? NemotronHMoE)?.sharedExperts?.downProj }
        if NemotronPrefillSharedDownCopy.verified, let first = downs.first {
            // The shared up's ReLU2 output, except the scaled case.
            let xs = inputs(first.weight.dim(1) * 8, seed: 0x5eed_de80, zeros: { $0 != 1 })
            if xs.count == rowCounts.count {
                let o = evaluate(.copy, cases: xs.count, layers: downs.count, run: { v, c, layer in
                    if let v { return launch(v, source: .copy, x: xs[c], linear: downs[layer], relu2: false) }
                    return NemotronPrefillSharedDownCopy.launch(xs[c], downs[layer])
                }, chain: chainTiming ? { v, layer, dep in
                    if let v { return launch(v, source: .copy, x: xs[0], linear: downs[layer], relu2: false, dep: dep) }
                    return NemotronPrefillSharedDownCopy.launch(xs[0], downs[layer], dep: dep)
                } : nil)
                if let pick = o.choice { choice["shared_down"] = pick }
                note("shared_down", o)
            } else {
                detail.append("shared_down off (inputs)")
            }
        } else {
            detail.append("shared_down off (not verified)")
        }
        report(head.joined(separator: ", ") + " (per launch, 488 rows, median of \(NemotronTuneAdoption.minCycles) over"
            + " every layer; "
            + detail.joined(separator: "; ") + ")")
    }
}

// MARK: - Seed stock quantized-matmul tune (NEMOTRON_SEED_QMM_TUNE)

/// Seed request prefill only: each MoE block's shared expert runs the request-prefill launches of
/// `NemotronHMLP.callAsFunction` with the down projection's input taking the routed FC1 launch's output as an MLX
/// dependency (`depends`: no launch, the same values). MLX then encodes the shared down after FC1, whose expert bank
/// ends that command buffer, so it runs beside the routed FC2 instead of beside the counting sort. Evaluation order
/// only; every kernel and value is unchanged. NEMOTRON_SEED_SHARED_OVERLAP=0 keeps the current order.
enum NemotronSeedSharedOverlap {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_SEED_SHARED_OVERLAP"] != "0"

    /// The block's shared expert when the reordered build applies (the seed's combine-add route), else nil.
    static func mlp(_ shared: NemotronHMLP?) -> NemotronHMLP? {
        guard enabled, NemotronHPrefillX5.requestPrefill, NemotronHPrefillX5.active,
              NemotronPrefillCombineAdd.verified, let shared, type(of: shared) == NemotronHMLP.self
        else { return nil }
        return shared
    }
}

extension NemotronHMLP {
    /// `callAsFunction(x)` for request-prefill rows (more than one), the down projection's input depending on `dep`.
    func seedShared(_ x: MLXArray, after dep: MLXArray) -> MLXArray {
        if let a = NemotronPrefillDenseDict.apply(x, upProj, kind: "up_proj", relu2: true)
            ?? NemotronPrefillSharedUpGather.apply(x, upProj) {
            let d = depends(input: a, dependencies: [dep])
            return NemotronPrefillSharedDownCopy.apply(d, downProj) ?? downProj(d)
        }
        let up = upProj(x)
        if let a = NemotronPrefillSharedRelu2.apply(up) {
            let d = depends(input: a, dependencies: [dep])
            return NemotronPrefillSharedDownCopy.apply(d, downProj) ?? downProj(d)
        }
        let r = MLX.maximum(up, MLXArray(0))
        return downProj(depends(input: r * r, dependencies: [dep]))
    }
}

/// Seed-only (request prefill, `NemotronHPrefillX5.active`, NAX hosts): load-time choice for the dense projections
/// the seed runs through MLX's stock quantized matmul (`affine_qmm_t_nax`) because their pair-dictionary or
/// decode-copy launch did not qualify: Mamba `out_proj` and the shared `up_proj` (kinds `NemotronPrefillDenseDict`
/// did not admit) and the shared `down_proj` (`NemotronPrefillSharedDownCopy` not verified). Those qualifies also
/// try row counts at which the stock product does not run `qmm_nax` (split K, or qmv below MLX's batch limit), and
/// fail there. Candidates are `NemotronSeedDenseRetile`'s launches in each of its geometries (`variants`;
/// 64x64x64w22 is the stock kernel's geometry) reading the projection's pair dictionary (out_proj; up_proj with the
/// typed ReLU2 store) or the shared down's decode copy. Each keeps every output's `BaseNAXFrag::mma` sequence over
/// k chunks 0, 16, ..., K - 16 in order on the BF16 values MLX's `QuantizedBlockLoader` dequantizes, as the stock
/// kernel does. `tune` (load time, after the seed retiles) compares every candidate bit for bit with the seed's
/// stock path (`linear(x)`, then relu2 for up_proj) at row counts where `stockRunsNAX` holds: 488 rows on every
/// layer of the kind; 488 rows x 40, 300 rows, the smallest admitted row count and 488 rows with specials on the
/// first and last layers (ReLU2-like rows for the shared down), three passes; any mismatch or Metal error drops a
/// candidate. It then times the stock path and the survivors round robin (`NemotronTuneAdoption.order`, median) as
/// chains in which every launch reads one operand from a view of the previous launch's output (the stock product
/// its biases, a dictionary candidate its pair table, a copy candidate its biases), so the launches run one after
/// another as in the seed, and adopts the fastest candidate only when `NemotronTuneAdoption.decide` admits it (at
/// least 3% faster than the stock path by median and faster than it in at least 75% of the cycles, same-cycle
/// samples). Production
/// (`NemotronPrefillDenseDict.apply`, `NemotronPrefillSharedDownCopy.apply`) launches the adopted candidate only
/// where `stockRunsNAX` holds for the request's rows; the residency target groups then treat the kind as admitted.
/// NEMOTRON_SEED_QMM_TUNE=0 keeps the stock projections; NEMOTRON_SEED_QMM_FORCE=<n> adopts variant n (1-based,
/// `NemotronSeedQMMTune.variants`: `NemotronSeedDenseRetile`'s base rows, then its pf rows, then its lean rows,
/// then `NemotronSeedDenseRetile.moreVariants` of at most 128 rows per tile, then its h rows, offered to the pair
/// dictionary kinds) wherever it qualified, 0 keeps the stock projections.
/// NEMOTRON_SEED_QMM_TUNE_X=0 leaves the pf rows out, NEMOTRON_SEED_QMM_TUNE_X2=0 the `moreVariants` rows,
/// NEMOTRON_SEED_DENSE_HB=0 the h rows.
enum NemotronSeedQMMTune {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_SEED_QMM_TUNE"] != "0"
    static let forced = Int(ProcessInfo.processInfo.environment["NEMOTRON_SEED_QMM_FORCE"] ?? "")
    /// Launches per timing sample: the kind's first layers in model order.
    static let chainLength = 12

    typealias Variant = NemotronSeedDenseRetile.Variant

    /// `NemotronSeedDenseRetile`'s base rows, then the stock geometry's K-loop prefetch twin (64x64x64w22pf);
    /// NEMOTRON_SEED_QMM_TUNE_X=0 leaves the pf row out. FORCE indices 1 ... 7 name the base rows either way.
    static let offerX = ProcessInfo.processInfo.environment["NEMOTRON_SEED_QMM_TUNE_X"] != "0"
    /// Last `NemotronSeedDenseRetile.moreVariants` of at most 128 rows per tile (pf with two stages on the
    /// incumbent and the 128-row 4 x 2 geometry, 128-row tiles on a 2 x 2 grid; the 256-row tiles leave these kinds'
    /// narrower products about two threadgroups per core); NEMOTRON_SEED_QMM_TUNE_X2=0 leaves them out. Appended
    /// after the pf and lean rows, so every earlier row keeps its FORCE index (15 ... 17 with every set offered).
    static let offerX2 = ProcessInfo.processInfo.environment["NEMOTRON_SEED_QMM_TUNE_X2"] != "0"
    /// Last `NemotronSeedDenseRetile.halfVariants` (h: direct B, one 16-bit code load per quad, row-guarded mma;
    /// pair dictionary kinds only; NEMOTRON_SEED_DENSE_HB=0 leaves them out), after every other row so each keeps
    /// its FORCE index (the h rows are 18 ... 22 with every set offered).
    static let variants: [Variant] = NemotronSeedDenseRetile.baseVariants
        + (offerX ? NemotronSeedDenseRetile.extraVariants : [])
        + NemotronSeedDenseRetile.leanVariants
        + (offerX2 ? NemotronSeedDenseRetile.moreVariants.filter { $0.bm <= 128 } : [])
        + NemotronSeedDenseRetile.halfVariants
    /// Soft cap on the tune's own load time: no kind, variant or timing phase starts past it. 6 s for the base
    /// rows, scaled by the candidate count so the base rows keep their share of it when the pf, lean, X2 and h rows
    /// are offered.
    static let budgetSeconds = 6.0 * nemotronTuneBudgetScale * Double(variants.count)
        / Double(NemotronSeedDenseRetile.baseVariants.count)

    /// Adopted variant per kind ("out_proj", "up_proj", "shared_down"), set once at load before any request.
    nonisolated(unsafe) private(set) static var choice: [String: Variant] = [:]

    static func adopted(_ kind: String) -> Bool { choice[kind] != nil }

    /// Whether the seed's attention q_proj / o_proj join the tune (NEMOTRON_SEED_QMM_ATTN=0 leaves them stock).
    static let attentionKinds = ProcessInfo.processInfo.environment["NEMOTRON_SEED_QMM_ATTN"] != "0"

    /// The seed's attention projection `kind` ("q_proj", "o_proj") through the adopted launch, where the stock
    /// product runs `qmm_nax` on the request's rows; nil otherwise (the caller runs its stock projection).
    static func attention(_ x: MLXArray, _ linear: Linear, kind: String) -> MLXArray? {
        guard attentionKinds, NemotronHPrefillX5.active, choice[kind] != nil, x.ndim == 3, x.dim(0) == 1,
              x.dim(1) > 32, NemotronSBDict.dict(linear) != nil else { return nil }
        return dense(x, linear, kind: kind, relu2: false)
    }

    /// Whether MLX's quantized matmul runs `qmm_nax` for a BF16 [m, k] x [n, k]^T product (one batch, transposed,
    /// group size 64, NAX host): m at or above every `get_qmv_batch_limit` (at most 33) and `qmm_splitk`'s split
    /// count 1 (quantized.cpp: 512 / (ceil(n / 32) * ceil(m / 32)), at least 1, capped at k / 64 and lowered until
    /// split * 64 divides k; 1 hands the product to `qmm`, which runs `qmm_nax`).
    static func stockRunsNAX(rows m: Int, n: Int, k: Int) -> Bool {
        guard m >= 33, n > 0, k >= 64, k % 64 == 0 else { return false }
        var split = max(1, 512 / (((n + 31) / 32) * ((m + 31) / 32)))
        split = min(split, k / 64)
        while split > 1 && k % (split * 64) != 0 { split -= 1 }
        return split <= 1
    }

    /// A contiguous view of `dep`'s leading bytes with `a`'s dtype and shape (timing chains only: a launch reading
    /// it waits for `dep`'s producer); nil when `dep` is too small.
    static func operandView(_ dep: MLXArray, like a: MLXArray) -> MLXArray? {
        var flat = dep.reshaped(-1)
        if flat.dtype != a.dtype {
            guard (flat.size * flat.dtype.size) % a.dtype.size == 0, (flat.size * flat.dtype.size) / a.dtype.size >= a.size
            else { return nil }
            flat = flat.view(dtype: a.dtype)
        }
        guard flat.ndim == 1, flat.size >= a.size else { return nil }
        return flat[0 ..< a.size].reshaped(a.shape)
    }

    /// From `NemotronPrefillDenseDict.apply` for a kind it did not admit: the adopted launch where the stock product
    /// runs `qmm_nax`; nil otherwise (the stock projection then runs).
    static func dense(_ x: MLXArray, _ linear: Linear, kind: String, relu2: Bool) -> MLXArray? {
        guard let v = choice[kind], relu2 == (kind == "up_proj"), x.ndim == 3, let q = linear as? QuantizedLinear,
              stockRunsNAX(rows: x.dim(1), n: q.weight.dim(0), k: x.dim(2))
        else { return nil }
        return NemotronSeedDenseRetile.launch(v, source: .dict, x: x, linear: linear, relu2: relu2)
    }

    /// From `NemotronPrefillSharedDownCopy.apply` when the copy launch was not verified: as `dense`.
    static func sharedDown(_ x: MLXArray, _ down: Linear) -> MLXArray? {
        guard let v = choice["shared_down"], x.ndim == 3, let q = down as? QuantizedLinear,
              stockRunsNAX(rows: x.dim(1), n: q.weight.dim(0), k: x.dim(2))
        else { return nil }
        return NemotronSeedDenseRetile.launch(v, source: .copy, x: x, linear: down, relu2: false)
    }

    private struct Kind {
        let name: String
        let linears: [Linear]
        let source: NemotronSeedDenseRetile.Source
        let relu2: Bool
        /// ReLU2-like check inputs (mostly exact zeros), as the shared down sees.
        let zeros: Bool
    }

    /// The seed's stock path: `linear(x)`, then for up_proj relu2 as the seed takes it (`NemotronPrefillSharedRelu2`
    /// once verified, else the stock relu2). With `dep` (timing chains only) the product reads its biases from a view
    /// of `dep`.
    private static func stock(_ x: MLXArray, _ linear: Linear, relu2: Bool, dep: MLXArray?) -> MLXArray? {
        let y: MLXArray
        if let dep {
            guard let q = linear as? QuantizedLinear, let b = q.biases, let view = operandView(dep, like: b)
            else { return nil }
            y = quantizedMM(x, q.weight, scales: q.scales, biases: view, transpose: true, groupSize: q.groupSize,
                            bits: q.bits, mode: q.mode)
        } else {
            y = linear(x)
        }
        guard relu2 else { return y }
        if NemotronPrefillSharedRelu2.verified, let fused = NemotronPrefillSharedRelu2.launch(y) { return fused }
        let r = MLX.maximum(y, MLXArray(0))
        return r * r
    }

    /// Check, time and pick for one kind; returns the pick and a report fragment.
    private static func evaluate(_ kind: Kind, seed: UInt64, specials: MLXArray, elapsed: () -> Double)
        -> (Variant?, String) {
        let variants = NemotronSeedDenseRetile.offered(Self.variants, kind.source)
        guard let first = kind.linears.first as? QuantizedLinear, let last = kind.linears.last,
              kind.linears.allSatisfy({ ($0 as? QuantizedLinear)?.weight.shape == first.weight.shape })
        else { return (nil, "off (no quantized layers)") }
        let k = first.weight.dim(1) * 8, n = first.weight.dim(0)
        guard let boundary = (33 ... 488).first(where: { stockRunsNAX(rows: $0, n: n, k: k) }) else {
            return (nil, "off (stock never runs qmm_nax)")
        }
        // Rows: 488, 488 x 40, 300, the smallest admitted count, 488 with specials (first row and mid-rows).
        let rowCounts = [488, 488, 300, boundary, 488]
        var inputs: [MLXArray] = []
        do {
            inputs = try withError { () -> [MLXArray] in
                rowCounts.enumerated().map { index, rows in
                    var a = MLXRandom.normal([1, rows, k], scale: 2, key: MLXRandom.key(seed + UInt64(index)))
                    if index == 1 { a = a * 40 }
                    if kind.zeros && index != 1 {
                        a = MLX.maximum(a - 1.5, MLXArray(Float(0)))
                        a = a * a
                    }
                    if index == 4 {
                        let flat = a.reshaped(-1), mid = (rows / 2) * k + 5
                        a = concatenated([specials, flat[8 ..< mid], specials, flat[(mid + 8)...]]).reshaped(1, rows, k)
                    }
                    let x = a.asType(.bfloat16)
                    eval(x)
                    return x
                }
            }
        } catch {
            return (nil, "off (inputs: \(error))")
        }
        guard inputs.count == rowCounts.count else { return (nil, "off (inputs)") }
        let edges: [Linear] = kind.linears.count > 1 ? [first, last] : [first]
        func layers(_ c: Int) -> [Linear] { c == 0 ? kind.linears : edges }
        func candidate(_ v: Variant, _ x: MLXArray, _ linear: Linear, dep: MLXArray?) -> MLXArray? {
            NemotronSeedDenseRetile.launch(v, source: kind.source, x: x, linear: linear, relu2: kind.relu2, dep: dep)
        }

        var references: [[MLXArray]] = []
        do {
            references = try withError { () -> [[MLXArray]] in
                var all: [[MLXArray]] = []
                for (c, x) in inputs.enumerated() {
                    var outs: [MLXArray] = []
                    for linear in layers(c) {
                        guard let y = stock(x, linear, relu2: kind.relu2, dep: nil) else { return [] }
                        outs.append(y)
                    }
                    eval(outs)
                    all.append(outs)
                }
                return all
            }
        } catch {
            return (nil, "off (stock: \(error))")
        }
        guard references.count == inputs.count else { return (nil, "off (stock unavailable)") }

        var alive: [Variant] = []
        var dropped: [String] = []
        for v in variants {
            guard elapsed() < budgetSeconds else {
                dropped.append("\(v.name) budget")
                continue
            }
            do {
                // Three passes over every case and layer: a race would have to repeat to pass.
                let same = try withError { () -> Bool in
                    for _ in 0 ..< 3 {
                        for (c, x) in inputs.enumerated() {
                            var ok = MLXArray(true)
                            for (i, linear) in layers(c).enumerated() {
                                guard let y = candidate(v, x, linear, dep: nil) else { return false }
                                ok = ok .&& (y.view(dtype: .uint16) .== references[c][i].view(dtype: .uint16)).all()
                            }
                            if !ok.item(Bool.self) { return false }
                        }
                    }
                    return true
                }
                if same { alive.append(v) } else { dropped.append("\(v.name) differs") }
            } catch {
                dropped.append("\(v.name) error")
            }
        }
        // Report only: one row below the smallest admitted count the stock product leaves `qmm_nax`, so the
        // stock geometry's twin is expected to differ there.
        var below = ""
        if boundary > 33, let v0 = variants.first, alive.contains(v0) {
            let same = try? withError { () -> Bool in
                let x = MLXRandom.normal([1, boundary - 1, k], scale: 2, key: MLXRandom.key(seed + 9)).asType(.bfloat16)
                guard let a = stock(x, first, relu2: kind.relu2, dep: nil), let b = candidate(v0, x, first, dep: nil)
                else { return false }
                return (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all().item(Bool.self)
            }
            below = ", \(boundary - 1) rows " + (same == true ? "equal" : "differ")
        }

        let candidates: [Variant?] = [nil] + alive.map { Optional($0) }
        let chain = Array(kind.linears.prefix(chainLength))
        var medians: [Double] = []
        var samples: [[Double]] = []
        var verdict: NemotronTuneAdoption.Verdict? = nil
        var reps = 0
        if !alive.isEmpty, elapsed() >= budgetSeconds {
            dropped.append("timing skipped (load budget)")
        } else if !alive.isEmpty {
            samples = Array(repeating: [Double](), count: candidates.count)
            do {
                try withError {
                    let x = inputs[0]
                    func build(_ i: Int) -> MLXArray? {
                        var previous: MLXArray? = nil
                        for linear in chain {
                            let out = candidates[i].map { candidate($0, x, linear, dep: previous) }
                                ?? stock(x, linear, relu2: kind.relu2, dep: previous)
                            guard let out else { return nil }
                            previous = out
                        }
                        return previous
                    }
                    func round(_ order: [Int]) -> [Double?] {
                        let chains = order.map { build($0) }
                        return chains.map { last -> Double? in
                            guard let last else { return nil }
                            let start = DispatchTime.now().uptimeNanoseconds
                            eval(last)
                            Stream.gpu.synchronize()
                            return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(chain.count)
                        }
                    }
                    var wall = 0.0
                    for _ in 0 ..< 2 {
                        let start = DispatchTime.now().uptimeNanoseconds
                        _ = round(Array(candidates.indices))
                        wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    }
                    reps = NemotronTuneAdoption.cycles(target: 250, wall: wall, cap: 31, seed: true)
                    for rep in 0 ..< reps {
                        let order = NemotronTuneAdoption.order(candidates.count, cycle: rep)
                        for (i, ms) in zip(order, round(order)) {
                            if let ms { samples[i].append(ms) }
                        }
                    }
                }
                if samples.allSatisfy({ $0.count == reps }) {
                    verdict = NemotronTuneAdoption.decide(samples, seed: true)
                    medians = verdict?.medians ?? []
                } else {
                    dropped.append("timing incomplete")
                }
            } catch {
                dropped.append("timing error")
            }
        }
        var picked: Variant? = nil
        var why = "kept"
        if let f = forced {
            if f >= 1, f <= variants.count, alive.contains(variants[f - 1]) {
                picked = variants[f - 1]
                why = "forced"
            } else {
                why = f == 0 ? "forced stock" : "forced \(f) not qualified, kept"
            }
        } else if !medians.isEmpty, let best = verdict?.adopted {
            picked = candidates[best]
            why = "adopted"
        }
        func ms(_ v: Variant?) -> String {
            guard let i = candidates.firstIndex(of: v), i < medians.count else { return "n/a" }
            return String(format: "%.4f", medians[i])
        }
        func won(_ v: Variant) -> String {
            guard let i = candidates.firstIndex(of: v), let verdict else { return "" }
            return " " + verdict.won(i)
        }
        let times = medians.isEmpty ? [] : alive.map { "\($0.name) \(ms($0))\(won($0))" }
        let rule = verdict?.rule({ candidates[$0]?.name ?? "stock" }) ?? "no candidate timed"
        return (picked, "\(why) \(picked?.name ?? "stock") \(ms(picked)) ms vs stock \(ms(nil)) ms per launch (\(rule);"
            + " median of \(reps), chains of \(chain.count); \(alive.count)/\(variants.count) bitwise over"
            + " \(rowCounts.count) cases x 3, smallest \(boundary) rows\(below)"
            + (times.isEmpty ? "" : "; [" + times.joined(separator: ", ") + "]")
            + (dropped.isEmpty ? "" : "; dropped: " + dropped.prefix(16).joined(separator: ", ")) + ")")
    }

    /// Load time, after the seed retiles; NAX hosts only (`NemotronHPrefillX5.runGather`).
    static func tune(_ target: Module) {
        choice = [:]
        let t0 = DispatchTime.now().uptimeNanoseconds
        func elapsed() -> Double { Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 }
        func report(_ message: String) {
            FileHandle.standardError.write(Data(
                "nemotron seed qmm tune: \(message) (\(String(format: "%.2f", elapsed())) s)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.runGather else { return report("off (setting)") }
        var kinds: [Kind] = []
        var parts: [String] = []
        for kind in ["out_proj", "up_proj"] {
            if NemotronPrefillDenseDict.kinds.contains(kind) {
                parts.append("\(kind) off (dictionary launch admitted)")
            } else if kind == "up_proj" && NemotronPrefillSharedUpGather.verified {
                parts.append("up_proj off (shared up gather verified)")
            } else {
                kinds.append(Kind(name: kind, linears: NemotronPrefillDenseDict.linears(target, kind: kind),
                                  source: .dict, relu2: kind == "up_proj", zeros: false))
            }
        }
        // The seed's attention q_proj and o_proj (stock `linear(x)` on 488 rows, `qmm_nax`), reading the pair
        // dictionaries `NemotronSBDict` builds for every 4-bit target projection; k_proj and v_proj run split K at
        // 488 rows and stay stock. NEMOTRON_SEED_QMM_ATTN=0 leaves them out.
        if attentionKinds {
            let attentions = target.modules().compactMap { $0 as? NemotronHAttention }
            for (kind, linears) in [("q_proj", attentions.map { $0.wq as Linear }),
                                    ("o_proj", attentions.map { $0.wo as Linear })] where !linears.isEmpty {
                if linears.allSatisfy({ ($0 as? QuantizedLinear).flatMap { NemotronSBDict.dict($0) } != nil }) {
                    kinds.append(Kind(name: kind, linears: linears, source: .dict, relu2: false, zeros: false))
                } else {
                    parts.append("\(kind) off (no dictionary)")
                }
            }
        }
        if NemotronPrefillSharedDownCopy.verified {
            parts.append("shared_down off (copy launch verified)")
        } else {
            kinds.append(Kind(name: "shared_down",
                              linears: target.modules().compactMap { ($0 as? NemotronHMoE)?.sharedExperts?.downProj },
                              source: .copy, relu2: false, zeros: true))
        }
        let specials = MLXArray([Float(-0.0), 0, .infinity, -.infinity, .nan, Float(bitPattern: 0x0001_0000),
                                 Float(bitPattern: 0x8001_0000), 3e38])
        for (index, kind) in kinds.enumerated() {
            guard elapsed() < budgetSeconds else {
                parts.append("\(kind.name) skipped (load budget)")
                continue
            }
            let (pick, message) = evaluate(kind, seed: UInt64(0x5eed_9a00 + 16 * index), specials: specials,
                                           elapsed: elapsed)
            if let pick { choice[kind.name] = pick }
            parts.append(kind.name + " " + message)
        }
        report(parts.joined(separator: "; "))
    }
}

// MARK: - Seed scan-window NAX GEMM tune (NEMOTRON_SEED_SSM_TUNE)

/// Seed-only (request prefill, `NemotronHPrefillX5.active`, NAX hosts): load-time geometry choice of the Mamba
/// scan-window GEMMs, the state product `nemotron_ssm_state_gemm` (`NemotronPrefillStateGemm`) and the scan decay
/// GEMM `nemotron_ssm_scan_decay_gemm` with its fresh-window form `_fresh_out` (`NemotronPrefillScanDecayGemm`).
/// Each variant is the incumbent's text through counted replacements (nil when a statement moved) and keeps every
/// output's `tile_matmad_nax` sequence: the same 16 x 32 x 16 multiply-accumulates on the same operand fragments
/// into the same accumulator, 32-k chunks in increasing order, the A tile staged by the same expressions, the same
/// stores and epilogue. A variant moves only
/// - rows per threadgroup (`m2`: the state product's 64 rows as two threadgroups of 32 rows, four SIMD groups);
/// - K staging: 64 k staged per barrier pair and consumed as two 32-k chunks in order (`sk64`), or two staging
///   buffers with one barrier per chunk (`dbuf`);
/// - staging SIMD groups (`st8`, `st16`: eight or sixteen SIMD groups stage the scan's A tile; the incumbent's four
///   run the multiply-accumulates and the store).
/// Variants: state product 1 `m2dbuf`, 2 `m2sk64`, 3 `sk64`, 4 `dbuf`, 5 `m2`; scan decay (and its fresh form)
/// 1 `sk64st8`, 2 `st8dbuf`, 3 `sk64`, 4 `dbuf`, 5 `st8`, 6 `sk64st16`, 7 `st16dbuf` (3: NEMOTRON_SEED_SSM_TUNE_X
/// != 0; 4 and later: NEMOTRON_SEED_SSM_TUNE_X2 != 0, one lower without the X row).
/// `tune` (load time, after the seed retiles) checks every variant bit for bit against the production launch on
/// `qualify`'s kinds of seeded cases plus window lengths around the 32- and 64-k staging steps (33, 65 or 31, 96,
/// 161), with specials (also at chunk boundaries inside the window), overflowing and denormal products, `passes`
/// passes each, and drops any mismatch or Metal error; it times the incumbent and the survivors round robin
/// (`NemotronTuneAdoption.order`, median) as chains in which each launch reads the previous launch's output (the
/// state product takes its decay row from the previous product, the scan its decay input from the previous decay
/// row), drops a variant whose chain output differs from the incumbent chain's in any round, and adopts the fastest
/// remaining variant only when `NemotronTuneAdoption.decide` admits it (at least 3% faster by median and faster than
/// the incumbent in at least 75% of the cycles, same-cycle samples). The fresh-window kernel takes the scan's
/// adopted variant once that variant's fresh form passes the same kind of check (seven cases).
/// On hosts without NAX the seed path does not run these kernels and the tune reports off.
/// NEMOTRON_SEED_SSM_TUNE=0 keeps the incumbents; NEMOTRON_SEED_SSM_FORCE=<n> adopts variant n (1-based, per kernel
/// list) wherever it qualified, 0 keeps the incumbents.
enum NemotronSeedSSMTune {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_SEED_SSM_TUNE"] != "0"
    static let forced = Int(ProcessInfo.processInfo.environment["NEMOTRON_SEED_SSM_FORCE"] ?? "")
    /// Check passes over every case: a staging race would have to repeat in every pass to go unseen.
    static let passes = 5

    enum Staging: Sendable { case sk32, sk64, dbuf }

    struct Variant: Sendable, Equatable {
        let name: String
        /// Output rows per threadgroup (the incumbents': 64).
        let rows: Int
        let staging: Staging
        /// SIMD groups staging the scan's A tile (nil: the four computing ones).
        let stage: Int?
    }

    /// NEMOTRON_SEED_SSM_TUNE_X=0 leaves out the third row of each list (`sk64` at the incumbent's shape: the
    /// state product's 64 rows per threadgroup, the scan's four staging SIMD groups). Appended, so FORCE 1 and 2
    /// keep their rows.
    static let offerX = ProcessInfo.processInfo.environment["NEMOTRON_SEED_SSM_TUNE_X"] != "0"
    /// NEMOTRON_SEED_SSM_TUNE_X2=0 leaves out the rows after those: the state product's `dbuf` (two staging buffers
    /// at the incumbent's 64 rows per threadgroup, 16,896 B) and `m2` (two threadgroups of 32 rows, the incumbent's
    /// 32-k staging); the scan's `dbuf` (two staging buffers, the incumbent's four staging SIMD groups), `st8`
    /// (eight staging SIMD groups, the incumbent's 32-k staging), `sk64st16` and `st16dbuf` (sixteen staging SIMD
    /// groups, 512 threads, each staging two or four of a step's columns; the first four SIMD groups compute and
    /// store). Appended after the X row, so FORCE 1 ... 3 keep their rows.
    static let offerX2 = ProcessInfo.processInfo.environment["NEMOTRON_SEED_SSM_TUNE_X2"] != "0"
    static let stateVariants: [Variant] = [
        Variant(name: "m2dbuf", rows: 32, staging: .dbuf, stage: nil),
        Variant(name: "m2sk64", rows: 32, staging: .sk64, stage: nil),
    ] + (offerX ? [Variant(name: "sk64", rows: 64, staging: .sk64, stage: nil)] : [])
        + (offerX2 ? [
            Variant(name: "dbuf", rows: 64, staging: .dbuf, stage: nil),
            Variant(name: "m2", rows: 32, staging: .sk32, stage: nil),
        ] : [])
    static let scanVariants: [Variant] = [
        Variant(name: "sk64st8", rows: 64, staging: .sk64, stage: 8),
        Variant(name: "st8dbuf", rows: 64, staging: .dbuf, stage: 8),
    ] + (offerX ? [Variant(name: "sk64", rows: 64, staging: .sk64, stage: nil)] : [])
        + (offerX2 ? [
            Variant(name: "dbuf", rows: 64, staging: .dbuf, stage: nil),
            Variant(name: "st8", rows: 64, staging: .sk32, stage: 8),
            Variant(name: "sk64st16", rows: 64, staging: .sk64, stage: 16),
            Variant(name: "st16dbuf", rows: 64, staging: .dbuf, stage: 16),
        ] : [])
    /// Soft cap on the tune's own load time: no variant or timing phase starts past it. 1.5 s for the two base
    /// rows per list, scaled by the candidate count so they keep their share of it when the X and X2 rows are
    /// offered.
    static let budgetSeconds = 1.5 * nemotronTuneBudgetScale * Double(stateVariants.count + scanVariants.count) / 4.0

    /// Adopted variants (nil: the incumbent), set once at load before any request.
    nonisolated(unsafe) private(set) static var stateChoice: Variant? = nil
    nonisolated(unsafe) private(set) static var scanChoice: Variant? = nil
    nonisolated(unsafe) private(set) static var freshChoice: Variant? = nil

    private static let barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"

    /// `text` with `old` replaced; nil unless `old` occurs exactly once.
    private static func once(_ text: String, _ old: String, _ new: String) -> String? {
        text.components(separatedBy: old).count == 2 ? text.replacingOccurrences(of: old, with: new) : nil
    }

    /// The K loop opened by `head`: the text before it, its staging statements (between its two barriers), its
    /// compute statements (after the second) and the text after it; nil when the loop moved.
    private static func split(_ body: String, head: String) -> (String, String, String, String)? {
        let lines = body.components(separatedBy: "\n")
        let heads = lines.indices.filter { lines[$0].trimmingCharacters(in: .whitespaces) == head }
        guard heads.count == 1 else { return nil }
        let hi = heads[0]
        let close = String(lines[hi].prefix(while: { $0 == " " })) + "}"
        guard let ei = lines[(hi + 1)...].firstIndex(where: { $0 == close }) else { return nil }
        let inner = Array(lines[(hi + 1) ..< ei])
        let bars = inner.indices.filter { inner[$0].trimmingCharacters(in: .whitespaces) == barrier }
        guard bars.count == 2, bars[0] == 0 else { return nil }
        return (lines[..<hi].joined(separator: "\n"), inner[1 ..< bars[1]].joined(separator: "\n"),
                inner[(bars[1] + 1)...].joined(separator: "\n"), lines[(ei + 1)...].joined(separator: "\n"))
    }

    /// The K loop rebuilt over `ext` for `staging` (`rows` staged rows per buffer; `only`: the SIMD groups that
    /// run the compute statements, every SIMD group reaching every barrier).
    private static func loop(_ staging: String, _ compute: String, ext: String, _ form: Staging, rows: Int,
                             only: String?) -> String? {
        let open = only.map { "if (\($0)) {" } ?? "{"
        let load = "Atile.template load<float, AS, 1>(As + tm * AS);"
        switch form {
        case .sk32:
            return ["for (int k0 = 0; k0 < \(ext); k0 += SK) {", barrier, staging, barrier, open, compute, "}", "}"]
                .joined(separator: "\n")
        case .sk64:
            guard let c = once(compute, load, "Atile.template load<float, AS, 1>(As + tm * AS + (k0 - kb));")
            else { return nil }
            return ["for (int k0 = 0; k0 < \(ext); k0 += 2 * SK) {", barrier, staging, barrier, open,
                    "const int kb = k0;", "for (int k0 = kb; k0 < kb + 2 * SK && k0 < \(ext); k0 += SK) {", c, "}",
                    "}", "}"].joined(separator: "\n")
        case .dbuf:
            guard let c = once(compute, load,
                               "Atile.template load<float, AS, 1>(As + ((k0 / SK) & 1) * (\(rows) * AS) + tm * AS);")
            else { return nil }
            return ["for (int ks = 0; ks < \(ext) + SK; ks += SK) {", "if (ks > 0) {", barrier, "}",
                    "if (ks < \(ext)) {", "const int k0 = ks;",
                    "threadgroup float* Ab = As + ((ks / SK) & 1) * (\(rows) * AS);",
                    staging.replacingOccurrences(of: "As[", with: "Ab["), "}",
                    "if (" + (only.map { "ks > 0 && (\($0))" } ?? "ks > 0") + ") {", "const int k0 = ks - SK;", c,
                    "}", "}"].joined(separator: "\n")
        }
    }

    /// The state product's text: `v.rows` of the 64 rows per threadgroup (threadgroup y picks them), all 128
    /// columns, one 32 x 32 tile per SIMD group (four per 32 rows).
    static func stateSource(_ v: Variant) -> String? {
        let ks = v.staging == .sk64 ? 64 : 32, buffers = v.staging == .dbuf ? 2 : 1, threads = 4 * v.rows
        guard v.rows == 32 || v.rows == 64,
              case let (p0, s0, compute, q0)? = split(NemotronPrefillStateGemm.body,
                                                      head: "for (int k0 = 0; k0 < n; k0 += SK) {")
        else { return nil }
        var pre: String? = p0
        if ks == 64 { pre = pre.flatMap { once($0, "constexpr int AS = SK + 1;", "constexpr int AS = 2 * SK + 1;") } }
        pre = pre.flatMap { once($0, "threadgroup float As[DH * AS];", "threadgroup float As[\(buffers * v.rows) * AS];") }
        pre = pre.flatMap { once($0, "const short tm = SM * short(sg / 4);",
                                 "const int r0 = \(v.rows) * int(threadgroup_position_in_grid.y);\n"
                                     + "    const short tm = SM * short(sg / 4);") }
        var stg: String? = once(s0, "for (int e = tid; e < DH * SK; e += 256) {",
                                "for (int e = tid; e < \(v.rows) * \(ks); e += \(threads)) {")
        stg = stg.flatMap { once($0, "const int i = e % DH;", "const int i = e % \(v.rows);") }
        stg = stg.flatMap { once($0, "const int t = e / DH;", "const int t = e / \(v.rows);") }
        stg = stg.flatMap { once($0, "const float a = xr[size_t(tt) * H * DH + i];",
                                 "const float a = xr[size_t(tt) * H * DH + r0 + i];") }
        let post = once(q0, "Dtile.store(out + (size_t(bh) * DH + tm) * DS + tn, DS);",
                        "Dtile.store(out + (size_t(bh) * DH + r0 + tm) * DS + tn, DS);")
        guard let pre, let stg, let post,
              let body = loop(stg, compute, ext: "n", v.staging, rows: v.rows, only: nil) else { return nil }
        return [pre, body, post].joined(separator: "\n")
    }

    /// The scan decay GEMM's text (`fresh`: the fresh-window form): 64 time rows x 64 columns per threadgroup as
    /// before, the A tile staged by `v.stage` SIMD groups (the computing four only when nil).
    static func scanSource(_ v: Variant, fresh: Bool) -> String? {
        let ks = v.staging == .sk64 ? 64 : 32, buffers = v.staging == .dbuf ? 2 : 1, stagers = v.stage ?? 4
        guard v.rows == 64, stagers >= 4,
              case let (p0, s0, compute, q0)? = split(NemotronPrefillScanDecayGemm.source(freshOut: fresh),
                                                      head: "for (int k0 = 0; k0 < L; k0 += SK) {")
        else { return nil }
        var pre: String? = p0
        if ks == 64 { pre = pre.flatMap { once($0, "constexpr int AS = SK + 1;", "constexpr int AS = 2 * SK + 1;") } }
        pre = pre.flatMap { once($0, "threadgroup float As[64 * AS];", "threadgroup float As[\(buffers * 64) * AS];") }
        let stg = once(s0, "for (int c = int(sg); c < SK; c += 4) {", "for (int c = int(sg); c < \(ks); c += \(stagers)) {")
        let only = stagers > 4 ? "sg < 4" : nil
        guard let pre, let stg,
              let body = loop(stg, compute, ext: "L", v.staging, rows: 64, only: only) else { return nil }
        return [pre, body, only.map { "if (\($0)) {\n" + q0 + "\n}" } ?? q0].joined(separator: "\n")
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel?] = [:]

    private static func kernel(_ kind: String, _ v: Variant) -> MLXFast.MLXFastKernel? {
        lock.withLock { () -> MLXFast.MLXFastKernel? in
            let key = kind + "_" + v.name
            if let made = kernels[key] { return made }
            var made: MLXFast.MLXFastKernel? = nil
            switch kind {
            case "state":
                made = stateSource(v).map {
                    MLXFast.metalKernel(
                        name: "nemotron_seed_ssm_state_" + v.name, inputNames: ["dtx", "last", "B", "n"],
                        outputNames: ["out"], source: $0, header: nemotronRunGatherHeader)
                }
            case "scan":
                made = scanSource(v, fresh: false).map {
                    MLXFast.metalKernel(
                        name: "nemotron_seed_ssm_scan_" + v.name, inputNames: ["dtA", "CB", "dtx", "L"],
                        outputNames: ["y", "last"], source: $0, header: nemotronRunGatherHeader)
                }
            default:
                made = scanSource(v, fresh: true).map {
                    MLXFast.metalKernel(
                        name: "nemotron_seed_ssm_fresh_" + v.name,
                        inputNames: ["dtA", "CB", "dtx", "L", "ec", "X", "Dw"], outputNames: ["o", "last"],
                        source: $0, header: nemotronRunGatherHeader, ensureRowContiguous: false)
                }
            }
            kernels[key] = made
            return made
        }
    }

    /// One state-product launch of `v` under `NemotronPrefillStateGemm.launch`'s guards; nil outside them.
    static func launchState(_ v: Variant, _ dtx: MLXArray, _ last: MLXArray, _ bWindow: MLXArray, n: Int, h: Int,
                            g: Int, dh: Int, d: Int) -> MLXArray? {
        guard NemotronHPrefillX5.runGather, dh == 64, d == 128, g > 0, h % g == 0,
              dtx.dtype == .float32, last.dtype == .float32, bWindow.dtype == .float32,
              dtx.shape == [1, n, h, dh], last.shape == [1, h, 1, n], bWindow.shape == [1, g, 1, n, d],
              let kernel = kernel("state", v)
        else { return nil }
        return kernel(
            [dtx, last, bWindow, n],
            template: [("H", h), ("R", h / g), ("DH", dh), ("DS", d)],
            grid: (4 * v.rows, 64 / v.rows, h), threadGroup: (4 * v.rows, 1, 1),
            outputShapes: [[1, h, dh, d]], outputDTypes: [.float32])[0]
    }

    /// One scan-decay launch of `v` under `NemotronPrefillScanDecayGemm.launch`'s guards; nil outside them.
    static func launchScan(_ v: Variant, _ dtA: MLXArray, _ CB: MLXArray, _ dtx: MLXArray, n: Int, b: Int, h: Int,
                           g: Int, dh: Int) -> (y: MLXArray, last: MLXArray)? {
        guard NemotronHPrefillX5.runGather, b == 1, dh == 64, g > 0, h % g == 0,
              dtA.dtype == .float32, dtx.dtype == .float32,
              CB.dtype == .bfloat16 || CB.dtype == .float16 || CB.dtype == .float32,
              dtA.shape == [b, n, h], dtx.shape == [b, n, h, dh], CB.shape == [b, g, n, n],
              let kernel = kernel("scan", v)
        else { return nil }
        let threads = 32 * (v.stage ?? 4)
        let out = kernel(
            [dtA, CB, dtx, n],
            template: [("CBT", CB.dtype), ("H", h), ("R", h / g), ("DH", dh)],
            grid: (((n + 63) / 64) * threads, 1, b * h), threadGroup: (threads, 1, 1),
            outputShapes: [[b, h, n, dh], [b, h, 1, n]], outputDTypes: [.float32, .float32])
        return (out[0], out[1])
    }

    /// One fresh-window launch of `v` under `NemotronPrefillScanDecayGemm.launchFresh`'s guards; nil outside them.
    static func launchFresh(_ v: Variant, _ dtA: MLXArray, _ CB: MLXArray, _ dtx: MLXArray, n: Int, h: Int, g: Int,
                            dh: Int, ec: MLXArray, x: MLXArray, D: MLXArray, lt: Int)
        -> (o: MLXArray, last: MLXArray)? {
        guard NemotronHPrefillX5.runGather, dh == 64, g > 0, h % g == 0,
              dtA.dtype == .float32, dtx.dtype == .float32,
              CB.dtype == .bfloat16 || CB.dtype == .float16 || CB.dtype == .float32,
              dtA.shape == [1, n, h], dtx.shape == [1, n, h, dh], CB.shape == [1, g, n, n],
              dtA.strides.last == 1, dtx.strides.last == 1, CB.strides.last == 1,
              dtA.strides[1] == h, dtx.strides[1] == h * dh, dtx.strides[2] == dh, CB.strides[2] == n,
              CB.strides[1] == n * n,
              ec.dtype == .float32, ec.shape == [1, n, h], ec.strides[1] == h, ec.strides[2] == 1,
              x.dtype == .bfloat16 || x.dtype == .float16, x.ndim == 4, x.dim(0) == 1, x.dim(1) >= n,
              x.dim(2) == h, x.dim(3) == dh, D.dtype == x.dtype, D.shape == [h], lt >= n,
              let kernel = kernel("fresh", v)
        else { return nil }
        let threads = 32 * (v.stage ?? 4)
        let out = kernel(
            [dtA, CB, dtx, n, ec, x, D],
            template: [("CBT", CB.dtype), ("H", h), ("R", h / g), ("DH", dh), ("OT", x.dtype), ("LT", lt)],
            grid: (((n + 63) / 64) * threads, 1, h), threadGroup: (threads, 1, 1),
            outputShapes: [[1, lt, h, dh], [1, h, 1, n]], outputDTypes: [x.dtype, .float32])
        return (out[0], out[1])
    }

    /// Production hooks (`apply` of the two enums): the adopted variant's launch, nil for the incumbent's.
    static func state(_ dtx: MLXArray, _ last: MLXArray, _ bWindow: MLXArray, n: Int, h: Int, g: Int, dh: Int,
                      d: Int) -> MLXArray? {
        stateChoice.flatMap { launchState($0, dtx, last, bWindow, n: n, h: h, g: g, dh: dh, d: d) }
    }
    static func scan(_ dtA: MLXArray, _ CB: MLXArray, _ dtx: MLXArray, n: Int, b: Int, h: Int, g: Int, dh: Int)
        -> (y: MLXArray, last: MLXArray)? {
        scanChoice.flatMap { launchScan($0, dtA, CB, dtx, n: n, b: b, h: h, g: g, dh: dh) }
    }
    static func fresh(_ dtA: MLXArray, _ CB: MLXArray, _ dtx: MLXArray, n: Int, h: Int, g: Int, dh: Int,
                      ec: MLXArray, x: MLXArray, D: MLXArray, lt: Int) -> (o: MLXArray, last: MLXArray)? {
        freshChoice.flatMap { launchFresh($0, dtA, CB, dtx, n: n, h: h, g: g, dh: dh, ec: ec, x: x, D: D, lt: lt) }
    }

    /// `a` with `specials` written at each flat offset of `at` (ascending; offsets that would overlap or leave the
    /// array are skipped).
    private static func plant(_ a: MLXArray, _ specials: MLXArray, at offsets: [Int]) -> MLXArray {
        let flat = a.reshaped(-1)
        var pieces: [MLXArray] = []
        var from = 0
        for o in offsets.sorted() where o >= from && o + specials.size <= flat.size {
            if o > from { pieces.append(flat[from ..< o]) }
            pieces.append(specials.asType(a.dtype))
            from = o + specials.size
        }
        if from < flat.size { pieces.append(flat[from...]) }
        return concatenated(pieces).reshaped(a.shape)
    }

    /// Check, time and pick over `variants`: `run(v, i)` is check case i's outputs (nil `v`: the production
    /// launch); `link(v, previous)` one timing-chain link after `previous` (nil for the first), returning what the
    /// next link reads. Returns the pick and a report fragment.
    private static func choose(_ variants: [Variant], cases: Int, chain: Int, elapsed: () -> Double,
                               run: (Variant?, Int) -> [MLXArray]?, link: (Variant?, MLXArray?) -> MLXArray?)
        -> (Variant?, String) {
        var references: [[MLXArray]] = []
        do {
            references = try withError { () -> [[MLXArray]] in
                let outs = (0 ..< cases).compactMap { run(nil, $0) }
                guard outs.count == cases else { return [] }
                eval(outs.flatMap { $0 })
                return outs
            }
        } catch {
            return (nil, "off (incumbent: \(error))")
        }
        guard references.count == cases else { return (nil, "off (incumbent unavailable)") }
        var alive: [Variant] = []
        var dropped: [String] = []
        for v in variants {
            guard elapsed() < budgetSeconds else {
                dropped.append("\(v.name) budget")
                continue
            }
            do {
                // `passes` passes over every case: a staging race would have to repeat to pass.
                let same = try withError { () -> Bool in
                    var ok = MLXArray(true)
                    for _ in 0 ..< passes {
                        for i in 0 ..< cases {
                            guard let out = run(v, i), out.count == references[i].count else { return false }
                            for (a, b) in zip(out, references[i]) {
                                ok = ok .&& (a.view(dtype: a.dtype == .float32 ? .uint32 : .uint16)
                                    .== b.view(dtype: b.dtype == .float32 ? .uint32 : .uint16)).all()
                            }
                        }
                        eval(ok)
                    }
                    return ok.item(Bool.self)
                }
                if same { alive.append(v) } else { dropped.append("\(v.name) differs") }
            } catch {
                dropped.append("\(v.name) error")
            }
        }
        let candidates: [Variant?] = [nil] + alive.map { Optional($0) }
        var medians: [Double] = []
        var verdict: NemotronTuneAdoption.Verdict? = nil
        var samples: [[Double]] = []
        var reps = 0
        if !alive.isEmpty, elapsed() >= budgetSeconds {
            dropped.append("timing skipped (load budget)")
            alive = []
        }
        if !alive.isEmpty {
            samples = Array(repeating: [Double](), count: candidates.count)
            // Every timed chain's last output against the incumbent chain's of the same round (the chains are the
            // same computation): a variant must also match through `chain` dependent launches in every round.
            var chainSame = Array(repeating: MLXArray(true), count: candidates.count)
            do {
                try withError {
                    func build(_ i: Int) -> MLXArray? {
                        var previous: MLXArray? = nil
                        for _ in 0 ..< chain {
                            guard let out = link(candidates[i], previous) else { return nil }
                            previous = out
                        }
                        return previous
                    }
                    func bits(_ a: MLXArray) -> MLXArray { a.view(dtype: a.dtype == .float32 ? .uint32 : .uint16) }
                    func round(_ order: [Int]) -> [Double?] {
                        let chains = order.map { build($0) }
                        let times = chains.map { last -> Double? in
                            guard let last else { return nil }
                            let start = DispatchTime.now().uptimeNanoseconds
                            eval(last)
                            Stream.gpu.synchronize()
                            return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(chain)
                        }
                        if let at = order.firstIndex(of: 0), let reference = chains[at] {
                            for (i, last) in zip(order, chains) where i != 0 {
                                chainSame[i] = chainSame[i] .&& (last.map { (bits($0) .== bits(reference)).all() }
                                    ?? MLXArray(false))
                            }
                        }
                        return times
                    }
                    var wall = 0.0
                    for _ in 0 ..< 2 {
                        let start = DispatchTime.now().uptimeNanoseconds
                        _ = round(Array(candidates.indices))
                        wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    }
                    reps = NemotronTuneAdoption.cycles(target: 120, wall: wall, cap: 31, seed: true)
                    for rep in 0 ..< reps {
                        let order = NemotronTuneAdoption.order(candidates.count, cycle: rep)
                        for (i, ms) in zip(order, round(order)) {
                            if let ms { samples[i].append(ms) }
                        }
                        eval(chainSame)
                    }
                }
            } catch {
                dropped.append("timing error")
                alive = []
            }
            if !alive.isEmpty, samples.allSatisfy({ $0.count == reps }) {
                for (i, v) in candidates.enumerated() {
                    if let v, !chainSame[i].item(Bool.self) {
                        dropped.append("\(v.name) chain differs")
                        alive.removeAll { $0 == v }
                    }
                }
                verdict = NemotronTuneAdoption.decide(
                    samples, eligible: { candidates[$0].map(alive.contains) ?? false }, seed: true)
                medians = verdict?.medians ?? []
            } else {
                if !alive.isEmpty { dropped.append("timing incomplete") }
                alive = []
            }
        }
        var picked: Variant? = nil
        var why = "kept"
        if !medians.isEmpty, !alive.isEmpty {
            if let f = forced {
                if f >= 1, f <= variants.count, alive.contains(variants[f - 1]) {
                    picked = variants[f - 1]
                    why = "forced"
                } else {
                    why = f == 0 ? "forced V0" : "forced \(f) not qualified, kept"
                }
            } else if let best = verdict?.adopted {
                picked = candidates[best]
                why = "adopted"
            }
        }
        func ms(_ v: Variant?) -> String {
            guard let i = candidates.firstIndex(of: v), i < medians.count else { return "n/a" }
            return String(format: "%.4f", medians[i])
        }
        let times = alive.map { v in
            "\(v.name) \(ms(v)) \(candidates.firstIndex(of: v).flatMap { verdict?.won($0) } ?? "n/a")"
        }
        let rule = verdict?.rule({ candidates[$0]?.name ?? "V0" }) ?? "no candidate timed"
        return (picked, "\(why) \(picked?.name ?? "V0") \(ms(picked)) ms vs V0 \(ms(nil)) ms per launch (\(rule);"
            + " median of \(reps), \(chain) chained launches/sample, chain outputs bitwise; \(alive.count)/\(variants.count)"
            + " bitwise over \(cases) cases x \(passes)" + (times.isEmpty ? "" : "; [" + times.joined(separator: ", ") + "]")
            + (dropped.isEmpty ? "" : "; dropped: " + dropped.prefix(6).joined(separator: ", ")) + ")")
    }

    /// Load time, after the seed retiles; NAX hosts only (`NemotronHPrefillX5.runGather`).
    static func tune(_ target: Module) {
        stateChoice = nil
        scanChoice = nil
        freshChoice = nil
        let t0 = DispatchTime.now().uptimeNanoseconds
        func elapsed() -> Double { Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 }
        func report(_ message: String) {
            FileHandle.standardError.write(Data(
                "nemotron seed ssm tune: \(message) (\(String(format: "%.2f", elapsed())) s)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.runGather else { return report("off (setting)") }
        let layers = max(1, target.modules().filter { $0 is NemotronHMamba2Mixer }.count)
        let (h, g, dh, d) = (64, 8, 64, 128)
        let specials = MLXArray([Float(-0.0), 0, .infinity, -.infinity, .nan, Float(bitPattern: 0x0001_0000),
                                 Float(bitPattern: 0x8001_0000), 3e38])
        var parts: [String] = []

        // State product: `NemotronPrefillStateGemm.qualify`'s case kinds; the chain alternates the seed's two
        // window lengths, each link's decay row read from the previous product.
        if NemotronPrefillStateGemm.verified {
            typealias StateCase = (dtx: MLXArray, last: MLXArray, b: MLXArray, n: Int)
            var cases: [StateCase] = []
            do {
                cases = try withError { () -> [StateCase] in
                    var made: [StateCase] = []
                    // Window lengths of the seed (256, 232) and others around the 32- and 64-k staging steps;
                    // specials inside the window at chunk boundaries (dtx: times 0, 31, 32, 63, 64 of heads 0 and
                    // 63; decay rows: times 31, 32, 63, 64), overflowing products (x 1e37) and denormal ones.
                    for (seed, (n, l, start)) in [(256, 488, 0), (232, 488, 256), (256, 512, 256), (100, 356, 256),
                                                  (64, 64, 0), (33, 289, 256), (161, 417, 256), (96, 96, 0),
                                                  (31, 31, 0)].enumerated() {
                        let key = MLXRandom.split(key: MLXRandom.key(UInt64(0x57b7 + seed)), into: 3)
                        var dtxFull = MLXRandom.normal([1, l, h, dh], key: key[0]) * Float(1 + seed)
                        if seed == 1 || seed == 5 {
                            dtxFull = concatenated([specials, dtxFull.reshaped(-1)[8...]]).reshaped(1, l, h, dh)
                            var at: [Int] = []
                            for t in [0, 31, 32, 63, 64] where t < n {
                                let row: Int = (start + t) * h
                                at.append(row * dh + 3)
                                at.append((row + h - 1) * dh + 40)
                            }
                            dtxFull = plant(dtxFull, specials, at: at)
                        }
                        if seed == 6 { dtxFull = dtxFull * Float(1e37) }
                        if seed == 8 { dtxFull = dtxFull * Float(1e-39) }
                        var last = MLX.exp(-MLX.abs(MLXRandom.normal([1, h, 1, n], key: key[1])))
                        if seed == 2 || seed == 6 {
                            last = concatenated([MLXArray([Float(0), .infinity, .nan, 1e-30]),
                                                 last.reshaped(-1)[4...]]).reshaped(1, h, 1, n)
                            var at: [Int] = []
                            for t in [31, 63] where t + 8 <= n { at.append(5 * n + t) }
                            last = plant(last, specials, at: at)
                        }
                        let flat = MLXRandom.normal([l * g * d], key: key[2]) * Float(2 + seed)
                        let b = nemotronPrefillBF32Window(flat, start: start, end: start + n, groups: g, state: d)
                        let dtx = dtxFull[0..., start ..< (start + n), 0..., 0...]
                        eval(dtx, last, b)
                        made.append((dtx, last, b, n))
                    }
                    return made
                }
            } catch {
                cases = []
            }
            if cases.count == 9 {
                func launch(_ v: Variant?, _ c: StateCase, _ last: MLXArray) -> MLXArray? {
                    v.map { launchState($0, c.dtx, last, c.b, n: c.n, h: h, g: g, dh: dh, d: d) }
                        ?? NemotronPrefillStateGemm.launch(dtx: c.dtx, last: last, bWindow: c.b, n: c.n, h: h, g: g,
                                                           dh: dh, d: d)
                }
                var step = 0
                let (pick, message) = choose(
                    stateVariants, cases: cases.count, chain: 2 * layers, elapsed: elapsed,
                    run: { v, i in launch(v, cases[i], cases[i].last).map { [$0] } },
                    link: { v, previous in
                        if previous == nil { step = 0 }
                        let c = cases[step % 2]
                        step += 1
                        return launch(v, c, previous.map { $0.reshaped(-1)[0 ..< (h * c.n)].reshaped(1, h, 1, c.n) }
                            ?? c.last)
                    })
                stateChoice = pick
                parts.append("state " + message)
            } else {
                parts.append("state off (inputs)")
            }
        } else {
            parts.append("state off (not verified)")
        }

        // Scan decay GEMM: `NemotronPrefillScanDecayGemm.qualify`'s case kinds; the chain runs the seed's second
        // window length, each link's decay input read from the previous decay row.
        if NemotronPrefillScanDecayGemm.verified, elapsed() < budgetSeconds {
            typealias ScanCase = (dtA: MLXArray, cb: MLXArray, dtx: MLXArray, n: Int)
            var cases: [ScanCase] = []
            do {
                cases = try withError { () -> [ScanCase] in
                    var made: [ScanCase] = []
                    // As the state cases: the seed's window lengths and others around the 32- and 64-k staging
                    // steps, specials at chunk boundaries (dtx times 31, 32, 63, 64; CB rows and columns 31, 32, 63,
                    // 64), overflowing products and a decay that underflows to zero.
                    for (seed, (n, l, start)) in [(256, 488, 0), (232, 488, 256), (256, 512, 256), (100, 356, 256),
                                                  (17, 17, 0), (33, 289, 256), (161, 417, 256), (96, 96, 0),
                                                  (65, 65, 0)].enumerated() {
                        let key = MLXRandom.split(key: MLXRandom.key(UInt64(0x5cb9 + seed)), into: 4)
                        var dtAFull = -MLX.abs(MLXRandom.normal([1, l, h], key: key[0])) * Float(0.05 * Double(1 + seed))
                        if seed == 8 { dtAFull = dtAFull * Float(4000) }
                        var dtxFull = MLXRandom.normal([1, l, h, dh], key: key[1]) * Float(1 + seed)
                        if seed == 1 || seed == 5 {
                            dtxFull = concatenated([dtxFull.reshaped(-1)[..<(start * h * dh + 3 * dh)], specials,
                                                    dtxFull.reshaped(-1)[(start * h * dh + 3 * dh + 8)...]])
                                .reshaped(1, l, h, dh)
                            var at: [Int] = []
                            for t in [31, 32, 63, 64] where t < n {
                                let row: Int = (start + t) * h + 7
                                at.append(row * dh + 20)
                            }
                            dtxFull = plant(dtxFull, specials, at: at)
                        }
                        if seed == 6 { dtxFull = dtxFull * Float(1e37) }
                        var cb = MLXRandom.normal([1, g, n, n], key: key[2]) * Float(2 + seed)
                        if seed == 2 || seed == 6 {
                            cb = concatenated([specials, cb.reshaped(-1)[8...]]).reshaped(1, g, n, n)
                            var at: [Int] = []
                            for (r, c) in [(31, 28), (32, 28), (63, 60), (64, 60), (100, 28)] where r < n && c + 8 <= n {
                                at.append(r * n + c)
                            }
                            cb = plant(cb, specials, at: at)
                        }
                        cb = cb.asType(.bfloat16)
                        let dtA = dtAFull[0..., start ..< (start + n), 0...]
                        let dtx = dtxFull[0..., start ..< (start + n), 0..., 0...]
                        eval(dtA, cb, dtx)
                        made.append((dtA, cb, dtx, n))
                    }
                    return made
                }
            } catch {
                cases = []
            }
            if cases.count == 9 {
                func launch(_ v: Variant?, _ c: ScanCase, _ dtA: MLXArray) -> (y: MLXArray, last: MLXArray)? {
                    v.map { launchScan($0, dtA, c.cb, c.dtx, n: c.n, b: 1, h: h, g: g, dh: dh) }
                        ?? NemotronPrefillScanDecayGemm.launch(dtA: dtA, CB: c.cb, dtx: c.dtx, n: c.n, b: 1, h: h,
                                                               g: g, dh: dh)
                }
                let c1 = cases[1]
                let (pick, message) = choose(
                    scanVariants, cases: cases.count, chain: layers, elapsed: elapsed,
                    run: { v, i in launch(v, cases[i], cases[i].dtA).map { [$0.y, $0.last] } },
                    link: { v, previous in
                        launch(v, c1, previous.map { $0.reshaped(1, c1.n, h) } ?? c1.dtA)?.last
                    })
                scanChoice = pick
                parts.append("scan " + message)
                // The fresh window takes the scan's pick once its own form matches on
                // `qualifyFresh`'s case kinds (three passes).
                if let pick, NemotronPrefillScanDecayGemm.freshVerified {
                    var same = false
                    do {
                        same = try withError { () -> Bool in
                            var ok = MLXArray(true)
                            for (seed, (n, l)) in [(256, 488), (256, 300), (100, 100), (17, 17), (33, 33),
                                                   (161, 300), (96, 96)].enumerated() {
                                let key = MLXRandom.split(key: MLXRandom.key(UInt64(0xf5f0 + seed)), into: 6)
                                let dtAFull = -MLX.abs(MLXRandom.normal([1, l, h], key: key[0]))
                                    * Float(0.05 * Double(1 + seed))
                                var dtxFull = MLXRandom.normal([1, l, h, dh], key: key[1]) * Float(1 + seed)
                                if seed == 1 {
                                    dtxFull = concatenated([specials, dtxFull.reshaped(-1)[8...]]).reshaped(1, l, h, dh)
                                }
                                if seed == 4 || seed == 5 {
                                    var at: [Int] = []
                                    for t in [31, 32, 63, 64] where t < n {
                                        let row: Int = t * h + 9
                                        at.append(row * dh + 30)
                                    }
                                    dtxFull = plant(dtxFull, specials, at: at)
                                }
                                let cb = (MLXRandom.normal([1, g, n, n], key: key[2]) * Float(2 + seed))
                                    .asType(.bfloat16)
                                let ecFull = MLX.exp(-MLX.abs(MLXRandom.normal([1, l, h], key: key[3])))
                                let wide = MLXRandom.normal([1, l, h * dh + 64], key: key[4]).asType(.bfloat16)
                                let x = wide[0..., 0..., 0 ..< (h * dh)].reshaped(1, l, h, dh)
                                let D = MLXRandom.normal([h], key: key[5]).asType(.bfloat16)
                                let dtA = dtAFull[0..., 0 ..< n, 0...]
                                let dtx = dtxFull[0..., 0 ..< n, 0..., 0...]
                                let ec = ecFull[0..., 0 ..< n, 0...]
                                let lt = n == l ? l : n
                                guard let a = NemotronPrefillScanDecayGemm.launchFresh(
                                    dtA: dtA, CB: cb, dtx: dtx, n: n, h: h, g: g, dh: dh, ec: ec, x: x, D: D, lt: lt)
                                else { return false }
                                for _ in 0 ..< passes {
                                    guard let b = launchFresh(pick, dtA, cb, dtx, n: n, h: h, g: g, dh: dh, ec: ec,
                                                              x: x, D: D, lt: lt) else { return false }
                                    ok = ok .&& (a.o[0..., 0 ..< n, 0..., 0...].view(dtype: .uint16)
                                        .== b.o[0..., 0 ..< n, 0..., 0...].view(dtype: .uint16)).all()
                                        .&& (a.last.view(dtype: .uint32) .== b.last.view(dtype: .uint32)).all()
                                    eval(ok)
                                }
                            }
                            return ok.item(Bool.self)
                        }
                    } catch {
                        same = false
                    }
                    freshChoice = same ? pick : nil
                    parts.append("fresh " + (same ? "on \(pick.name) (7 cases x \(passes) bitwise)" : "kept (\(pick.name) differs)"))
                }
            } else {
                parts.append("scan off (inputs)")
            }
        } else {
            parts.append(NemotronPrefillScanDecayGemm.verified ? "scan skipped (load budget)" : "scan off (not verified)")
        }
        report(parts.joined(separator: "; "))
    }
}
