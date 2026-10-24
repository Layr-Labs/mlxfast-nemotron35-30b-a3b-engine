import Foundation
import MLX
import MLXFast

// P0-42. Input-independent, separately allocated affine4/gs64 cache.
// Original parameters are read-only. Codes and BF16 scale/bias bit patterns
// are preserved exactly; inverse() is the full round-trip verification path.
public final class NemotronZSkipKMajorKernel: NemotronZSkipKernel {
    public let version = "p042-kmajor-v2s-weighted-same-tg-v1"
    public init() {}
    public func derivedByteCount(_ stock: NemotronZSkipStock) -> Int {
        stock.experts * stock.n * (stock.k / 2 + stock.k / 64 * 4)
    }
    public func supports(site: NemotronZSkipSite, rows: Int, stock: NemotronZSkipStock) -> Bool {
        (site == .routedWeighted || site == .routedLeader) && (1...4).contains(rows)
            && stock.k == 1856 && stock.n == 2688 && stock.experts == 128
            && stock.weight.dtype == .uint32 && stock.scales.dtype == .bfloat16
            && stock.biases.dtype == .bfloat16
    }
    private static let repackCodes = MLXFast.metalKernel(name: "p042_kmajor_codes_v2",
        inputNames: ["w"], outputNames: ["p"], source: """
        uint i = thread_position_in_grid.x;
        if (i >= E*N*K/2) return;
        uint c = i % 32, k = (i / 32) % K;
        uint tile = (i / (32*K)) % (N/64), e = i / (32*K*(N/64));
        uint n = tile*64 + c;
        uint a = w[(e*N+n)*(K/8)+k/8];
        uint b = w[(e*N+n+32)*(K/8)+k/8];
        p[i] = uchar(((a >> (4*(k%8))) & 15) | (((b >> (4*(k%8))) & 15) << 4));
        """)
    private static let repackSB = MLXFast.metalKernel(name: "p042_kmajor_sb_v2",
        inputNames: ["s", "b"], outputNames: ["ps", "pb"], source: """
        uint i = thread_position_in_grid.x;
        if (i >= E*N*(K/64)) return;
        uint c = i % 64, g = (i / 64) % (K/64);
        uint tile = (i / K) % (N/64), e = i / (K*(N/64));
        uint old = (e*N+tile*64+c)*(K/64)+g;
        ps[i] = s[old]; pb[i] = b[old];
        """)
    public func build(_ stock: NemotronZSkipStock) -> NemotronZSkipDerived {
        let t: [(String, any KernelTemplateArg)] = [("E",stock.experts),("N",stock.n),("K",stock.k)]
        let w = Self.repackCodes([stock.weight], template: t,
            grid: (stock.experts*stock.n*stock.k/2,1,1), threadGroup: (256,1,1),
            outputShapes: [[stock.experts,stock.n/64,stock.k,32]], outputDTypes: [.uint8])[0]
        let sb = Self.repackSB([stock.scales.view(dtype: .uint16),stock.biases.view(dtype: .uint16)], template: t,
            grid: (stock.experts*stock.n*(stock.k/64),1,1), threadGroup: (256,1,1),
            outputShapes: [[stock.experts,stock.n/64,stock.k/64,64], [stock.experts,stock.n/64,stock.k/64,64]],
            outputDTypes: [.uint16,.uint16])
        return NemotronZSkipDerived(arrays: [w,sb[0].view(dtype: .bfloat16),sb[1].view(dtype: .bfloat16)],
            layout: "p042-tile64-kmajor-half-paired-v2")
    }
    private static let inverseCodes = MLXFast.metalKernel(name: "p042_kmajor_inverse_codes_v2",
        inputNames: ["p"], outputNames: ["w"], source: """
        uint i = thread_position_in_grid.x;
        if (i >= E*N*(K/8)) return;
        uint kw = i % (K/8), n = (i / (K/8)) % N, e = i / ((K/8)*N);
        uint value = 0;
        for (uint j=0; j<8; j++) {
            uint pos = ((e*(N/64)+n/64)*K+kw*8+j)*32+n%32;
            value |= ((uint(p[pos]) >> ((n%64)/32*4)) & 15) << (4*j);
        }
        w[i] = value;
        """)
    private static let inverseSB = MLXFast.metalKernel(name: "p042_kmajor_inverse_sb_v2",
        inputNames: ["ps","pb"], outputNames: ["s","b"], source: """
        uint i = thread_position_in_grid.x;
        if (i >= E*N*(K/64)) return;
        uint g=i%(K/64), n=(i/(K/64))%N, e=i/((K/64)*N);
        uint pos = ((e*(N/64)+n/64)*(K/64)+g)*64+n%64;
        s[i] = ps[pos]; b[i] = pb[pos];
        """)
    public func inverse(_ d: NemotronZSkipDerived, stock: NemotronZSkipStock) -> [MLXArray] {
        let t: [(String, any KernelTemplateArg)] = [("E",stock.experts),("N",stock.n),("K",stock.k)]
        let w = Self.inverseCodes([d.arrays[0]], template: t,
            grid: (stock.weight.size,1,1), threadGroup: (256,1,1),
            outputShapes: [stock.weight.shape], outputDTypes: [.uint32])[0]
        let sb = Self.inverseSB([d.arrays[1].view(dtype: .uint16),d.arrays[2].view(dtype: .uint16)], template: t,
            grid: (stock.scales.size,1,1), threadGroup: (256,1,1),
            outputShapes: [stock.scales.shape,stock.biases.shape], outputDTypes: [.uint16,.uint16])
        return [w,sb[0].view(dtype: .bfloat16),sb[1].view(dtype: .bfloat16)]
    }
    // Every virtual stock lane retains the same nml_load_vector operation order.
    private static let prepare = MLXFast.metalKernel(name: "p042_kmajor_prepare_v2",
        inputNames: ["x"], outputNames: ["px","mask"], source: """
        uint lane=thread_index_in_simdgroup, slot=threadgroup_position_in_grid.y;
        constexpr int STEPS=(K+255)/256;
        for (int step=0; step<STEPS; step++) {
            int k=step*256+int(lane)*8;
            ulong off=((ulong(slot)*STEPS+step)*9)*32+lane;
            float sum=0; uint bits=0;
            for (int i=0; i<8; i+=4) {
                float a=k+i<K?float(x[slot*K+k+i]):0.0f;
                float b=k+i+1<K?float(x[slot*K+k+i+1]):0.0f;
                float c=k+i+2<K?float(x[slot*K+k+i+2]):0.0f;
                float d=k+i+3<K?float(x[slot*K+k+i+3]):0.0f;
                sum += a+b+c+d;
                px[off+i*32]=a; px[off+(i+1)*32]=b/16.0f;
                px[off+(i+2)*32]=c/256.0f; px[off+(i+3)*32]=d/4096.0f;
                bits |= (a!=0?1u:0u)<<i; bits |= (b!=0?1u:0u)<<(i+1);
                bits |= (c!=0?1u:0u)<<(i+2); bits |= (d!=0?1u:0u)<<(i+3);
            }
            px[off+8*32]=sum;
            mask[(slot*STEPS+step)*32+lane]=bits;
        }
        """)
    private static let downKernel = MLXFast.metalKernel(name: "p042_kmajor_down_v2s",
        inputNames: ["w","sc","bs","px","mask","ids"], outputNames: ["y"], source: """
        uint lid=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
        uint tile=threadgroup_position_in_grid.x, slot=threadgroup_position_in_grid.z;
        uint expert=ids[slot];
        constexpr int STEPS=(K+255)/256;
        thread float acc0[4]={0},acc1[4]={0};
        threadgroup float partial[32*65];
        for (int step=0; step<STEPS; step++) {
            #pragma unroll
            for (int v=0; v<4; v++) {
                int vl=sg+v*8, k=step*256+vl*8;
                // Stock has no tail operation for an invalid virtual k8 lane.
                if (k>=K) continue;
                uint m=mask[(slot*STEPS+step)*32+vl];
                ulong si=((ulong(expert)*(N/64)+tile)*(K/64)+k/64)*64+lid;
                float s0=sc[si],b0=bs[si],s1=sc[si+32],b1=bs[si+32];
                if (m==0) {
                    // Affine scales may be signed. Preserve the stock zero
                    // contribution and accumulator step; skip only code loads.
                    acc0[v] += s0*0.0f + 0.0f*b0;
                    acc1[v] += s1*0.0f + 0.0f*b1;
                    continue;
                }
                ulong off=((ulong(slot)*STEPS+step)*9)*32+vl;
                thread float xt[8];
                #pragma unroll
                for (int j=0; j<8; j++) xt[j]=px[off+j*32];
                uint lo[2]={0},hi[2]={0};
                #pragma unroll
                for (int j=0; j<8; j++) {
                    if ((m & (1u<<j)) != 0) {
                        ulong p=((ulong(expert)*(N/64)+tile)*K+k+j)*32+lid;
                        uint q=w[p];
                        lo[j/4] |= (q&15) << (4*(j%4));
                        hi[j/4] |= (q>>4) << (4*(j%4));
                    }
                }
                float dot0=0,dot1=0;
                #pragma unroll
                for (int j=0; j<2; j++) {
                    dot0 += (xt[4*j]*(lo[j]&0x000f) + xt[4*j+1]*(lo[j]&0x00f0)
                        + xt[4*j+2]*(lo[j]&0x0f00) + xt[4*j+3]*(lo[j]&0xf000));
                    dot1 += (xt[4*j]*(hi[j]&0x000f) + xt[4*j+1]*(hi[j]&0x00f0)
                        + xt[4*j+2]*(hi[j]&0x0f00) + xt[4*j+3]*(hi[j]&0xf000));
                }
                float sum=px[off+8*32];
                acc0[v] += s0*dot0 + sum*b0;
                acc1[v] += s1*dot1 + sum*b1;
            }
        }
        #pragma unroll
        for (int v=0; v<4; v++) {
            partial[(sg+v*8)*65+lid]=acc0[v];
            partial[(sg+v*8)*65+lid+32]=acc1[v];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int r=0; r<8; r++) {
            int col=sg*8+r;
            float value=partial[lid*65+col];
            value=simd_sum(value);
            if (lid==0) y[slot*N+tile*64+col]=static_cast<T>(value);
        }
        """)
    // One threadgroup owns all six experts for one row/tile. The low/high
    // halves reuse a padded virtual-stock-lane transpose so shared memory stays
    // below 32 KiB. BF16 raw expert words live only until the weighted epilogue.
    private static let fusedWeightedEnabled =
        ProcessInfo.processInfo.environment["NEMOTRON_ZSKIP_FUSED_WEIGHTED"] != "0"
    private static let fusedWeighted = MLXFast.metalKernel(name: "p042_kmajor_weighted_same_tg_v1",
        inputNames: ["w","sc","bs","px","mask","ids","scores"], outputNames: ["y"], source: """
        uint lid=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
        uint tile=threadgroup_position_in_grid.x, row=threadgroup_position_in_grid.z;
        uint expertSlot=sg/2, localSG=sg%2;
        uint slot=row*6+expertSlot, expert=ids[slot];
        constexpr int STEPS=(K+255)/256;
        thread float acc0[16]={0},acc1[16]={0};
        threadgroup float partial[6*32*33];
        threadgroup T raw[6*64];
        for (int step=0; step<STEPS; step++) {
            #pragma unroll
            for (int v=0; v<16; v++) {
                int vl=localSG+v*2, k=step*256+vl*8;
                // Stock has no tail operation for an invalid virtual k8 lane.
                if (k>=K) continue;
                uint m=mask[(slot*STEPS+step)*32+vl];
                ulong si=((ulong(expert)*(N/64)+tile)*(K/64)+k/64)*64+lid;
                float s0=sc[si],b0=bs[si],s1=sc[si+32],b1=bs[si+32];
                if (m==0) {
                    // Affine scales may be signed. Preserve the stock zero
                    // contribution and accumulator step; skip only code loads.
                    acc0[v] += s0*0.0f + 0.0f*b0;
                    acc1[v] += s1*0.0f + 0.0f*b1;
                    continue;
                }
                ulong off=((ulong(slot)*STEPS+step)*9)*32+vl;
                thread float xt[8];
                #pragma unroll
                for (int j=0; j<8; j++) xt[j]=px[off+j*32];
                uint lo[2]={0},hi[2]={0};
                #pragma unroll
                for (int j=0; j<8; j++) {
                    if ((m & (1u<<j)) != 0) {
                        ulong p=((ulong(expert)*(N/64)+tile)*K+k+j)*32+lid;
                        uint q=w[p];
                        lo[j/4] |= (q&15) << (4*(j%4));
                        hi[j/4] |= (q>>4) << (4*(j%4));
                    }
                }
                float dot0=0,dot1=0;
                #pragma unroll
                for (int j=0; j<2; j++) {
                    dot0 += (xt[4*j]*(lo[j]&0x000f) + xt[4*j+1]*(lo[j]&0x00f0)
                        + xt[4*j+2]*(lo[j]&0x0f00) + xt[4*j+3]*(lo[j]&0xf000));
                    dot1 += (xt[4*j]*(hi[j]&0x000f) + xt[4*j+1]*(hi[j]&0x00f0)
                        + xt[4*j+2]*(hi[j]&0x0f00) + xt[4*j+3]*(hi[j]&0xf000));
                }
                float sum=px[off+8*32];
                acc0[v] += s0*dot0 + sum*b0;
                acc1[v] += s1*dot1 + sum*b1;
            }
        }
        #pragma unroll
        for (int v=0; v<16; v++) {
            partial[(expertSlot*32+localSG+v*2)*33+lid]=acc0[v];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int r=0; r<16; r++) {
            int col=int(localSG)*16+r;
            float value=partial[(expertSlot*32+lid)*33+col];
            value=simd_sum(value);
            if (lid==0) raw[expertSlot*64+col]=static_cast<T>(value);
        }
        // Every reader finishes before the high half reuses partial.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int v=0; v<16; v++) {
            partial[(expertSlot*32+localSG+v*2)*33+lid]=acc1[v];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int r=0; r<16; r++) {
            int col=int(localSG)*16+r;
            float value=partial[(expertSlot*32+lid)*33+col];
            value=simd_sum(value);
            if (lid==0) raw[expertSlot*64+col+32]=static_cast<T>(value);
        }
        // All six experts have narrowed to the original BF16 boundary.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg==0) {
            #pragma unroll
            for (int columnHalf=0; columnHalf<2; columnHalf++) {
                int col=int(lid)+32*columnHalf;
                float total=float(raw[col])*scores[row*6]+0.0f;
                for (int j=1; j<6; j++) {
                    float product=float(raw[j*64+col])*scores[row*6+j]+0.0f;
                    total=product+total;
                }
                y[row*N+tile*64+col]=static_cast<T>(total);
            }
        }
        """)
    private static let reduce = MLXFast.metalKernel(name: "p042_kmajor_weight_reduce_v2",
        inputNames: ["raw","scores"], outputNames: ["y"], source: """
        uint n=thread_position_in_grid.x, row=thread_position_in_grid.y;
        if (n>=N || row>=R) return;
        float total=float(raw[(row*TOPK)*N+n])*scores[row*TOPK]+0.0f;
        for (int j=1; j<TOPK; j++) {
            float p=float(raw[(row*TOPK+j)*N+n])*scores[row*TOPK+j]+0.0f;
            total=p+total;
        }
        y[row*N+n]=static_cast<T>(total);
        """)
    public func down(_ d: NemotronZSkipDerived, stock: NemotronZSkipStock,
                     x: MLXArray, indices: MLXArray?, scores: MLXArray?,
                     site: NemotronZSkipSite, rows: Int, topK: Int,
                     original: () -> MLXArray) -> MLXArray {
        guard supports(site: site, rows: rows, stock: stock), let indices,
              d.layout == "p042-tile64-kmajor-half-paired-v2" else { return original() }
        let slots=rows*topK, steps=(stock.k+255)/256
        let p=Self.prepare([x], template: [("K",stock.k)], grid: (32,slots,1), threadGroup: (32,1,1),
            outputShapes: [[slots,steps,9,32],[slots,steps,32]], outputDTypes: [.float32,.uint32])
        if Self.fusedWeightedEnabled, site == .routedWeighted, rows == 3, topK == 6, let scores {
            return Self.fusedWeighted(
                [d.arrays[0],d.arrays[1],d.arrays[2],p[0],p[1],indices,scores],
                template: [("T",x.dtype),("K",stock.k),("N",stock.n)],
                grid: (32*(stock.n/64),12,rows), threadGroup: (32,12,1),
                outputShapes: [[1,rows,stock.n]], outputDTypes: [x.dtype])[0]
        }
        let raw=Self.downKernel([d.arrays[0],d.arrays[1],d.arrays[2],p[0],p[1],indices],
            template: [("T",x.dtype),("K",stock.k),("N",stock.n)],
            grid: (32*(stock.n/64),8,slots), threadGroup: (32,8,1),
            outputShapes: [[slots,stock.n]], outputDTypes: [x.dtype])[0]
        if site == .routedWeighted, let scores {
            return Self.reduce([raw,scores], template: [("T",x.dtype),("N",stock.n),("R",rows),("TOPK",topK)],
                grid: (stock.n,rows,1), threadGroup: (256,1,1),
                outputShapes: [[1,rows,stock.n]], outputDTypes: [x.dtype])[0]
        }
        return raw
    }
}
