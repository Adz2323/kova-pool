// kangaroo.cu: Pollard kangaroo (SOTA-style, with symmetry) for secp256k1 on one NVIDIA GPU. Prototype.
//
//   Finds k in [start, start + 2^range) with k*G == Q for a KNOWN public key Q.
//
//   Method (the design of RetiredCoder's RCKangaroo, written fresh here on CUDACyclone's field maths):
//     * Every kangaroo is an affine point with a signed 192-bit distance. Tame: T = d*G. Wild: W = d*G - P' where
//       P' = Q - start*G + x32*G (x32 = 2^(range-5), shifts the key away from the range edges).
//     * A jump is chosen by x (index = x[0] & 511) and its DIRECTION by the parity of y: y even -> +J, y odd -> -J.
//       So a point and its negation walk mirror paths with identical x: the walk is a walk on {P, -P} classes and the
//       expected work drops to about K = 1.15 * 2^(range/2) (classic: 2.1). Distinguished points (DP, top bits of x
//       zero) are sent to the host; a tame/wild or wild/wild collision gives the key.
//     * Loops: the symmetric walk can cycle. 2-cycles (P -> P+J -> P, one per ~1024 jumps) are detected right away and
//       escaped with a jump from a second table of huge jumps. Longer (even) cycles are detected from the distance
//       history (a ring of the last MD_LEN distances per kangaroo) and escaped with a jump from a third table.
//     * GPU: 24 kangaroos per thread, one Montgomery batch inversion per step; jump table 1 in shared memory; the hot
//       kernel (k_jump) writes a 2-byte record per step per kangaroo (jump index, direction, table, DP flag) and a
//       light second kernel (k_dist) replays them to update distances and detect loops; k_escape does the table-3 jumps.
//
//   --cpu runs the identical algorithm on the host (HostOps) for small ranges: proves the logic without a GPU.
//
// Build: make  (see Makefile; field maths from ../CUDAMath.h with EOMUL=1, as bsgs)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cuda.h>          // driver API: --cubin loads a hand-assembled jump kernel (SASS work)
#include "CUDAMath.h"
#include "modinv30.h"      // safegcd inversion (libsecp256k1 port): replaces CUDAMath's _ModInv in the walk
#include "fred32.h"        // field mul/sqr with a 32-bit carry-chained reduction (137/129 SASS vs EOMUL's 165/157)

typedef unsigned __int128 u128;
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;

#define JMP_CNT   512            // jumps per table
#define JMP_MASK  (JMP_CNT - 1)
// kangaroos per thread (points per batch inversion): --groups 8, 16 or 24
#define STEPS_DEF 1000           // jumps per kangaroo per launch (default; --steps)
#define MD_LEN    10             // distance history per kangaroo (loop detection: cycles up to MD_LEN long)
#define DP_SLOTS  16             // DP x-slots per kangaroo per launch
#define REC_INV   0x0200         // record flags: direction -J
#define REC_JMP2  0x0400         // jump from table 2 (2-cycle escape)
#define REC_DP    0x0800         // the point after this jump is a DP
#define MAX_DP_OUT (1u << 20)

// ============================================================================ portable 256-bit helpers (as bsgs.cu)
__host__ __device__ __forceinline__ void cp4(u64* d, const u64* s) { d[0]=s[0]; d[1]=s[1]; d[2]=s[2]; d[3]=s[3]; }
__host__ __device__ __forceinline__ bool iszero4(const u64* a) { return (a[0]|a[1]|a[2]|a[3]) == 0; }
__host__ __device__ __forceinline__ bool eq4(const u64* a, const u64* b) { return a[0]==b[0]&&a[1]==b[1]&&a[2]==b[2]&&a[3]==b[3]; }
#define P0 0xFFFFFFFEFFFFFC2FULL
#define PM 0xFFFFFFFFFFFFFFFFULL
__host__ __device__ __forceinline__ bool geq_p(const u64* a) { if (a[3] != PM || a[2] != PM || a[1] != PM) return false; return a[0] >= P0; }
__host__ __device__ __forceinline__ void p_add(u64* r, const u64* a, const u64* b) {
    u128 c = 0; u64 t[4];
    for (int i = 0; i < 4; ++i) { c += (u128)a[i] + b[i]; t[i] = (u64)c; c >>= 64; }
    if (c || geq_p(t)) { u128 d = (u128)t[0] + 0x1000003D1ULL; t[0] = (u64)d; d >>= 64; for (int i = 1; i < 4; ++i) { d += t[i]; t[i] = (u64)d; d >>= 64; } }
    cp4(r, t);
}
__host__ __device__ __forceinline__ void p_sub(u64* r, const u64* a, const u64* b) {
    u64 t[4]; u64 br = 0;
    for (int i = 0; i < 4; ++i) { u128 d = (u128)a[i] - b[i] - br; t[i] = (u64)d; br = (u64)(d >> 64) ? 1 : 0; }
    if (br) { const u64 p[4] = {P0, PM, PM, PM}; u128 c = 0; for (int i = 0; i < 4; ++i) { c += (u128)t[i] + p[i]; t[i] = (u64)c; c >>= 64; } }
    cp4(r, t);
}
__host__ __device__ __forceinline__ void p_neg(u64* r, const u64* a) { const u64 z[4] = {0,0,0,0}; p_sub(r, z, a); }

// ============================================================================ field ops: host / device
struct HostOps {
    static void mul(u64* r, const u64* a, const u64* b) {
        u64 t[8] = {0};
        for (int i = 0; i < 4; ++i) { u128 c = 0; for (int j = 0; j < 4; ++j) { c += (u128)a[i] * b[j] + t[i + j]; t[i + j] = (u64)c; c >>= 64; } t[i + 4] = (u64)c; }
        const u64 C = 0x1000003D1ULL; u64 s[4]; u128 c = 0;
        for (int k = 0; k < 4; ++k) { c += (u128)t[4 + k] * C + t[k]; s[k] = (u64)c; c >>= 64; }
        u128 d = (u128)(u64)c * C + s[0]; s[0] = (u64)d; d >>= 64;
        for (int k = 1; k < 4; ++k) { d += s[k]; s[k] = (u64)d; d >>= 64; }
        if (d) { u128 e = (u128)s[0] + C; s[0] = (u64)e; e >>= 64; for (int k = 1; k < 4; ++k) { e += s[k]; s[k] = (u64)e; e >>= 64; } }
        if (geq_p(s)) { u128 e = (u128)s[0] + C; s[0] = (u64)e; e >>= 64; for (int k = 1; k < 4; ++k) { e += s[k]; s[k] = (u64)e; e >>= 64; } }
        cp4(r, s);
    }
    static void sqr(u64* r, const u64* a) { mul(r, a, a); }
    static void sub(u64* r, const u64* a, const u64* b) { p_sub(r, a, b); }
    static void neg(u64* r) { u64 t[4]; p_neg(t, r); cp4(r, t); }
    static void inv(u64* r, const u64* a) { mi_inv(r, a); }
    static void inv_fermat(u64* r, const u64* a) {
        u64 e[4] = {P0 - 2, PM, PM, PM}, res[4] = {1,0,0,0}, base[4]; cp4(base, a);
        for (int i = 0; i < 256; ++i) { if ((e[i >> 6] >> (i & 63)) & 1) mul(res, res, base); sqr(base, base); }
        cp4(r, res);
    }
};
struct DevOps {     // field ops on the GPU: fred32 mul/sqr (EOMUL products + 32-bit reduction), safegcd inversion
    static __device__ __forceinline__ void mul(u64* r, const u64* a, const u64* b) { fr_mul(r, a, b); }
    static __device__ __forceinline__ void sqr(u64* r, const u64* a) { fr_sqr(r, a); }
    static __device__ __forceinline__ void sub(u64* r, const u64* a, const u64* b) { ModSub256(r, (u64*)a, (u64*)b); }
    static __device__ __forceinline__ void neg(u64* r) { ModNeg256(r); }
    static __device__ __forceinline__ void inv(u64* r, const u64* a) { mi_inv(r, a); }
};
struct DevOpsV {    // CUDAMath's EOMUL mul/sqr and _ModInv (VanitySearch DivStep62): the v1 field ops, for comparison (--inv 0)
    static __device__ __forceinline__ void mul(u64* r, const u64* a, const u64* b) { _ModMult(r, (u64*)a, (u64*)b); }
    static __device__ __forceinline__ void sqr(u64* r, const u64* a) { _ModSqr(r, a); }
    static __device__ __forceinline__ void sub(u64* r, const u64* a, const u64* b) { ModSub256(r, (u64*)a, (u64*)b); }
    static __device__ __forceinline__ void neg(u64* r) { ModNeg256(r); }
    static __device__ __forceinline__ void inv(u64* r, const u64* a) { u64 t[5] = {a[0], a[1], a[2], a[3], 0}; _ModInv(t); cp4(r, t); }
};

struct DevOpsF {   // as DevOps, but the inversion is a^(p-2) by square-and-multiply: no data-dependent loop, cannot hang
    static __device__ __forceinline__ void mul(u64* r, const u64* a, const u64* b) { _ModMult(r, (u64*)a, (u64*)b); }
    static __device__ __forceinline__ void sqr(u64* r, const u64* a) { _ModSqr(r, a); }
    static __device__ __forceinline__ void sub(u64* r, const u64* a, const u64* b) { ModSub256(r, (u64*)a, (u64*)b); }
    static __device__ __forceinline__ void neg(u64* r) { ModNeg256(r); }
    static __device__ void inv(u64* r, const u64* a) {
        const u64 e[4] = {P0 - 2, PM, PM, PM}; u64 res[4] = {1, 0, 0, 0}, base[4]; cp4(base, a);
        for (int i = 0; i < 256; ++i) { if ((e[i >> 6] >> (i & 63)) & 1) mul(res, res, base); sqr(base, base); }
        cp4(r, res);
    }
};

// ============================================================================ EC helpers
struct Pt { u64 x[4], y[4]; };
template <class F> __host__ __device__ void jac_madd(u64* X, u64* Y, u64* Z, const u64* x2, const u64* y2) {
    u64 Z1Z1[4], U2[4], S2[4], H[4], HH[4], I[4], J[4], r[4], V[4], t[4];
    F::sqr(Z1Z1, Z); F::mul(U2, x2, Z1Z1); F::mul(S2, y2, Z); F::mul(S2, S2, Z1Z1);
    F::sub(H, U2, X); F::sqr(HH, H); p_add(I, HH, HH); p_add(I, I, I); F::mul(J, H, I);
    F::sub(r, S2, Y); p_add(r, r, r); F::mul(V, X, I);
    F::sqr(t, r); F::sub(t, t, J); F::sub(t, t, V); F::sub(t, t, V);
    u64 Y3[4], Z3[4]; F::sub(Y3, V, t); F::mul(Y3, r, Y3);
    u64 w[4]; F::mul(w, Y, J); p_add(w, w, w); F::sub(Y3, Y3, w);
    p_add(Z3, Z, H); F::sqr(Z3, Z3); F::sub(Z3, Z3, Z1Z1); F::sub(Z3, Z3, HH);
    cp4(X, t); cp4(Y, Y3); cp4(Z, Z3);
}
template <class F> __host__ __device__ void aff_add(Pt& R, const Pt& A, const Pt& B) {
    u64 dx[4], dy[4], inv[4], lam[4], x3[4], y3[4];
    F::sub(dx, B.x, A.x); F::sub(dy, B.y, A.y); F::inv(inv, dx); F::mul(lam, dy, inv);
    F::sqr(x3, lam); F::sub(x3, x3, A.x); F::sub(x3, x3, B.x);
    F::sub(y3, A.x, x3); F::mul(y3, lam, y3); F::sub(y3, y3, A.y);
    cp4(R.x, x3); cp4(R.y, y3);
}
// R = s * D[0], D[b] = 2^b * D[0]; false for s == 0
template <class F> __host__ __device__ bool mul_table(Pt& R, const u64* s, const Pt* D, int nb) {
    u64 X[4], Y[4], Z[4]; bool inf = true;
    for (int b = 0; b < nb; ++b) {
        if (!((s[b >> 6] >> (b & 63)) & 1)) continue;
        if (inf) { cp4(X, D[b].x); cp4(Y, D[b].y); Z[0] = 1; Z[1] = Z[2] = Z[3] = 0; inf = false; }
        else jac_madd<F>(X, Y, Z, D[b].x, D[b].y);
    }
    if (inf) return false;
    u64 zi[4], zi2[4], zi3[4]; F::inv(zi, Z); F::sqr(zi2, zi); F::mul(zi3, zi2, zi);
    F::mul(R.x, X, zi2); F::mul(R.y, Y, zi3); return true;
}

// ============================================================================ 192-bit signed distances
__host__ __device__ __forceinline__ void d_add(u64* d, const u64* j) {
    u128 c = (u128)d[0] + j[0]; d[0] = (u64)c; c >>= 64; c += (u128)d[1] + j[1]; d[1] = (u64)c; c >>= 64; d[2] += j[2] + (u64)c;
}
__host__ __device__ __forceinline__ void d_sub(u64* d, const u64* j) {
    u128 b = (u128)d[0] - j[0]; d[0] = (u64)b; u64 br = (u64)(b >> 64) ? 1 : 0;
    b = (u128)d[1] - j[1] - br; d[1] = (u64)b; br = (u64)(b >> 64) ? 1 : 0; d[2] -= j[2] + br;
}

// jump-table read: on the device jt1 lives in shared memory; tell the compiler so it emits LDS instead of generic LD
__host__ __device__ __forceinline__ void ld_jump(u64* jx, u64* jy, const u64* tab, u32 j, bool shared) {
#ifdef __CUDA_ARCH__
    if (shared) {
        const unsigned sa = (unsigned)__cvta_generic_to_shared(tab + 8 * j);
        asm volatile("ld.shared.v2.u64 {%0,%1}, [%2];" : "=l"(jx[0]), "=l"(jx[1]) : "r"(sa));
        asm volatile("ld.shared.v2.u64 {%0,%1}, [%2+16];" : "=l"(jx[2]), "=l"(jx[3]) : "r"(sa));
        asm volatile("ld.shared.v2.u64 {%0,%1}, [%2+32];" : "=l"(jy[0]), "=l"(jy[1]) : "r"(sa));
        asm volatile("ld.shared.v2.u64 {%0,%1}, [%2+48];" : "=l"(jy[2]), "=l"(jy[3]) : "r"(sa));
        return;
    }
#endif
    cp4(jx, tab + 8 * j); cp4(jy, tab + 8 * j + 4);
}

// ============================================================================ kangaroo state (SoA) + context
// Kangaroo i (i < NK): x at X + 4*i, y at Y + 4*i (coalesced: kangaroo index = thread + 256*block + group*(blocks*256)).
struct Ctx {
    u64 *X, *Y, *S;        // points and prefix products, 4 words each
    u64* D;                // distances: 4 words each (3 used), signed 192-bit
    u64* HIST;             // MD_LEN words each: ring of distance low words (loop detection)
    u32* FLAG;             // k_dist: history ring position
    u64* T2;               // per thread: bit g = kangaroo g of this thread jumps from table 2 next (2-cycle escape)
    u64 stride;            // kangaroo i = thread + g * stride
    u16* REC;              // steps x NK jump records
    u64* DPX;              // DP_SLOTS x 2 words x NK: x[0], x[1] of DPs found this launch
    u32* DPN;              // NK: DPs found this launch per kangaroo (low 16: written by k_jump, high 16: consumed)
    u64* OUT; u32* NOUT;   // DP records out: 6 words each (x0, x1, d0, d1, d2, kang|type<<31)
    u32* LOOPED; u32* NLOOP;
    u64 NK; u64 NT; u32 dpBits; int steps; u64 aliasMask;   // NT: kangaroos 0..NT-1 are tame, the rest wild (--species). aliasMask: ~0 normally; --alias N (power of two) folds all kangaroos onto N (speed probe, wrong results)
    const u64* J1;         // table 1: x(4) y(4) per jump (global copy; k_jump stages it in shared memory)
    const u64* J2; const u64* J3;   // tables 2, 3: x(4) y(4)
    const u64* JD1; const u64* JD2; const u64* JD3;   // jump distances: 3 words each
};

__host__ __device__ __forceinline__ bool is_dp(const u64* x, u32 dpBits) { return (x[3] >> (64 - dpBits)) == 0; }

// one jump of one kangaroo, affine, given 1/(x - jx) as `ik`. Returns the record.
template <class F>
__host__ __device__ __forceinline__ u16 jump_point(u64* x, u64* y, const u64* jx, const u64* jy, const u64* ik, u16 rec) {
    u64 dy[4], lam[4], nx[4], ny[4], jyy[4], njy[4];
    cp4(njy, jy); F::neg(njy);
    const bool odd = y[0] & 1;                                              // y odd: jump -J (select, no branch)
    jyy[0] = odd ? njy[0] : jy[0]; jyy[1] = odd ? njy[1] : jy[1]; jyy[2] = odd ? njy[2] : jy[2]; jyy[3] = odd ? njy[3] : jy[3];
    rec |= odd ? REC_INV : 0;
    F::sub(dy, y, jyy); F::mul(lam, dy, ik);      // lambda = (y - jy) / (x - jx)
    F::sqr(nx, lam); F::sub(nx, nx, x); F::sub(nx, nx, jx);
    F::sub(ny, x, nx); F::mul(ny, lam, ny); F::sub(ny, ny, y);
    cp4(x, nx); cp4(y, ny);
    return rec;
}

// one step for the G kangaroos of one "thread": indices base + g*stride. jt1: table 1 (x,y per jump; shared memory on
// the GPU). pre: prefix products, G x 4 words: a per-thread local array, or (PRE_STRIDE > 1) a shared-memory array laid
// out [g][word][thread] so that consecutive threads hit consecutive words. t2mask: bit g = next jump from table 2.
template <class F, int G, int PRE_STRIDE, int UNR = 1>
__host__ __device__ __forceinline__ void step_group(const Ctx& c, u64 base, u64 stride, const u64* jt1, int stepIdx, u64* pre, u64& t2mask) {
    u64 acc[4], dx[4];
#define PRE(g, k) pre[((g) * 4 + (k)) * PRE_STRIDE]
    // forward: prefix products of (x_g - jx_g)
    #pragma unroll UNR
    for (int g = 0; g < G; ++g) {
        const u64 i = (base + g * stride) & c.aliasMask; const u64* x = c.X + 4 * i;
        const u32 j = (u32)x[0] & JMP_MASK;
        const u64* jx = ((t2mask >> g) & 1) ? c.J2 + 8 * j : jt1 + 8 * j;
        F::sub(dx, x, jx);
        if (g == 0) cp4(acc, dx); else F::mul(acc, acc, dx);
        PRE(g, 0) = acc[0]; PRE(g, 1) = acc[1]; PRE(g, 2) = acc[2]; PRE(g, 3) = acc[3];
    }
    u64 inv[4]; F::inv(inv, acc);
    // backward
    #pragma unroll UNR
    for (int g = G - 1; g >= 0; --g) {
        const u64 i = (base + g * stride) & c.aliasMask; u64* x = c.X + 4 * i; u64* y = c.Y + 4 * i;
        const u32 j = (u32)x[0] & JMP_MASK; const bool t2 = (t2mask >> g) & 1;
        const u64* jx = t2 ? c.J2 + 8 * j : jt1 + 8 * j; const u64* jy = jx + 4;
        u64 ik[4];
        if (g > 0) { u64 pg[4] = {PRE(g - 1, 0), PRE(g - 1, 1), PRE(g - 1, 2), PRE(g - 1, 3)}; F::mul(ik, inv, pg); F::sub(dx, x, jx); F::mul(inv, inv, dx); }
        else cp4(ik, inv);
        u64 xx[4], yy[4]; cp4(xx, x); cp4(yy, y);
        u16 rec = jump_point<F>(xx, yy, jx, jy, ik, (u16)j);
        if (t2) { rec |= REC_JMP2; t2mask &= ~(1ULL << g); }
        else {
            // 2-cycle: the next jump would be the same index in the opposite direction -> back to where we came from
            const u32 jn = (u32)xx[0] & JMP_MASK; const bool invn = yy[0] & 1;
            if (jn == j && invn != ((rec & REC_INV) != 0)) t2mask |= 1ULL << g;
        }
        if (is_dp(xx, c.dpBits)) {
            rec |= REC_DP;
#ifdef __CUDA_ARCH__
            u32 n = atomicAdd(c.DPN + i, 1u);
#else
            u32 n = c.DPN[i]++;
#endif
            if (n < DP_SLOTS) { c.DPX[(i * DP_SLOTS + n) * 2] = xx[0]; c.DPX[(i * DP_SLOTS + n) * 2 + 1] = xx[1]; }
        }
        cp4(x, xx); cp4(y, yy);
#ifdef __CUDA_ARCH__
        __stcs(c.REC + (u64)stepIdx * c.NK + i, rec);         // streaming: the record list must not evict the state from L2
#else
        c.REC[(u64)stepIdx * c.NK + i] = rec;
#endif
    }
#undef PRE
}


// FUSED step: processes the G kangaroos in order g = G-1..0 (DIR = 0) or 0..G-1 (DIR = 1). preIn holds the prefix
// products accumulated by the previous step in the opposite order (so the Montgomery backward order is right), preOut
// receives the prefix products of the NEW points, accumulated in this step's order, for the next step.
template <class F, int G, int PRE_STRIDE, int DIR>
__host__ __device__ __forceinline__ void step_fused(const Ctx& c, u64 base, u64 stride, const u64* jt1, int stepIdx,
                                                    const u64* preIn, u64* preOut, u64& t2mask, u64* accOut) {
#define PRE_(p, g, k) (p)[((g) * 4 + (k)) * PRE_STRIDE]
    u64 inv[4]; { u64 tot[4] = {PRE_(preIn, G - 1, 0), PRE_(preIn, G - 1, 1), PRE_(preIn, G - 1, 2), PRE_(preIn, G - 1, 3)}; F::inv(inv, tot); }
    u64 acc[4]; bool first = true; u64 dx[4];
    #pragma unroll 1
    for (int n = 0; n < G; ++n) {
        const int g = DIR ? n : G - 1 - n;           // processing order
        const int k = G - 1 - n;                     // this kangaroo's position in preIn's accumulation order
        const u64 i = (base + g * stride) & c.aliasMask; u64* x = c.X + 4 * i; u64* y = c.Y + 4 * i;
        const u32 j = (u32)x[0] & JMP_MASK; const bool t2 = (t2mask >> g) & 1;
        u64 jx[4], jy[4]; ld_jump(jx, jy, t2 ? c.J2 : jt1, j, !t2);
        u64 ik[4];
        if (k > 0) { u64 pg[4] = {PRE_(preIn, k - 1, 0), PRE_(preIn, k - 1, 1), PRE_(preIn, k - 1, 2), PRE_(preIn, k - 1, 3)};
                     F::mul(ik, inv, pg); F::sub(dx, x, jx); F::mul(inv, inv, dx); }
        else cp4(ik, inv);
        u64 xx[4], yy[4]; cp4(xx, x); cp4(yy, y);
        u16 rec = jump_point<F>(xx, yy, jx, jy, ik, (u16)j);
        if (t2) { rec |= REC_JMP2; t2mask &= ~(1ULL << g); }
        else {
            const u32 jn = (u32)xx[0] & JMP_MASK; const bool invn = yy[0] & 1;
            if (jn == j && invn != ((rec & REC_INV) != 0)) t2mask |= 1ULL << g;
        }
        if (is_dp(xx, c.dpBits)) {
            rec |= REC_DP;
#ifdef __CUDA_ARCH__
            u32 nn = atomicAdd(c.DPN + i, 1u);
#else
            u32 nn = c.DPN[i]++;
#endif
            if (nn < DP_SLOTS) { c.DPX[(i * DP_SLOTS + nn) * 2] = xx[0]; c.DPX[(i * DP_SLOTS + nn) * 2 + 1] = xx[1]; }
        }
        cp4(x, xx); cp4(y, yy);
#ifdef __CUDA_ARCH__
        __stcs(c.REC + (u64)stepIdx * c.NK + i, rec);
#else
        c.REC[(u64)stepIdx * c.NK + i] = rec;
#endif
        // next step's prefix product for this kangaroo (its NEW x, and the table it will use next)
        {
            const u32 j2 = (u32)xx[0] & JMP_MASK; const bool nt2 = (t2mask >> g) & 1;
            u64 jx2[4], jy2[4]; ld_jump(jx2, jy2, nt2 ? c.J2 : jt1, j2, !nt2);
            u64 d2[4]; F::sub(d2, xx, jx2);
            if (first) { cp4(acc, d2); first = false; } else F::mul(acc, acc, d2);
            PRE_(preOut, n, 0) = acc[0]; PRE_(preOut, n, 1) = acc[1]; PRE_(preOut, n, 2) = acc[2]; PRE_(preOut, n, 3) = acc[3];
        }
    }
    cp4(accOut, acc);
#undef PRE_
}

// FUSED step, two kangaroos per iteration with their field arithmetic interleaved in one straight-line block (ILP).
// Same contract as step_fused. G must be even.
template <class F, int G, int PRE_STRIDE, int DIR>
__host__ __device__ __forceinline__ void step_fused2(const Ctx& c, u64 base, u64 stride, const u64* jt1, int stepIdx,
                                                     const u64* preIn, u64* preOut, u64& t2mask, u64* accOut) {
#define PRE_(p, g, k) (p)[((g) * 4 + (k)) * PRE_STRIDE]
    u64 inv[4]; { u64 tot[4] = {PRE_(preIn, G - 1, 0), PRE_(preIn, G - 1, 1), PRE_(preIn, G - 1, 2), PRE_(preIn, G - 1, 3)}; F::inv(inv, tot); }
    u64 acc[4];
    #pragma unroll 1
    for (int n = 0; n < G; n += 2) {
        const int gA = DIR ? n : G - 1 - n, gB = DIR ? n + 1 : G - 2 - n;   // processing order
        const int kA = G - 1 - n, kB = G - 2 - n;                            // positions in preIn
        const u64 iA = (base + gA * stride) & c.aliasMask, iB = (base + gB * stride) & c.aliasMask;
        u64* xA = c.X + 4 * iA; u64* yA = c.Y + 4 * iA; u64* xB = c.X + 4 * iB; u64* yB = c.Y + 4 * iB;
        const u32 jA = (u32)xA[0] & JMP_MASK, jB = (u32)xB[0] & JMP_MASK;
        const bool tA = (t2mask >> gA) & 1, tB = (t2mask >> gB) & 1;
        u64 jxA[4], jyA[4], jxB[4], jyB[4];
        ld_jump(jxA, jyA, tA ? c.J2 : jt1, jA, !tA); ld_jump(jxB, jyB, tB ? c.J2 : jt1, jB, !tB);
        // Montgomery chain: ikA = inv * pre[kA-1]; inv *= dxA; ikB = inv * pre[kB-1]; inv *= dxB
        u64 ikA[4], ikB[4], dxA[4], dxB[4];
        F::sub(dxA, xA, jxA); F::sub(dxB, xB, jxB);
        { u64 pg[4] = {PRE_(preIn, kA - 1, 0), PRE_(preIn, kA - 1, 1), PRE_(preIn, kA - 1, 2), PRE_(preIn, kA - 1, 3)}; F::mul(ikA, inv, pg); }
        F::mul(inv, inv, dxA);
        if (kB > 0) { u64 pg[4] = {PRE_(preIn, kB - 1, 0), PRE_(preIn, kB - 1, 1), PRE_(preIn, kB - 1, 2), PRE_(preIn, kB - 1, 3)}; F::mul(ikB, inv, pg); F::mul(inv, inv, dxB); }
        else cp4(ikB, inv);
        // both jumps, interleaved
        u64 nA[4], nB[4], sA[4], sB[4]; cp4(nA, jyA); cp4(nB, jyB); F::neg(nA); F::neg(nB);
        const bool oA = yA[0] & 1, oB = yB[0] & 1;
        for (int k = 0; k < 4; ++k) { sA[k] = oA ? nA[k] : jyA[k]; sB[k] = oB ? nB[k] : jyB[k]; }
        u64 dyA[4], dyB[4], lA[4], lB[4], xxA[4], xxB[4], yyA[4], yyB[4], tA4[4], tB4[4];
        F::sub(dyA, yA, sA); F::sub(dyB, yB, sB);
        F::mul(lA, dyA, ikA); F::mul(lB, dyB, ikB);
        F::sqr(xxA, lA); F::sqr(xxB, lB);
        F::sub(xxA, xxA, xA); F::sub(xxB, xxB, xB); F::sub(xxA, xxA, jxA); F::sub(xxB, xxB, jxB);
        F::sub(tA4, xA, xxA); F::sub(tB4, xB, xxB);
        F::mul(yyA, lA, tA4); F::mul(yyB, lB, tB4);
        F::sub(yyA, yyA, yA); F::sub(yyB, yyB, yB);
        u16 rA = (u16)jA | (oA ? REC_INV : 0) | (tA ? REC_JMP2 : 0), rB = (u16)jB | (oB ? REC_INV : 0) | (tB ? REC_JMP2 : 0);
        // 2-cycle detection and table-2 flags (selects)
        const u32 jnA = (u32)xxA[0] & JMP_MASK, jnB = (u32)xxB[0] & JMP_MASK;
        const bool cycA = !tA && jnA == jA && ((yyA[0] & 1) != oA), cycB = !tB && jnB == jB && ((yyB[0] & 1) != oB);
        t2mask = (t2mask & ~(1ULL << gA) & ~(1ULL << gB)) | ((u64)cycA << gA) | ((u64)cycB << gB);
        // next step's prefix products (new x, table the kangaroo will use next)
        u64 j2xA[4], j2yA[4], j2xB[4], j2yB[4]; ld_jump(j2xA, j2yA, cycA ? c.J2 : jt1, jnA, !cycA); ld_jump(j2xB, j2yB, cycB ? c.J2 : jt1, jnB, !cycB);
        u64 d2A[4], d2B[4]; F::sub(d2A, xxA, j2xA); F::sub(d2B, xxB, j2xB);
        if (n == 0) cp4(acc, d2A); else F::mul(acc, acc, d2A);
        PRE_(preOut, n, 0) = acc[0]; PRE_(preOut, n, 1) = acc[1]; PRE_(preOut, n, 2) = acc[2]; PRE_(preOut, n, 3) = acc[3];
        F::mul(acc, acc, d2B);
        PRE_(preOut, n + 1, 0) = acc[0]; PRE_(preOut, n + 1, 1) = acc[1]; PRE_(preOut, n + 1, 2) = acc[2]; PRE_(preOut, n + 1, 3) = acc[3];
        // stores
        cp4(xA, xxA); cp4(yA, yyA); cp4(xB, xxB); cp4(yB, yyB);
        const bool dA = is_dp(xxA, c.dpBits), dB = is_dp(xxB, c.dpBits);
        rA |= dA ? REC_DP : 0; rB |= dB ? REC_DP : 0;
#ifdef __CUDA_ARCH__
        __stcs(c.REC + (u64)stepIdx * c.NK + iA, rA); __stcs(c.REC + (u64)stepIdx * c.NK + iB, rB);
        if (dA) { u32 nn = atomicAdd(c.DPN + iA, 1u); if (nn < DP_SLOTS) { c.DPX[(iA * DP_SLOTS + nn) * 2] = xxA[0]; c.DPX[(iA * DP_SLOTS + nn) * 2 + 1] = xxA[1]; } }
        if (dB) { u32 nn = atomicAdd(c.DPN + iB, 1u); if (nn < DP_SLOTS) { c.DPX[(iB * DP_SLOTS + nn) * 2] = xxB[0]; c.DPX[(iB * DP_SLOTS + nn) * 2 + 1] = xxB[1]; } }
#else
        c.REC[(u64)stepIdx * c.NK + iA] = rA; c.REC[(u64)stepIdx * c.NK + iB] = rB;
        if (dA) { u32 nn = c.DPN[iA]++; if (nn < DP_SLOTS) { c.DPX[(iA * DP_SLOTS + nn) * 2] = xxA[0]; c.DPX[(iA * DP_SLOTS + nn) * 2 + 1] = xxA[1]; } }
        if (dB) { u32 nn = c.DPN[iB]++; if (nn < DP_SLOTS) { c.DPX[(iB * DP_SLOTS + nn) * 2] = xxB[0]; c.DPX[(iB * DP_SLOTS + nn) * 2 + 1] = xxB[1]; } }
#endif
    }
    cp4(accOut, acc);
#undef PRE_
}

// initial prefix products (order g = 0..G-1), used before the first fused step of a launch
template <class F, int G, int PRE_STRIDE>
__host__ __device__ __forceinline__ void prefix_init(const Ctx& c, u64 base, u64 stride, const u64* jt1, u64* pre, u64 t2mask) {
    u64 acc[4], dx[4];
    for (int g = 0; g < G; ++g) {
        const u64 i = (base + g * stride) & c.aliasMask; const u64* x = c.X + 4 * i;
        const u32 j = (u32)x[0] & JMP_MASK; const u64* jx = ((t2mask >> g) & 1) ? c.J2 + 8 * j : jt1 + 8 * j;
        F::sub(dx, x, jx);
        if (g == 0) cp4(acc, dx); else F::mul(acc, acc, dx);
        pre[(g * 4 + 0) * PRE_STRIDE] = acc[0]; pre[(g * 4 + 1) * PRE_STRIDE] = acc[1]; pre[(g * 4 + 2) * PRE_STRIDE] = acc[2]; pre[(g * 4 + 3) * PRE_STRIDE] = acc[3];
    }
}

// replay the records of kangaroo i for this launch: distances, DP output, loop detection
__host__ __device__ void replay(const Ctx& c, u64 i, const u64* jd1, const u64* jd2) {
    u64 d[3] = {c.D[4 * i], c.D[4 * i + 1], c.D[4 * i + 2]};
    u64 hist[MD_LEN]; for (int k = 0; k < MD_LEN; ++k) hist[k] = c.HIST[i * MD_LEN + k];
    u32 pos = c.FLAG[i]; bool looped = false; u32 dpc = 0;
    for (int s = 0; s < c.steps; ++s) {
        const u16 r = c.REC[(u64)s * c.NK + i];
        const u64* jd = (r & REC_JMP2) ? jd2 + 3 * (r & JMP_MASK) : jd1 + 3 * (r & JMP_MASK);
        if (r & REC_INV) d_sub(d, jd); else d_add(d, jd);
        if (r & REC_DP) {
            if (dpc < DP_SLOTS) {
#ifdef __CUDA_ARCH__
                u32 n = atomicAdd(c.NOUT, 1u);
#else
                u32 n = __atomic_fetch_add(c.NOUT, 1u, __ATOMIC_RELAXED);
#endif
                if (n < MAX_DP_OUT) {
                    u64* o = c.OUT + 6 * n;
                    o[0] = c.DPX[(i * DP_SLOTS + dpc) * 2]; o[1] = c.DPX[(i * DP_SLOTS + dpc) * 2 + 1];
                    o[2] = d[0]; o[3] = d[1]; o[4] = d[2]; o[5] = i;
                }
            }
            ++dpc;
        }
        if (!looped) {
            for (int k = 0; k < MD_LEN; ++k) if (hist[k] == d[0]) { looped = true; break; }
        }
        hist[pos] = d[0]; pos = (pos + 1) % MD_LEN;
    }
    c.D[4 * i] = d[0]; c.D[4 * i + 1] = d[1]; c.D[4 * i + 2] = d[2];
    for (int k = 0; k < MD_LEN; ++k) c.HIST[i * MD_LEN + k] = hist[k];
    c.FLAG[i] = pos;
    c.DPN[i] = 0;
    if (looped) {
#ifdef __CUDA_ARCH__
        u32 n = atomicAdd(c.NLOOP, 1u);
#else
        u32 n = __atomic_fetch_add(c.NLOOP, 1u, __ATOMIC_RELAXED);
#endif
        c.LOOPED[n] = (u32)i;
    }
}

// escape a loop: one jump from table 3 (huge jumps) from the current point
template <class F>
__host__ __device__ void escape(const Ctx& c, u64 i) {
    u64* x = c.X + 4 * i; u64* y = c.Y + 4 * i;
    const u32 j = (u32)x[0] & JMP_MASK; const u64* jx = c.J3 + 8 * j; const u64* jy = jx + 4;
    u64 dx[4], ik[4]; F::sub(dx, x, jx); F::inv(ik, dx);
    u16 rec = jump_point<F>(x, y, jx, jy, ik, 0);
    u64 d[3] = {c.D[4 * i], c.D[4 * i + 1], c.D[4 * i + 2]};
    if (rec & REC_INV) d_sub(d, c.JD3 + 3 * j); else d_add(d, c.JD3 + 3 * j);
    c.D[4 * i] = d[0]; c.D[4 * i + 1] = d[1]; c.D[4 * i + 2] = d[2];
    for (int k = 0; k < MD_LEN; ++k) c.HIST[i * MD_LEN + k] = 0;      // forget the loop's history
    const u64 bit = 1ULL << (u32)(i / c.stride);
#ifdef __CUDA_ARCH__
    atomicAnd((unsigned long long*)(c.T2 + (i % c.stride)), (unsigned long long)~bit);
#else
    c.T2[i % c.stride] &= ~bit;
#endif
}

// ============================================================================ GPU kernels
static void die(const char* m) { fprintf(stderr, "error: %s\n", m); exit(1); }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
template <int G, bool SMEM_PRE, int MINB, int UNR>
__global__ void __launch_bounds__(256, MINB)
k_jump(Ctx c, u64 blocks) {
    extern __shared__ u64 smem[];                          // [0, 4096): table 1 (32 KB); then prefix products if SMEM_PRE
    u64* jt1 = smem;
    for (int k = threadIdx.x; k < JMP_CNT * 8; k += 256) jt1[k] = c.J1[k];
    __syncthreads();
    const u64 base = (u64)blockIdx.x * 256 + threadIdx.x, stride = blocks * 256;
    u64 t2mask = c.T2[base];
    if (SMEM_PRE) {
        u64* pre = smem + JMP_CNT * 8 + threadIdx.x;       // [g][word][thread]
        for (int s = 0; s < c.steps; ++s) step_group<DevOps, G, 256, UNR>(c, base, stride, jt1, s, pre, t2mask);
    } else {
        u64 pre[G * 4];
        for (int s = 0; s < c.steps; ++s) step_group<DevOps, G, 1, UNR>(c, base, stride, jt1, s, pre, t2mask);
    }
    c.T2[base] = t2mask;
}
template <int G, int MINB, class OPS, int ILP>
__global__ void __launch_bounds__(256, MINB)
k_jump_fused(Ctx c, u64 blocks) {
    extern __shared__ u64 smem[];
    u64* jt1 = smem;
    for (int k = threadIdx.x; k < JMP_CNT * 8; k += 256) jt1[k] = c.J1[k];
    __syncthreads();
    const u64 base = (u64)blockIdx.x * 256 + threadIdx.x, stride = blocks * 256;
    u64 t2mask = c.T2[base];
    u64 preA[G * 4], preB[G * 4], acc[4];
    prefix_init<OPS, G, 1>(c, base, stride, jt1, preA, t2mask);
    for (int s = 0; s + 1 < c.steps; s += 2) {
        if (ILP == 2) { step_fused2<OPS, G, 1, 0>(c, base, stride, jt1, s, preA, preB, t2mask, acc); step_fused2<OPS, G, 1, 1>(c, base, stride, jt1, s + 1, preB, preA, t2mask, acc); }
        else { step_fused<OPS, G, 1, 0>(c, base, stride, jt1, s, preA, preB, t2mask, acc); step_fused<OPS, G, 1, 1>(c, base, stride, jt1, s + 1, preB, preA, t2mask, acc); }
    }
    c.T2[base] = t2mask;
}
// variants: --groups 8|12|16|24|32|48, --unroll 1|2 (two kangaroo chains interleaved), --minblocks 1|2, --smem 1 (8 only)
static size_t launch_jump(int G, bool smemPre, int minb, int unr, const Ctx& c, u64 blocks) {
    const size_t sm = JMP_CNT * 64 + (smemPre ? (size_t)G * 4 * 256 * 8 : 0);
#define LJ(GG, SP, MB, UN) if (G == GG && smemPre == SP && minb == MB && unr == UN) { static bool set = false; \
        if (!set) { CK(cudaFuncSetAttribute(k_jump<GG, SP, MB, UN>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm)); set = true; } \
        k_jump<GG, SP, MB, UN><<<(unsigned)blocks, 256, sm>>>(c, blocks); return sm; }
    LJ(8, true, 1, 1) LJ(8, false, 1, 1)
    LJ(12, false, 1, 1) LJ(12, false, 2, 1) LJ(12, false, 2, 2)
    LJ(16, false, 1, 1) LJ(16, false, 1, 2)
    LJ(24, false, 1, 1) LJ(24, false, 1, 2) LJ(24, false, 2, 2)
    LJ(32, false, 1, 1) LJ(32, false, 1, 2)
    LJ(48, false, 1, 1) LJ(48, false, 1, 2)
#undef LJ
    die("unsupported --groups/--smem/--minblocks/--unroll combination"); return 0;
}
// --cubin FILE: the jump kernel comes from an external cubin (hand-written / hand-edited SASS, CuAssembler), launched
// through the driver API with the same arguments (Ctx, blocks), grid, block and shared memory as the built-in one.
static CUfunction gCubinFn = nullptr;
static void cubin_load(const std::string& file, const std::string& kernel, size_t sm) {
    CUmodule mod; CUresult r = cuModuleLoad(&mod, file.c_str());
    if (r != CUDA_SUCCESS) { const char* e = "?"; cuGetErrorString(r, &e); die((std::string("cuModuleLoad ") + file + ": " + e).c_str()); }
    r = cuModuleGetFunction(&gCubinFn, mod, kernel.c_str());
    if (r != CUDA_SUCCESS) { const char* e = "?"; cuGetErrorString(r, &e); die((std::string("kernel ") + kernel + " not in " + file + ": " + e).c_str()); }
    r = cuFuncSetAttribute(gCubinFn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)sm);
    if (r != CUDA_SUCCESS) die("cuFuncSetAttribute(shared memory) failed");
    int nreg = 0; cuFuncGetAttribute(&nreg, CU_FUNC_ATTRIBUTE_NUM_REGS, gCubinFn);
    int lmem = 0; cuFuncGetAttribute(&lmem, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, gCubinFn);
    printf("[+] cubin kernel  : %s from %s (%d registers, %d B local memory per thread)\n", kernel.c_str(), file.c_str(), nreg, lmem);
}
static size_t launch_cubin(const Ctx& c, u64 blocks) {
    const size_t sm = JMP_CNT * 64;
    Ctx cc = c; u64 b = blocks; void* args[] = {&cc, &b};
    CUresult r = cuLaunchKernel(gCubinFn, (unsigned)blocks, 1, 1, 256, 1, 1, (unsigned)sm, 0, args, nullptr);
    if (r != CUDA_SUCCESS) { const char* e = "?"; cuGetErrorString(r, &e); die((std::string("cuLaunchKernel: ") + e).c_str()); }
    return sm;
}
// the mangled name of the built-in fused kernel for this configuration (what a hand-edited cubin keeps)
static std::string fused_kernel_name(int G, int minb, int invKind, int ilp) {
    char b[128]; snprintf(b, sizeof b, "_Z12k_jump_fusedILi%dELi%dE%sLi%dEEv3Ctxm", G, minb, invKind ? "6DevOps" : "7DevOpsV", ilp); return b;
}
static size_t launch_fused(int G, int minb, int invKind, int ilp, const Ctx& c, u64 blocks) {
    if (gCubinFn) return launch_cubin(c, blocks);
    const size_t sm = JMP_CNT * 64;
#define LF(GG, MB, OPS, IK, IL) if (G == GG && minb == MB && invKind == IK && ilp == IL) { static bool set = false; \
        if (!set) { CK(cudaFuncSetAttribute(k_jump_fused<GG, MB, OPS, IL>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm)); set = true; } \
        k_jump_fused<GG, MB, OPS, IL><<<(unsigned)blocks, 256, sm>>>(c, blocks); return sm; }
    LF(12, 1, DevOps, 1, 1) LF(12, 2, DevOps, 1, 1) LF(12, 2, DevOps, 1, 2) LF(12, 1, DevOps, 1, 2)
    LF(16, 1, DevOps, 1, 1) LF(16, 1, DevOps, 1, 2)
    LF(24, 1, DevOps, 1, 1) LF(24, 2, DevOps, 1, 1) LF(24, 1, DevOps, 1, 2) LF(24, 2, DevOps, 1, 2)
    LF(32, 1, DevOps, 1, 1) LF(32, 1, DevOps, 1, 2)
    LF(24, 1, DevOpsV, 0, 1)
#undef LF
    die("--fused: unsupported --groups/--minblocks/--inv/--ilp combination"); return 0;
}
__global__ void k_dist(Ctx c) {
    __shared__ u64 jd1[JMP_CNT * 3], jd2[JMP_CNT * 3];    // 24 KB
    for (int k = threadIdx.x; k < JMP_CNT * 3; k += blockDim.x) { jd1[k] = c.JD1[k]; jd2[k] = c.JD2[k]; }
    __syncthreads();
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c.NK) replay(c, i, jd1, jd2);
}
__global__ void k_escape(Ctx c) {
    const u32 n = (u32)blockIdx.x * blockDim.x + threadIdx.x;
    if (n < *c.NLOOP) escape<DevOpsF>(c, c.LOOPED[n]);   // Fermat inversion: _ModInv hung here and in k_start
}
// start points: kangaroo i: P = D[i]*G (+ base for wild, i >= NT)
__global__ void k_start(Ctx c, const Pt* G2, int nb, Pt wildBase, int mode) {
    const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.NK) return;
    u64 s[4] = {c.D[4 * i], c.D[4 * i + 1], c.D[4 * i + 2], 0};
    Pt P;
    // mode 0/1/2: Fermat inversion (DevOpsF). mode 3: mul_table with _ModInv (bisect); 4: nb = 1; 5: nb = 2
    if (mode == 2) { cp4(P.x, G2[0].x); cp4(P.y, G2[0].y); }
    else if (mode == 3) mul_table<DevOps>(P, s, G2, nb);
    else if (mode == 4) mul_table<DevOps>(P, s, G2, 1);
    else if (mode == 5) mul_table<DevOps>(P, s, G2, 2);
    else mul_table<DevOpsF>(P, s, G2, nb);
    if ((mode == 0 || mode == 2) && i >= c.NT) { Pt R; aff_add<DevOpsF>(R, P, wildBase); P = R; }
    cp4(c.X + 4 * i, P.x); cp4(c.Y + 4 * i, P.y);
}

// regenerate kangaroos caught in a long cycle (same DP reported twice): fresh random distance (host-chosen, in nd),
// new start point, history and 2-cycle flag cleared. RCKangaroo does the same (KernelGen on looped kangaroos).
__global__ void k_regen(Ctx c, const Pt* G2, int nb, Pt wildBase, const u32* idx, const u64* nd, u32 n) {
    const u32 k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const u64 i = idx[k];
    u64 s[4] = {nd[3 * k], nd[3 * k + 1], nd[3 * k + 2], 0};
    c.D[4 * i] = s[0]; c.D[4 * i + 1] = s[1]; c.D[4 * i + 2] = s[2]; c.D[4 * i + 3] = 0;
    Pt P; mul_table<DevOpsF>(P, s, G2, nb);
    if (i >= c.NT) { Pt R; aff_add<DevOpsF>(R, P, wildBase); P = R; }
    cp4(c.X + 4 * i, P.x); cp4(c.Y + 4 * i, P.y);
    if (!c.HIST) return;          // RC backend: the walk kernels keep their own loop history (stale entries are harmless)
    for (int h = 0; h < MD_LEN; ++h) c.HIST[i * MD_LEN + h] = 0;
    c.FLAG[i] = 0; c.DPN[i] = 0;
    atomicAnd((unsigned long long*)(c.T2 + (i % c.stride)), (unsigned long long)~(1ULL << (u32)(i / c.stride)));
}

static std::string human(double v);
// ============================================================================ RC backend: RCKangaroo's kernels
// --rc runs RetiredCoder's RCKangaroo v4 walk (GPLv3, rc/): the turbo SASS kernels KernelA (walk + dedicated inversion
// SMs) and KernelB (distance replay, loops) from rc/kernel_sm89.cubin / kernel_sm120.cubin on RTX 40 / 50, or the CUDA
// KernelA/KernelB from rc/RCGpuCore.cu elsewhere (--rc cuda); KernelC (table-3 escape jumps) is always the CUDA one.
// Everything else stays ours: jump tables (same seeded table 1 as every other worker, so the fleet walks one graph),
// start points (k_start), DP handling, collision resolution, --dpstream, --on-find. RC's state layout is ours:
// x at L2[4*i], y at L2[4*i + NK*4], distance 4 words per kangaroo, kangaroo i = thread + group * (blocks*256).
// Differences a fleet must know: RC's DP test is (x bits 224..224+dp) == 0, ours is the top dp bits of x; RC's
// 2-cycle / loop handling is its own. A console DP table mixes the two kinds of worker safely (points never match
// across them, so no collision is ever found between an RC worker and a non-RC one: run a herd of one kind).
struct RcParams {                 // == RCKangaroo's TKparams (rc/defs.h), field for field; passed to the kernels by value
    u64* L2; u32* Jumps12; u32* DPTable; u32* Reserved1; u64* JumpsList; u64* LastPnts; u32* dbg_buf; u32* L1S2; u64* Reserved2;
    u32 iter_cnt, BlockCnt, StopThr, dp_mask;
    u32* DPs_out; u64* LoopTable; u32* LoopedKangs; u64* dists; u64* JmpDists12; u32 KangCnt;
    u64* Jumps1; u64* Jumps2; u64* Jumps3; u32 BlockSize, GroupCnt; u64 DP; bool IsGenMode;
    u32 KernelA_LDS_Size, KernelB_LDS_Size, KernelC_LDS_Size;
};
extern "C" {
size_t rc_params_size(void); int rc_set_params(const void* p, const unsigned long long* jmp2_table);
void rc_call_gen(const void* p); void rc_call_a(const void* p); void rc_call_b(const void* p); void rc_call_c(const void* p);
}
#define RC_STEP_CNT 1000
#define RC_GROUP 24
#define RC_DP_SIZE 48
#define RC_MAX_DP (256 * 1024)
struct RcState {
    RcParams k{}; bool cubin = false; int invSM = 0; size_t invData = 0, l2size = 0; u64 NK = 0;
    CUmodule mod = nullptr; CUfunction fnA = nullptr, fnB = nullptr;
    std::vector<u32> dpHost;      // DPs_out staging
};
// the cubin for this GPU: --rc-cubin FILE, else kernel_sm<cc>.cubin next to the binary, in ./rc, ../kangaroo/rc or /app/rc
static std::string rc_find_cubin(const std::string& opt, int ccMajor, int ccMinor, const char* argv0) {
    if (!opt.empty()) return opt;
    const std::string name = ccMajor == 12 ? "kernel_sm120.cubin" : (ccMajor == 8 && ccMinor == 9) ? "kernel_sm89.cubin" : "";
    if (name.empty()) return "";
    // kernel_sm89_sqr.cubin = RC's kernel rebuilt by sass/rcasm.py with sass/gen_sqr.py's true square (+2%): preferred when present
    const std::string sqr = name.substr(0, name.size() - 6) + "_sqr.cubin";
    std::string exe = argv0 ? argv0 : ""; size_t s = exe.find_last_of('/'); std::string dir = s == std::string::npos ? "." : exe.substr(0, s);
    for (const std::string& n : {sqr, name})
        for (const std::string& d : {dir + "/rc/", dir + "/", std::string("rc/"), std::string("kangaroo/rc/"), std::string("/app/rc/")}) {
            std::ifstream f(d + n); if (f.good()) return d + n;
        }
    return "";
}
static const char* gArgv0 = nullptr;
static void rc_cu(CUresult r, const char* what) { if (r != CUDA_SUCCESS) { const char* e = "?"; cuGetErrorString(r, &e); die((std::string(what) + ": " + e).c_str()); } }
// allocate RC's buffers, pack the jump tables in RC's three layouts, load the cubin. Returns the state; c.X/c.Y/c.D are set.
static void rc_setup(RcState& rc, Ctx& c, u64 blocks, int dp, const std::vector<u64>& J1, const std::vector<u64>& J2, const std::vector<u64>& J3,
                     const std::vector<u64>& JD1, const std::vector<u64>& JD2, const std::vector<u64>& JD3, const std::string& cubinPath, int sms) {
    if (sizeof(RcParams) != rc_params_size()) die("RcParams does not match RCKangaroo's TKparams (rc/defs.h changed?)");
    RcParams& k = rc.k; const u64 NK = blocks * 256 * RC_GROUP; rc.NK = NK;
    k.BlockCnt = (u32)blocks; k.BlockSize = 256; k.GroupCnt = RC_GROUP; k.KangCnt = (u32)NK; k.DP = dp; k.dp_mask = (u32)((1ull << dp) - 1);
    k.iter_cnt = RC_STEP_CNT; k.StopThr = (u32)(0.5 * (blocks * 8)); k.IsGenMode = false;
    k.KernelA_LDS_Size = 98 * 1024; k.KernelB_LDS_Size = 48 * 1024; k.KernelC_LDS_Size = 96 * JMP_CNT;
    // L2: x | y | (distance copy, unused: we use k_start) | inversion mailboxes (asm kernels)
    rc.invData = 1024 + 4 * 256 * 8 + 4 * (32 * 1024) + 2 * ((32 * 4) * 256 * 8) + 4 * 256 * 8;
    rc.l2size = (size_t)NK * 96 + rc.invData;
    auto dalloc = [&](void** p, size_t bytes) { CK(cudaMalloc(p, bytes)); CK(cudaMemset(*p, 0, bytes)); };
    dalloc((void**)&k.L2, rc.l2size);
    dalloc((void**)&k.DPs_out, (size_t)RC_MAX_DP * RC_DP_SIZE + 16);
    dalloc((void**)&k.Jumps1, JMP_CNT * 96); dalloc((void**)&k.Jumps2, JMP_CNT * 96); dalloc((void**)&k.Jumps3, JMP_CNT * 96);
    dalloc((void**)&k.JumpsList, 2 * (size_t)NK * (RC_STEP_CNT + MD_LEN));
    dalloc((void**)&k.DPTable, (size_t)NK * (16 * DP_SLOTS + 4));
    dalloc((void**)&k.L1S2, (size_t)sms * 256 * 8);
    dalloc((void**)&k.LastPnts, (size_t)NK * MD_LEN * 64);
    dalloc((void**)&k.LoopTable, (size_t)NK * MD_LEN * 8);
    dalloc((void**)&k.dbg_buf, 1024);
    dalloc((void**)&k.LoopedKangs, (size_t)NK * 4 + 8);
    dalloc((void**)&k.dists, (size_t)NK * 32);
    dalloc((void**)&k.Jumps12, JMP_CNT * 192); dalloc((void**)&k.JmpDists12, JMP_CNT * 96);
    // Jumps1/2/3: x(32) y(32) dist(32) per entry (CUDA KernelA/KernelC, KernelGen)
    auto pack96 = [&](u64* dst, const std::vector<u64>& J, const std::vector<u64>& JD) {
        std::vector<u64> b(JMP_CNT * 12, 0);
        for (int i = 0; i < JMP_CNT; ++i) { cp4(&b[12 * i], &J[8 * i]); cp4(&b[12 * i + 4], &J[8 * i + 4]); b[12 * i + 8] = JD[3 * i]; b[12 * i + 9] = JD[3 * i + 1]; b[12 * i + 10] = JD[3 * i + 2]; }
        CK(cudaMemcpy(dst, b.data(), JMP_CNT * 96, cudaMemcpyHostToDevice));
    };
    pack96(k.Jumps1, J1, JD1); pack96(k.Jumps2, J2, JD2); pack96(k.Jumps3, J3, JD3);
    // Jumps12 (asm KernelA): tables 1 and 2, each 48 KB in six 8 KB parts: x lo16, x hi16, y lo16, y hi16, -y lo16, -y hi16
    {
        std::vector<u32> b(JMP_CNT * 48, 0); const int part = 4 * JMP_CNT;
        auto put = [&](u32* base, int i, int p, const u64* w2) { memcpy(base + i * 4 + p * part, w2, 16); };
        for (int t = 0; t < 2; ++t) {
            const std::vector<u64>& J = t ? J2 : J1; u32* base = b.data() + t * (48 * 1024 / 4);
            for (int i = 0; i < JMP_CNT; ++i) {
                u64 ny[4]; p_neg(ny, &J[8 * i + 4]);
                put(base, i, 0, &J[8 * i]); put(base, i, 1, &J[8 * i + 2]); put(base, i, 2, &J[8 * i + 4]); put(base, i, 3, &J[8 * i + 6]); put(base, i, 4, ny); put(base, i, 5, ny + 2);
            }
        }
        CK(cudaMemcpy(k.Jumps12, b.data(), JMP_CNT * 192, cudaMemcpyHostToDevice));
    }
    // JmpDists12 (KernelB): +d1, -d1, +d2, -d2 as 16-byte low parts (4 x 8 KB) then 8-byte high words (4 x 4 KB); -d = 2^192 - d
    {
        std::vector<u32> b(JMP_CNT * 24, 0);
        auto neg192 = [](const u64* d, u64* n) { u128 t = (u128)0 - d[0]; n[0] = (u64)t; u64 br = d[0] ? 1 : 0; t = (u128)0 - d[1] - br; n[1] = (u64)t; br = (d[1] | br) ? 1 : 0; n[2] = (u64)0 - d[2] - br; };
        for (int i = 0; i < JMP_CNT; ++i) {
            u64 d1[3] = {JD1[3 * i], JD1[3 * i + 1], JD1[3 * i + 2]}, d2[3] = {JD2[3 * i], JD2[3 * i + 1], JD2[3 * i + 2]}, n1[3], n2[3];
            neg192(d1, n1); neg192(d2, n2);
            const u64* tabs[4] = {d1, n1, d2, n2};
            for (int t = 0; t < 4; ++t) { memcpy(b.data() + i * 4 + t * (8 * 1024 / 4), tabs[t], 16); memcpy(b.data() + i * 2 + (32 * 1024 / 4) + t * (4 * 1024 / 4), tabs[t] + 2, 8); }
        }
        CK(cudaMemcpy(k.JmpDists12, b.data(), JMP_CNT * 96, cudaMemcpyHostToDevice));
    }
    // CUDA kernels: shared-memory attributes + table 2 in constant memory (CUDA KernelA)
    { std::vector<u64> t2(JMP_CNT * 8); memcpy(t2.data(), J2.data(), JMP_CNT * 64);
      int e = rc_set_params(&k, (const unsigned long long*)t2.data()); if (e) die("rc_set_params (RCGpuCore.cu) failed"); }
    if (rc.cubin) {
        cudaFree(0);
        rc_cu(cuModuleLoad(&rc.mod, cubinPath.c_str()), ("cuModuleLoad " + cubinPath).c_str());
        rc_cu(cuModuleGetFunction(&rc.fnA, rc.mod, "KernelA"), "KernelA not in cubin");
        rc_cu(cuModuleGetFunction(&rc.fnB, rc.mod, "KernelB"), "KernelB not in cubin");
        rc_cu(cuFuncSetAttribute(rc.fnA, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)k.KernelA_LDS_Size), "KernelA shared memory");
        rc_cu(cuFuncSetAttribute(rc.fnB, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)k.KernelB_LDS_Size), "KernelB shared memory");
        int nreg = 0; cuFuncGetAttribute(&nreg, CU_FUNC_ATTRIBUTE_NUM_REGS, rc.fnA);
        printf("[+] RC turbo      : %s (KernelA %d registers, %d inversion SMs, %llu walk blocks)\n", cubinPath.c_str(), nreg, rc.invSM, (unsigned long long)blocks);
    } else printf("[+] RC cuda       : RCKangaroo CUDA KernelA/KernelB (%llu blocks)\n", (unsigned long long)blocks);
    // L2 persistence window over the state + mailboxes (as RCKangaroo)
    { int maxPersist = 0; int dev = 0; cudaGetDevice(&dev); CK(cudaDeviceGetAttribute(&maxPersist, cudaDevAttrMaxPersistingL2CacheSize, dev));
      size_t win = std::min<size_t>(rc.l2size, (size_t)maxPersist);
      if (maxPersist > 0) { CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, win));
          cudaStreamAttrValue attr{}; attr.accessPolicyWindow.base_ptr = k.L2; attr.accessPolicyWindow.num_bytes = win; attr.accessPolicyWindow.hitRatio = 1.0f;
          attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
          CK(cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr));
          printf("[+] L2 window     : %s B of %s B state persisting\n", human((double)win).c_str(), human((double)rc.l2size).c_str()); } }
    c.X = k.L2; c.Y = k.L2 + NK * 4; c.D = k.dists; c.NK = NK; c.stride = blocks * 256; c.steps = RC_STEP_CNT;   // c.NT stays as set by the caller
    c.HIST = nullptr; c.FLAG = nullptr; c.T2 = nullptr; c.DPN = nullptr; c.REC = nullptr; c.DPX = nullptr;
    rc.dpHost.resize((size_t)RC_MAX_DP * RC_DP_SIZE / 4);
}
// one launch: KernelA (+ inversion SMs) -> KernelB -> KernelC; DPs converted to our 6-word records in hOUT
static void rc_launch(RcState& rc, std::vector<u64>& hOUT, u32& nout, u32& nloop) {
    RcParams& k = rc.k;
    CK(cudaMemset(k.DPs_out, 0, 16)); CK(cudaMemset(k.DPTable, 0, (size_t)k.KangCnt * 4)); CK(cudaMemset(k.LoopedKangs, 0, 8));
    if (rc.cubin) {
        CK(cudaMemset((char*)k.L2 + (size_t)k.KangCnt * 96, 0, rc.invData));
        RcParams a = k; void* args[] = {&a};
        rc_cu(cuLaunchKernel(rc.fnA, k.BlockCnt + rc.invSM, 1, 1, 256, 1, 1, k.KernelA_LDS_Size, 0, args, nullptr), "KernelA launch");
        rc_cu(cuLaunchKernel(rc.fnB, k.BlockCnt, 1, 1, 256, 1, 1, k.KernelB_LDS_Size, 0, args, nullptr), "KernelB launch");
    } else { rc_call_a(&k); rc_call_b(&k); }
    rc_call_c(&k);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    u32 cnt = 0; CK(cudaMemcpy(&cnt, k.DPs_out, 4, cudaMemcpyDeviceToHost));
    if (cnt > RC_MAX_DP) { printf("[!] RC DP buffer overflow: %u DPs in one launch, raise --dp\n", cnt); cnt = RC_MAX_DP; }
    CK(cudaMemcpy(&nloop, k.LoopedKangs, 4, cudaMemcpyDeviceToHost));
    nout = std::min(cnt, MAX_DP_OUT);
    if (cnt) {
        CK(cudaMemcpy(rc.dpHost.data(), k.DPs_out + 4, (size_t)cnt * RC_DP_SIZE, cudaMemcpyDeviceToHost));
        for (u32 i = 0; i < nout; ++i) {        // RC record: x[0..1] (16 B) | d[0..1] | d[2] | kang (u32) | pad
            const u32* p = rc.dpHost.data() + i * (RC_DP_SIZE / 4); u64* o = &hOUT[6 * i];
            memcpy(&o[0], p, 16); memcpy(&o[2], p + 4, 24); o[5] = p[10];
        }
    }
}

// --invtest: device inversion / multiplication of random field elements vs the host
template <class OPS>
__global__ void k_invtest(const u64* in, u64* out, int n, int reps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    u64 r[4]; cp4(r, in + 4 * i);
    for (int k = 0; k < reps; ++k) { u64 t[4]; OPS::inv(t, r); cp4(r, t); }     // reps odd: r = a^-1
    cp4(out + 8 * i, r);
    u64 m[4]; OPS::mul(m, in + 4 * i, in + 4 * ((i + 1) % n)); cp4(out + 8 * i + 4, m);
}

// ============================================================================ host: scalars, points, parsing (as bsgs.cu)
static const u64 GX[4] = {0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL};
static const u64 GY[4] = {0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL};
static const u64 NORD[4] = {0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL};

struct U256 { u64 w[4] = {0,0,0,0}; };
static U256 u_from_hex(std::string h) {
    U256 r; if (h.rfind("0x", 0) == 0 || h.rfind("0X", 0) == 0) h = h.substr(2);
    if (h.empty() || h.size() > 64) die("bad hex number");
    for (char ch : h) {
        int v = isdigit(ch) ? ch - '0' : (ch >= 'a' && ch <= 'f') ? ch - 'a' + 10 : (ch >= 'A' && ch <= 'F') ? ch - 'A' + 10 : -1;
        if (v < 0) die("bad hex digit");
        for (int i = 3; i > 0; --i) r.w[i] = (r.w[i] << 4) | (r.w[i - 1] >> 60);
        r.w[0] = (r.w[0] << 4) | (u64)v;
    }
    return r;
}
static std::string u_hex(const U256& a, bool trim = true) {
    char buf[80]; snprintf(buf, sizeof buf, "%016llx%016llx%016llx%016llx", (unsigned long long)a.w[3], (unsigned long long)a.w[2], (unsigned long long)a.w[1], (unsigned long long)a.w[0]);
    std::string s(buf); if (trim) { size_t p = s.find_first_not_of('0'); s = p == std::string::npos ? "0" : s.substr(p); } return s;
}
static U256 u_add(const U256& a, const U256& b) { U256 r; u128 c = 0; for (int i = 0; i < 4; ++i) { c += (u128)a.w[i] + b.w[i]; r.w[i] = (u64)c; c >>= 64; } return r; }
static U256 u_sub(const U256& a, const U256& b) { U256 r; u64 br = 0; for (int i = 0; i < 4; ++i) { u128 d = (u128)a.w[i] - b.w[i] - br; r.w[i] = (u64)d; br = (u64)(d >> 64) ? 1 : 0; } return r; }
static int u_cmp(const U256& a, const U256& b) { for (int i = 3; i >= 0; --i) if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1; return 0; }
static U256 u_shl1(const U256& a, int n) { U256 r; r.w[n / 64] = 1ULL << (n % 64); return r; }       // 2^n
static U256 u_shr1(const U256& a) { U256 r; for (int i = 0; i < 4; ++i) r.w[i] = (a.w[i] >> 1) | (i < 3 ? a.w[i + 1] << 63 : 0); return r; }
static bool u_neg(const U256& a) { return a.w[3] >> 63; }
static U256 u_negate(const U256& a) { U256 z; return u_sub(z, a); }
static U256 u_modn(const U256& a) { return u_cmp(a, *(const U256*)NORD) >= 0 ? u_sub(a, *(const U256*)NORD) : a; }   // a < 2n
static U256 u_addn(const U256& a, const U256& b) { return u_modn(u_add(a, b)); }
static U256 u_subn(const U256& a, const U256& b) { return u_cmp(a, b) >= 0 ? u_sub(a, b) : u_add(u_sub(a, b), *(const U256*)NORD); }


static Pt hp_G() { Pt p; cp4(p.x, GX); cp4(p.y, GY); return p; }
static Pt hp_neg(const Pt& a) { Pt r = a; p_neg(r.y, a.y); return r; }
static Pt hp_dbl(const Pt& a) {
    u64 x2[4], t[4], lam[4], inv[4], y2[4];
    HostOps::sqr(x2, a.x); p_add(t, x2, x2); p_add(t, t, x2); p_add(y2, a.y, a.y); HostOps::inv(inv, y2); HostOps::mul(lam, t, inv);
    Pt r; HostOps::sqr(r.x, lam); HostOps::sub(r.x, r.x, a.x); HostOps::sub(r.x, r.x, a.x);
    HostOps::sub(t, a.x, r.x); HostOps::mul(r.y, lam, t); HostOps::sub(r.y, r.y, a.y); return r;
}
static Pt hp_add(const Pt& a, const Pt& b) { if (eq4(a.x, b.x)) { if (eq4(a.y, b.y)) return hp_dbl(a); die("point add hit infinity"); } Pt r; aff_add<HostOps>(r, a, b); return r; }
static std::vector<Pt> GT;
static bool hp_mulG(Pt& R, const U256& k) { return mul_table<HostOps>(R, k.w, GT.data(), 256); }
static bool parse_pubkey(const std::string& hex, Pt& Q) {
    std::string h = hex;
    if (h.size() == 66 && (h.substr(0, 2) == "02" || h.substr(0, 2) == "03")) {
        U256 x = u_from_hex(h.substr(2)); cp4(Q.x, x.w);
        u64 y2[4], t[4], seven[4] = {7, 0, 0, 0}; HostOps::sqr(t, Q.x); HostOps::mul(t, t, Q.x); p_add(y2, t, seven);
        u64 e[4] = {0xFFFFFFFFBFFFFF0CULL, PM, PM, 0x3FFFFFFFFFFFFFFFULL}, r[4] = {1, 0, 0, 0}, b[4]; cp4(b, y2);
        for (int i = 0; i < 256; ++i) { if ((e[i >> 6] >> (i & 63)) & 1) HostOps::mul(r, r, b); HostOps::sqr(b, b); }
        HostOps::sqr(t, r); if (!eq4(t, y2)) return false;
        if ((int)(r[0] & 1) != (h[1] == '3')) p_neg(r, r);
        cp4(Q.y, r); return true;
    }
    if (h.size() == 130 && h.substr(0, 2) == "04") {
        U256 x = u_from_hex(h.substr(2, 64)), y = u_from_hex(h.substr(66, 64)); cp4(Q.x, x.w); cp4(Q.y, y.w);
        u64 l[4], r[4], seven[4] = {7, 0, 0, 0}; HostOps::sqr(l, Q.y); HostOps::sqr(r, Q.x); HostOps::mul(r, r, Q.x); p_add(r, r, seven);
        return eq4(l, r);
    }
    return false;
}
static std::string pub_hex(const Pt& P) { U256 x; cp4(x.w, P.x); return std::string((P.y[0] & 1) ? "03" : "02") + u_hex(x, false); }
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::string human(double v) {
    const char* u[] = {"", "K", "M", "G", "T", "P", "E"}; int i = 0;
    while (v >= 1000 && i < 6) { v /= 1000; ++i; } char b[32]; snprintf(b, sizeof b, "%.2f %s", v, u[i]); return b;
}

// ============================================================================ the solver
struct Opts {
    std::string pubkey, start = "0"; int range = 0, dp = 0; bool cpu = false, bench = false; int gpu = 0, cpuThreads = 0;
    int selftest = 0; u64 seed = 0; int report = 10; int maxSeconds = 0; double maxK = 0; std::string out = "kangaroo_found.txt";
    int bps = 0; u64 blocksOpt = 0; int steps = STEPS_DEF; int groups = 24; bool smemPre = false; int minBlocks = 1; int l2 = 1; int unroll = 1; int fused = 1; u64 alias = 0; int invKind = 1; int ilp = 2; bool debug = false, invtest = false, hostStart = false; int startTest = 0; std::string onFind; bool agent = false; int benchRounds = 3; bool check = false; bool dpstream = false; int dplocal = 1; std::string cubin, kernel; std::string rc = "off", rcCubin; int rcInv = -1; std::string species = "both", blind;
};

struct DPRec { u64 x1; u64 d[3]; u32 type; u32 kang; };
static double gLastK = 0, gLastRate = 0;

static int run(Opts o) {
    GT.resize(256); GT[0] = hp_G(); for (int i = 1; i < 256; ++i) GT[i] = hp_dbl(GT[i - 1]);
    std::mt19937_64 rng(o.seed ? o.seed : (u64)std::random_device{}() ^ (u64)time(nullptr));
    auto rnd_below = [&](const U256& m) -> U256 {   // uniform in [0, m) (rejection on the top word)
        int bits = 0; for (int i = 3; i >= 0; --i) if (m.w[i]) { bits = 64 * i + 64 - __builtin_clzll(m.w[i]); break; }
        while (true) { U256 r; for (int i = 0; i < 4; ++i) r.w[i] = rng();
            for (int i = 0; i < 4; ++i) { int lo = i * 64; if (lo >= bits) r.w[i] = 0; else if (lo + 64 > bits) r.w[i] &= (bits - lo == 64) ? ~0ULL : ((1ULL << (bits - lo)) - 1); }
            if (u_cmp(r, m) < 0) return r; }
    };

    // ---- device
    cudaDeviceProp prop{}; int sms = 0;
    if (!o.cpu) { CK(cudaSetDevice(o.gpu)); CK(cudaGetDeviceProperties(&prop, o.gpu)); sms = prop.multiProcessorCount; }
    const int R = o.range; if (R < 20 || R > 170) die("--range must be 20..170 (bits)");
    int dp = o.dp;
    // --rc: RCKangaroo's kernels. auto = turbo cubin on RTX 40 (sm_89) / RTX 50 (sm_120), RC's CUDA kernels elsewhere
    RcState rc; bool rcMode = false; std::string rcCubinPath;
    if (o.rc != "off") {
        if (o.cpu) die("--rc needs a GPU");
        rcMode = true;
        rcCubinPath = o.rc == "cuda" ? "" : rc_find_cubin(o.rcCubin, prop.major, prop.minor, gArgv0);
        if (o.rc == "cubin" && rcCubinPath.empty()) die("--rc cubin: no RCKangaroo cubin for this GPU (RTX 40: rc/kernel_sm89.cubin, RTX 50: rc/kernel_sm120.cubin; or --rc-cubin FILE)");
        rc.cubin = !rcCubinPath.empty();
        rc.invSM = rc.cubin ? (o.rcInv >= 0 ? o.rcInv : std::max(1, prop.major == 12 ? sms / 24 : sms / 32)) : 0;
        if (o.bps > 1) printf("[!] --rc: --bps ignored (one block per SM, RC's layout); --blocks N (>= 12) sets the walk blocks for small tests\n");
        if (o.blocksOpt && o.blocksOpt < 12) die("--rc: --blocks must be >= 12 (RC stop threshold: 0.5*blocks*8 < blocks*8 - 32)");
        o.groups = RC_GROUP; o.steps = RC_STEP_CNT; o.fused = 1; o.ilp = 2; o.smemPre = false; o.alias = 0;
        if (!o.cubin.empty()) die("--rc and --cubin are different things: --rc-cubin FILE picks the RC cubin");
    }
    u64 blocks = o.cpu ? 0 : rcMode ? (o.blocksOpt ? o.blocksOpt : (u64)(sms - rc.invSM)) : (o.blocksOpt ? o.blocksOpt : (u64)sms * (o.bps ? o.bps : 1));
    const int G = o.groups; if (G != 6 && G != 8 && G != 12 && G != 16 && G != 24 && G != 32 && G != 48) die("--groups must be 6, 8, 12, 16, 24, 32 or 48");
    u64 NK = o.cpu ? (u64)(o.cpuThreads ? o.cpuThreads : std::max(1u, std::thread::hardware_concurrency())) * G : blocks * 256 * G;
    // --species: pool workers run ONE species, so a worker alone can never derive the key (only the pool's table can)
    const u64 NT = o.species == "tame" ? NK : o.species == "wild" ? 0 : NK / 3;
    if (!dp) { // default: ~8 DPs per kangaroo over the expected run, within 14..32
        double ops = 1.15 * std::pow(2.0, R / 2.0); double perKang = ops / (double)NK; dp = (int)std::floor(std::log2(std::max(perKang / 8.0, 2.0)));
        dp = std::max(o.cpu ? 4 : 14, std::min(32, dp));
    }
    printf("[+] kangaroos     : %s (%s tame, %s wild)%s\n", human((double)NK).c_str(), human((double)NT).c_str(), human((double)(NK - NT)).c_str(),
           o.cpu ? " on the CPU (test mode)" : (" on " + std::string(prop.name)).c_str());
    const double expOps = 1.15 * std::pow(2.0, R / 2.0);
    {   // RCKangaroo's empirical DP-overhead estimate
        double perKang = expOps / (double)NK / std::ldexp(1.0, dp); if (perKang < 0.001) perKang = 0.001;
        double K = 1.15 + (0.07 + 0.76 / std::sqrt(perKang)) / (1 + 0.30 * perKang);
        printf("[+] range %d bits : expected ops 1.15 * 2^%.1f = 2^%.2f; DP bits %d (~%.1f DPs per kangaroo, est. K with DP overhead %.2f, ~%s DPs = %s of host RAM)\n",
               R, R / 2.0, std::log2(expOps), dp, perKang, K, human(K * std::pow(2.0, R / 2.0) / std::ldexp(1.0, dp)).c_str(),
               human(K * std::pow(2.0, R / 2.0) / std::ldexp(1.0, dp) * 64).c_str());
    }

    if (o.invtest && !o.cpu) {
        const int n = 4096; std::vector<u64> in(n * 4), out(n * 8);
        for (int i = 0; i < n; ++i) { for (int k = 0; k < 4; ++k) in[4 * i + k] = rng(); in[4 * i + 3] &= 0x7FFFFFFFFFFFFFFFULL; if (i < 8) { in[4 * i + 3] = 0; in[4 * i + 2] = 0; } }
        u64 *din, *dout; CK(cudaMalloc(&din, n * 32)); CK(cudaMalloc(&dout, n * 64)); CK(cudaMemcpy(din, in.data(), n * 32, cudaMemcpyHostToDevice));
        for (int kind = 1; kind >= 0; --kind) {
            const char* nm = kind ? "safegcd (modinv30)" : "CUDAMath _ModInv (DivStep62)";
            printf("[invtest] %s: %d device inversions (x1) ...\n", nm, n); fflush(stdout); double ti = now_s();
            if (kind) k_invtest<DevOps><<<(n + 255) / 256, 256>>>(din, dout, n, 1); else k_invtest<DevOpsV><<<(n + 255) / 256, 256>>>(din, dout, n, 1);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            printf("[invtest]    done in %.4f s\n", now_s() - ti); fflush(stdout);
            CK(cudaMemcpy(out.data(), dout, n * 64, cudaMemcpyDeviceToHost));
            int badI = 0, badM = 0;
            for (int i = 0; i < n; ++i) { u64 r[4], m[4]; HostOps::inv_fermat(r, &in[4 * i]); HostOps::mul(m, &in[4 * i], &in[4 * ((i + 1) % n)]);
                if (!eq4(r, &out[8 * i])) ++badI; if (!eq4(m, &out[8 * i + 4])) ++badM; }
            printf("[invtest]    inversions wrong: %d / %d, multiplications wrong: %d / %d\n", badI, n, badM, n); fflush(stdout);
            // throughput: 786k threads x 101 chained inversions
            const int big = 786432, reps = 101; u64 *bin, *bout; CK(cudaMalloc(&bin, (size_t)big * 32)); CK(cudaMalloc(&bout, (size_t)big * 64));
            { std::vector<u64> bi((size_t)big * 4); for (auto& w : bi) w = rng() & 0x7FFFFFFFFFFFFFFFULL; CK(cudaMemcpy(bin, bi.data(), (size_t)big * 32, cudaMemcpyHostToDevice)); }
            ti = now_s();
            if (kind) k_invtest<DevOps><<<big / 256, 256>>>(bin, bout, big, reps); else k_invtest<DevOpsV><<<big / 256, 256>>>(bin, bout, big, reps);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            double el = now_s() - ti;
            printf("[invtest]    throughput: %.2f G inversions/s (%d x %d in %.3f s)\n", (double)big * reps / el / 1e9, big, reps, el); fflush(stdout);
            CK(cudaFree(bin)); CK(cudaFree(bout));
        }
        CK(cudaFree(din)); CK(cudaFree(dout));
    }

    // ---- jump tables (host): 1: [2^(R/2+3), 2^(R/2+4)) even; 2: [2^(R-10), 2^(R-9)); 3: [2^(R-12), 2^(R-11))
    std::vector<u64> J1(JMP_CNT * 8), J2(JMP_CNT * 8), J3(JMP_CNT * 8), JD1(JMP_CNT * 3), JD2(JMP_CNT * 3), JD3(JMP_CNT * 3);
    auto make_jumps = [&](int bits, std::vector<u64>& J, std::vector<u64>& JD) {
        U256 lo = u_shl1(U256(), bits);
        for (int i = 0; i < JMP_CNT; ++i) {
            U256 d = u_add(lo, rnd_below(lo)); d.w[0] &= ~1ULL;
            Pt P; hp_mulG(P, d); cp4(&J[8 * i], P.x); cp4(&J[8 * i + 4], P.y); JD[3 * i] = d.w[0]; JD[3 * i + 1] = d.w[1]; JD[3 * i + 2] = d.w[2];
        }
    };
    std::mt19937_64 jrng(12345); std::swap(rng, jrng);      // fixed jumps (reproducible), then back to the run RNG
    make_jumps(R / 2 + 3, J1, JD1); make_jumps(std::max(R - 10, R / 2 + 8), J2, JD2); make_jumps(std::max(R - 12, R / 2 + 6), J3, JD3);
    std::swap(rng, jrng);

    // ---- the point to solve: P' = Q - start*G + x32*G;  wild base = -P'
    Pt Q; U256 startK = u_from_hex(o.start); U256 secret; bool haveSecret = false;
    if (o.selftest || o.bench) {
        // random key in [start, start + 2^R)
        secret = u_add(startK, rnd_below(u_shl1(U256(), R))); if (iszero4(secret.w)) secret.w[0] = 1;
        hp_mulG(Q, secret); haveSecret = true;
        printf("[selftest] secret key %s (%d-bit range from %s) -> pubkey %s\n", u_hex(secret).c_str(), R, u_hex(startK).c_str(), pub_hex(Q).c_str());
    } else if (!parse_pubkey(o.pubkey, Q)) die("bad --pubkey");
    const U256 x32 = u_shl1(U256(), R - 5);
    Pt Pp = Q;
    if (!iszero4(startK.w)) { Pt S; hp_mulG(S, startK); Pp = hp_add(Pp, hp_neg(S)); }
    { Pt X; hp_mulG(X, x32); Pp = hp_add(Pp, X); }
    // --blind R: wild kangaroos start from -P' + R instead of -P' (R = -r*G, r known only to the pool server; r < 2^(R-4)
    // so the shifted key stays under the wild herd). A wild/wild collision inside this worker then gives key + r (useless
    // here); the server subtracts r from this worker's wild distances and resolves as usual. Local resolution is
    // therefore off (--dplocal 0) with --blind.
    Pt wildBase = hp_neg(Pp);
    if (!o.blind.empty()) { Pt Rb; if (!parse_pubkey(o.blind, Rb)) die("bad --blind point"); wildBase = hp_add(wildBase, Rb); o.dplocal = 0;
                            if (!o.dpstream) die("--blind needs --dpstream (DPs are resolved by the pool server)"); }
    printf("[+] target pubkey : %s   start %s   range 2^%d\n", pub_hex(Q).c_str(), u_hex(startK).c_str(), R);

    // ---- state
    std::vector<u64> hD(NK * 4, 0);
    U256 tameMax = x32; U256 wildMax; { U256 t; for (int i = 0; i < 34; ++i) t = u_add(t, x32); wildMax = t; }
    for (u64 i = 0; i < NK; ++i) {
        U256 d = i < NT ? rnd_below(tameMax) : rnd_below(wildMax); if (i >= NT) d.w[0] &= ~1ULL;
        hD[4 * i] = d.w[0]; hD[4 * i + 1] = d.w[1]; hD[4 * i + 2] = d.w[2];
    }
    Ctx c{}; c.NK = NK; c.NT = NT; c.dpBits = dp; c.steps = o.steps; c.aliasMask = o.alias ? (o.alias - 1) : ~0ULL;
    if (o.alias && (o.alias & (o.alias - 1))) die("--alias must be a power of two");
    if (o.fused && (o.steps & 1)) die("--fused needs an even --steps");
    if (!o.cubin.empty()) {
        if (o.cpu || !o.fused) die("--cubin needs the GPU fused kernel");
        cudaFree(0);   // make sure the runtime's primary context exists before the driver API is used
        cubin_load(o.cubin, o.kernel.empty() ? fused_kernel_name(G, o.minBlocks, o.invKind, o.ilp) : o.kernel, JMP_CNT * 64);
    }
    std::vector<u64> hX, hY, hS, hHIST, hDPX, hOUT; std::vector<u32> hFLAG, hDPN, hLOOPED; std::vector<u64> hT2; std::vector<u16> hREC; u32 hNOUT = 0, hNLOOP = 0;
    const int NB2 = R + 8;   // powers of G for start points: distances < 2^(R+6)
    std::vector<Pt> G2(GT.begin(), GT.begin() + NB2); Pt* dG2 = nullptr;
    if (o.cpu) {
        hX.assign(NK * 4, 0); hY.assign(NK * 4, 0); hS.assign(NK * 4, 0); hHIST.assign(NK * MD_LEN, 0); hDPX.assign(NK * DP_SLOTS * 2, 0);
        hOUT.assign((size_t)MAX_DP_OUT * 6, 0); hFLAG.assign(NK, 0); hDPN.assign(NK, 0); hLOOPED.assign(NK, 0); hREC.assign((size_t)o.steps * NK, 0); hT2.assign(NK / G, 0);
        c.X = hX.data(); c.Y = hY.data(); c.S = hS.data(); c.D = hD.data(); c.HIST = hHIST.data(); c.FLAG = hFLAG.data(); c.REC = hREC.data(); c.T2 = hT2.data(); c.stride = NK / G;
        c.DPX = hDPX.data(); c.DPN = hDPN.data(); c.OUT = hOUT.data(); c.NOUT = &hNOUT; c.LOOPED = hLOOPED.data(); c.NLOOP = &hNLOOP;
        c.J1 = J1.data(); c.J2 = J2.data(); c.J3 = J3.data(); c.JD1 = JD1.data(); c.JD2 = JD2.data(); c.JD3 = JD3.data();
        for (u64 i = 0; i < NK; ++i) {
            u64 s[4] = {hD[4 * i], hD[4 * i + 1], hD[4 * i + 2], 0}; Pt P; mul_table<HostOps>(P, s, G2.data(), NB2);
            if (i >= NT) P = hp_add(P, wildBase);
            cp4(&hX[4 * i], P.x); cp4(&hY[4 * i], P.y);
        }
    } else if (rcMode) {
        rc_setup(rc, c, blocks, dp, J1, J2, J3, JD1, JD2, JD3, rcCubinPath, sms);
        CK(cudaMemcpy(c.D, hD.data(), NK * 32, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dG2, NB2 * sizeof(Pt))); CK(cudaMemcpy(dG2, G2.data(), NB2 * sizeof(Pt), cudaMemcpyHostToDevice));
        if (o.debug) { printf("[debug] k_start (%llu kangaroos) ...\n", (unsigned long long)NK); fflush(stdout); }
        double tks = now_s();
        k_start<<<(unsigned)((NK + 255) / 256), 256>>>(c, dG2, NB2, wildBase, 0); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        if (o.debug) { printf("[debug] k_start done in %.3f s\n", now_s() - tks); fflush(stdout); }
        hOUT.assign((size_t)MAX_DP_OUT * 6, 0);
        size_t freeB, totB; CK(cudaMemGetInfo(&freeB, &totB));
        printf("[+] GPU memory    : %s B used for %s kangaroos (RC layout: %d per thread, %llu walk blocks x 256%s)\n", human((double)(totB - freeB)).c_str(), human((double)NK).c_str(), G, (unsigned long long)blocks,
               rc.cubin ? (" + " + std::to_string(rc.invSM) + " inversion blocks").c_str() : "");
    } else {
        auto dalloc = [&](void** p, size_t bytes) { CK(cudaMalloc(p, bytes)); CK(cudaMemset(*p, 0, bytes)); };
        dalloc((void**)&c.X, NK * 64); c.Y = c.X + NK * 4; c.S = nullptr; dalloc((void**)&c.D, NK * 32); dalloc((void**)&c.T2, blocks * 256 * 8); c.stride = blocks * 256;
        dalloc((void**)&c.HIST, NK * MD_LEN * 8); dalloc((void**)&c.FLAG, NK * 4); dalloc((void**)&c.REC, (size_t)o.steps * NK * 2);
        dalloc((void**)&c.DPX, NK * DP_SLOTS * 16); dalloc((void**)&c.DPN, NK * 4); dalloc((void**)&c.OUT, (size_t)MAX_DP_OUT * 48); dalloc((void**)&c.NOUT, 4);
        dalloc((void**)&c.LOOPED, NK * 4); dalloc((void**)&c.NLOOP, 4);
        u64 *dJ1, *dJ2, *dJ3, *dJD1, *dJD2, *dJD3;
        dalloc((void**)&dJ1, JMP_CNT * 64); dalloc((void**)&dJ2, JMP_CNT * 64); dalloc((void**)&dJ3, JMP_CNT * 64);
        dalloc((void**)&dJD1, JMP_CNT * 24); dalloc((void**)&dJD2, JMP_CNT * 24); dalloc((void**)&dJD3, JMP_CNT * 24);
        CK(cudaMemcpy(dJ1, J1.data(), JMP_CNT * 64, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dJ2, J2.data(), JMP_CNT * 64, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dJ3, J3.data(), JMP_CNT * 64, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dJD1, JD1.data(), JMP_CNT * 24, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dJD2, JD2.data(), JMP_CNT * 24, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dJD3, JD3.data(), JMP_CNT * 24, cudaMemcpyHostToDevice));
        c.J1 = dJ1; c.J2 = dJ2; c.J3 = dJ3; c.JD1 = dJD1; c.JD2 = dJD2; c.JD3 = dJD3;
        if (o.l2) {   // keep the kangaroo state (x, y) resident in L2 (RCKangaroo does the same)
            int maxPersist = 0; CK(cudaDeviceGetAttribute(&maxPersist, cudaDevAttrMaxPersistingL2CacheSize, o.gpu));
            size_t win = std::min<size_t>((size_t)NK * 64, (size_t)maxPersist);
            if (maxPersist > 0) {
                CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, win));
                cudaStreamAttrValue attr{}; attr.accessPolicyWindow.base_ptr = c.X; attr.accessPolicyWindow.num_bytes = win;
                attr.accessPolicyWindow.hitRatio = 1.0f; attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
                CK(cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr));
                printf("[+] L2 window     : %s B of %s B state persisting (device max %s B)\n", human((double)win).c_str(), human((double)NK * 64).c_str(), human((double)maxPersist).c_str());
            } else printf("[+] L2 window     : not supported on this device\n");
        }
        CK(cudaMemcpy(c.D, hD.data(), NK * 32, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dG2, NB2 * sizeof(Pt))); CK(cudaMemcpy(dG2, G2.data(), NB2 * sizeof(Pt), cudaMemcpyHostToDevice));
        if (o.startTest) {   // isolated start-kernel tests on a few threads: 1 = scalar mult only, 2 = base add only, 0 = both
            for (int mode : {1, 2, 0, 4, 5, 3}) {
                static const char* names[] = {"Fermat: mul_table + aff_add", "Fermat: mul_table only", "Fermat: aff_add only", "_ModInv: mul_table full", "_ModInv: mul_table nb=1 (inv of Z=1)", "_ModInv: mul_table nb=2 (one jac_madd)"};
                printf("[starttest] k_start mode %d (%s) on %d threads ...\n", mode, names[mode], o.startTest); fflush(stdout);
                double tt = now_s(); Ctx ct = c; ct.NK = o.startTest;
                k_start<<<(unsigned)((o.startTest + 255) / 256), 256>>>(ct, dG2, NB2, wildBase, mode); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                printf("[starttest]    done in %.3f s\n", now_s() - tt); fflush(stdout);
            }
        }
        if (o.hostStart) {
            if (o.debug) { printf("[debug] start points on the host (%llu kangaroos) ...\n", (unsigned long long)NK); fflush(stdout); }
            double tks = now_s(); std::vector<u64> hx(NK * 4), hy(NK * 4);
            const unsigned nt = std::max(1u, std::thread::hardware_concurrency()); std::vector<std::thread> th;
            for (unsigned t = 0; t < nt; ++t) th.emplace_back([&, t]() {
                for (u64 i = t; i < NK; i += nt) {
                    u64 sc[4] = {hD[4 * i], hD[4 * i + 1], hD[4 * i + 2], 0}; Pt P; mul_table<HostOps>(P, sc, G2.data(), NB2);
                    if (i >= NT) P = hp_add(P, wildBase);
                    cp4(&hx[4 * i], P.x); cp4(&hy[4 * i], P.y);
                }
            });
            for (auto& x : th) x.join();
            CK(cudaMemcpy(c.X, hx.data(), NK * 32, cudaMemcpyHostToDevice)); CK(cudaMemcpy(c.Y, hy.data(), NK * 32, cudaMemcpyHostToDevice));
            if (o.debug) { printf("[debug] host start points done in %.3f s\n", now_s() - tks); fflush(stdout); }
        } else {
            if (o.debug) { printf("[debug] k_start (%llu kangaroos) ...\n", (unsigned long long)NK); fflush(stdout); }
            double tks = now_s();
            k_start<<<(unsigned)((NK + 255) / 256), 256>>>(c, dG2, NB2, wildBase, 0); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            if (o.debug) { printf("[debug] k_start done in %.3f s\n", now_s() - tks); fflush(stdout); }
        }
        hOUT.assign((size_t)MAX_DP_OUT * 6, 0);
        size_t freeB, totB; CK(cudaMemGetInfo(&freeB, &totB));
        printf("[+] GPU memory    : %s B used for %s kangaroos (%d per thread, %llu blocks x 256, prefix products in %s, L2 window %s)\n", human((double)(totB - freeB)).c_str(), human((double)NK).c_str(), G, (unsigned long long)blocks, o.smemPre ? "shared memory" : "local memory", o.l2 ? "on" : "off");
    }

    // ---- host DP table + collision resolution
    std::unordered_map<u64, DPRec> dps; dps.reserve(1 << 20);
    U256 key; bool found = false; u64 dupCount = 0, collisions = 0, dpStreamed = 0, regenTotal = 0;
    std::vector<u32> regen;                      // kangaroos that reported the same DP twice: stuck in a long cycle
    // Recent-DP table: a DP with the same x AND the same distance as one seen before is useless: either the kangaroo is
    // circling in a cycle longer than the in-kernel history catches, or two kangaroos have become clones (same x, same
    // distance: they walk identical paths from then on; measured on the CPU, most streamed DPs were such repeats).
    // The local table (--dplocal 1) caught both as "same x, same type, same distance -> regenerate". Streaming workers
    // have no table, so this open-addressing table of the last 2^24 (x, distance) pairs (~256 MB; ~35 min of DPs at
    // 8K DPs/s) does the same: the kangaroo that repeats gets a fresh random start.
    const u64 RECENT_MASK = (1ULL << 24) - 1;
    std::vector<u64> recentX(RECENT_MASK + 1, 0), recentD(RECENT_MASK + 1, 0);
    auto sgn_mag = [](const u64* d, U256& mag) -> bool { U256 v; v.w[0] = d[0]; v.w[1] = d[1]; v.w[2] = d[2]; v.w[3] = (d[2] >> 63) ? ~0ULL : 0;
        bool neg = u_neg(v); mag = neg ? u_negate(v) : v; return neg; };
    auto to_modn = [&](const u64* d) -> U256 { U256 m; bool neg = sgn_mag(d, m); m = u_modn(m); return neg ? u_subn(U256(), m) : m; };
    auto try_key = [&](const U256& kp) -> bool {   // kp mod n: is kp*G == P' ?  then k = kp - x32 + start
        if (iszero4(kp.w)) return false; Pt T; if (!hp_mulG(T, kp)) return false;
        if (!(eq4(T.x, Pp.x) && eq4(T.y, Pp.y))) return false;
        U256 k = u_addn(u_subn(kp, x32), startK); Pt C; if (!hp_mulG(C, k)) return false;
        if (eq4(C.x, Q.x) && eq4(C.y, Q.y)) { key = k; found = true; return true; }
        return false;
    };
    auto resolve = [&](const DPRec& a, const DPRec& b) -> bool {
        ++collisions;
        U256 da = to_modn(a.d), db = to_modn(b.d);
        if (a.type != b.type) {   // tame t, wild w:  P' = (w - t)G  or  (w + t)G
            const U256& t = a.type == 0 ? da : db; const U256& w = a.type == 0 ? db : da;
            U256 c1 = u_subn(w, t), c2 = u_addn(w, t);
            return try_key(c1) || try_key(c2) || try_key(u_subn(U256(), c1)) || try_key(u_subn(U256(), c2));
        }
        if (a.type == 1) {        // wild/wild: W1 = -W2 -> 2P' = (w1 + w2)G
            auto halve = [](U256 v) { U256 one; one.w[0] = 1; if (v.w[0] & 1) return u_add(u_add(u_shr1(v), u_shr1(*(const U256*)NORD)), one); return u_shr1(v); };
            U256 h = halve(u_addn(da, db)), h2 = halve(u_subn(da, db));
            return try_key(h) || try_key(u_subn(U256(), h)) || try_key(h2) || try_key(u_subn(U256(), h2));
        }
        return false;
    };
    auto process_dps = [&](u32 n) {
        if (n > MAX_DP_OUT) { printf("[!] DP buffer overflow: %u DPs in one launch, raise --dp\n", n); n = MAX_DP_OUT; }
        for (u32 k = 0; k < n && !found; ++k) {
            const u64* r = &hOUT[6 * k]; DPRec rec; rec.x1 = r[1]; rec.d[0] = r[2]; rec.d[1] = r[3]; rec.d[2] = r[4];
            rec.kang = (u32)r[5]; rec.type = rec.kang < NT ? 0 : 1;
            {   // a kangaroo reporting the same DP as its previous one is circling in a cycle longer than MD_LEN
                const u64 slot = (r[0] ^ (r[0] >> 29)) & RECENT_MASK;
                if (recentX[slot] == r[0] && recentD[slot] == r[2]) { ++dupCount; regen.push_back(rec.kang); continue; }   // (not streamed again)
                recentX[slot] = r[0]; recentD[slot] = r[2];
            }
            if (o.dpstream)   // fleet mode: one line per DP for the agent -> console (x 128 bits, distance 192-bit two's complement, type)
                printf("DP %016llx%016llx%016llx%016llx%016llx%u\n", (unsigned long long)r[0], (unsigned long long)r[1],
                       (unsigned long long)r[4], (unsigned long long)r[3], (unsigned long long)r[2], rec.type);
            if (!o.dplocal) { ++dpStreamed; continue; }
            auto it = dps.find(r[0]);
            if (it == dps.end()) { dps.emplace(r[0], rec); continue; }
            const DPRec& old = it->second;
            if (old.x1 != rec.x1) continue;                                         // different point, same 64 bits
            if (old.type == rec.type && old.d[0] == rec.d[0] && old.d[1] == rec.d[1] && old.d[2] == rec.d[2]) { ++dupCount; regen.push_back(rec.kang); continue; }
            if (old.type == 0 && rec.type == 0) continue;                           // tame/tame: nothing
            if (!resolve(old, rec)) printf("[!] collision did not yield the key (x %016llx)\n", (unsigned long long)r[0]);
        }
        if (o.dpstream) fflush(stdout);
    };

    // ---- CPU launch (same three phases as the GPU)
    u64 cpuT = o.cpu ? NK / G : 0;
    auto cpu_launch = [&]() {
        std::vector<std::thread> th;
        for (u64 t = 0; t < cpuT; ++t) th.emplace_back([&, t]() {
            std::vector<u64> pre(4 * 48), preB(4 * 48), accv(4); u64 m = hT2[t];
            if (o.fused) {
                auto run = [&](auto tag) { constexpr int GG = decltype(tag)::value;
                    prefix_init<HostOps, GG, 1>(c, t, cpuT, c.J1, pre.data(), m);
                    for (int s = 0; s + 1 < c.steps; s += 2) {
                        if (o.ilp == 2) { step_fused2<HostOps, GG, 1, 0>(c, t, cpuT, c.J1, s, pre.data(), preB.data(), m, accv.data());
                                          step_fused2<HostOps, GG, 1, 1>(c, t, cpuT, c.J1, s + 1, preB.data(), pre.data(), m, accv.data()); }
                        else { step_fused<HostOps, GG, 1, 0>(c, t, cpuT, c.J1, s, pre.data(), preB.data(), m, accv.data());
                               step_fused<HostOps, GG, 1, 1>(c, t, cpuT, c.J1, s + 1, preB.data(), pre.data(), m, accv.data()); }
                    } };
                if (G == 6) run(std::integral_constant<int, 6>{}); else if (G == 8) run(std::integral_constant<int, 8>{});
                else if (G == 12) run(std::integral_constant<int, 12>{}); else if (G == 16) run(std::integral_constant<int, 16>{});
                else if (G == 32) run(std::integral_constant<int, 32>{}); else run(std::integral_constant<int, 24>{});
                hT2[t] = m; return;
            }
            for (int s = 0; s < c.steps; ++s) {
                switch (G) {
                    case 8: step_group<HostOps, 8, 1>(c, t, cpuT, c.J1, s, pre.data(), m); break;
                    case 12: step_group<HostOps, 12, 1>(c, t, cpuT, c.J1, s, pre.data(), m); break;
                    case 16: step_group<HostOps, 16, 1>(c, t, cpuT, c.J1, s, pre.data(), m); break;
                    case 32: step_group<HostOps, 32, 1>(c, t, cpuT, c.J1, s, pre.data(), m); break;
                    case 48: step_group<HostOps, 48, 1>(c, t, cpuT, c.J1, s, pre.data(), m); break;
                    default: step_group<HostOps, 24, 1>(c, t, cpuT, c.J1, s, pre.data(), m);
                }
            }
            hT2[t] = m;
        });
        for (auto& x : th) x.join(); th.clear();
        hNOUT = 0; hNLOOP = 0;
        for (u64 t = 0; t < cpuT; ++t) th.emplace_back([&, t]() { for (u64 i = t; i < NK; i += cpuT) replay(c, i, c.JD1, c.JD2); });
        for (auto& x : th) x.join();
        for (u32 k = 0; k < hNLOOP; ++k) escape<HostOps>(c, hLOOPED[k]);
    };

    // ---- --check: every kangaroo's point must equal dist*G (tame) or dist*G - P' (wild)
    auto check_all = [&]() -> u64 {
        std::vector<u64> X(NK * 4), Y(NK * 4), D(NK * 4);
        if (o.cpu) { X = hX; Y = hY; D = hD; }
        else { CK(cudaMemcpy(X.data(), c.X, NK * 32, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(Y.data(), c.Y, NK * 32, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(D.data(), c.D, NK * 32, cudaMemcpyDeviceToHost)); }
        u64 bad = 0;
        for (u64 i = 0; i < NK; ++i) {
            U256 mag; bool neg = sgn_mag(&D[4 * i], mag); Pt P; if (!hp_mulG(P, mag)) { ++bad; continue; }
            if (neg) P = hp_neg(P);
            if (i >= NT) P = hp_add(P, wildBase);
            if (!eq4(P.x, &X[4 * i]) || !eq4(P.y, &Y[4 * i])) ++bad;
        }
        return bad;
    };
    if (o.check) { u64 bad = check_all(); printf("[check] start points: %llu / %llu wrong\n", (unsigned long long)bad, (unsigned long long)NK); }

    // ---- run
    const double t0 = now_s(); double tLast = t0; u64 launches = 0, loopsTotal = 0; double opsLast = 0;
    const double maxOps = o.maxK > 0 ? o.maxK * expOps : 0;
    while (!found) {
        u32 nout = 0, nloop = 0;
        if (o.cpu) { cpu_launch(); nout = hNOUT; nloop = hNLOOP; }
        else if (rcMode) {
            const bool dbg = o.debug && launches < 3; double tk = now_s();
            rc_launch(rc, hOUT, nout, nloop);
            if (dbg) { printf("[debug] launch %llu: RC KernelA/B/C done in %.3f s (%u DPs, %u loops)\n", (unsigned long long)launches + 1, now_s() - tk, nout, nloop); fflush(stdout); }
        } else {
            CK(cudaMemset(c.NOUT, 0, 4)); CK(cudaMemset(c.NLOOP, 0, 4));
            const bool dbg = o.debug && launches < 3; double tk = now_s();
            if (dbg) { printf("[debug] launch %llu: k_jump (%llu blocks, %d steps) ...\n", (unsigned long long)launches + 1, (unsigned long long)blocks, o.steps); fflush(stdout); }
            if (o.fused) launch_fused(G, o.minBlocks, o.invKind, o.ilp, c, blocks); else launch_jump(G, o.smemPre, o.minBlocks, o.unroll, c, blocks);
            CK(cudaGetLastError());
            if (dbg) { CK(cudaDeviceSynchronize()); printf("[debug]    k_jump done in %.3f s; k_dist ...\n", now_s() - tk); fflush(stdout); tk = now_s(); }
            k_dist<<<(unsigned)((NK + 255) / 256), 256>>>(c); CK(cudaGetLastError());
            if (dbg) { CK(cudaDeviceSynchronize()); printf("[debug]    k_dist done in %.3f s; k_escape ...\n", now_s() - tk); fflush(stdout); tk = now_s(); }
            k_escape<<<(unsigned)((NK + 255) / 256), 256>>>(c); CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            if (dbg) { printf("[debug]    k_escape done in %.3f s\n", now_s() - tk); fflush(stdout); }
            CK(cudaMemcpy(&nout, c.NOUT, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&nloop, c.NLOOP, 4, cudaMemcpyDeviceToHost));
            if (nout) CK(cudaMemcpy(hOUT.data(), c.OUT, (size_t)std::min(nout, MAX_DP_OUT) * 48, cudaMemcpyDeviceToHost));
        }
        ++launches; loopsTotal += nloop;
        if (o.check) { u64 bad = check_all(); printf("[check] launch %llu: %llu / %llu kangaroos off their distance (%u loops escaped)\n", (unsigned long long)launches, (unsigned long long)bad, (unsigned long long)NK, nloop); if (bad) return 1; }
        process_dps(nout);
        if (!regen.empty() && !found) {
            std::sort(regen.begin(), regen.end()); regen.erase(std::unique(regen.begin(), regen.end()), regen.end());
            std::vector<u64> nd(regen.size() * 3);
            for (size_t k = 0; k < regen.size(); ++k) {
                const u64 i = regen[k]; U256 d = i < NT ? rnd_below(tameMax) : rnd_below(wildMax); if (i >= NT) d.w[0] &= ~1ULL;
                nd[3 * k] = d.w[0]; nd[3 * k + 1] = d.w[1]; nd[3 * k + 2] = d.w[2];
            }
            if (o.cpu) {
                for (size_t k = 0; k < regen.size(); ++k) {
                    const u64 i = regen[k]; hD[4 * i] = nd[3 * k]; hD[4 * i + 1] = nd[3 * k + 1]; hD[4 * i + 2] = nd[3 * k + 2]; hD[4 * i + 3] = 0;
                    u64 sc[4] = {nd[3 * k], nd[3 * k + 1], nd[3 * k + 2], 0}; Pt P; mul_table<HostOps>(P, sc, G2.data(), NB2);
                    if (i >= NT) P = hp_add(P, wildBase);
                    cp4(&hX[4 * i], P.x); cp4(&hY[4 * i], P.y);
                    for (int h = 0; h < MD_LEN; ++h) hHIST[i * MD_LEN + h] = 0;
                    hFLAG[i] = 0; hDPN[i] = 0; hT2[i % cpuT] &= ~(1ULL << (u32)(i / cpuT));
                }
            } else {
                u32* dIdx; u64* dNd; CK(cudaMalloc(&dIdx, regen.size() * 4)); CK(cudaMalloc(&dNd, nd.size() * 8));
                CK(cudaMemcpy(dIdx, regen.data(), regen.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dNd, nd.data(), nd.size() * 8, cudaMemcpyHostToDevice));
                k_regen<<<(unsigned)((regen.size() + 255) / 256), 256>>>(c, dG2, NB2, wildBase, dIdx, dNd, (u32)regen.size()); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                CK(cudaFree(dIdx)); CK(cudaFree(dNd));
            }
            regenTotal += regen.size(); regen.clear();
        }
        const double ops = (double)launches * (double)NK * o.steps, t = now_s();
        if (t - tLast >= o.report || found) {
            const double rate = (ops - opsLast) / (t - tLast);
            printf("[+] %6.0f s | %s jumps/s | 2^%.2f ops = %.2f x expected | %s DPs (%llu dup) | %s loops escaped, %s kangaroos regenerated\n", t - t0, human(rate).c_str(),
                   std::log2(ops), ops / expOps, human((double)(o.dplocal ? dps.size() : dpStreamed)).c_str(), (unsigned long long)dupCount, human((double)loopsTotal).c_str(), human((double)regenTotal).c_str());
            if (o.agent) printf("STAT jumps=%.6e ops=%.6e exp=%.6e dps=%llu loops=%llu dp=%d kangs=%llu regen=%llu\n", rate, ops, expOps,
                                (unsigned long long)(o.dplocal ? dps.size() : dpStreamed), (unsigned long long)loopsTotal, dp, (unsigned long long)NK, (unsigned long long)regenTotal);
            fflush(stdout); tLast = t; opsLast = ops;
        }
        if (o.maxSeconds && t - t0 > o.maxSeconds) break;
        if (maxOps && ops > maxOps) break;
    }
    const double ops = (double)launches * (double)NK * o.steps, el = now_s() - t0;
    gLastK = ops / std::pow(2.0, R / 2.0); gLastRate = ops / el;
    printf("[+] %s jumps in %.1f s (%s jumps/s); K = ops / 2^(range/2) = %.3f; %zu DPs, %llu collisions checked, %s loops escaped\n",
           human(ops).c_str(), el, human(ops / el).c_str(), gLastK, dps.size(), (unsigned long long)collisions, human((double)loopsTotal).c_str());
    if (found) {
        if (haveSecret && u_cmp(key, secret) != 0) { printf("[!] FOUND KEY %s DOES NOT MATCH THE SECRET %s\n", u_hex(key).c_str(), u_hex(secret).c_str()); return 1; }
        if (!o.onFind.empty()) {
            std::string k = u_hex(key, false);
            setenv("CYCLONE_PRIV", k.c_str(), 1); setenv("CYCLONE_ADDR", "", 1); setenv("CYCLONE_TOOL", "kangaroo", 1);
            printf("\n[+] KEY FOUND (hidden: handed to the on-find hook)\n[+] pubkey   : %s\n", pub_hex(Q).c_str()); fflush(stdout);
            int rc = std::system(o.onFind.c_str()); printf("[on-find] exit code %d\n", rc); unsetenv("CYCLONE_PRIV");
        } else {
            printf("\n[+] KEY FOUND: %s\n[+] pubkey   : %s\n", u_hex(key).c_str(), pub_hex(Q).c_str());
            std::ofstream f(o.out, std::ios::app); f << u_hex(key) << " " << pub_hex(Q) << "\n"; printf("[+] saved to %s\n", o.out.c_str());
        }
        return 0;
    }
    printf("[-] stopped without finding the key\n");
    return 2;
}

static void usage() {
    printf(
"kangaroo: Pollard kangaroo (SOTA-style, symmetric walk) for secp256k1 on an NVIDIA GPU\n\n"
"  kangaroo --pubkey 02ABC... --range R [--start HEX] [options]   solve k in [start, start + 2^R)\n"
"  kangaroo --selftest --range R [options]                        random key in that range, solve it (proves the build)\n"
"  kangaroo --bench --range R [--bench-rounds N]                  solve N random keys, print K and jumps/s\n\n"
"  --dp N            distinguished-point bits (default: ~8 DPs per kangaroo; GPU 14..32)\n"
"  --bps N           thread blocks per SM (default 1 = 256 threads x 24 kangaroos per SM); --blocks N sets the block count directly\n"
"  --gpu N, --time S, --max K (stop after K x expected ops), --seed N, --out FILE, --report S\n"
"  --on-find CMD     on a find, run CMD with the key in $CYCLONE_PRIV and do NOT print or save it (worker mode)\n"
"  --agent           machine-readable STAT lines\n"
"  --dpstream        print every distinguished point as a DP line (fleet mode: the console keeps the DP table)\n"
"  --dplocal 0|1     keep the local DP table and resolve collisions here too (default 1; 0 = console only, saves RAM)\n"
"  --steps N         jumps per kangaroo per launch (default 1000)\n"
"  --groups N        kangaroos per thread: 8,12,16,24,32,48 (default 24); --unroll 1|2; --minblocks 1|2; --smem 1 (groups 8)\n"
"  --l2 0|1          L2 persistence window over the kangaroo state (default 1)\n"
"  --fused 0|1       fused step (default 1: no forward pass, steps alternate direction); --inv 1|0: safegcd (default) | DivStep62 inversion\n"
"  --ilp 1|2         fused step: one kangaroo per iteration, or two with interleaved arithmetic (default 2)\n"
"  --alias N         fold the herd onto N kangaroos (speed probe only, wrong results)\n"
"  --hoststart       compute the start points on the host instead of the GPU; --starttest N: isolated start-kernel tests\n"
"  --debug           print and time each kernel of the first launches; --invtest: device field inversion/multiplication vs host\n"
"  --species S       both (default: 1/3 tame, 2/3 wild) | tame | wild: one species only (pool workers; a lone worker then cannot solve)\n"
"  --blind R         pool wild workers: start wild kangaroos from -P' + R (compressed point from the pool); forces --dplocal 0\n"
"  --check           after every launch verify every kangaroo: point == dist*G (+wild base) (slow; CPU mode, or GPU with few kangaroos)\n"
"  --cubin FILE      run the jump kernel from this cubin (hand-written SASS); --kernel NAME overrides the kernel symbol\n"
"  --rc [auto|cubin|cuda]  RCKangaroo's kernels (GPLv3, rc/): auto = turbo SASS cubin on RTX 40/50 (~14.5 G/4090, 19.3 G/5090),\n"
"                    RC's CUDA kernels elsewhere; --rc-cubin FILE picks the cubin; --rc-inv N = inversion SMs (default SMs/32, RTX 50: /24)\n"
"  --cpu [--threads N]  run the same algorithm on the CPU (testing, small ranges)\n");
}

int main(int argc, char** argv) {
    Opts o; gArgv0 = argv[0];
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i]; auto nx = [&]() -> std::string { if (i + 1 >= argc) { usage(); die(("missing value for " + a).c_str()); } return argv[++i]; };
        if (a == "--pubkey") o.pubkey = nx();
        else if (a == "--range") o.range = atoi(nx().c_str());
        else if (a == "--start") o.start = nx();
        else if (a == "--dp") o.dp = atoi(nx().c_str());
        else if (a == "--bps") o.bps = atoi(nx().c_str());
        else if (a == "--blocks") o.blocksOpt = strtoull(nx().c_str(), 0, 0);
        else if (a == "--gpu") o.gpu = atoi(nx().c_str());
        else if (a == "--time") o.maxSeconds = atoi(nx().c_str());
        else if (a == "--max") o.maxK = atof(nx().c_str());
        else if (a == "--seed") o.seed = strtoull(nx().c_str(), 0, 0);
        else if (a == "--out") o.out = nx();
        else if (a == "--report") o.report = atoi(nx().c_str());
        else if (a == "--selftest") o.selftest = 1;
        else if (a == "--bench") o.bench = true;
        else if (a == "--bench-rounds") o.benchRounds = atoi(nx().c_str());
        else if (a == "--cpu") o.cpu = true;
        else if (a == "--threads") o.cpuThreads = atoi(nx().c_str());
        else if (a == "--on-find") o.onFind = nx();
        else if (a == "--agent") o.agent = true;
        else if (a == "--dpstream") o.dpstream = true;
        else if (a == "--dplocal") o.dplocal = atoi(nx().c_str());
        else if (a == "--blind") o.blind = nx();
        else if (a == "--species") { o.species = nx(); if (o.species != "both" && o.species != "tame" && o.species != "wild") die("--species both|tame|wild"); }
        else if (a == "--check") o.check = true;
        else if (a == "--cubin") o.cubin = nx();
        else if (a == "--rc") { o.rc = (i + 1 < argc && argv[i + 1][0] != '-') ? nx() : "auto"; if (o.rc != "auto" && o.rc != "cubin" && o.rc != "cuda" && o.rc != "off") { usage(); return 1; } }
        else if (a == "--rc-cubin") { o.rcCubin = nx(); if (o.rc == "off") o.rc = "cubin"; }
        else if (a == "--rc-inv") o.rcInv = atoi(nx().c_str());
        else if (a == "--kernel") o.kernel = nx();
        else if (a == "--steps") o.steps = atoi(nx().c_str());
        else if (a == "--groups") o.groups = atoi(nx().c_str());
        else if (a == "--smem") o.smemPre = atoi(nx().c_str()) != 0;
        else if (a == "--minblocks") o.minBlocks = atoi(nx().c_str());
        else if (a == "--unroll") o.unroll = atoi(nx().c_str());
        else if (a == "--fused") o.fused = atoi(nx().c_str());
        else if (a == "--inv") o.invKind = atoi(nx().c_str());
        else if (a == "--ilp") o.ilp = atoi(nx().c_str());
        else if (a == "--alias") o.alias = strtoull(nx().c_str(), 0, 0);
        else if (a == "--l2") o.l2 = atoi(nx().c_str());
        else if (a == "--debug") o.debug = true;
        else if (a == "--invtest") o.invtest = true;
        else if (a == "--hoststart") o.hostStart = true;
        else if (a == "--starttest") o.startTest = atoi(nx().c_str());
        else { usage(); return a == "-h" || a == "--help" ? 0 : 1; }
    }
    if (!o.range) { usage(); return 1; }
    if (!o.selftest && !o.bench && o.pubkey.empty()) { usage(); return 1; }
    if (o.bench) {
        double sumK = 0; int done = 0;
        for (int r = 0; r < o.benchRounds; ++r) {
            Opts b = o; b.seed = o.seed ? o.seed + r : 0; b.selftest = 1; b.bench = false;
            printf("\n[bench] round %d / %d\n", r + 1, o.benchRounds);
            int rc = run(b); if (rc != 0) return rc; ++done; sumK += gLastK;
            printf("[bench] rounds %d: average K %.3f (RCKangaroo: 1.15 + DP overhead), last speed %s jumps/s\n", done, sumK / done, human(gLastRate).c_str());
        }
        return 0;
    }
    return run(o);
}
