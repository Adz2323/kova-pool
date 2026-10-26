// kangaroo_core.h: the Kova Pool kangaroo walk for CPUs, phones and browsers. Freestanding C++17: no libc, no
// threads, no allocation, so the same code builds natively (kangaroo_cpu.cpp) and for WebAssembly (kangaroo_wasm.cpp,
// clang --target=wasm32 -nostdlib). Everything that decides WHICH points a kangaroo visits lives here, so every CPU
// and phone worker walks exactly the graph the pool's GPU workers walk (RCKangaroo's turbo kernels, see
// kangaroo/rc/RCGpuCore.cu): same seeded jump tables (kangaroo.cu make_jumps, mt19937_64(12345)), same jump rule
// (index x[0] & 511, direction by y parity), same 2-cycle escape via table 2, same 4..10-cycle detection on the
// distance history with one table-3 jump, same DP criterion (low dp bits of x's top 32-bit word zero).
#pragma once
#include <stdint.h>

#define __host__
#define __device__
#define __forceinline__ inline
#include "modinv30.h"      // safegcd field inversion (libsecp256k1 port): the file the GPU solver uses

typedef uint64_t u64; typedef uint32_t u32; typedef uint16_t u16;

#define JMP_CNT   512
#define JMP_MASK  (JMP_CNT - 1)
#define STEP_CNT  1000           // jumps per kangaroo per launch (RC: STEP_CNT)
#define MD_LEN    10             // distance history (RC: cycles up to MD_LEN caught)
#define GROUP     24             // kangaroos per group = one batch inversion (RC: PNT_GROUP_CNT)
#define INV_FLAG  0x0200
#define JMP2_FLAG 0x0400

// ============================================================================ 64-bit primitives
#if defined(__wasm__)
// no 64x64 -> 128 multiply instruction and no compiler-rt (__multi3) in a freestanding wasm build: four 32-bit products
static inline u64 kc_mul64(u64 a, u64 b, u64* hi) {
    const u64 a0 = (u32)a, a1 = a >> 32, b0 = (u32)b, b1 = b >> 32;
    const u64 p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const u64 mid = (p00 >> 32) + (u32)p01 + (u32)p10;
    *hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return (mid << 32) | (u32)p00;
}
#else
typedef unsigned __int128 kc_u128;
static inline u64 kc_mul64(u64 a, u64 b, u64* hi) { kc_u128 p = (kc_u128)a * b; *hi = (u64)(p >> 64); return (u64)p; }
#endif
static inline u64 kc_addc(u64 a, u64 b, u64* c) { u64 s = a + b; u64 c1 = s < a; s += *c; *c = c1 + (s < *c); return s; }   // *c in: 0/1 carry, out: carry
static inline u64 kc_subb(u64 a, u64 b, u64* br) { u64 d = a - b; u64 b1 = a < b; u64 d2 = d - *br; *br = b1 + (d < *br); return d2; }

// ============================================================================ field arithmetic mod p (secp256k1)
#define P0 0xFFFFFFFEFFFFFC2FULL
#define PM 0xFFFFFFFFFFFFFFFFULL
static inline void cp4(u64* d, const u64* s) { d[0]=s[0]; d[1]=s[1]; d[2]=s[2]; d[3]=s[3]; }
static inline bool iszero4(const u64* a) { return (a[0]|a[1]|a[2]|a[3]) == 0; }
static inline bool eq4(const u64* a, const u64* b) { return a[0]==b[0]&&a[1]==b[1]&&a[2]==b[2]&&a[3]==b[3]; }
static inline bool geq_p(const u64* a) { if (a[3] != PM || a[2] != PM || a[1] != PM) return false; return a[0] >= P0; }
static inline void p_reduce_once(u64* t) {       // t += 2^32 + 977 (i.e. subtract p from a value >= p, mod 2^256)
    u64 c = 0; t[0] = kc_addc(t[0], 0x1000003D1ULL, &c); t[1] = kc_addc(t[1], 0, &c); t[2] = kc_addc(t[2], 0, &c); t[3] = kc_addc(t[3], 0, &c);
}
static inline void p_add(u64* r, const u64* a, const u64* b) {
    u64 t[4], c = 0;
    for (int i = 0; i < 4; ++i) t[i] = kc_addc(a[i], b[i], &c);
    if (c || geq_p(t)) p_reduce_once(t);
    cp4(r, t);
}
static inline void p_sub(u64* r, const u64* a, const u64* b) {
    u64 t[4], br = 0;
    for (int i = 0; i < 4; ++i) t[i] = kc_subb(a[i], b[i], &br);
    if (br) { const u64 p[4] = {P0, PM, PM, PM}; u64 c = 0; for (int i = 0; i < 4; ++i) t[i] = kc_addc(t[i], p[i], &c); }
    cp4(r, t);
}
static inline void p_neg(u64* r, const u64* a) { const u64 z[4] = {0,0,0,0}; p_sub(r, z, a); }
static inline void p_mul(u64* r, const u64* a, const u64* b) {
    u64 t[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 4; ++i) {
        u64 c = 0;
        for (int j = 0; j < 4; ++j) { u64 hi, lo = kc_mul64(a[i], b[j], &hi); u64 cc = 0; lo = kc_addc(lo, t[i + j], &cc); hi += cc; cc = 0; lo = kc_addc(lo, c, &cc); hi += cc; t[i + j] = lo; c = hi; }
        t[i + 4] = c;
    }
    const u64 C = 0x1000003D1ULL; u64 s[4]; u64 c = 0;
    for (int k = 0; k < 4; ++k) { u64 hi, lo = kc_mul64(t[4 + k], C, &hi); u64 cc = 0; lo = kc_addc(lo, t[k], &cc); hi += cc; cc = 0; lo = kc_addc(lo, c, &cc); hi += cc; s[k] = lo; c = hi; }
    { u64 hi, lo = kc_mul64(c, C, &hi); u64 cc = 0; s[0] = kc_addc(s[0], lo, &cc); u64 d = hi + cc;   // d < 2^33
      cc = 0; s[1] = kc_addc(s[1], d, &cc); s[2] = kc_addc(s[2], 0, &cc); s[3] = kc_addc(s[3], 0, &cc);
      if (cc) p_reduce_once(s); }
    if (geq_p(s)) p_reduce_once(s);
    cp4(r, s);
}
static inline void p_sqr(u64* r, const u64* a) { p_mul(r, a, a); }
static inline void p_inv(u64* r, const u64* a) { mi_inv(r, a); }

// ============================================================================ points
struct Pt { u64 x[4], y[4]; };
static const u64 GX[4] = {0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL};
static const u64 GY[4] = {0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL};

static inline void aff_add(Pt& R, const Pt& A, const Pt& B) {
    u64 dx[4], dy[4], inv[4], lam[4], x3[4], y3[4];
    p_sub(dx, B.x, A.x); p_sub(dy, B.y, A.y); p_inv(inv, dx); p_mul(lam, dy, inv);
    p_sqr(x3, lam); p_sub(x3, x3, A.x); p_sub(x3, x3, B.x);
    p_sub(y3, A.x, x3); p_mul(y3, lam, y3); p_sub(y3, y3, A.y);
    cp4(R.x, x3); cp4(R.y, y3);
}
static inline Pt hp_dbl(const Pt& a) {
    u64 x2[4], t[4], lam[4], inv[4], y2[4];
    p_sqr(x2, a.x); p_add(t, x2, x2); p_add(t, t, x2); p_add(y2, a.y, a.y); p_inv(inv, y2); p_mul(lam, t, inv);
    Pt r; p_sqr(r.x, lam); p_sub(r.x, r.x, a.x); p_sub(r.x, r.x, a.x);
    p_sub(t, a.x, r.x); p_mul(r.y, lam, t); p_sub(r.y, r.y, a.y); return r;
}
static inline Pt hp_neg(const Pt& a) { Pt r = a; p_neg(r.y, a.y); return r; }
// a + b for a != +-b (callers never add a point to itself here except through hp_dbl); returns false at infinity
static inline bool hp_add(Pt& r, const Pt& a, const Pt& b) {
    if (eq4(a.x, b.x)) { if (eq4(a.y, b.y)) { r = hp_dbl(a); return true; } return false; }
    aff_add(r, a, b); return true;
}
static inline void jac_madd(u64* X, u64* Y, u64* Z, const u64* x2, const u64* y2) {
    u64 Z1Z1[4], U2[4], S2[4], H[4], HH[4], I[4], J[4], r[4], V[4], t[4];
    p_sqr(Z1Z1, Z); p_mul(U2, x2, Z1Z1); p_mul(S2, y2, Z); p_mul(S2, S2, Z1Z1);
    p_sub(H, U2, X); p_sqr(HH, H); p_add(I, HH, HH); p_add(I, I, I); p_mul(J, H, I);
    p_sub(r, S2, Y); p_add(r, r, r); p_mul(V, X, I);
    p_sqr(t, r); p_sub(t, t, J); p_sub(t, t, V); p_sub(t, t, V);
    u64 Y3[4], Z3[4]; p_sub(Y3, V, t); p_mul(Y3, r, Y3);
    u64 w[4]; p_mul(w, Y, J); p_add(w, w, w); p_sub(Y3, Y3, w);
    p_add(Z3, Z, H); p_sqr(Z3, Z3); p_sub(Z3, Z3, Z1Z1); p_sub(Z3, Z3, HH);
    cp4(X, t); cp4(Y, Y3); cp4(Z, Z3);
}
// R = s * D[0] with D[b] = 2^b D[0]; false for s == 0
static inline bool mul_table(Pt& R, const u64* s, const Pt* D, int nb) {
    u64 X[4], Y[4], Z[4]; bool inf = true;
    for (int b = 0; b < nb; ++b) {
        if (!((s[b >> 6] >> (b & 63)) & 1)) continue;
        if (inf) { cp4(X, D[b].x); cp4(Y, D[b].y); Z[0] = 1; Z[1] = Z[2] = Z[3] = 0; inf = false; }
        else jac_madd(X, Y, Z, D[b].x, D[b].y);
    }
    if (inf) return false;
    u64 zi[4], zi2[4], zi3[4]; p_inv(zi, Z); p_sqr(zi2, zi); p_mul(zi3, zi2, zi);
    p_mul(R.x, X, zi2); p_mul(R.y, Y, zi3); return true;
}
// y from x and parity (compressed public key); false if x is not on the curve
static inline bool decompress(Pt& Q, const u64* x, bool odd) {
    cp4(Q.x, x);
    u64 y2[4], t[4], seven[4] = {7, 0, 0, 0}; p_sqr(t, Q.x); p_mul(t, t, Q.x); p_add(y2, t, seven);
    u64 e[4] = {0xFFFFFFFFBFFFFF0CULL, PM, PM, 0x3FFFFFFFFFFFFFFFULL}, r[4] = {1, 0, 0, 0}, b[4]; cp4(b, y2);
    for (int i = 0; i < 256; ++i) { if ((e[i >> 6] >> (i & 63)) & 1) p_mul(r, r, b); p_sqr(b, b); }
    p_sqr(t, r); if (!eq4(t, y2)) return false;
    if ((bool)(r[0] & 1) != odd) p_neg(r, r);
    cp4(Q.y, r); return true;
}

// ============================================================================ 256-bit integers (as kangaroo.cu)
struct U256 { u64 w[4]; };
static inline U256 u_zero() { U256 r; r.w[0] = r.w[1] = r.w[2] = r.w[3] = 0; return r; }
static inline U256 u_add(const U256& a, const U256& b) { U256 r; u64 c = 0; for (int i = 0; i < 4; ++i) r.w[i] = kc_addc(a.w[i], b.w[i], &c); return r; }
static inline int u_cmp(const U256& a, const U256& b) { for (int i = 3; i >= 0; --i) if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1; return 0; }
static inline U256 u_pow2(int n) { U256 r = u_zero(); r.w[n / 64] = 1ULL << (n % 64); return r; }
static inline int u_bits(const U256& m) { for (int i = 3; i >= 0; --i) if (m.w[i]) return 64 * i + 64 - __builtin_clzll(m.w[i]); return 0; }

// ============================================================================ mt19937_64 (the C++ standard engine, so the tables match std::mt19937_64(12345))
struct MT64 {
    u64 mt[312]; int idx;
    void seed(u64 s) { mt[0] = s; for (int i = 1; i < 312; ++i) mt[i] = 6364136223846793005ULL * (mt[i - 1] ^ (mt[i - 1] >> 62)) + (u64)i; idx = 312; }
    u64 next() {
        if (idx >= 312) {
            for (int i = 0; i < 312; ++i) {
                u64 x = (mt[i] & 0xFFFFFFFF80000000ULL) | (mt[(i + 1) % 312] & 0x7FFFFFFFULL);
                u64 xa = x >> 1; if (x & 1) xa ^= 0xB5026F5AA96619E9ULL;
                mt[i] = mt[(i + 156) % 312] ^ xa;
            }
            idx = 0;
        }
        u64 y = mt[idx++];
        y ^= (y >> 29) & 0x5555555555555555ULL; y ^= (y << 17) & 0x71D67FFFEDA60000ULL; y ^= (y << 37) & 0xFFF7EEE000000000ULL; y ^= y >> 43;
        return y;
    }
};
// uniform in [0, m): 4 words from the engine (word 0 first), masked to m's bit length, rejected if >= m (kangaroo.cu rnd_below)
static inline U256 rnd_below(MT64& rng, const U256& m) {
    const int bits = u_bits(m);
    for (;;) {
        U256 r; for (int i = 0; i < 4; ++i) r.w[i] = rng.next();
        for (int i = 0; i < 4; ++i) { int lo = i * 64; if (lo >= bits) r.w[i] = 0; else if (lo + 64 > bits) r.w[i] &= (bits - lo == 64) ? ~0ULL : ((1ULL << (bits - lo)) - 1); }
        if (u_cmp(r, m) < 0) return r;
    }
}

// ============================================================================ tables and the walk
struct Tables {
    Pt GT[256];                              // 2^b G
    u64 J1[JMP_CNT * 8], J2[JMP_CNT * 8], J3[JMP_CNT * 8];   // x(4) y(4) per jump
    u64 JD1[JMP_CNT * 3], JD2[JMP_CNT * 3], JD3[JMP_CNT * 3]; // distances, 3 words per jump
    u32 dpMask;
};
static inline bool t_mulG(const Tables& T, Pt& R, const U256& k) { return mul_table(R, k.w, T.GT, 256); }
static inline void make_jumps(Tables& T, MT64& rng, int bits, u64* J, u64* JD) {
    const U256 lo = u_pow2(bits);
    for (int i = 0; i < JMP_CNT; ++i) {
        U256 d = u_add(lo, rnd_below(rng, lo)); d.w[0] &= ~1ULL;
        Pt P; t_mulG(T, P, d); cp4(&J[8 * i], P.x); cp4(&J[8 * i + 4], P.y); JD[3 * i] = d.w[0]; JD[3 * i + 1] = d.w[1]; JD[3 * i + 2] = d.w[2];
    }
}
// G table, the three jump tables (identical on every worker: seed 12345) and the DP mask for a range / dp
static inline void tables_init(Tables& T, int R, int dp) {
    cp4(T.GT[0].x, GX); cp4(T.GT[0].y, GY); for (int i = 1; i < 256; ++i) T.GT[i] = hp_dbl(T.GT[i - 1]);
    MT64 rng; rng.seed(12345);
    const int b2 = R - 10 > R / 2 + 8 ? R - 10 : R / 2 + 8, b3 = R - 12 > R / 2 + 6 ? R - 12 : R / 2 + 6;
    make_jumps(T, rng, R / 2 + 3, T.J1, T.JD1); make_jumps(T, rng, b2, T.J2, T.JD2); make_jumps(T, rng, b3, T.J3, T.JD3);
    T.dpMask = dp >= 32 ? 0xFFFFFFFFu : ((1u << dp) - 1);
}

// 192-bit signed distances (3 words)
static inline void d_add(u64* d, const u64* j) { u64 c = 0; d[0] = kc_addc(d[0], j[0], &c); d[1] = kc_addc(d[1], j[1], &c); d[2] = kc_addc(d[2], j[2], &c); }
static inline void d_sub(u64* d, const u64* j) { u64 b = 0; d[0] = kc_subb(d[0], j[0], &b); d[1] = kc_subb(d[1], j[1], &b); d[2] = kc_subb(d[2], j[2], &b); }

struct DPOut { u64 x0, x1, d[3]; u32 kang; };

// one group of GROUP kangaroos: RC's KernelA/KernelB/KernelC rules
struct Walker {
    u64 X[GROUP][4], Y[GROUP][4], D[GROUP][4];       // point, distance (3 words used)
    u64 hist[GROUP][MD_LEN]; u32 hpos[GROUP];        // RC KernelB: ring of the last MD_LEN distance low words
    u64 lastX0[GROUP], lastD0[GROUP];                // the previous DP of each kangaroo (same twice = stuck in a long cycle)
    u32 l1s2;                                         // bit g: kangaroo g jumps from table 2 next (2-cycle escape)
    u64 base;                                         // global index of kangaroo 0 of this group
    u64 loops;
    DPOut* out; u32 nout, outMax;                     // DP records of the current launch
    u32 stuck[GROUP]; u32 nstuck;                     // kangaroos that reported the same DP twice (restart them)
};
static inline void escape3(const Tables& T, Walker& w, int g) {   // RC KernelC: one jump from table 3 from the current point
    u64* x = w.X[g]; u64* y = w.Y[g];
    const u32 j = (u32)x[0] & JMP_MASK; const u64* jx = T.J3 + 8 * j; u64 jy[4]; cp4(jy, jx + 4);
    const bool odd = y[0] & 1; if (odd) p_neg(jy, jy);
    u64 dx[4], inv[4], dy[4], lam[4], nx[4], ny[4];
    p_sub(dx, x, jx); p_inv(inv, dx); p_sub(dy, y, jy); p_mul(lam, dy, inv);
    p_sqr(nx, lam); p_sub(nx, nx, jx); p_sub(nx, nx, x);
    p_sub(ny, x, nx); p_mul(ny, lam, ny); p_sub(ny, ny, y);
    cp4(x, nx); cp4(y, ny);
    if (odd) d_sub(w.D[g], T.JD3 + 3 * j); else d_add(w.D[g], T.JD3 + 3 * j);
    for (int h = 0; h < MD_LEN; ++h) w.hist[g][h] = 0;
    w.hpos[g] = 0; w.l1s2 &= ~(1u << g); ++w.loops;
}
static inline void walker_reset(Walker& w, int g) { for (int h = 0; h < MD_LEN; ++h) w.hist[g][h] = 0; w.hpos[g] = 0; w.l1s2 &= ~(1u << g); w.lastX0[g] = w.lastD0[g] = 0; }
// `steps` jumps for every kangaroo of the group; DPs go to w.out (w.nout), stuck kangaroos to w.stuck
static inline void walker_run(const Tables& T, Walker& w, int steps) {
    u64 pre[GROUP][4], acc[4], dx[4];
    w.nout = 0; w.nstuck = 0;
    for (int step = 0; step < steps; ++step) {
        for (int g = 0; g < GROUP; ++g) {                                  // forward: prefix products of (x_g - jx_g)
            const u64* x = w.X[g]; const u32 j = (u32)x[0] & JMP_MASK;
            const u64* jx = ((w.l1s2 >> g) & 1) ? T.J2 + 8 * j : T.J1 + 8 * j;
            p_sub(dx, x, jx);
            if (g == 0) cp4(acc, dx); else p_mul(acc, acc, dx);
            cp4(pre[g], acc);
        }
        u64 inv[4]; p_inv(inv, acc);
        for (int g = GROUP - 1; g >= 0; --g) {                             // backward: one jump per kangaroo
            u64* x = w.X[g]; u64* y = w.Y[g];
            const u32 j = (u32)x[0] & JMP_MASK; const bool t2 = (w.l1s2 >> g) & 1;
            const u64* jx = t2 ? T.J2 + 8 * j : T.J1 + 8 * j; u64 jy[4]; cp4(jy, jx + 4);
            u64 ik[4];
            if (g > 0) { p_mul(ik, inv, pre[g - 1]); p_sub(dx, x, jx); p_mul(inv, inv, dx); }
            else cp4(ik, inv);
            const bool odd = y[0] & 1;                                     // RC: inv_flag = y odd -> jump -J
            const u32 rec = j | (odd ? INV_FLAG : 0);
            if (odd) p_neg(jy, jy);
            u64 dy[4], lam[4], nx[4], ny[4];
            p_sub(dy, y, jy); p_mul(lam, dy, ik);
            p_sqr(nx, lam); p_sub(nx, nx, jx); p_sub(nx, nx, x);
            p_sub(ny, x, nx); p_mul(ny, lam, ny); p_sub(ny, ny, y);
            cp4(x, nx); cp4(y, ny);
            if (!t2) {                                                     // 2-cycle: next jump = same index, opposite direction
                const u32 jn = ((u32)nx[0] & JMP_MASK) | ((ny[0] & 1) ? 0 : INV_FLAG);
                if (rec == jn) w.l1s2 |= 1u << g;
            } else w.l1s2 &= ~(1u << g);
            const u64* jd = (t2 ? T.JD2 : T.JD1) + 3 * j;                  // distance (RC KernelB): +-d from table 1 or 2
            if (odd) d_sub(w.D[g], jd); else d_add(w.D[g], jd);
            u64* tab = w.hist[g]; const u32 it = w.hpos[g]; const u64 d0 = w.D[g][0];   // loops of 4, 6, 8, 10 (RC ProcessJumpDistance)
            const bool looped = tab[(it + MD_LEN - 4) % MD_LEN] == d0 || tab[(it + MD_LEN - 6) % MD_LEN] == d0 || tab[(it + MD_LEN - 8) % MD_LEN] == d0 || tab[it] == d0;
            tab[it] = d0; w.hpos[g] = (it + 1) % MD_LEN;
            if ((((u32)(nx[3] >> 32)) & T.dpMask) == 0) {
                if (w.lastX0[g] == nx[0] && w.lastD0[g] == d0) { if (w.nstuck < GROUP) w.stuck[w.nstuck++] = (u32)g; }
                else {
                    w.lastX0[g] = nx[0]; w.lastD0[g] = d0;
                    if (w.nout < w.outMax) { DPOut& o = w.out[w.nout++]; o.x0 = nx[0]; o.x1 = nx[1]; o.d[0] = w.D[g][0]; o.d[1] = w.D[g][1]; o.d[2] = w.D[g][2]; o.kang = (u32)(w.base + g); }
                }
            }
            if (looped && d0) escape3(T, w, g);
        }
    }
}
