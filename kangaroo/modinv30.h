// modinv30.h: modular inversion mod p (secp256k1 field) by Bernstein-Yang "safegcd" divsteps in 30-bit batches,
// variable time. A port of libsecp256k1's modinv32_var (src/modinv32_impl.h, MIT licence, (c) 2020 Peter Dettman),
// written host+device so the same code runs under --cpu. 32-bit integer arithmetic only: a good fit for the GPU.
//
//   Representation: signed 30-bit limbs, 9 limbs (270 bits). Modulus p = 2^256 - 2^32 - 977.
#pragma once
#include <stdint.h>

#define MI_M30 ((int32_t)0x3FFFFFFF)

struct mi_s30 { int32_t v[9]; };

// p in signed30: p = 2^256 - 0x1000003D1 = 2^256 - 4*2^30 - 0x3D1
__host__ __device__ __forceinline__ void mi_modulus(mi_s30& m) {
    m.v[0] = -0x3D1; m.v[1] = -4; m.v[2] = 0; m.v[3] = 0; m.v[4] = 0; m.v[5] = 0; m.v[6] = 0; m.v[7] = 0; m.v[8] = 65536;
}
#define MI_MODULUS_INV30 0x2DDACACFu  // p^-1 mod 2^30

__host__ __device__ __forceinline__ int mi_ctz32(uint32_t x) {
#ifdef __CUDA_ARCH__
    return __ffs(x) - 1;
#else
    return __builtin_ctz(x);
#endif
}

// 30 divsteps on the low limbs; returns the new eta and the transition matrix t = {u, v, q, r}
__host__ __device__ __forceinline__ int32_t mi_divsteps_30_var(int32_t eta, uint32_t f0, uint32_t g0, int32_t t[4]) {
    uint32_t u = 1, v = 0, q = 0, r = 1;
    uint32_t f = f0, g = g0, m;
    uint16_t w;
    int i = 30, limit, zeros;
    for (;;) {
        zeros = mi_ctz32(g | (0xFFFFFFFFu << i));
        g >>= zeros; u <<= zeros; v <<= zeros; eta -= zeros; i -= zeros;
        if (i == 0) break;
        if (eta < 0) {
            uint32_t tmp;
            eta = -eta;
            tmp = f; f = g; g = (uint32_t)(-(int32_t)tmp);
            tmp = u; u = q; q = (uint32_t)(-(int32_t)tmp);
            tmp = v; v = r; r = (uint32_t)(-(int32_t)tmp);
            limit = ((int)eta + 1) > i ? i : ((int)eta + 1);
            // f*g*(f*f-2) is -g/f only mod 64 (libsecp256k1's 2020 formula): cancel at most 6 bits. With the 8-bit mask
            // (& 255, which belongs to the later inv256-table version) the transition matrix exceeds its bound and about
            // 1 inverse in 10^7 came out off by 2^256 - p. Found 22-24 Oct 2026 by the division tool's walk checks.
            m = (0xFFFFFFFFu >> (32 - limit)) & 63u;
            w = (uint16_t)((f * g * (f * f - 2)) & m);
        } else {
            limit = ((int)eta + 1) > i ? i : ((int)eta + 1);
            m = (0xFFFFFFFFu >> (32 - limit)) & 15u;
            w = (uint16_t)(f + (((f + 1) & 4) << 1));
            w = (uint16_t)((-(int32_t)w * (int32_t)g) & m);
        }
        g += f * w; q += u * w; r += v * w;
    }
    t[0] = (int32_t)u; t[1] = (int32_t)v; t[2] = (int32_t)q; t[3] = (int32_t)r;
    return eta;
}

// (d, e) = t * (d, e) / 2^30 mod p, keeping d, e in (-2p, p)
__host__ __device__ __forceinline__ void mi_update_de_30(mi_s30& d, mi_s30& e, const int32_t t[4], const mi_s30& mod) {
    const int32_t u = t[0], v = t[1], q = t[2], r = t[3];
    int32_t di, ei, md, me, sd, se;
    int64_t cd, ce;
    sd = d.v[8] >> 31; se = e.v[8] >> 31;
    md = (u & sd) + (v & se); me = (q & sd) + (r & se);
    di = d.v[0]; ei = e.v[0];
    cd = (int64_t)u * di + (int64_t)v * ei;
    ce = (int64_t)q * di + (int64_t)r * ei;
    md -= (int32_t)((MI_MODULUS_INV30 * (uint32_t)cd + (uint32_t)md) & MI_M30);
    me -= (int32_t)((MI_MODULUS_INV30 * (uint32_t)ce + (uint32_t)me) & MI_M30);
    cd += (int64_t)mod.v[0] * md; ce += (int64_t)mod.v[0] * me;
    cd >>= 30; ce >>= 30;
    #pragma unroll
    for (int i = 1; i < 9; ++i) {
        di = d.v[i]; ei = e.v[i];
        cd += (int64_t)u * di + (int64_t)v * ei;
        ce += (int64_t)q * di + (int64_t)r * ei;
        cd += (int64_t)mod.v[i] * md; ce += (int64_t)mod.v[i] * me;
        d.v[i - 1] = (int32_t)cd & MI_M30; cd >>= 30;
        e.v[i - 1] = (int32_t)ce & MI_M30; ce >>= 30;
    }
    d.v[8] = (int32_t)cd; e.v[8] = (int32_t)ce;
}

// (f, g) = t * (f, g) / 2^30 (exact), over len limbs
__host__ __device__ __forceinline__ void mi_update_fg_30_var(int len, mi_s30& f, mi_s30& g, const int32_t t[4]) {
    const int32_t u = t[0], v = t[1], q = t[2], r = t[3];
    int32_t fi, gi; int64_t cf, cg;
    fi = f.v[0]; gi = g.v[0];
    cf = (int64_t)u * fi + (int64_t)v * gi;
    cg = (int64_t)q * fi + (int64_t)r * gi;
    cf >>= 30; cg >>= 30;
    for (int i = 1; i < len; ++i) {
        fi = f.v[i]; gi = g.v[i];
        cf += (int64_t)u * fi + (int64_t)v * gi;
        cg += (int64_t)q * fi + (int64_t)r * gi;
        f.v[i - 1] = (int32_t)cf & MI_M30; cf >>= 30;
        g.v[i - 1] = (int32_t)cg & MI_M30; cg >>= 30;
    }
    f.v[len - 1] = (int32_t)cf; g.v[len - 1] = (int32_t)cg;
}

// r in (-2p, p) -> [0, p), negated if sign < 0
__host__ __device__ __forceinline__ void mi_normalize_30(mi_s30& r, int32_t sign, const mi_s30& mod) {
    int32_t r0 = r.v[0], r1 = r.v[1], r2 = r.v[2], r3 = r.v[3], r4 = r.v[4], r5 = r.v[5], r6 = r.v[6], r7 = r.v[7], r8 = r.v[8];
    int32_t cond_add, cond_negate;
    cond_add = r8 >> 31;
    r0 += mod.v[0] & cond_add; r1 += mod.v[1] & cond_add; r2 += mod.v[2] & cond_add; r3 += mod.v[3] & cond_add; r4 += mod.v[4] & cond_add;
    r5 += mod.v[5] & cond_add; r6 += mod.v[6] & cond_add; r7 += mod.v[7] & cond_add; r8 += mod.v[8] & cond_add;
    cond_negate = sign >> 31;
    r0 = (r0 ^ cond_negate) - cond_negate; r1 = (r1 ^ cond_negate) - cond_negate; r2 = (r2 ^ cond_negate) - cond_negate;
    r3 = (r3 ^ cond_negate) - cond_negate; r4 = (r4 ^ cond_negate) - cond_negate; r5 = (r5 ^ cond_negate) - cond_negate;
    r6 = (r6 ^ cond_negate) - cond_negate; r7 = (r7 ^ cond_negate) - cond_negate; r8 = (r8 ^ cond_negate) - cond_negate;
    r1 += r0 >> 30; r0 &= MI_M30; r2 += r1 >> 30; r1 &= MI_M30; r3 += r2 >> 30; r2 &= MI_M30; r4 += r3 >> 30; r3 &= MI_M30;
    r5 += r4 >> 30; r4 &= MI_M30; r6 += r5 >> 30; r5 &= MI_M30; r7 += r6 >> 30; r6 &= MI_M30; r8 += r7 >> 30; r7 &= MI_M30;
    cond_add = r8 >> 31;
    r0 += mod.v[0] & cond_add; r1 += mod.v[1] & cond_add; r2 += mod.v[2] & cond_add; r3 += mod.v[3] & cond_add; r4 += mod.v[4] & cond_add;
    r5 += mod.v[5] & cond_add; r6 += mod.v[6] & cond_add; r7 += mod.v[7] & cond_add; r8 += mod.v[8] & cond_add;
    r1 += r0 >> 30; r0 &= MI_M30; r2 += r1 >> 30; r1 &= MI_M30; r3 += r2 >> 30; r2 &= MI_M30; r4 += r3 >> 30; r3 &= MI_M30;
    r5 += r4 >> 30; r4 &= MI_M30; r6 += r5 >> 30; r5 &= MI_M30; r7 += r6 >> 30; r6 &= MI_M30; r8 += r7 >> 30; r7 &= MI_M30;
    r.v[0] = r0; r.v[1] = r1; r.v[2] = r2; r.v[3] = r3; r.v[4] = r4; r.v[5] = r5; r.v[6] = r6; r.v[7] = r7; r.v[8] = r8;
}

// 256-bit (4 x u64, < p) <-> signed30
__host__ __device__ __forceinline__ void mi_from_u64(mi_s30& r, const uint64_t a[4]) {
    r.v[0] = (int32_t)(a[0] & MI_M30);
    r.v[1] = (int32_t)((a[0] >> 30) & MI_M30);
    r.v[2] = (int32_t)(((a[0] >> 60) | (a[1] << 4)) & MI_M30);
    r.v[3] = (int32_t)((a[1] >> 26) & MI_M30);
    r.v[4] = (int32_t)(((a[1] >> 56) | (a[2] << 8)) & MI_M30);
    r.v[5] = (int32_t)((a[2] >> 22) & MI_M30);
    r.v[6] = (int32_t)(((a[2] >> 52) | (a[3] << 12)) & MI_M30);
    r.v[7] = (int32_t)((a[3] >> 18) & MI_M30);
    r.v[8] = (int32_t)(a[3] >> 48);
}
__host__ __device__ __forceinline__ void mi_to_u64(uint64_t r[4], const mi_s30& a) {
    r[0] = (uint64_t)(uint32_t)a.v[0] | ((uint64_t)(uint32_t)a.v[1] << 30) | ((uint64_t)(uint32_t)a.v[2] << 60);
    r[1] = ((uint64_t)(uint32_t)a.v[2] >> 4) | ((uint64_t)(uint32_t)a.v[3] << 26) | ((uint64_t)(uint32_t)a.v[4] << 56);
    r[2] = ((uint64_t)(uint32_t)a.v[4] >> 8) | ((uint64_t)(uint32_t)a.v[5] << 22) | ((uint64_t)(uint32_t)a.v[6] << 52);
    r[3] = ((uint64_t)(uint32_t)a.v[6] >> 12) | ((uint64_t)(uint32_t)a.v[7] << 18) | ((uint64_t)(uint32_t)a.v[8] << 48);
}

// r = a^-1 mod p (a < p, a != 0). Variable time.
__host__ __device__ __forceinline__ void mi_inv(uint64_t r[4], const uint64_t a[4]) {
    mi_s30 mod; mi_modulus(mod);
    mi_s30 d = {{0, 0, 0, 0, 0, 0, 0, 0, 0}}, e = {{1, 0, 0, 0, 0, 0, 0, 0, 0}}, f = mod, g;
    mi_from_u64(g, a);
    int len = 9; int32_t eta = -1; int32_t t[4];
    for (;;) {
        eta = mi_divsteps_30_var(eta, (uint32_t)f.v[0], (uint32_t)g.v[0], t);
        mi_update_de_30(d, e, t, mod);
        mi_update_fg_30_var(len, f, g, t);
        if (g.v[0] == 0) {
            int32_t cond = 0;
            for (int j = 1; j < len; ++j) cond |= g.v[j];
            if (cond == 0) break;
        }
        {
            int32_t fn = f.v[len - 1], gn = g.v[len - 1];
            int32_t cond = ((int32_t)len - 2) >> 31;
            cond |= fn ^ (fn >> 31);
            cond |= gn ^ (gn >> 31);
            if (cond == 0) { f.v[len - 2] |= (int32_t)((uint32_t)fn << 30); g.v[len - 2] |= (int32_t)((uint32_t)gn << 30); --len; }
        }
    }
    mi_normalize_30(d, f.v[len - 1], mod);
    mi_to_u64(r, d);
}
