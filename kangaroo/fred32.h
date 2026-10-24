// fred32.h: secp256k1 field multiply / square with a 32-bit carry-chained reduction.
//   Products: fmul32.h's even/odd scanning (ea_mul256 / ea_sqr256, 64 resp. 36 IMAD.WIDE).
//   Reduction (this file): r = lo + hi * (2^32 + 0x3D1) done with 32-bit mad chains whose addends carry the "+ lo" and
//   "+ hi<<32" terms for free, then one small fold of the top 35 bits. ~50 instructions vs ~70 for the 64-bit
//   UMultSpecial reduction. Result < 2^256 and congruent mod p; may be in [p, 2^256) (lazy, as the original).
//   Carry drop within 2^-188 of 2^256, as the original: never hit in practice.
// Host fallback macros come from fmul32.h, so the same code is testable on the CPU.
#pragma once
#include "fmul32.h"

#define FR_C 0x3D1u

// p16: 512-bit product, little-endian 32-bit words. r8: 256-bit result.
F32_FN void f32_reduce32(uint32_t r[8], const uint32_t p16[16]) {
  F_CARRY_DECL
  const uint32_t C = FR_C;
  const uint32_t* lo = p16; const uint32_t* h = p16 + 8;
  uint32_t u0, u1, u2, u3, u4, u5, u6, u7, u8;        // even chain: h_even * C + lo
  uint32_t v1, v2, v3, v4, v5, v6, v7, v8, v9;        // odd chain:  h_odd * C + (h << 32)
  F_MAD_LO_CC (u0, h[0], C, lo[0]); F_MADC_HI_CC(u1, h[0], C, lo[1]);
  F_MADC_LO_CC(u2, h[2], C, lo[2]); F_MADC_HI_CC(u3, h[2], C, lo[3]);
  F_MADC_LO_CC(u4, h[4], C, lo[4]); F_MADC_HI_CC(u5, h[4], C, lo[5]);
  F_MADC_LO_CC(u6, h[6], C, lo[6]); F_MADC_HI_CC(u7, h[6], C, lo[7]);
  F_ADDC(u8, 0u, 0u);
  F_MAD_LO_CC (v1, h[1], C, h[0]);  F_MADC_HI_CC(v2, h[1], C, h[1]);
  F_MADC_LO_CC(v3, h[3], C, h[2]);  F_MADC_HI_CC(v4, h[3], C, h[3]);
  F_MADC_LO_CC(v5, h[5], C, h[4]);  F_MADC_HI_CC(v6, h[5], C, h[5]);
  F_MADC_LO_CC(v7, h[7], C, h[6]);  F_MADC_HI_CC(v8, h[7], C, h[7]);
  F_ADDC(v9, 0u, 0u);
  // w = u + (v at words 1..9): w0..w9
  uint32_t w1, w2, w3, w4, w5, w6, w7, w8, w9;
  F_ADD_CC (w1, u1, v1); F_ADDC_CC(w2, u2, v2); F_ADDC_CC(w3, u3, v3); F_ADDC_CC(w4, u4, v4);
  F_ADDC_CC(w5, u5, v5); F_ADDC_CC(w6, u6, v6); F_ADDC_CC(w7, u7, v7); F_ADDC_CC(w8, u8, v8);
  F_ADDC(w9, v9, 0u);
  // fold top = (w8, w9) < 2^35:  r = w[0..7] + top * (2^32 + C)
  //   top*C = lo(w8*C) + (hi(w8*C) + lo(w9*C)) << 32 + hi(w9*C) << 64; top << 32 = w8 << 32 + w9 << 64.
  //   Three carry chains. (The earlier two-chain version fed the carry out of word 1 back into word 1 through
  //   MADC_LO: wrong by 2^64 - 2^96 about once per 2^24 products. Found 22 Oct 2026 by the division tool's checks.)
  uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
  F_MAD_LO_CC (r0, w8, C, u0);  F_MADC_HI_CC(r1, w8, C, w1);  F_MADC_HI_CC(r2, w9, C, w2);
  F_ADDC_CC(r3, w3, 0u); F_ADDC_CC(r4, w4, 0u); F_ADDC_CC(r5, w5, 0u); F_ADDC_CC(r6, w6, 0u); F_ADDC(r7, w7, 0u);
  F_MAD_LO_CC (r1, w9, C, r1);
  F_ADDC_CC(r2, r2, 0u); F_ADDC_CC(r3, r3, 0u); F_ADDC_CC(r4, r4, 0u); F_ADDC_CC(r5, r5, 0u); F_ADDC_CC(r6, r6, 0u); F_ADDC(r7, r7, 0u);
  F_ADD_CC (r1, r1, w8); F_ADDC_CC(r2, r2, w9);
  F_ADDC_CC(r3, r3, 0u); F_ADDC_CC(r4, r4, 0u); F_ADDC_CC(r5, r5, 0u); F_ADDC_CC(r6, r6, 0u); F_ADDC(r7, r7, 0u);
  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3; r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

#define FR_SPLIT(D, S) { D[0]=(uint32_t)S[0]; D[1]=(uint32_t)(S[0]>>32); D[2]=(uint32_t)S[1]; D[3]=(uint32_t)(S[1]>>32); \
                         D[4]=(uint32_t)S[2]; D[5]=(uint32_t)(S[2]>>32); D[6]=(uint32_t)S[3]; D[7]=(uint32_t)(S[3]>>32); }
#define FR_JOIN(D, S)  { D[0]=(uint64_t)S[0]|((uint64_t)S[1]<<32); D[1]=(uint64_t)S[2]|((uint64_t)S[3]<<32); \
                         D[2]=(uint64_t)S[4]|((uint64_t)S[5]<<32); D[3]=(uint64_t)S[6]|((uint64_t)S[7]<<32); }

F32_FN void fr_mul(uint64_t* r, const uint64_t* a, const uint64_t* b) {
  uint32_t A[8], B[8], P[16], R[8];
  FR_SPLIT(A, a); FR_SPLIT(B, b);
  ea_mul256(P, A, B);
  f32_reduce32(R, P);
  FR_JOIN(r, R);
}
F32_FN void fr_sqr(uint64_t* r, const uint64_t* a) {
  uint32_t A[8], P[16], R[8];
  FR_SPLIT(A, a);
  ea_sqr256(P, A);
  f32_reduce32(R, P);
  FR_JOIN(r, R);
}
