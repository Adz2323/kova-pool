// fmul32.h - secp256k1 256x256 -> 512-bit product with 32-bit limbs and PTX carry chains (turbo branch).
// Generated, fully unrolled. The same code runs on the host (carry flag emulated) so it can be unit-tested on a CPU.
// Only the PRODUCT is new: the reduction mod p stays the original VanitySearch code, so results are bit-identical.
#pragma once
#include <stdint.h>

#if defined(__CUDA_ARCH__)
#define F32_FN __device__ __forceinline__
#define F_MULLO(r,a,b)      asm volatile("mul.lo.u32 %0,%1,%2;"        : "=r"(r) : "r"(a), "r"(b))
#define F_MAD_LO_CC(r,a,b,c)  asm volatile("mad.lo.cc.u32 %0,%1,%2,%3;"  : "=r"(r) : "r"(a), "r"(b), "r"(c))
#define F_MADC_LO_CC(r,a,b,c) asm volatile("madc.lo.cc.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c))
#define F_MAD_HI_CC(r,a,b,c)  asm volatile("mad.hi.cc.u32 %0,%1,%2,%3;"  : "=r"(r) : "r"(a), "r"(b), "r"(c))
#define F_MADC_HI_CC(r,a,b,c) asm volatile("madc.hi.cc.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c))
#define F_MADC_HI(r,a,b,c)    asm volatile("madc.hi.u32 %0,%1,%2,%3;"    : "=r"(r) : "r"(a), "r"(b), "r"(c))
#define F_MAD_HI(r,a,b,c)     asm volatile("mad.hi.u32 %0,%1,%2,%3;"     : "=r"(r) : "r"(a), "r"(b), "r"(c))
#define F_ADDC(r,a,b)         asm volatile("addc.u32 %0,%1,%2;"          : "=r"(r) : "r"(a), "r"(b))
#define F_SHL1(r,hi,lo)       (r) = __funnelshift_l((lo), (hi), 1)
#define F_CARRY_DECL
#else
#define F32_FN static inline
#define F_CARRY_DECL uint32_t f_cf = 0;
#define F_MULLO(r,a,b)        do{ (r) = (uint32_t)((uint64_t)(a)*(b)); }while(0)
#define F_MAD_LO_CC(r,a,b,c)  do{ uint64_t s_=(uint64_t)(uint32_t)((uint64_t)(a)*(b))+(c); (r)=(uint32_t)s_; f_cf=(uint32_t)(s_>>32);}while(0)
#define F_MADC_LO_CC(r,a,b,c) do{ uint64_t s_=(uint64_t)(uint32_t)((uint64_t)(a)*(b))+(c)+f_cf; (r)=(uint32_t)s_; f_cf=(uint32_t)(s_>>32);}while(0)
#define F_MAD_HI_CC(r,a,b,c)  do{ uint64_t s_=(((uint64_t)(a)*(b))>>32)+(c); (r)=(uint32_t)s_; f_cf=(uint32_t)(s_>>32);}while(0)
#define F_MADC_HI_CC(r,a,b,c) do{ uint64_t s_=(((uint64_t)(a)*(b))>>32)+(c)+f_cf; (r)=(uint32_t)s_; f_cf=(uint32_t)(s_>>32);}while(0)
#define F_MADC_HI(r,a,b,c)    do{ uint64_t s_=(((uint64_t)(a)*(b))>>32)+(c)+f_cf; (r)=(uint32_t)s_; }while(0)
#define F_MAD_HI(r,a,b,c)     do{ uint64_t s_=(((uint64_t)(a)*(b))>>32)+(c); (r)=(uint32_t)s_; }while(0)
#define F_ADDC(r,a,b)         do{ uint64_t s_=(uint64_t)(a)+(b)+f_cf; (r)=(uint32_t)s_; }while(0)
#define F_SHL1(r,hi,lo)       (r) = (uint32_t)(((hi)<<1)|((lo)>>31))
#endif

// r[16] = a[8] * b[8]   (little-endian 32-bit limbs)
F32_FN void f32_mul256(uint32_t r[16], const uint32_t a[8], const uint32_t b[8]) {
  F_CARRY_DECL
  F_MULLO(r[0], a[0], b[0]);
  F_MULLO(r[1], a[1], b[0]);
  F_MULLO(r[2], a[2], b[0]);
  F_MULLO(r[3], a[3], b[0]);
  F_MULLO(r[4], a[4], b[0]);
  F_MULLO(r[5], a[5], b[0]);
  F_MULLO(r[6], a[6], b[0]);
  F_MULLO(r[7], a[7], b[0]);
  r[8] = 0;
  F_MAD_HI_CC(r[1], a[0], b[0], r[1]);
  F_MADC_HI_CC(r[2], a[1], b[0], r[2]);
  F_MADC_HI_CC(r[3], a[2], b[0], r[3]);
  F_MADC_HI_CC(r[4], a[3], b[0], r[4]);
  F_MADC_HI_CC(r[5], a[4], b[0], r[5]);
  F_MADC_HI_CC(r[6], a[5], b[0], r[6]);
  F_MADC_HI_CC(r[7], a[6], b[0], r[7]);
  F_MADC_HI(r[8], a[7], b[0], r[8]);
  r[9] = 0;
  F_MAD_LO_CC(r[1], a[0], b[1], r[1]);
  F_MADC_LO_CC(r[2], a[1], b[1], r[2]);
  F_MADC_LO_CC(r[3], a[2], b[1], r[3]);
  F_MADC_LO_CC(r[4], a[3], b[1], r[4]);
  F_MADC_LO_CC(r[5], a[4], b[1], r[5]);
  F_MADC_LO_CC(r[6], a[5], b[1], r[6]);
  F_MADC_LO_CC(r[7], a[6], b[1], r[7]);
  F_MADC_LO_CC(r[8], a[7], b[1], r[8]);
  F_ADDC(r[9], r[9], 0u);
  F_MAD_HI_CC(r[2], a[0], b[1], r[2]);
  F_MADC_HI_CC(r[3], a[1], b[1], r[3]);
  F_MADC_HI_CC(r[4], a[2], b[1], r[4]);
  F_MADC_HI_CC(r[5], a[3], b[1], r[5]);
  F_MADC_HI_CC(r[6], a[4], b[1], r[6]);
  F_MADC_HI_CC(r[7], a[5], b[1], r[7]);
  F_MADC_HI_CC(r[8], a[6], b[1], r[8]);
  F_MADC_HI(r[9], a[7], b[1], r[9]);
  r[10] = 0;
  F_MAD_LO_CC(r[2], a[0], b[2], r[2]);
  F_MADC_LO_CC(r[3], a[1], b[2], r[3]);
  F_MADC_LO_CC(r[4], a[2], b[2], r[4]);
  F_MADC_LO_CC(r[5], a[3], b[2], r[5]);
  F_MADC_LO_CC(r[6], a[4], b[2], r[6]);
  F_MADC_LO_CC(r[7], a[5], b[2], r[7]);
  F_MADC_LO_CC(r[8], a[6], b[2], r[8]);
  F_MADC_LO_CC(r[9], a[7], b[2], r[9]);
  F_ADDC(r[10], r[10], 0u);
  F_MAD_HI_CC(r[3], a[0], b[2], r[3]);
  F_MADC_HI_CC(r[4], a[1], b[2], r[4]);
  F_MADC_HI_CC(r[5], a[2], b[2], r[5]);
  F_MADC_HI_CC(r[6], a[3], b[2], r[6]);
  F_MADC_HI_CC(r[7], a[4], b[2], r[7]);
  F_MADC_HI_CC(r[8], a[5], b[2], r[8]);
  F_MADC_HI_CC(r[9], a[6], b[2], r[9]);
  F_MADC_HI(r[10], a[7], b[2], r[10]);
  r[11] = 0;
  F_MAD_LO_CC(r[3], a[0], b[3], r[3]);
  F_MADC_LO_CC(r[4], a[1], b[3], r[4]);
  F_MADC_LO_CC(r[5], a[2], b[3], r[5]);
  F_MADC_LO_CC(r[6], a[3], b[3], r[6]);
  F_MADC_LO_CC(r[7], a[4], b[3], r[7]);
  F_MADC_LO_CC(r[8], a[5], b[3], r[8]);
  F_MADC_LO_CC(r[9], a[6], b[3], r[9]);
  F_MADC_LO_CC(r[10], a[7], b[3], r[10]);
  F_ADDC(r[11], r[11], 0u);
  F_MAD_HI_CC(r[4], a[0], b[3], r[4]);
  F_MADC_HI_CC(r[5], a[1], b[3], r[5]);
  F_MADC_HI_CC(r[6], a[2], b[3], r[6]);
  F_MADC_HI_CC(r[7], a[3], b[3], r[7]);
  F_MADC_HI_CC(r[8], a[4], b[3], r[8]);
  F_MADC_HI_CC(r[9], a[5], b[3], r[9]);
  F_MADC_HI_CC(r[10], a[6], b[3], r[10]);
  F_MADC_HI(r[11], a[7], b[3], r[11]);
  r[12] = 0;
  F_MAD_LO_CC(r[4], a[0], b[4], r[4]);
  F_MADC_LO_CC(r[5], a[1], b[4], r[5]);
  F_MADC_LO_CC(r[6], a[2], b[4], r[6]);
  F_MADC_LO_CC(r[7], a[3], b[4], r[7]);
  F_MADC_LO_CC(r[8], a[4], b[4], r[8]);
  F_MADC_LO_CC(r[9], a[5], b[4], r[9]);
  F_MADC_LO_CC(r[10], a[6], b[4], r[10]);
  F_MADC_LO_CC(r[11], a[7], b[4], r[11]);
  F_ADDC(r[12], r[12], 0u);
  F_MAD_HI_CC(r[5], a[0], b[4], r[5]);
  F_MADC_HI_CC(r[6], a[1], b[4], r[6]);
  F_MADC_HI_CC(r[7], a[2], b[4], r[7]);
  F_MADC_HI_CC(r[8], a[3], b[4], r[8]);
  F_MADC_HI_CC(r[9], a[4], b[4], r[9]);
  F_MADC_HI_CC(r[10], a[5], b[4], r[10]);
  F_MADC_HI_CC(r[11], a[6], b[4], r[11]);
  F_MADC_HI(r[12], a[7], b[4], r[12]);
  r[13] = 0;
  F_MAD_LO_CC(r[5], a[0], b[5], r[5]);
  F_MADC_LO_CC(r[6], a[1], b[5], r[6]);
  F_MADC_LO_CC(r[7], a[2], b[5], r[7]);
  F_MADC_LO_CC(r[8], a[3], b[5], r[8]);
  F_MADC_LO_CC(r[9], a[4], b[5], r[9]);
  F_MADC_LO_CC(r[10], a[5], b[5], r[10]);
  F_MADC_LO_CC(r[11], a[6], b[5], r[11]);
  F_MADC_LO_CC(r[12], a[7], b[5], r[12]);
  F_ADDC(r[13], r[13], 0u);
  F_MAD_HI_CC(r[6], a[0], b[5], r[6]);
  F_MADC_HI_CC(r[7], a[1], b[5], r[7]);
  F_MADC_HI_CC(r[8], a[2], b[5], r[8]);
  F_MADC_HI_CC(r[9], a[3], b[5], r[9]);
  F_MADC_HI_CC(r[10], a[4], b[5], r[10]);
  F_MADC_HI_CC(r[11], a[5], b[5], r[11]);
  F_MADC_HI_CC(r[12], a[6], b[5], r[12]);
  F_MADC_HI(r[13], a[7], b[5], r[13]);
  r[14] = 0;
  F_MAD_LO_CC(r[6], a[0], b[6], r[6]);
  F_MADC_LO_CC(r[7], a[1], b[6], r[7]);
  F_MADC_LO_CC(r[8], a[2], b[6], r[8]);
  F_MADC_LO_CC(r[9], a[3], b[6], r[9]);
  F_MADC_LO_CC(r[10], a[4], b[6], r[10]);
  F_MADC_LO_CC(r[11], a[5], b[6], r[11]);
  F_MADC_LO_CC(r[12], a[6], b[6], r[12]);
  F_MADC_LO_CC(r[13], a[7], b[6], r[13]);
  F_ADDC(r[14], r[14], 0u);
  F_MAD_HI_CC(r[7], a[0], b[6], r[7]);
  F_MADC_HI_CC(r[8], a[1], b[6], r[8]);
  F_MADC_HI_CC(r[9], a[2], b[6], r[9]);
  F_MADC_HI_CC(r[10], a[3], b[6], r[10]);
  F_MADC_HI_CC(r[11], a[4], b[6], r[11]);
  F_MADC_HI_CC(r[12], a[5], b[6], r[12]);
  F_MADC_HI_CC(r[13], a[6], b[6], r[13]);
  F_MADC_HI(r[14], a[7], b[6], r[14]);
  r[15] = 0;
  F_MAD_LO_CC(r[7], a[0], b[7], r[7]);
  F_MADC_LO_CC(r[8], a[1], b[7], r[8]);
  F_MADC_LO_CC(r[9], a[2], b[7], r[9]);
  F_MADC_LO_CC(r[10], a[3], b[7], r[10]);
  F_MADC_LO_CC(r[11], a[4], b[7], r[11]);
  F_MADC_LO_CC(r[12], a[5], b[7], r[12]);
  F_MADC_LO_CC(r[13], a[6], b[7], r[13]);
  F_MADC_LO_CC(r[14], a[7], b[7], r[14]);
  F_ADDC(r[15], r[15], 0u);
  F_MAD_HI_CC(r[8], a[0], b[7], r[8]);
  F_MADC_HI_CC(r[9], a[1], b[7], r[9]);
  F_MADC_HI_CC(r[10], a[2], b[7], r[10]);
  F_MADC_HI_CC(r[11], a[3], b[7], r[11]);
  F_MADC_HI_CC(r[12], a[4], b[7], r[12]);
  F_MADC_HI_CC(r[13], a[5], b[7], r[13]);
  F_MADC_HI_CC(r[14], a[6], b[7], r[14]);
  F_MADC_HI(r[15], a[7], b[7], r[15]);
}

// r[16] = a[8]^2: cross products once, doubled, plus the squares
F32_FN void f32_sqr256(uint32_t r[16], const uint32_t a[8]) {
  F_CARRY_DECL
  r[0] = 0; r[15] = 0;
  F_MULLO(r[1], a[0], a[1]);
  F_MULLO(r[2], a[0], a[2]);
  F_MULLO(r[3], a[0], a[3]);
  F_MULLO(r[4], a[0], a[4]);
  F_MULLO(r[5], a[0], a[5]);
  F_MULLO(r[6], a[0], a[6]);
  F_MULLO(r[7], a[0], a[7]);
  r[8] = 0;
  F_MAD_HI_CC(r[2], a[0], a[1], r[2]);
  F_MADC_HI_CC(r[3], a[0], a[2], r[3]);
  F_MADC_HI_CC(r[4], a[0], a[3], r[4]);
  F_MADC_HI_CC(r[5], a[0], a[4], r[5]);
  F_MADC_HI_CC(r[6], a[0], a[5], r[6]);
  F_MADC_HI_CC(r[7], a[0], a[6], r[7]);
  F_MADC_HI(r[8], a[0], a[7], r[8]);
  r[9] = 0;
  F_MAD_LO_CC(r[3], a[1], a[2], r[3]);
  F_MADC_LO_CC(r[4], a[1], a[3], r[4]);
  F_MADC_LO_CC(r[5], a[1], a[4], r[5]);
  F_MADC_LO_CC(r[6], a[1], a[5], r[6]);
  F_MADC_LO_CC(r[7], a[1], a[6], r[7]);
  F_MADC_LO_CC(r[8], a[1], a[7], r[8]);
  F_ADDC(r[9], r[9], 0u);
  F_MAD_HI_CC(r[4], a[1], a[2], r[4]);
  F_MADC_HI_CC(r[5], a[1], a[3], r[5]);
  F_MADC_HI_CC(r[6], a[1], a[4], r[6]);
  F_MADC_HI_CC(r[7], a[1], a[5], r[7]);
  F_MADC_HI_CC(r[8], a[1], a[6], r[8]);
  F_MADC_HI(r[9], a[1], a[7], r[9]);
  r[10] = 0;
  F_MAD_LO_CC(r[5], a[2], a[3], r[5]);
  F_MADC_LO_CC(r[6], a[2], a[4], r[6]);
  F_MADC_LO_CC(r[7], a[2], a[5], r[7]);
  F_MADC_LO_CC(r[8], a[2], a[6], r[8]);
  F_MADC_LO_CC(r[9], a[2], a[7], r[9]);
  F_ADDC(r[10], r[10], 0u);
  F_MAD_HI_CC(r[6], a[2], a[3], r[6]);
  F_MADC_HI_CC(r[7], a[2], a[4], r[7]);
  F_MADC_HI_CC(r[8], a[2], a[5], r[8]);
  F_MADC_HI_CC(r[9], a[2], a[6], r[9]);
  F_MADC_HI(r[10], a[2], a[7], r[10]);
  r[11] = 0;
  F_MAD_LO_CC(r[7], a[3], a[4], r[7]);
  F_MADC_LO_CC(r[8], a[3], a[5], r[8]);
  F_MADC_LO_CC(r[9], a[3], a[6], r[9]);
  F_MADC_LO_CC(r[10], a[3], a[7], r[10]);
  F_ADDC(r[11], r[11], 0u);
  F_MAD_HI_CC(r[8], a[3], a[4], r[8]);
  F_MADC_HI_CC(r[9], a[3], a[5], r[9]);
  F_MADC_HI_CC(r[10], a[3], a[6], r[10]);
  F_MADC_HI(r[11], a[3], a[7], r[11]);
  r[12] = 0;
  F_MAD_LO_CC(r[9], a[4], a[5], r[9]);
  F_MADC_LO_CC(r[10], a[4], a[6], r[10]);
  F_MADC_LO_CC(r[11], a[4], a[7], r[11]);
  F_ADDC(r[12], r[12], 0u);
  F_MAD_HI_CC(r[10], a[4], a[5], r[10]);
  F_MADC_HI_CC(r[11], a[4], a[6], r[11]);
  F_MADC_HI(r[12], a[4], a[7], r[12]);
  r[13] = 0;
  F_MAD_LO_CC(r[11], a[5], a[6], r[11]);
  F_MADC_LO_CC(r[12], a[5], a[7], r[12]);
  F_ADDC(r[13], r[13], 0u);
  F_MAD_HI_CC(r[12], a[5], a[6], r[12]);
  F_MADC_HI(r[13], a[5], a[7], r[13]);
  r[14] = 0;
  F_MAD_LO_CC(r[13], a[6], a[7], r[13]);
  F_ADDC(r[14], r[14], 0u);
  F_MAD_HI(r[14], a[6], a[7], r[14]);
  r[15] = r[14] >> 31;
  F_SHL1(r[14], r[14], r[13]);
  F_SHL1(r[13], r[13], r[12]);
  F_SHL1(r[12], r[12], r[11]);
  F_SHL1(r[11], r[11], r[10]);
  F_SHL1(r[10], r[10], r[9]);
  F_SHL1(r[9], r[9], r[8]);
  F_SHL1(r[8], r[8], r[7]);
  F_SHL1(r[7], r[7], r[6]);
  F_SHL1(r[6], r[6], r[5]);
  F_SHL1(r[5], r[5], r[4]);
  F_SHL1(r[4], r[4], r[3]);
  F_SHL1(r[3], r[3], r[2]);
  F_SHL1(r[2], r[2], r[1]);
  F_SHL1(r[1], r[1], r[0]);
  r[0] = 0;
  F_MAD_LO_CC(r[0], a[0], a[0], r[0]);
  F_MADC_HI_CC(r[1], a[0], a[0], r[1]);
  F_MADC_LO_CC(r[2], a[1], a[1], r[2]);
  F_MADC_HI_CC(r[3], a[1], a[1], r[3]);
  F_MADC_LO_CC(r[4], a[2], a[2], r[4]);
  F_MADC_HI_CC(r[5], a[2], a[2], r[5]);
  F_MADC_LO_CC(r[6], a[3], a[3], r[6]);
  F_MADC_HI_CC(r[7], a[3], a[3], r[7]);
  F_MADC_LO_CC(r[8], a[4], a[4], r[8]);
  F_MADC_HI_CC(r[9], a[4], a[4], r[9]);
  F_MADC_LO_CC(r[10], a[5], a[5], r[10]);
  F_MADC_HI_CC(r[11], a[5], a[5], r[11]);
  F_MADC_LO_CC(r[12], a[6], a[6], r[12]);
  F_MADC_HI_CC(r[13], a[6], a[6], r[13]);
  F_MADC_LO_CC(r[14], a[7], a[7], r[14]);
  F_MADC_HI(r[15], a[7], a[7], r[15]);
}

// ---------------------------------------------------------------------------------------------------------------
// EOMUL: aligned even/odd product scanning (the sppark idea, laid out for register pairs). The lo and hi halves of
// one 32x32 product land in adjacent limbs of one carry chain, and products are split by the parity of their limb
// position, so every pair sits on an even register pair. ptxas then emits IMAD.WIDE.U32(.X) with predicate carries
// (the instruction pattern RetiredCoder hand-writes in RCKangaroo's asm) and no register shuffles.
// sm_89 SASS (incl. moves): product 109 vs 201 for the original 64-bit code; square 97 vs ~150.
// Verified on CPU against a reference for 3M random/edge inputs each.
#if defined(__CUDA_ARCH__)
#define F_ADD_CC(r,a,b)  asm volatile("add.cc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b))
#define F_ADDC_CC(r,a,b) asm volatile("addc.cc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b))
#define F_MULHI(r,a,b)   asm volatile("mul.hi.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b))
#else
#define F_ADD_CC(r,a,b)  do{ uint64_t s_=(uint64_t)(a)+(b); (r)=(uint32_t)s_; f_cf=(uint32_t)(s_>>32);}while(0)
#define F_ADDC_CC(r,a,b) do{ uint64_t s_=(uint64_t)(a)+(b)+f_cf; (r)=(uint32_t)s_; f_cf=(uint32_t)(s_>>32);}while(0)
#define F_MULHI(r,a,b)   do{ (r)=(uint32_t)(((uint64_t)(a)*(b))>>32);}while(0)
#endif
// aligned even/odd product scanning: products whose limb position p=i+j is even go to A[p], odd ones to B[p-1],
// so every lo/hi pair sits on an even register pair (IMAD.WIDE needs that) and no shuffling moves are needed.
F32_FN void ea_mul256(uint32_t r[16], const uint32_t a[8], const uint32_t b[8]) {
  F_CARRY_DECL
  uint32_t A[18], B[18];
  _Pragma("unroll") for (int k = 0; k < 18; ++k) { A[k] = 0; B[k] = 0; }
  F_MAD_LO_CC(A[0], a[0], b[0], A[0]);
  F_MADC_HI_CC(A[1], a[0], b[0], A[1]);
  F_MADC_LO_CC(A[2], a[2], b[0], A[2]);
  F_MADC_HI_CC(A[3], a[2], b[0], A[3]);
  F_MADC_LO_CC(A[4], a[4], b[0], A[4]);
  F_MADC_HI_CC(A[5], a[4], b[0], A[5]);
  F_MADC_LO_CC(A[6], a[6], b[0], A[6]);
  F_MADC_HI_CC(A[7], a[6], b[0], A[7]);
  F_ADDC(A[8], A[8], 0u);
  F_MAD_LO_CC(B[0], a[1], b[0], B[0]);
  F_MADC_HI_CC(B[1], a[1], b[0], B[1]);
  F_MADC_LO_CC(B[2], a[3], b[0], B[2]);
  F_MADC_HI_CC(B[3], a[3], b[0], B[3]);
  F_MADC_LO_CC(B[4], a[5], b[0], B[4]);
  F_MADC_HI_CC(B[5], a[5], b[0], B[5]);
  F_MADC_LO_CC(B[6], a[7], b[0], B[6]);
  F_MADC_HI_CC(B[7], a[7], b[0], B[7]);
  F_ADDC(B[8], B[8], 0u);
  F_MAD_LO_CC(A[2], a[1], b[1], A[2]);
  F_MADC_HI_CC(A[3], a[1], b[1], A[3]);
  F_MADC_LO_CC(A[4], a[3], b[1], A[4]);
  F_MADC_HI_CC(A[5], a[3], b[1], A[5]);
  F_MADC_LO_CC(A[6], a[5], b[1], A[6]);
  F_MADC_HI_CC(A[7], a[5], b[1], A[7]);
  F_MADC_LO_CC(A[8], a[7], b[1], A[8]);
  F_MADC_HI_CC(A[9], a[7], b[1], A[9]);
  F_ADDC(A[10], A[10], 0u);
  F_MAD_LO_CC(B[0], a[0], b[1], B[0]);
  F_MADC_HI_CC(B[1], a[0], b[1], B[1]);
  F_MADC_LO_CC(B[2], a[2], b[1], B[2]);
  F_MADC_HI_CC(B[3], a[2], b[1], B[3]);
  F_MADC_LO_CC(B[4], a[4], b[1], B[4]);
  F_MADC_HI_CC(B[5], a[4], b[1], B[5]);
  F_MADC_LO_CC(B[6], a[6], b[1], B[6]);
  F_MADC_HI_CC(B[7], a[6], b[1], B[7]);
  F_ADDC(B[8], B[8], 0u);
  F_MAD_LO_CC(A[2], a[0], b[2], A[2]);
  F_MADC_HI_CC(A[3], a[0], b[2], A[3]);
  F_MADC_LO_CC(A[4], a[2], b[2], A[4]);
  F_MADC_HI_CC(A[5], a[2], b[2], A[5]);
  F_MADC_LO_CC(A[6], a[4], b[2], A[6]);
  F_MADC_HI_CC(A[7], a[4], b[2], A[7]);
  F_MADC_LO_CC(A[8], a[6], b[2], A[8]);
  F_MADC_HI_CC(A[9], a[6], b[2], A[9]);
  F_ADDC(A[10], A[10], 0u);
  F_MAD_LO_CC(B[2], a[1], b[2], B[2]);
  F_MADC_HI_CC(B[3], a[1], b[2], B[3]);
  F_MADC_LO_CC(B[4], a[3], b[2], B[4]);
  F_MADC_HI_CC(B[5], a[3], b[2], B[5]);
  F_MADC_LO_CC(B[6], a[5], b[2], B[6]);
  F_MADC_HI_CC(B[7], a[5], b[2], B[7]);
  F_MADC_LO_CC(B[8], a[7], b[2], B[8]);
  F_MADC_HI_CC(B[9], a[7], b[2], B[9]);
  F_ADDC(B[10], B[10], 0u);
  F_MAD_LO_CC(A[4], a[1], b[3], A[4]);
  F_MADC_HI_CC(A[5], a[1], b[3], A[5]);
  F_MADC_LO_CC(A[6], a[3], b[3], A[6]);
  F_MADC_HI_CC(A[7], a[3], b[3], A[7]);
  F_MADC_LO_CC(A[8], a[5], b[3], A[8]);
  F_MADC_HI_CC(A[9], a[5], b[3], A[9]);
  F_MADC_LO_CC(A[10], a[7], b[3], A[10]);
  F_MADC_HI_CC(A[11], a[7], b[3], A[11]);
  F_ADDC(A[12], A[12], 0u);
  F_MAD_LO_CC(B[2], a[0], b[3], B[2]);
  F_MADC_HI_CC(B[3], a[0], b[3], B[3]);
  F_MADC_LO_CC(B[4], a[2], b[3], B[4]);
  F_MADC_HI_CC(B[5], a[2], b[3], B[5]);
  F_MADC_LO_CC(B[6], a[4], b[3], B[6]);
  F_MADC_HI_CC(B[7], a[4], b[3], B[7]);
  F_MADC_LO_CC(B[8], a[6], b[3], B[8]);
  F_MADC_HI_CC(B[9], a[6], b[3], B[9]);
  F_ADDC(B[10], B[10], 0u);
  F_MAD_LO_CC(A[4], a[0], b[4], A[4]);
  F_MADC_HI_CC(A[5], a[0], b[4], A[5]);
  F_MADC_LO_CC(A[6], a[2], b[4], A[6]);
  F_MADC_HI_CC(A[7], a[2], b[4], A[7]);
  F_MADC_LO_CC(A[8], a[4], b[4], A[8]);
  F_MADC_HI_CC(A[9], a[4], b[4], A[9]);
  F_MADC_LO_CC(A[10], a[6], b[4], A[10]);
  F_MADC_HI_CC(A[11], a[6], b[4], A[11]);
  F_ADDC(A[12], A[12], 0u);
  F_MAD_LO_CC(B[4], a[1], b[4], B[4]);
  F_MADC_HI_CC(B[5], a[1], b[4], B[5]);
  F_MADC_LO_CC(B[6], a[3], b[4], B[6]);
  F_MADC_HI_CC(B[7], a[3], b[4], B[7]);
  F_MADC_LO_CC(B[8], a[5], b[4], B[8]);
  F_MADC_HI_CC(B[9], a[5], b[4], B[9]);
  F_MADC_LO_CC(B[10], a[7], b[4], B[10]);
  F_MADC_HI_CC(B[11], a[7], b[4], B[11]);
  F_ADDC(B[12], B[12], 0u);
  F_MAD_LO_CC(A[6], a[1], b[5], A[6]);
  F_MADC_HI_CC(A[7], a[1], b[5], A[7]);
  F_MADC_LO_CC(A[8], a[3], b[5], A[8]);
  F_MADC_HI_CC(A[9], a[3], b[5], A[9]);
  F_MADC_LO_CC(A[10], a[5], b[5], A[10]);
  F_MADC_HI_CC(A[11], a[5], b[5], A[11]);
  F_MADC_LO_CC(A[12], a[7], b[5], A[12]);
  F_MADC_HI_CC(A[13], a[7], b[5], A[13]);
  F_ADDC(A[14], A[14], 0u);
  F_MAD_LO_CC(B[4], a[0], b[5], B[4]);
  F_MADC_HI_CC(B[5], a[0], b[5], B[5]);
  F_MADC_LO_CC(B[6], a[2], b[5], B[6]);
  F_MADC_HI_CC(B[7], a[2], b[5], B[7]);
  F_MADC_LO_CC(B[8], a[4], b[5], B[8]);
  F_MADC_HI_CC(B[9], a[4], b[5], B[9]);
  F_MADC_LO_CC(B[10], a[6], b[5], B[10]);
  F_MADC_HI_CC(B[11], a[6], b[5], B[11]);
  F_ADDC(B[12], B[12], 0u);
  F_MAD_LO_CC(A[6], a[0], b[6], A[6]);
  F_MADC_HI_CC(A[7], a[0], b[6], A[7]);
  F_MADC_LO_CC(A[8], a[2], b[6], A[8]);
  F_MADC_HI_CC(A[9], a[2], b[6], A[9]);
  F_MADC_LO_CC(A[10], a[4], b[6], A[10]);
  F_MADC_HI_CC(A[11], a[4], b[6], A[11]);
  F_MADC_LO_CC(A[12], a[6], b[6], A[12]);
  F_MADC_HI_CC(A[13], a[6], b[6], A[13]);
  F_ADDC(A[14], A[14], 0u);
  F_MAD_LO_CC(B[6], a[1], b[6], B[6]);
  F_MADC_HI_CC(B[7], a[1], b[6], B[7]);
  F_MADC_LO_CC(B[8], a[3], b[6], B[8]);
  F_MADC_HI_CC(B[9], a[3], b[6], B[9]);
  F_MADC_LO_CC(B[10], a[5], b[6], B[10]);
  F_MADC_HI_CC(B[11], a[5], b[6], B[11]);
  F_MADC_LO_CC(B[12], a[7], b[6], B[12]);
  F_MADC_HI_CC(B[13], a[7], b[6], B[13]);
  F_ADDC(B[14], B[14], 0u);
  F_MAD_LO_CC(A[8], a[1], b[7], A[8]);
  F_MADC_HI_CC(A[9], a[1], b[7], A[9]);
  F_MADC_LO_CC(A[10], a[3], b[7], A[10]);
  F_MADC_HI_CC(A[11], a[3], b[7], A[11]);
  F_MADC_LO_CC(A[12], a[5], b[7], A[12]);
  F_MADC_HI_CC(A[13], a[5], b[7], A[13]);
  F_MADC_LO_CC(A[14], a[7], b[7], A[14]);
  F_MADC_HI_CC(A[15], a[7], b[7], A[15]);
  F_ADDC(A[16], A[16], 0u);
  F_MAD_LO_CC(B[6], a[0], b[7], B[6]);
  F_MADC_HI_CC(B[7], a[0], b[7], B[7]);
  F_MADC_LO_CC(B[8], a[2], b[7], B[8]);
  F_MADC_HI_CC(B[9], a[2], b[7], B[9]);
  F_MADC_LO_CC(B[10], a[4], b[7], B[10]);
  F_MADC_HI_CC(B[11], a[4], b[7], B[11]);
  F_MADC_LO_CC(B[12], a[6], b[7], B[12]);
  F_MADC_HI_CC(B[13], a[6], b[7], B[13]);
  F_ADDC(B[14], B[14], 0u);
  r[0] = A[0];
  F_ADD_CC(r[1], A[1], B[0]);
  F_ADDC_CC(r[2], A[2], B[1]);
  F_ADDC_CC(r[3], A[3], B[2]);
  F_ADDC_CC(r[4], A[4], B[3]);
  F_ADDC_CC(r[5], A[5], B[4]);
  F_ADDC_CC(r[6], A[6], B[5]);
  F_ADDC_CC(r[7], A[7], B[6]);
  F_ADDC_CC(r[8], A[8], B[7]);
  F_ADDC_CC(r[9], A[9], B[8]);
  F_ADDC_CC(r[10], A[10], B[9]);
  F_ADDC_CC(r[11], A[11], B[10]);
  F_ADDC_CC(r[12], A[12], B[11]);
  F_ADDC_CC(r[13], A[13], B[12]);
  F_ADDC_CC(r[14], A[14], B[13]);
  F_ADDC_CC(r[15], A[15], B[14]);
}

F32_FN void ea_sqr256(uint32_t r[16], const uint32_t a[8]) {
  F_CARRY_DECL
  uint32_t A[18], B[18];
  _Pragma("unroll") for (int k = 0; k < 18; ++k) { A[k] = 0; B[k] = 0; }
  F_MAD_LO_CC(A[2], a[2], a[0], A[2]);
  F_MADC_HI_CC(A[3], a[2], a[0], A[3]);
  F_MADC_LO_CC(A[4], a[4], a[0], A[4]);
  F_MADC_HI_CC(A[5], a[4], a[0], A[5]);
  F_MADC_LO_CC(A[6], a[6], a[0], A[6]);
  F_MADC_HI_CC(A[7], a[6], a[0], A[7]);
  F_ADDC(A[8], A[8], 0u);
  F_MAD_LO_CC(B[0], a[1], a[0], B[0]);
  F_MADC_HI_CC(B[1], a[1], a[0], B[1]);
  F_MADC_LO_CC(B[2], a[3], a[0], B[2]);
  F_MADC_HI_CC(B[3], a[3], a[0], B[3]);
  F_MADC_LO_CC(B[4], a[5], a[0], B[4]);
  F_MADC_HI_CC(B[5], a[5], a[0], B[5]);
  F_MADC_LO_CC(B[6], a[7], a[0], B[6]);
  F_MADC_HI_CC(B[7], a[7], a[0], B[7]);
  F_ADDC(B[8], B[8], 0u);
  F_MAD_LO_CC(A[4], a[3], a[1], A[4]);
  F_MADC_HI_CC(A[5], a[3], a[1], A[5]);
  F_MADC_LO_CC(A[6], a[5], a[1], A[6]);
  F_MADC_HI_CC(A[7], a[5], a[1], A[7]);
  F_MADC_LO_CC(A[8], a[7], a[1], A[8]);
  F_MADC_HI_CC(A[9], a[7], a[1], A[9]);
  F_ADDC(A[10], A[10], 0u);
  F_MAD_LO_CC(B[2], a[2], a[1], B[2]);
  F_MADC_HI_CC(B[3], a[2], a[1], B[3]);
  F_MADC_LO_CC(B[4], a[4], a[1], B[4]);
  F_MADC_HI_CC(B[5], a[4], a[1], B[5]);
  F_MADC_LO_CC(B[6], a[6], a[1], B[6]);
  F_MADC_HI_CC(B[7], a[6], a[1], B[7]);
  F_ADDC(B[8], B[8], 0u);
  F_MAD_LO_CC(A[6], a[4], a[2], A[6]);
  F_MADC_HI_CC(A[7], a[4], a[2], A[7]);
  F_MADC_LO_CC(A[8], a[6], a[2], A[8]);
  F_MADC_HI_CC(A[9], a[6], a[2], A[9]);
  F_ADDC(A[10], A[10], 0u);
  F_MAD_LO_CC(B[4], a[3], a[2], B[4]);
  F_MADC_HI_CC(B[5], a[3], a[2], B[5]);
  F_MADC_LO_CC(B[6], a[5], a[2], B[6]);
  F_MADC_HI_CC(B[7], a[5], a[2], B[7]);
  F_MADC_LO_CC(B[8], a[7], a[2], B[8]);
  F_MADC_HI_CC(B[9], a[7], a[2], B[9]);
  F_ADDC(B[10], B[10], 0u);
  F_MAD_LO_CC(A[8], a[5], a[3], A[8]);
  F_MADC_HI_CC(A[9], a[5], a[3], A[9]);
  F_MADC_LO_CC(A[10], a[7], a[3], A[10]);
  F_MADC_HI_CC(A[11], a[7], a[3], A[11]);
  F_ADDC(A[12], A[12], 0u);
  F_MAD_LO_CC(B[6], a[4], a[3], B[6]);
  F_MADC_HI_CC(B[7], a[4], a[3], B[7]);
  F_MADC_LO_CC(B[8], a[6], a[3], B[8]);
  F_MADC_HI_CC(B[9], a[6], a[3], B[9]);
  F_ADDC(B[10], B[10], 0u);
  F_MAD_LO_CC(A[10], a[6], a[4], A[10]);
  F_MADC_HI_CC(A[11], a[6], a[4], A[11]);
  F_ADDC(A[12], A[12], 0u);
  F_MAD_LO_CC(B[8], a[5], a[4], B[8]);
  F_MADC_HI_CC(B[9], a[5], a[4], B[9]);
  F_MADC_LO_CC(B[10], a[7], a[4], B[10]);
  F_MADC_HI_CC(B[11], a[7], a[4], B[11]);
  F_ADDC(B[12], B[12], 0u);
  F_MAD_LO_CC(A[12], a[7], a[5], A[12]);
  F_MADC_HI_CC(A[13], a[7], a[5], A[13]);
  F_ADDC(A[14], A[14], 0u);
  F_MAD_LO_CC(B[10], a[6], a[5], B[10]);
  F_MADC_HI_CC(B[11], a[6], a[5], B[11]);
  F_ADDC(B[12], B[12], 0u);
  F_MAD_LO_CC(B[12], a[7], a[6], B[12]);
  F_MADC_HI_CC(B[13], a[7], a[6], B[13]);
  F_ADDC(B[14], B[14], 0u);
  r[0] = A[0];
  F_ADD_CC(r[1], A[1], B[0]);
  F_ADDC_CC(r[2], A[2], B[1]);
  F_ADDC_CC(r[3], A[3], B[2]);
  F_ADDC_CC(r[4], A[4], B[3]);
  F_ADDC_CC(r[5], A[5], B[4]);
  F_ADDC_CC(r[6], A[6], B[5]);
  F_ADDC_CC(r[7], A[7], B[6]);
  F_ADDC_CC(r[8], A[8], B[7]);
  F_ADDC_CC(r[9], A[9], B[8]);
  F_ADDC_CC(r[10], A[10], B[9]);
  F_ADDC_CC(r[11], A[11], B[10]);
  F_ADDC_CC(r[12], A[12], B[11]);
  F_ADDC_CC(r[13], A[13], B[12]);
  F_ADDC_CC(r[14], A[14], B[13]);
  F_ADDC_CC(r[15], A[15], B[14]);
  F_SHL1(r[15], r[15], r[14]);
  F_SHL1(r[14], r[14], r[13]);
  F_SHL1(r[13], r[13], r[12]);
  F_SHL1(r[12], r[12], r[11]);
  F_SHL1(r[11], r[11], r[10]);
  F_SHL1(r[10], r[10], r[9]);
  F_SHL1(r[9], r[9], r[8]);
  F_SHL1(r[8], r[8], r[7]);
  F_SHL1(r[7], r[7], r[6]);
  F_SHL1(r[6], r[6], r[5]);
  F_SHL1(r[5], r[5], r[4]);
  F_SHL1(r[4], r[4], r[3]);
  F_SHL1(r[3], r[3], r[2]);
  F_SHL1(r[2], r[2], r[1]);
  F_SHL1(r[1], r[1], r[0]);
  r[0] = r[0] << 1;
  F_MAD_LO_CC(r[0], a[0], a[0], r[0]);
  F_MADC_HI_CC(r[1], a[0], a[0], r[1]);
  F_MADC_LO_CC(r[2], a[1], a[1], r[2]);
  F_MADC_HI_CC(r[3], a[1], a[1], r[3]);
  F_MADC_LO_CC(r[4], a[2], a[2], r[4]);
  F_MADC_HI_CC(r[5], a[2], a[2], r[5]);
  F_MADC_LO_CC(r[6], a[3], a[3], r[6]);
  F_MADC_HI_CC(r[7], a[3], a[3], r[7]);
  F_MADC_LO_CC(r[8], a[4], a[4], r[8]);
  F_MADC_HI_CC(r[9], a[4], a[4], r[9]);
  F_MADC_LO_CC(r[10], a[5], a[5], r[10]);
  F_MADC_HI_CC(r[11], a[5], a[5], r[11]);
  F_MADC_LO_CC(r[12], a[6], a[6], r[12]);
  F_MADC_HI_CC(r[13], a[6], a[6], r[13]);
  F_MADC_LO_CC(r[14], a[7], a[7], r[14]);
  F_MADC_HI(r[15], a[7], a[7], r[15]);
}
