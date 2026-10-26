// kangaroo_wasm.cpp: the Kova Pool kangaroo for phones and browsers. Freestanding WebAssembly build of
// kangaroo_core.h (no libc): one group of GROUP kangaroos per instance; the page runs one instance per Web Worker.
//
// Build:  clang++ --target=wasm32 -O3 -std=c++17 -nostdlib -ffreestanding -fno-exceptions -fno-rtti -I.. \
//           -Wl,--no-entry -Wl,--export-dynamic -Wl,-z,stack-size=262144 -o kangaroo.wasm kangaroo_wasm.cpp
//
// Protocol (all numbers are little-endian u64 words at the exported buffers; see pool/site/phone-worker.js):
//   kw_buf()            -> address of the 256-byte input buffer: pubkey x (4 words) | pubkey y parity (word 4) |
//                          start (4 words, 5..8) | blind x (4 words, 9..12) | blind parity (13) | has blind (14)
//   kw_init(R, dp, species, seedLo, seedHi) -> 0 ok, 1 bad pubkey, 2 bad blind, 3 bad range. species 0 tame, 1 wild,
//                          2 both (kangaroos 0..7 tame, 8..23 wild: the GPU solver's split); kw_nt() = tame count.
//                          Builds the tables (about a second) and the group's start points.
//   kw_run(steps)       -> number of DP records written to kw_out(); restarts kangaroos that got stuck
//   kw_out()            -> DP records: x0 x1 d0 d1 d2 kang (6 words each), as the native solver's lines
//   kw_loops()          -> loops escaped so far;  kw_regen() -> kangaroos restarted so far
#include "kangaroo_core.h"

extern "C" {
// the few libc symbols clang may emit calls to for struct copies / zeroing
void* memcpy(void* d, const void* s, unsigned long n) { unsigned char* dd = (unsigned char*)d; const unsigned char* ss = (const unsigned char*)s; while (n--) *dd++ = *ss++; return d; }
void* memset(void* d, int v, unsigned long n) { unsigned char* dd = (unsigned char*)d; while (n--) *dd++ = (unsigned char)v; return d; }
}

static Tables T;
static Walker W;
static DPOut OUT[GROUP * 64];
static u64 IN[32];
static MT64 rng;
static int R_; static u64 NT_;
static Pt wildBase; static U256 tameMax, wildMax;
static u64 regenTotal;

static void start_kang(int g) {
    const bool wild = (u64)g >= NT_;
    U256 d = rnd_below(rng, wild ? wildMax : tameMax); if (wild) d.w[0] &= ~1ULL;
    W.D[g][0] = d.w[0]; W.D[g][1] = d.w[1]; W.D[g][2] = d.w[2]; W.D[g][3] = 0;
    Pt P; mul_table(P, d.w, T.GT, R_ + 8);
    if (wild) { Pt S; hp_add(S, P, wildBase); P = S; }
    cp4(W.X[g], P.x); cp4(W.Y[g], P.y); walker_reset(W, g);
}

extern "C" {
__attribute__((export_name("kw_buf"))) u64* kw_buf() { return IN; }
__attribute__((export_name("kw_out"))) DPOut* kw_out() { return OUT; }
__attribute__((export_name("kw_loops"))) u64 kw_loops() { return W.loops; }
__attribute__((export_name("kw_regen"))) u64 kw_regen() { return regenTotal; }
__attribute__((export_name("kw_nt"))) int kw_nt() { return (int)NT_; }
__attribute__((export_name("kw_init"))) int kw_init(int R, int dp, int species, u64 seedLo, u64 seedHi) {
    if (R < 20 || R > 170) return 3;
    R_ = R; tables_init(T, R, dp > 32 ? 32 : dp);
    Pt Q; if (!decompress(Q, IN, IN[4] & 1)) return 1;
    U256 startK; cp4(startK.w, IN + 5);
    const U256 x32 = u_pow2(R - 5);
    Pt Pp = Q, tmp;
    if (!iszero4(startK.w)) { Pt S; t_mulG(T, S, startK); if (!hp_add(tmp, Pp, hp_neg(S))) return 1; Pp = tmp; }
    { Pt X; t_mulG(T, X, x32); if (!hp_add(tmp, Pp, X)) return 1; Pp = tmp; }
    wildBase = hp_neg(Pp);
    if (IN[14]) { Pt Rb; if (!decompress(Rb, IN + 9, IN[13] & 1)) return 2; if (!hp_add(tmp, wildBase, Rb)) return 2; wildBase = tmp; }
    tameMax = x32; wildMax = u_zero(); for (int i = 0; i < 34; ++i) wildMax = u_add(wildMax, x32);
    NT_ = species == 1 ? 0 : species == 2 ? GROUP / 3 : GROUP;
    rng.seed(seedLo ^ (seedHi * 6364136223846793005ULL) ^ 0x9E3779B97F4A7C15ULL);
    memset(&W, 0, sizeof W); W.out = OUT; W.outMax = GROUP * 64; W.base = 0; regenTotal = 0;
    for (int g = 0; g < GROUP; ++g) start_kang(g);
    return 0;
}
__attribute__((export_name("kw_run"))) int kw_run(int steps) {
    walker_run(T, W, steps);
    for (u32 k = 0; k < W.nstuck; ++k) { start_kang((int)W.stuck[k]); ++regenTotal; }
    return (int)W.nout;
}
}
