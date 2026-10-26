// kangaroo_cpu.cpp: the Kova Pool kangaroo for CPUs and phones. Plain C++17, no CUDA: builds with clang++/g++ on
// macOS, Linux, Android (NDK) and with Emscripten for the browser.
//
//   It walks EXACTLY the RCKangaroo walk the pool's GPU workers run (kangaroo --rc: RetiredCoder's KernelA/B/C
//   semantics, kangaroo/rc/RCGpuCore.cu), on the SAME jump tables (mt19937_64 seed 12345, as kangaroo.cu builds
//   them), with the same start-point rules, DP criterion and DP line format. So a CPU kangaroo that lands on a GPU
//   kangaroo's trail follows it and reports the same distinguished points: CPU and GPU points collide in one table.
//
//   The walk (per kangaroo: affine point + signed 192-bit distance):
//     jump index j = x[0] & 511; direction by y parity: y even -> P + J_j (d += jd), y odd -> P - J_j (d -= jd)
//     table 1 normally; when the NEXT jump would undo this one (same index, opposite direction: a 2-cycle) the next
//     jump comes from table 2 instead; cycles of 4..10 are caught from the distance history and escaped with one jump
//     from table 3 (RC's KernelB/KernelC rule); a kangaroo that reports the same DP twice is restarted (RC's regen).
//     DP: the low dp bits of the top 32-bit word of x are zero (RC's dp_mask on x bits 224..224+dp).
//
//   Output (--agent --dpstream, what pool_agent.py reads):  DP <x low 128 bits, 32 hex><distance, 192-bit two's
//   complement, 48 hex><type 0 tame / 1 wild>   and   STAT jumps=... dps=... kangs=... every --report seconds.
//
// Build:  c++ -O3 -std=c++17 -pthread -I.. -o kangaroo_cpu kangaroo_cpu.cpp   (modinv30.h is in ../; macOS: clang++ from Xcode tools)
// Test:   ./kangaroo_cpu --selftest --range 32          (random key, both species, solved locally)
// Pool:   curl -fsSL https://pool.kova23.com/cpu.sh | bash -s -- pk_TOKEN   (or KANG_CPU=1 KANG_BIN=./kangaroo_cpu POOL_TOKEN=pk_... python3 pool_agent.py)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <random>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <unordered_map>

#define __host__
#define __device__
#define __forceinline__ inline
#include "modinv30.h"      // safegcd field inversion (libsecp256k1 port), the same file the GPU solver uses

typedef unsigned __int128 u128;
typedef uint64_t u64; typedef uint32_t u32; typedef uint16_t u16;

#define JMP_CNT   512
#define JMP_MASK  (JMP_CNT - 1)
#define STEP_CNT  1000           // jumps per kangaroo per launch (RC: STEP_CNT)
#define MD_LEN    10             // distance history (RC: cycles up to MD_LEN caught)
#define GROUP     24             // kangaroos per thread = one batch inversion (RC: PNT_GROUP_CNT)
#define INV_FLAG  0x0200
#define JMP2_FLAG 0x0400
#define DP_FLAG   0x0800

// ============================================================================ field arithmetic mod p (secp256k1)
#define P0 0xFFFFFFFEFFFFFC2FULL
#define PM 0xFFFFFFFFFFFFFFFFULL
static inline void cp4(u64* d, const u64* s) { d[0]=s[0]; d[1]=s[1]; d[2]=s[2]; d[3]=s[3]; }
static inline bool iszero4(const u64* a) { return (a[0]|a[1]|a[2]|a[3]) == 0; }
static inline bool eq4(const u64* a, const u64* b) { return a[0]==b[0]&&a[1]==b[1]&&a[2]==b[2]&&a[3]==b[3]; }
static inline bool geq_p(const u64* a) { if (a[3] != PM || a[2] != PM || a[1] != PM) return false; return a[0] >= P0; }
static inline void p_add(u64* r, const u64* a, const u64* b) {
    u128 c = 0; u64 t[4];
    for (int i = 0; i < 4; ++i) { c += (u128)a[i] + b[i]; t[i] = (u64)c; c >>= 64; }
    if (c || geq_p(t)) { u128 d = (u128)t[0] + 0x1000003D1ULL; t[0] = (u64)d; d >>= 64; for (int i = 1; i < 4; ++i) { d += t[i]; t[i] = (u64)d; d >>= 64; } }
    cp4(r, t);
}
static inline void p_sub(u64* r, const u64* a, const u64* b) {
    u64 t[4]; u64 br = 0;
    for (int i = 0; i < 4; ++i) { u128 d = (u128)a[i] - b[i] - br; t[i] = (u64)d; br = (u64)(d >> 64) ? 1 : 0; }
    if (br) { const u64 p[4] = {P0, PM, PM, PM}; u128 c = 0; for (int i = 0; i < 4; ++i) { c += (u128)t[i] + p[i]; t[i] = (u64)c; c >>= 64; } }
    cp4(r, t);
}
static inline void p_neg(u64* r, const u64* a) { const u64 z[4] = {0,0,0,0}; p_sub(r, z, a); }
static inline void p_mul(u64* r, const u64* a, const u64* b) {
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
static inline void p_sqr(u64* r, const u64* a) { p_mul(r, a, a); }
static inline void p_inv(u64* r, const u64* a) { mi_inv(r, a); }

// ============================================================================ points
struct Pt { u64 x[4], y[4]; };
static const u64 GX[4] = {0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL};
static const u64 GY[4] = {0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL};
static const u64 NORD[4] = {0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL};
static void die(const char* m) { fprintf(stderr, "error: %s\n", m); exit(1); }

static void aff_add(Pt& R, const Pt& A, const Pt& B) {
    u64 dx[4], dy[4], inv[4], lam[4], x3[4], y3[4];
    p_sub(dx, B.x, A.x); p_sub(dy, B.y, A.y); p_inv(inv, dx); p_mul(lam, dy, inv);
    p_sqr(x3, lam); p_sub(x3, x3, A.x); p_sub(x3, x3, B.x);
    p_sub(y3, A.x, x3); p_mul(y3, lam, y3); p_sub(y3, y3, A.y);
    cp4(R.x, x3); cp4(R.y, y3);
}
static Pt hp_dbl(const Pt& a) {
    u64 x2[4], t[4], lam[4], inv[4], y2[4];
    p_sqr(x2, a.x); p_add(t, x2, x2); p_add(t, t, x2); p_add(y2, a.y, a.y); p_inv(inv, y2); p_mul(lam, t, inv);
    Pt r; p_sqr(r.x, lam); p_sub(r.x, r.x, a.x); p_sub(r.x, r.x, a.x);
    p_sub(t, a.x, r.x); p_mul(r.y, lam, t); p_sub(r.y, r.y, a.y); return r;
}
static Pt hp_neg(const Pt& a) { Pt r = a; p_neg(r.y, a.y); return r; }
static Pt hp_add(const Pt& a, const Pt& b) { if (eq4(a.x, b.x)) { if (eq4(a.y, b.y)) return hp_dbl(a); die("point add hit infinity"); } Pt r; aff_add(r, a, b); return r; }
// jacobian mixed add for the scalar multiplication table
static void jac_madd(u64* X, u64* Y, u64* Z, const u64* x2, const u64* y2) {
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
static std::vector<Pt> GT;   // GT[b] = 2^b G
static bool mul_table(Pt& R, const u64* s, const Pt* D, int nb) {
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

// ============================================================================ 256-bit integers (as kangaroo.cu)
struct U256 { u64 w[4] = {0,0,0,0}; };
static U256 u_from_hex(std::string h) {
    U256 r; if (h.rfind("0x", 0) == 0 || h.rfind("0X", 0) == 0) h = h.substr(2);
    if (h.empty() || h.size() > 64) die("bad hex number");
    for (char ch : h) {
        int v = isdigit((unsigned char)ch) ? ch - '0' : (ch >= 'a' && ch <= 'f') ? ch - 'a' + 10 : (ch >= 'A' && ch <= 'F') ? ch - 'A' + 10 : -1;
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
static U256 u_shl1(const U256&, int n) { U256 r; r.w[n / 64] = 1ULL << (n % 64); return r; }       // 2^n
static U256 u_shr1(const U256& a) { U256 r; for (int i = 0; i < 4; ++i) r.w[i] = (a.w[i] >> 1) | (i < 3 ? a.w[i + 1] << 63 : 0); return r; }
static bool u_neg(const U256& a) { return a.w[3] >> 63; }
static U256 u_negate(const U256& a) { U256 z; return u_sub(z, a); }
static U256 u_modn(const U256& a) { return u_cmp(a, *(const U256*)NORD) >= 0 ? u_sub(a, *(const U256*)NORD) : a; }
static U256 u_addn(const U256& a, const U256& b) { return u_modn(u_add(a, b)); }
static U256 u_subn(const U256& a, const U256& b) { return u_cmp(a, b) >= 0 ? u_sub(a, b) : u_add(u_sub(a, b), *(const U256*)NORD); }
static bool hp_mulG(Pt& R, const U256& k) { return mul_table(R, k.w, GT.data(), 256); }
static bool parse_pubkey(const std::string& hex, Pt& Q) {
    std::string h = hex;
    if (h.size() == 66 && (h.substr(0, 2) == "02" || h.substr(0, 2) == "03")) {
        U256 x = u_from_hex(h.substr(2)); cp4(Q.x, x.w);
        u64 y2[4], t[4], seven[4] = {7, 0, 0, 0}; p_sqr(t, Q.x); p_mul(t, t, Q.x); p_add(y2, t, seven);
        u64 e[4] = {0xFFFFFFFFBFFFFF0CULL, PM, PM, 0x3FFFFFFFFFFFFFFFULL}, r[4] = {1, 0, 0, 0}, b[4]; cp4(b, y2);
        for (int i = 0; i < 256; ++i) { if ((e[i >> 6] >> (i & 63)) & 1) p_mul(r, r, b); p_sqr(b, b); }
        p_sqr(t, r); if (!eq4(t, y2)) return false;
        if ((int)(r[0] & 1) != (h[1] == '3')) p_neg(r, r);
        cp4(Q.y, r); return true;
    }
    if (h.size() == 130 && h.substr(0, 2) == "04") {
        U256 x = u_from_hex(h.substr(2, 64)), y = u_from_hex(h.substr(66, 64)); cp4(Q.x, x.w); cp4(Q.y, y.w);
        u64 l[4], r[4], seven[4] = {7, 0, 0, 0}; p_sqr(l, Q.y); p_sqr(r, Q.x); p_mul(r, r, Q.x); p_add(r, r, seven);
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
// 192-bit signed distances (3 words)
static inline void d_add(u64* d, const u64* j) { u128 c = (u128)d[0] + j[0]; d[0] = (u64)c; c >>= 64; c += (u128)d[1] + j[1]; d[1] = (u64)c; c >>= 64; d[2] += j[2] + (u64)c; }
static inline void d_sub(u64* d, const u64* j) {
    u128 b = (u128)d[0] - j[0]; d[0] = (u64)b; u64 br = (u64)(b >> 64) ? 1 : 0;
    b = (u128)d[1] - j[1] - br; d[1] = (u64)b; br = (u64)(b >> 64) ? 1 : 0; d[2] -= j[2] + br;
}

// ============================================================================ the solver
struct Opts {
    std::string pubkey, start = "0", species = "both", blind, out = "kangaroo_found.txt";
    int range = 0, dp = 0, threads = 0, report = 10, maxSeconds = 0; double maxK = 0; u64 seed = 0;
    bool selftest = false, agent = false, dpstream = false; int dplocal = 1;
};
struct DPOut { u64 x0, x1, d[3]; u32 kang; };
struct DPRec { u64 x1; u64 d[3]; u32 type; u32 kang; };

// one worker thread: GROUP kangaroos, RC's KernelA/KernelB/KernelC rules, run STEP_CNT steps per launch
struct Walker {
    u64 X[GROUP][4], Y[GROUP][4], D[GROUP][4];       // point, distance (3 words used)
    u64 hist[GROUP][MD_LEN]; u32 hpos[GROUP];        // RC KernelB: ring of the last MD_LEN distance low words
    u32 l1s2 = 0;                                     // bit g: kangaroo g jumps from table 2 next (2-cycle escape)
    u64 base = 0;                                     // global index of kangaroo 0 of this thread
    std::vector<DPOut> out; u64 loops = 0;
};
static const u64 *J1, *J2, *J3, *JD1, *JD2, *JD3;    // tables: x(4) y(4) per jump; distances 3 words per jump
static u32 g_dpMask; static u64 g_NT;

static void escape3(Walker& w, int g) {                 // RC KernelC: one jump from table 3 from the current point
    u64* x = w.X[g]; u64* y = w.Y[g];
    const u32 j = (u32)x[0] & JMP_MASK; const u64* jx = J3 + 8 * j; u64 jy[4]; cp4(jy, jx + 4);
    const bool odd = y[0] & 1; if (odd) p_neg(jy, jy);
    u64 dx[4], inv[4], dy[4], lam[4], nx[4], ny[4];
    p_sub(dx, x, jx); p_inv(inv, dx); p_sub(dy, y, jy); p_mul(lam, dy, inv);
    p_sqr(nx, lam); p_sub(nx, nx, jx); p_sub(nx, nx, x);
    p_sub(ny, x, nx); p_mul(ny, lam, ny); p_sub(ny, ny, y);
    cp4(x, nx); cp4(y, ny);
    if (odd) d_sub(w.D[g], JD3 + 3 * j); else d_add(w.D[g], JD3 + 3 * j);
    for (int h = 0; h < MD_LEN; ++h) w.hist[g][h] = 0; w.hpos[g] = 0; w.l1s2 &= ~(1u << g); ++w.loops;
}

static void launch(Walker& w) {
    u64 pre[GROUP][4], acc[4], dx[4];
    for (int step = 0; step < STEP_CNT; ++step) {
        // forward: prefix products of (x_g - jx_g)
        for (int g = 0; g < GROUP; ++g) {
            const u64* x = w.X[g]; const u32 j = (u32)x[0] & JMP_MASK;
            const u64* jx = ((w.l1s2 >> g) & 1) ? J2 + 8 * j : J1 + 8 * j;
            p_sub(dx, x, jx);
            if (g == 0) cp4(acc, dx); else p_mul(acc, acc, dx);
            cp4(pre[g], acc);
        }
        u64 inv[4]; p_inv(inv, acc);
        // backward: one jump per kangaroo
        for (int g = GROUP - 1; g >= 0; --g) {
            u64* x = w.X[g]; u64* y = w.Y[g];
            u32 j = (u32)x[0] & JMP_MASK; const bool t2 = (w.l1s2 >> g) & 1;
            const u64* jx = t2 ? J2 + 8 * j : J1 + 8 * j; u64 jy[4]; cp4(jy, jx + 4);
            u64 ik[4];
            if (g > 0) { p_mul(ik, inv, pre[g - 1]); p_sub(dx, x, jx); p_mul(inv, inv, dx); }
            else cp4(ik, inv);
            const bool odd = y[0] & 1;                 // RC: inv_flag = y odd -> jump -J
            u32 rec = j | (odd ? INV_FLAG : 0);
            if (odd) p_neg(jy, jy);
            u64 dy[4], lam[4], nx[4], ny[4];
            p_sub(dy, y, jy); p_mul(lam, dy, ik);
            p_sqr(nx, lam); p_sub(nx, nx, jx); p_sub(nx, nx, x);
            p_sub(ny, x, nx); p_mul(ny, lam, ny); p_sub(ny, ny, y);
            cp4(x, nx); cp4(y, ny);
            if (!t2) {                                 // 2-cycle: next jump = same index, opposite direction
                const u32 jn = ((u32)nx[0] & JMP_MASK) | ((ny[0] & 1) ? 0 : INV_FLAG);
                if (rec == jn) w.l1s2 |= 1u << g;
            } else { w.l1s2 &= ~(1u << g); rec |= JMP2_FLAG; }
            // distance (RC KernelB): +-d from table 1 or 2
            const u64* jd = (t2 ? JD2 : JD1) + 3 * j;
            if (odd) d_sub(w.D[g], jd); else d_add(w.D[g], jd);
            // loop detection on the distance history (RC ProcessJumpDistance: cycles of 4, 6, 8, 10)
            u64* tab = w.hist[g]; const u32 it = w.hpos[g]; const u64 d0 = w.D[g][0];
            bool looped = tab[(it + MD_LEN - 4) % MD_LEN] == d0 || tab[(it + MD_LEN - 6) % MD_LEN] == d0 || tab[(it + MD_LEN - 8) % MD_LEN] == d0 || tab[it] == d0;
            tab[it] = d0; w.hpos[g] = (it + 1) % MD_LEN;
            if ((((u32)(nx[3] >> 32)) & g_dpMask) == 0) w.out.push_back({nx[0], nx[1], {w.D[g][0], w.D[g][1], w.D[g][2]}, (u32)(w.base + g)});
            if (looped && d0) escape3(w, g);
        }
    }
}

int main(int argc, char** argv) {
    Opts o;
    auto nx = [&](int& i) { if (i + 1 >= argc) die("missing value"); return std::string(argv[++i]); };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--pubkey") o.pubkey = nx(i); else if (a == "--start") o.start = nx(i); else if (a == "--range") o.range = atoi(nx(i).c_str());
        else if (a == "--dp") o.dp = atoi(nx(i).c_str()); else if (a == "--threads") o.threads = atoi(nx(i).c_str());
        else if (a == "--report") o.report = atoi(nx(i).c_str()); else if (a == "--time") o.maxSeconds = atoi(nx(i).c_str());
        else if (a == "--max") o.maxK = atof(nx(i).c_str()); else if (a == "--seed") o.seed = strtoull(nx(i).c_str(), 0, 0);
        else if (a == "--species") o.species = nx(i); else if (a == "--blind") o.blind = nx(i); else if (a == "--out") o.out = nx(i);
        else if (a == "--selftest") o.selftest = true; else if (a == "--agent") o.agent = true; else if (a == "--dpstream") o.dpstream = true;
        else if (a == "--dplocal") o.dplocal = atoi(nx(i).c_str());
        else if (a == "--cpu") {} else if (a == "--gpu" || a == "--rc" || a == "--rc-cubin" || a == "--blocks") nx(i);   // GPU options: accepted, ignored
        else if (a == "-h" || a == "--help") {
            printf("kangaroo_cpu: Pollard kangaroo for secp256k1 on the CPU, walking the pool's GPU graph\n"
                   "  --pubkey HEX --range BITS [--start HEX] [--dp N] [--threads N] [--species tame|wild|both] [--blind HEX]\n"
                   "  --agent --dpstream --dplocal 0   stream DP lines for pool_agent.py\n"
                   "  --selftest --range BITS          random key, solved locally (checks the walk)\n"
                   "  --time S, --max K, --report S, --seed N, --out FILE\n"); return 0;
        } else die(("unknown option " + a).c_str());
    }
    GT.resize(256); cp4(GT[0].x, GX); cp4(GT[0].y, GY); for (int i = 1; i < 256; ++i) GT[i] = hp_dbl(GT[i - 1]);
    std::mt19937_64 rng(o.seed ? o.seed : (u64)std::random_device{}() ^ (u64)time(nullptr));
    auto rnd_below = [&](const U256& m) -> U256 {   // uniform in [0, m): rejection on the top word (as kangaroo.cu)
        int bits = 0; for (int i = 3; i >= 0; --i) if (m.w[i]) { bits = 64 * i + 64 - __builtin_clzll(m.w[i]); break; }
        while (true) { U256 r; for (int i = 0; i < 4; ++i) r.w[i] = rng();
            for (int i = 0; i < 4; ++i) { int lo = i * 64; if (lo >= bits) r.w[i] = 0; else if (lo + 64 > bits) r.w[i] &= (bits - lo == 64) ? ~0ULL : ((1ULL << (bits - lo)) - 1); }
            if (u_cmp(r, m) < 0) return r; }
    };
    const int R = o.range; if (R < 20 || R > 170) die("--range must be 20..170 (bits)");
    int T = o.threads > 0 ? o.threads : (int)std::max(1u, std::thread::hardware_concurrency());
    const u64 NK = (u64)T * GROUP;
    const u64 NT = o.species == "tame" ? NK : o.species == "wild" ? 0 : NK / 3; g_NT = NT;
    int dp = o.dp;
    if (!dp) { double ops = 1.15 * std::pow(2.0, R / 2.0); double perKang = ops / (double)NK; dp = (int)std::floor(std::log2(std::max(perKang / 8.0, 2.0))); dp = std::max(4, std::min(32, dp)); }
    if (dp > 32) dp = 32;                               // the walk tests 32 bits at most (as the GPU turbo kernels)
    g_dpMask = dp == 32 ? 0xFFFFFFFFu : ((1u << dp) - 1);
    const double expOps = 1.15 * std::pow(2.0, R / 2.0);
    printf("[+] kangaroos     : %s (%s tame, %s wild) on %d CPU threads\n", human((double)NK).c_str(), human((double)NT).c_str(), human((double)(NK - NT)).c_str(), T);
    printf("[+] range %d bits : expected ops 1.15 * 2^%.1f = 2^%.2f; DP bits %d\n", R, R / 2.0, std::log2(expOps), dp);

    // ---- jump tables: identical to every pool worker (kangaroo.cu make_jumps, mt19937_64(12345))
    std::vector<u64> vJ1(JMP_CNT * 8), vJ2(JMP_CNT * 8), vJ3(JMP_CNT * 8), vJD1(JMP_CNT * 3), vJD2(JMP_CNT * 3), vJD3(JMP_CNT * 3);
    auto make_jumps = [&](int bits, std::vector<u64>& J, std::vector<u64>& JD) {
        U256 lo = u_shl1(U256(), bits);
        for (int i = 0; i < JMP_CNT; ++i) {
            U256 d = u_add(lo, rnd_below(lo)); d.w[0] &= ~1ULL;
            Pt P; hp_mulG(P, d); cp4(&J[8 * i], P.x); cp4(&J[8 * i + 4], P.y); JD[3 * i] = d.w[0]; JD[3 * i + 1] = d.w[1]; JD[3 * i + 2] = d.w[2];
        }
    };
    { std::mt19937_64 jrng(12345); std::swap(rng, jrng);
      make_jumps(R / 2 + 3, vJ1, vJD1); make_jumps(std::max(R - 10, R / 2 + 8), vJ2, vJD2); make_jumps(std::max(R - 12, R / 2 + 6), vJ3, vJD3);
      std::swap(rng, jrng); }
    J1 = vJ1.data(); J2 = vJ2.data(); J3 = vJ3.data(); JD1 = vJD1.data(); JD2 = vJD2.data(); JD3 = vJD3.data();
    if (getenv("KANG_DUMP_JUMPS")) { U256 a; cp4(a.w, &vJ1[0]); U256 b; cp4(b.w, &vJ1[8 * 511]); printf("[jumps] J1[0].x %s  J1[511].x %s  JD1[0] %016llx\n", u_hex(a).c_str(), u_hex(b).c_str(), (unsigned long long)vJD1[0]); }

    // ---- the point: P' = Q - start*G + x32*G;  wild base = -P' (+ blind)
    Pt Q; U256 startK = u_from_hex(o.start); U256 secret; bool haveSecret = false;
    if (o.selftest) {
        secret = u_add(startK, rnd_below(u_shl1(U256(), R))); if (iszero4(secret.w)) secret.w[0] = 1;
        hp_mulG(Q, secret); haveSecret = true;
        printf("[selftest] secret key %s (%d-bit range from %s) -> pubkey %s\n", u_hex(secret).c_str(), R, u_hex(startK).c_str(), pub_hex(Q).c_str());
    } else if (!parse_pubkey(o.pubkey, Q)) die("bad --pubkey");
    const U256 x32 = u_shl1(U256(), R - 5);
    Pt Pp = Q;
    if (!iszero4(startK.w)) { Pt S; hp_mulG(S, startK); Pp = hp_add(Pp, hp_neg(S)); }
    { Pt X; hp_mulG(X, x32); Pp = hp_add(Pp, X); }
    Pt wildBase = hp_neg(Pp);
    if (!o.blind.empty()) { Pt Rb; if (!parse_pubkey(o.blind, Rb)) die("bad --blind point"); wildBase = hp_add(wildBase, Rb); o.dplocal = 0;
                            if (!o.dpstream) die("--blind needs --dpstream (DPs are resolved by the pool server)"); }
    printf("[+] target pubkey : %s   start %s   range 2^%d\n", pub_hex(Q).c_str(), u_hex(startK).c_str(), R);

    // ---- start points
    const int NB2 = R + 8; std::vector<Pt> G2(GT.begin(), GT.begin() + NB2);
    U256 tameMax = x32; U256 wildMax; { U256 t; for (int i = 0; i < 34; ++i) t = u_add(t, x32); wildMax = t; }
    std::vector<Walker> W(T);
    auto start_kang = [&](Walker& w, int g) {
        const u64 i = w.base + g; U256 d = i < NT ? rnd_below(tameMax) : rnd_below(wildMax); if (i >= NT) d.w[0] &= ~1ULL;
        w.D[g][0] = d.w[0]; w.D[g][1] = d.w[1]; w.D[g][2] = d.w[2]; w.D[g][3] = 0;
        u64 s[4] = {d.w[0], d.w[1], d.w[2], 0}; Pt P; mul_table(P, s, G2.data(), NB2);
        if (i >= NT) P = hp_add(P, wildBase);
        cp4(w.X[g], P.x); cp4(w.Y[g], P.y);
        for (int h = 0; h < MD_LEN; ++h) w.hist[g][h] = 0; w.hpos[g] = 0; w.l1s2 &= ~(1u << g);
    };
    for (int t = 0; t < T; ++t) { W[t].base = (u64)t * GROUP; for (int g = 0; g < GROUP; ++g) start_kang(W[t], g); }

    // ---- host: DP table (local resolution) or stream; same-DP-twice -> regenerate (RC: looped kangs restart)
    std::unordered_map<u64, DPRec> dps; U256 key; bool found = false; u64 dupCount = 0, collisions = 0, dpStreamed = 0, regenTotal = 0, loopsTotal = 0;
    const u64 RECENT_MASK = (1ULL << 20) - 1; std::vector<u64> recentX(RECENT_MASK + 1, 0), recentD(RECENT_MASK + 1, 0);
    auto sgn_mag = [](const u64* d, U256& mag) -> bool { U256 v; v.w[0] = d[0]; v.w[1] = d[1]; v.w[2] = d[2]; v.w[3] = (d[2] >> 63) ? ~0ULL : 0;
        bool neg = u_neg(v); mag = neg ? u_negate(v) : v; return neg; };
    auto to_modn = [&](const u64* d) -> U256 { U256 m; bool neg = sgn_mag(d, m); m = u_modn(m); return neg ? u_subn(U256(), m) : m; };
    auto try_key = [&](const U256& kp) -> bool {
        if (iszero4(kp.w)) return false; Pt Tp; if (!hp_mulG(Tp, kp)) return false;
        if (!(eq4(Tp.x, Pp.x) && eq4(Tp.y, Pp.y))) return false;
        U256 k = u_addn(u_subn(kp, x32), startK); Pt C; if (!hp_mulG(C, k)) return false;
        if (eq4(C.x, Q.x) && eq4(C.y, Q.y)) { key = k; found = true; return true; }
        return false;
    };
    auto resolve = [&](const DPRec& a, const DPRec& b) -> bool {
        ++collisions; U256 da = to_modn(a.d), db = to_modn(b.d);
        if (a.type != b.type) {
            const U256& t = a.type == 0 ? da : db; const U256& w = a.type == 0 ? db : da;
            U256 c1 = u_subn(w, t), c2 = u_addn(w, t);
            return try_key(c1) || try_key(c2) || try_key(u_subn(U256(), c1)) || try_key(u_subn(U256(), c2));
        }
        if (a.type == 1) {
            auto halve = [](U256 v) { U256 one; one.w[0] = 1; if (v.w[0] & 1) return u_add(u_add(u_shr1(v), u_shr1(*(const U256*)NORD)), one); return u_shr1(v); };
            U256 h = halve(u_addn(da, db)), h2 = halve(u_subn(da, db));
            return try_key(h) || try_key(u_subn(U256(), h)) || try_key(h2) || try_key(u_subn(U256(), h2));
        }
        return false;
    };
    std::vector<u32> regen;
    auto process = [&](std::vector<DPOut>& out) {
        for (const DPOut& r : out) {
            if (found) break;
            DPRec rec; rec.x1 = r.x1; rec.d[0] = r.d[0]; rec.d[1] = r.d[1]; rec.d[2] = r.d[2]; rec.kang = r.kang; rec.type = r.kang < NT ? 0 : 1;
            const u64 slot = (r.x0 ^ (r.x0 >> 29)) & RECENT_MASK;
            if (recentX[slot] == r.x0 && recentD[slot] == r.d[0]) { ++dupCount; regen.push_back(r.kang); continue; }
            recentX[slot] = r.x0; recentD[slot] = r.d[0];
            if (o.dpstream) printf("DP %016llx%016llx%016llx%016llx%016llx%u\n", (unsigned long long)r.x0, (unsigned long long)r.x1,
                                   (unsigned long long)r.d[2], (unsigned long long)r.d[1], (unsigned long long)r.d[0], rec.type);
            if (!o.dplocal) { ++dpStreamed; continue; }
            auto it = dps.find(r.x0);
            if (it == dps.end()) { dps.emplace(r.x0, rec); continue; }
            const DPRec& old = it->second;
            if (old.x1 != rec.x1) continue;
            if (old.type == rec.type && old.d[0] == rec.d[0] && old.d[1] == rec.d[1] && old.d[2] == rec.d[2]) { ++dupCount; regen.push_back(rec.kang); continue; }
            if (old.type == 0 && rec.type == 0) continue;
            if (!resolve(old, rec)) printf("[!] collision did not yield the key (x %016llx)\n", (unsigned long long)r.x0);
        }
        out.clear(); if (o.dpstream) fflush(stdout);
    };

    // ---- run: all threads walk one launch, then the host processes their DPs
    const double t0 = now_s(); double tLast = t0, opsLast = 0; u64 launches = 0;
    const double maxOps = o.maxK > 0 ? o.maxK * expOps : 0;
    while (!found) {
        std::vector<std::thread> th;
        for (int t = 0; t < T; ++t) th.emplace_back([&, t]() { launch(W[t]); });
        for (auto& x : th) x.join();
        ++launches;
        for (int t = 0; t < T; ++t) { loopsTotal += W[t].loops; W[t].loops = 0; process(W[t].out); }
        if (!regen.empty() && !found) {
            std::sort(regen.begin(), regen.end()); regen.erase(std::unique(regen.begin(), regen.end()), regen.end());
            for (u32 i : regen) start_kang(W[i / GROUP], (int)(i % GROUP));
            regenTotal += regen.size(); regen.clear();
        }
        const double ops = (double)launches * (double)NK * STEP_CNT, t = now_s();
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
    const double ops = (double)launches * (double)NK * STEP_CNT, el = now_s() - t0;
    printf("[+] %s jumps in %.1f s (%s jumps/s); K = ops / 2^(range/2) = %.3f; %zu DPs, %llu collisions checked, %s loops escaped\n",
           human(ops).c_str(), el, human(ops / el).c_str(), ops / std::pow(2.0, R / 2.0), dps.size(), (unsigned long long)collisions, human((double)loopsTotal).c_str());
    if (found) {
        if (haveSecret && u_cmp(key, secret) != 0) { printf("[!] FOUND KEY %s DOES NOT MATCH THE SECRET %s\n", u_hex(key).c_str(), u_hex(secret).c_str()); return 1; }
        printf("\n[+] KEY FOUND: %s\n[+] pubkey   : %s\n", u_hex(key).c_str(), pub_hex(Q).c_str());
        if (!o.selftest) { std::ofstream f(o.out, std::ios::app); f << u_hex(key) << " " << pub_hex(Q) << "\n"; printf("[+] saved to %s\n", o.out.c_str()); }
        return 0;
    }
    printf("[-] stopped without finding the key\n");
    return 2;
}
