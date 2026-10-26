// kangaroo_cpu.cpp: the Kova Pool kangaroo for CPUs. Plain C++17, no CUDA: builds with clang++/g++ on macOS, Linux
// and Android (NDK). The walk itself is in kangaroo_core.h (shared with the browser build, kangaroo_wasm.cpp): it is
// EXACTLY the RCKangaroo walk the pool's GPU workers run, on the same jump tables, so CPU and GPU points collide in
// one table. This file is the host side: threads, the DP stream for pool_agent.py, local resolution for --selftest.
//
//   Output (--agent --dpstream, what pool_agent.py reads):  DP <x low 64 bits, 16 hex><x bits 64..127, 16 hex>
//   <distance, 192-bit two's complement, 48 hex><type 0 tame / 1 wild>   and   STAT jumps=... dps=... kangs=... every
//   --report seconds.
//
// Build:  c++ -O3 -std=c++17 -pthread -I.. -o kangaroo_cpu kangaroo_cpu.cpp   (modinv30.h is in ../; macOS: clang++ from Xcode tools)
// Test:   ./kangaroo_cpu --selftest --range 32          (random key, both species, solved locally)
// Pool:   curl -fsSL https://pool.kova23.com/cpu.sh | bash -s -- pk_TOKEN   (or KANG_CPU=1 KANG_BIN=./kangaroo_cpu POOL_TOKEN=pk_... python3 pool_agent.py)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <random>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <unordered_map>
#include <ctime>
#include "kangaroo_core.h"

static const u64 NORD[4] = {0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL};
static void die(const char* m) { fprintf(stderr, "error: %s\n", m); exit(1); }
static U256 u_from_hex(std::string h) {
    U256 r = u_zero(); if (h.rfind("0x", 0) == 0 || h.rfind("0X", 0) == 0) h = h.substr(2);
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
static U256 u_sub(const U256& a, const U256& b) { U256 r; u64 br = 0; for (int i = 0; i < 4; ++i) r.w[i] = kc_subb(a.w[i], b.w[i], &br); return r; }
static U256 u_shr1(const U256& a) { U256 r; for (int i = 0; i < 4; ++i) r.w[i] = (a.w[i] >> 1) | (i < 3 ? a.w[i + 1] << 63 : 0); return r; }
static bool u_neg(const U256& a) { return a.w[3] >> 63; }
static U256 u_negate(const U256& a) { return u_sub(u_zero(), a); }
static U256 u_modn(const U256& a) { return u_cmp(a, *(const U256*)NORD) >= 0 ? u_sub(a, *(const U256*)NORD) : a; }
static U256 u_addn(const U256& a, const U256& b) { return u_modn(u_add(a, b)); }
static U256 u_subn(const U256& a, const U256& b) { return u_cmp(a, b) >= 0 ? u_sub(a, b) : u_add(u_sub(a, b), *(const U256*)NORD); }
static Pt padd(const Pt& a, const Pt& b) { Pt r; if (!hp_add(r, a, b)) die("point add hit infinity"); return r; }
static bool parse_pubkey(const std::string& h, Pt& Q) {
    if (h.size() == 66 && (h.substr(0, 2) == "02" || h.substr(0, 2) == "03")) { U256 x = u_from_hex(h.substr(2)); return decompress(Q, x.w, h[1] == '3'); }
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

struct Opts {
    std::string pubkey, start = "0", species = "both", blind, out = "kangaroo_found.txt";
    int range = 0, dp = 0, threads = 0, report = 10, maxSeconds = 0; double maxK = 0; u64 seed = 0;
    bool selftest = false, agent = false, dpstream = false; int dplocal = 1;
};
struct DPRec { u64 x1; u64 d[3]; u32 type; u32 kang; };

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
    const int R = o.range; if (R < 20 || R > 170) die("--range must be 20..170 (bits)");
    int T = o.threads > 0 ? o.threads : (int)std::max(1u, std::thread::hardware_concurrency());
    const u64 NK = (u64)T * GROUP;
    const u64 NT = o.species == "tame" ? NK : o.species == "wild" ? 0 : NK / 3;
    int dp = o.dp;
    if (!dp) { double ops = 1.15 * std::pow(2.0, R / 2.0); double perKang = ops / (double)NK; dp = (int)std::floor(std::log2(std::max(perKang / 8.0, 2.0))); dp = std::max(4, std::min(32, dp)); }
    if (dp > 32) dp = 32;                               // the walk tests 32 bits at most (as the GPU turbo kernels)
    const double expOps = 1.15 * std::pow(2.0, R / 2.0);
    printf("[+] kangaroos     : %s (%s tame, %s wild) on %d CPU threads\n", human((double)NK).c_str(), human((double)NT).c_str(), human((double)(NK - NT)).c_str(), T);
    printf("[+] range %d bits : expected ops 1.15 * 2^%.1f = 2^%.2f; DP bits %d\n", R, R / 2.0, std::log2(expOps), dp);

    static Tables T_; tables_init(T_, R, dp); const Tables& TB = T_;
    if (getenv("KANG_DUMP_JUMPS")) { U256 a; cp4(a.w, &TB.J1[0]); U256 b; cp4(b.w, &TB.J1[8 * 511]); printf("[jumps] J1[0].x %s  J1[511].x %s  JD1[0] %016llx\n", u_hex(a).c_str(), u_hex(b).c_str(), (unsigned long long)TB.JD1[0]); }
    MT64 rng; rng.seed(o.seed ? o.seed : (u64)std::random_device{}() ^ (u64)time(nullptr));

    // ---- the point: P' = Q - start*G + x32*G;  wild base = -P' (+ blind)
    Pt Q; U256 startK = u_from_hex(o.start); U256 secret = u_zero(); bool haveSecret = false;
    if (o.selftest) {
        secret = u_add(startK, rnd_below(rng, u_pow2(R))); if (iszero4(secret.w)) secret.w[0] = 1;
        t_mulG(TB, Q, secret); haveSecret = true;
        printf("[selftest] secret key %s (%d-bit range from %s) -> pubkey %s\n", u_hex(secret).c_str(), R, u_hex(startK).c_str(), pub_hex(Q).c_str());
    } else if (!parse_pubkey(o.pubkey, Q)) die("bad --pubkey");
    const U256 x32 = u_pow2(R - 5);
    Pt Pp = Q;
    if (!iszero4(startK.w)) { Pt S; t_mulG(TB, S, startK); Pp = padd(Pp, hp_neg(S)); }
    { Pt X; t_mulG(TB, X, x32); Pp = padd(Pp, X); }
    Pt wildBase = hp_neg(Pp);
    if (!o.blind.empty()) { Pt Rb; if (!parse_pubkey(o.blind, Rb)) die("bad --blind point"); wildBase = padd(wildBase, Rb); o.dplocal = 0;
                            if (!o.dpstream) die("--blind needs --dpstream (DPs are resolved by the pool server)"); }
    printf("[+] target pubkey : %s   start %s   range 2^%d\n", pub_hex(Q).c_str(), u_hex(startK).c_str(), R);

    // ---- start points
    const int NB2 = R + 8;
    U256 tameMax = x32; U256 wildMax = u_zero(); for (int i = 0; i < 34; ++i) wildMax = u_add(wildMax, x32);
    std::vector<Walker> W(T); std::vector<std::vector<DPOut>> outs(T, std::vector<DPOut>(GROUP * STEP_CNT / 4 + 64));
    auto start_kang = [&](Walker& w, int g) {
        const u64 i = w.base + g; U256 d = i < NT ? rnd_below(rng, tameMax) : rnd_below(rng, wildMax); if (i >= NT) d.w[0] &= ~1ULL;
        w.D[g][0] = d.w[0]; w.D[g][1] = d.w[1]; w.D[g][2] = d.w[2]; w.D[g][3] = 0;
        Pt P; mul_table(P, d.w, TB.GT, NB2);
        if (i >= NT) P = padd(P, wildBase);
        cp4(w.X[g], P.x); cp4(w.Y[g], P.y); walker_reset(w, g);
    };
    for (int t = 0; t < T; ++t) { memset(&W[t], 0, sizeof(Walker)); W[t].base = (u64)t * GROUP; W[t].out = outs[t].data(); W[t].outMax = (u32)outs[t].size(); for (int g = 0; g < GROUP; ++g) start_kang(W[t], g); }

    // ---- host: DP table (local resolution) or stream
    std::unordered_map<u64, DPRec> dps; U256 key = u_zero(); bool found = false; u64 dupCount = 0, collisions = 0, dpStreamed = 0, regenTotal = 0, loopsTotal = 0;
    auto sgn_mag = [](const u64* d, U256& mag) -> bool { U256 v; v.w[0] = d[0]; v.w[1] = d[1]; v.w[2] = d[2]; v.w[3] = (d[2] >> 63) ? ~0ULL : 0;
        bool neg = u_neg(v); mag = neg ? u_negate(v) : v; return neg; };
    auto to_modn = [&](const u64* d) -> U256 { U256 m; bool neg = sgn_mag(d, m); m = u_modn(m); return neg ? u_subn(u_zero(), m) : m; };
    auto try_key = [&](const U256& kp) -> bool {
        if (iszero4(kp.w)) return false; Pt Tp; if (!t_mulG(TB, Tp, kp)) return false;
        if (!(eq4(Tp.x, Pp.x) && eq4(Tp.y, Pp.y))) return false;
        U256 k = u_addn(u_subn(kp, x32), startK); Pt C; if (!t_mulG(TB, C, k)) return false;
        if (eq4(C.x, Q.x) && eq4(C.y, Q.y)) { key = k; found = true; return true; }
        return false;
    };
    auto resolve = [&](const DPRec& a, const DPRec& b) -> bool {
        ++collisions; U256 da = to_modn(a.d), db = to_modn(b.d);
        if (a.type != b.type) {
            const U256& t = a.type == 0 ? da : db; const U256& w = a.type == 0 ? db : da;
            U256 c1 = u_subn(w, t), c2 = u_addn(w, t);
            return try_key(c1) || try_key(c2) || try_key(u_subn(u_zero(), c1)) || try_key(u_subn(u_zero(), c2));
        }
        if (a.type == 1) {
            auto halve = [](U256 v) { U256 one = u_zero(); one.w[0] = 1; if (v.w[0] & 1) return u_add(u_add(u_shr1(v), u_shr1(*(const U256*)NORD)), one); return u_shr1(v); };
            U256 h = halve(u_addn(da, db)), h2 = halve(u_subn(da, db));
            return try_key(h) || try_key(u_subn(u_zero(), h)) || try_key(h2) || try_key(u_subn(u_zero(), h2));
        }
        return false;
    };
    auto process = [&](Walker& w) {
        for (u32 k = 0; k < w.nout && !found; ++k) {
            const DPOut& r = w.out[k];
            DPRec rec; rec.x1 = r.x1; rec.d[0] = r.d[0]; rec.d[1] = r.d[1]; rec.d[2] = r.d[2]; rec.kang = r.kang; rec.type = r.kang < NT ? 0 : 1;
            if (o.dpstream) printf("DP %016llx%016llx%016llx%016llx%016llx%u\n", (unsigned long long)r.x0, (unsigned long long)r.x1,
                                   (unsigned long long)r.d[2], (unsigned long long)r.d[1], (unsigned long long)r.d[0], rec.type);
            if (!o.dplocal) { ++dpStreamed; continue; }
            auto it = dps.find(r.x0);
            if (it == dps.end()) { dps.emplace(r.x0, rec); continue; }
            const DPRec& old = it->second;
            if (old.x1 != rec.x1) continue;
            if (old.type == rec.type && old.d[0] == rec.d[0] && old.d[1] == rec.d[1] && old.d[2] == rec.d[2]) { ++dupCount; continue; }
            if (old.type == 0 && rec.type == 0) continue;
            if (!resolve(old, rec)) printf("[!] collision did not yield the key (x %016llx)\n", (unsigned long long)r.x0);
        }
        for (u32 k = 0; k < w.nstuck; ++k) { start_kang(w, (int)w.stuck[k]); ++regenTotal; ++dupCount; }   // same DP twice: restart (RC regen)
        if (o.dpstream) fflush(stdout);
    };

    // ---- run: all threads walk one launch, then the host processes their DPs
    const double t0 = now_s(); double tLast = t0, opsLast = 0; u64 launches = 0;
    const double maxOps = o.maxK > 0 ? o.maxK * expOps : 0;
    while (!found) {
        std::vector<std::thread> th;
        for (int t = 0; t < T; ++t) th.emplace_back([&, t]() { walker_run(TB, W[t], STEP_CNT); });
        for (auto& x : th) x.join();
        ++launches;
        for (int t = 0; t < T; ++t) { loopsTotal += W[t].loops; W[t].loops = 0; process(W[t]); }
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
