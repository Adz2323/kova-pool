// kp_int128.h: 128-bit integer emulation for the Windows (MSVC) build of the Kangaroo solver.
//
// MSVC has no __int128, and nvcc only accepts MSVC as the host compiler on Windows, so neither host nor device code can
// use the GCC 128-bit types there. This header provides kp_u128 / kp_i128 with exactly the operations the solver's
// sources use ((u128)x casts, + - * with u64, >> by 64 or less, (u64) and bool conversions, > u64, < 0) and the build
// script (winpatch.py) substitutes them in a COPY of the sources: the Linux sources in the repo are untouched.
// Define KP_INT128_SHIM to use the emulation on any compiler (that's how it is tested on Linux against the real type).
#pragma once
#include <cstdint>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

#if defined(__CUDACC__)
#define KP_HD __host__ __device__ __forceinline__
#else
#define KP_HD inline
#endif

struct kp_u128 {
    uint64_t lo, hi;
    KP_HD kp_u128() : lo(0), hi(0) {}
    KP_HD kp_u128(uint64_t v) : lo(v), hi(0) {}                 // (u128)x
    KP_HD kp_u128(uint64_t l, uint64_t h) : lo(l), hi(h) {}
    KP_HD explicit operator uint64_t() const { return lo; }
    KP_HD explicit operator bool() const { return (lo | hi) != 0; }
    KP_HD kp_u128 operator+(const kp_u128& o) const { uint64_t l = lo + o.lo; return kp_u128(l, hi + o.hi + (l < lo ? 1 : 0)); }
    KP_HD kp_u128 operator-(const kp_u128& o) const { uint64_t l = lo - o.lo; return kp_u128(l, hi - o.hi - (lo < o.lo ? 1 : 0)); }
    KP_HD kp_u128& operator+=(const kp_u128& o) { *this = *this + o; return *this; }
    KP_HD kp_u128& operator-=(const kp_u128& o) { *this = *this - o; return *this; }
    KP_HD kp_u128 operator*(const kp_u128& o) const {           // full 128x128 -> low 128 bits
        uint64_t h;
#if defined(__CUDA_ARCH__)
        uint64_t l = lo * o.lo; h = __umul64hi(lo, o.lo);
#elif defined(_MSC_VER) && !defined(__clang__)
        uint64_t l = _umul128(lo, o.lo, &h);
#else
        unsigned __int128 p = (unsigned __int128)lo * o.lo; uint64_t l = (uint64_t)p; h = (uint64_t)(p >> 64);
#endif
        return kp_u128(l, h + lo * o.hi + hi * o.lo);
    }
    KP_HD kp_u128 operator>>(int n) const {
        if (n == 0) return *this;
        if (n >= 128) return kp_u128();
        if (n >= 64) return kp_u128(hi >> (n - 64), 0);
        return kp_u128((lo >> n) | (hi << (64 - n)), hi >> n);
    }
    KP_HD kp_u128 operator<<(int n) const {
        if (n == 0) return *this;
        if (n >= 128) return kp_u128();
        if (n >= 64) return kp_u128(0, lo << (n - 64));
        return kp_u128(lo << n, (hi << n) | (lo >> (64 - n)));
    }
    KP_HD kp_u128& operator>>=(int n) { *this = *this >> n; return *this; }
    KP_HD kp_u128& operator<<=(int n) { *this = *this << n; return *this; }
    KP_HD bool operator>(uint64_t v) const { return hi != 0 || lo > v; }
    KP_HD bool operator<(uint64_t v) const { return hi == 0 && lo < v; }
    KP_HD bool operator>=(uint64_t v) const { return !(*this < v); }
    KP_HD bool operator==(uint64_t v) const { return hi == 0 && lo == v; }
    KP_HD bool operator!=(uint64_t v) const { return !(*this == v); }
    KP_HD kp_u128 operator&(uint64_t v) const { return kp_u128(lo & v, 0); }
    KP_HD bool operator!() const { return (lo | hi) == 0; }
};
// mixed-operand forms the sources use: (u128)a * b, (u128)a + b, (u128)a - b - borrow, with any integer type on the right
#include <type_traits>
#define KP_INT template <class T, class = typename std::enable_if<std::is_integral<T>::value>::type>
KP_INT KP_HD kp_u128 operator+(const kp_u128& a, T b) { return a + kp_u128((uint64_t)b); }
KP_INT KP_HD kp_u128 operator-(const kp_u128& a, T b) { return a - kp_u128((uint64_t)b); }
KP_INT KP_HD kp_u128 operator*(const kp_u128& a, T b) { return a * kp_u128((uint64_t)b); }
KP_INT KP_HD kp_u128 operator+(T a, const kp_u128& b) { return kp_u128((uint64_t)a) + b; }
KP_INT KP_HD kp_u128 operator*(T a, const kp_u128& b) { return kp_u128((uint64_t)a) * b; }

// signed: only used as t = (i128)a - b - borrow; out = (u64)t; borrow = (t < 0)
struct kp_i128 : kp_u128 {
    KP_HD kp_i128() : kp_u128() {}
    KP_HD kp_i128(uint64_t v) : kp_u128(v) {}
    KP_HD kp_i128(const kp_u128& v) : kp_u128(v) {}
    KP_HD bool operator<(int) const { return (hi >> 63) != 0; }   // negative <=> top bit set (only ever compared with 0)
};
KP_INT KP_HD kp_i128 operator-(const kp_i128& a, T b) { return kp_i128((const kp_u128&)a - kp_u128((uint64_t)b)); }

// host-only GCC builtins used by the solver
#if defined(_MSC_VER) && !defined(__clang__)
inline int kp_clzll(unsigned long long x) { unsigned long i; _BitScanReverse64(&i, x); return 63 - (int)i; }
inline int kp_ctz(unsigned int x) { unsigned long i; _BitScanForward(&i, x); return (int)i; }
#define __builtin_clzll(x) kp_clzll(x)
#define __builtin_ctz(x) kp_ctz(x)
#include <cstdlib>
inline int kp_setenv(const char* k, const char* v, int) { return _putenv_s(k, v); }
inline int kp_unsetenv(const char* k) { return _putenv_s(k, ""); }
#define setenv kp_setenv
#define unsetenv kp_unsetenv
#endif
