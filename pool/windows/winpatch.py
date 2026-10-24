#!/usr/bin/env python3
"""winpatch.py SRC_ROOT OUT_DIR: copy the Kangaroo solver sources into OUT_DIR and patch the COPY so MSVC/nvcc can build
them on Windows (128-bit integer emulation from kp_int128.h, GCC builtins, setenv). The repo sources are not changed.
Every substitution has an anchor that must match exactly once; a miss fails the build loudly rather than silently
producing a different solver. Used by .github/workflows/windows-worker.yml and by the Linux shim test."""
import os, shutil, sys
src, out = sys.argv[1], sys.argv[2]
files = ["CUDAMath.h", "fmul32.h", "kangaroo/kangaroo.cu", "kangaroo/modinv30.h", "kangaroo/fred32.h"]
os.makedirs(os.path.join(out, "kangaroo", "rc"), exist_ok=True)
for f in files: shutil.copy(os.path.join(src, f), os.path.join(out, f))
for f in os.listdir(os.path.join(src, "kangaroo", "rc")):
    if not f.endswith(".o"): shutil.copy(os.path.join(src, "kangaroo", "rc", f), os.path.join(out, "kangaroo", "rc", f))
shutil.copy(os.path.join(src, "pool", "windows", "kp_int128.h"), os.path.join(out, "kp_int128.h"))

def patch(path, subs):
    p = os.path.join(out, path); s = open(p, encoding="utf-8").read()
    for a, b, n in subs:
        if s.count(a) != n: sys.exit(f"winpatch: anchor count {s.count(a)} != {n} in {path}: {a[:60]!r}")
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)

INC = '#include "kp_int128.h"\n'
# CUDAMath.h: device field add/sub use GCC 128-bit types -> emulation
patch("CUDAMath.h", [("__uint128_t", "kp_u128", 15), ("__int128_t", "kp_i128", 5)])
with open(os.path.join(out, "CUDAMath.h"), encoding="utf-8") as f: cm = f.read()
open(os.path.join(out, "CUDAMath.h"), "w", encoding="utf-8").write(INC + cm)
# kangaroo.cu: host 128-bit type + builtins
patch("kangaroo/kangaroo.cu", [("typedef unsigned __int128 u128;", 'typedef kp_u128 u128;', 1),
                               ('#include "CUDAMath.h"\n', INC + '#include "CUDAMath.h"\n', 1)])
# modinv30.h / fred32.h are included by kangaroo.cu after kp_int128.h, so the builtin macros are already in scope
print(f"winpatch: patched copy in {out}")
