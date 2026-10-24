# kangaroo/rc: RCKangaroo's kernels (RetiredCoder, GPLv3)

Verbatim copies from https://github.com/RetiredC/RCKangaroo (commit 618473a, 15 Oct 2026), licence in `LICENSE.TXT`:

| File | What |
|---|---|
| `kernel_sm89.cubin`, `kernel_sm120.cubin` | the v4 **turbo** kernels `KernelA` (walk + dedicated inversion SMs) and `KernelB` (distance replay, loops), hand-written in RCAsm. RTX 40 / RTX 50 |
| `main.asm`, `fuse.asm`, `mod_mul.asm`, `mod_inv.asm`, `mod_sub.asm`, `newKernelB.asm` | their source (RCAsm syntax; `MulMod256` is 112 SASS instructions with 5 carry chains) |
| `RCGpuCore.cu`, `RCGpuUtils.h`, `defs.h` | the CUDA kernels: `KernelA`/`KernelB` (used when no cubin fits the GPU), `KernelC` (table-3 escape jumps, always), `KernelGen`; `TKparams` |
| `rc_glue.cu` | ours: C-linkage wrappers so `kangaroo.cu` drives the CUDA kernels without sharing RC's typedefs |

Ours, built with `../sass/`: `kernel_sm89_sqr.cubin` = `kernel_sm89.cubin` reassembled from this source with `SqrAddMod256`
replaced by `mod_mul_sqr.asm`'s true square (46 wide multiplies instead of 62; +2% measured, 14.58 vs 14.29 G on a 4090).
`kangaroo --rc` prefers it when present (GPLv3 as the rest; source = `*.asm` with `mod_mul_sqr.asm` in place of `SqrAddMod256`).

`kangaroo --rc` (see `../README.md`) runs these with our jump tables, start points, DP handling and fleet plumbing.
This folder ships inside the worker image at `/app/rc`, source included, as the GPL requires.
