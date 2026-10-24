# kangaroo: Pollard kangaroo (SOTA-style, symmetric walk) for secp256k1 on NVIDIA GPUs (prototype)

Solves **k·G = Q for a known public key Q** in a range `[start, start + 2^range)`. Like `bsgs`, it needs the public
key, so it applies to the puzzles whose key is known (135, 140, … 160) and to any pubkey-known target. Unlike BSGS it
needs almost no memory and its success probability grows with the **square** of the work: expected cost
**≈ 1.15 × 2^(range/2) jumps** (K = 1.15), which beats BSGS for ranges above roughly 2^70 (see `bsgs/README.md`).

The design follows RetiredCoder's RCKangaroo (SOTA v2 method); the code is written fresh on CUDACyclone's field maths.

## How it works

| Part | What |
|---|---|
| Kangaroos | ~786k on a 4090 (SMs × 256 threads × 24). 1/3 **tame** `T = d·G`, 2/3 **wild** `W = d·G − P'` with `P' = Q − start·G + x32·G` (x32 = 2^(range−5) keeps the key off the range edges). Each carries a signed 192-bit distance `d` |
| Jump | Index from x (`x[0] & 511`), **direction from the parity of y** (even → +J, odd → −J). A point and its negation then walk mirror paths with the same x: the walk is on {P, −P} classes, which is where the √2 saving (K 2.1 → 1.15) comes from. Jump table 1: 512 distances in [2^(range/2+3), 2^(range/2+4)), even |
| Batch | 24 kangaroos per thread, one Montgomery batch inversion per step; jump table 1 lives in shared memory |
| Loops | The symmetric walk cycles. 2-cycles (P → P+J → P, ~1 per 1024 jumps) are detected right after the jump and the next jump comes from table 2 (jumps ~2^(range−10)). Longer cycles (4, 6, … ≤ 10) are found by `k_dist` from a ring of the last 10 distances and escaped with one jump from table 3 (~2^(range−12)) |
| Kernels | `k_jump`: 1000 steps per launch, writes a 2-byte record per jump (index, direction, table, DP flag) · `k_dist`: replays the records to update distances, emit DPs and detect loops · `k_escape`: table-3 jump for looped kangaroos |
| DPs | Top `--dp` bits of x zero. Records (x, d, kangaroo) go to a host hash map. Tame/wild collision: `k' = w ∓ t`; wild/wild: `k' = (w1 + w2)/2`. Every candidate is verified by `k·G == Q` before it is reported |
| CPU mode | `--cpu` runs the identical algorithm on the host (small herds) · `--check` verifies every kangaroo against its distance after every launch |

## `--rc`: RCKangaroo's turbo kernels (14.5 G/4090, 19.3 G/5090)

RetiredCoder published the **source** of RCKangaroo v4's turbo kernels (RCAsm, GPLv3) along with the cubins. `--rc`
runs them under our solver: `rc/kernel_sm89.cubin` (RTX 40) / `rc/kernel_sm120.cubin` (RTX 50) supply `KernelA`
(the walk, with `SMs/32` SMs (RTX 50: `/24`) dedicated to inversions: the "triple Montgomery trick") and `KernelB`
(distance replay, loop detection); RC's CUDA `KernelC` (table-3 escapes) and, on other GPUs, his CUDA `KernelA`/`KernelB`
come from `rc/RCGpuCore.cu`. Everything else is ours and unchanged: seeded jump tables (every worker walks one graph),
`k_start`, DP handling, collision resolution, `--dpstream`, `--on-find`, `--check`. RC's state layout turned out to be
ours (x at `4i`, y at `4i + 4·NK`, 4-word distances, `i = thread + group·blocks·256`), so nothing is copied around.

| Option | Meaning |
|---|---|
| `--rc` / `--rc auto` | turbo cubin when one fits the GPU, else RC's CUDA kernels |
| `--rc cubin` / `--rc cuda` | insist on one of them; `--rc-cubin FILE` picks the cubin (default: `rc/kernel_sm<cc>.cubin` next to the binary, in `./rc`, `kangaroo/rc` or `/app/rc`) |
| `--rc-inv N` | inversion SMs (default `SMs/32`, RTX 50 `/24`, min 1); `--blocks N` (≥ 12) = walk blocks for small `--check` runs |

Fixed by RC's kernels: 24 kangaroos per thread, 1000 steps per launch, one block per SM. **DP test differs**: RC checks
bits 224..224+dp of x, ours the top dp bits, so run a herd of one kind only (a mixed console table is safe but the two
kinds never collide with each other). `make` builds `rc/RCGpuCore.o` in; `make rczip` packs `rc/` for the worker image.
Tests 5-7 of the pod test script (not in this repository) are the RC consistency (`--check` against the host), correctness and speed runs;
test 8 rebuilds the cubin with the SASS tooling (not in this repository). **Measured 15 Oct 2026 on a RunPod 4090: `--rc` 14.09 G jumps/s, `--rc cuda`
7.67 G, 40-bit `--check` clean, 48/64-bit keys found (K = 4.1 at 64 bits).**

## Build and test

```bash
make                                          # for the GPU in this machine (links rc/RCGpuCore.cu; --rc loads rc/*.cubin)
make GPU_ARCH= SM_ARCHS="86 89 90 100 120"    # RTX 30/40, H100, B200, RTX 50 (CUDA 12.8+)
./kangaroo --cpu --threads 2 --selftest --range 40     # CPU sanity check (no GPU needed), ~1 s
./kangaroo --selftest --range 64                       # GPU: random 64-bit key, ~1 s
./kangaroo --bench --range 70 --bench-rounds 5         # K over 5 random keys
./kangaroo --pubkey 02ABC... --range 135 --start 4000000000000000000000000000000000   # puzzle 135
```

| Option | Meaning |
|---|---|
| `--pubkey HEX --range R [--start HEX]` | solve k in [start, start + 2^R) (R = 20..170) |
| `--selftest --range R`, `--bench --range R --bench-rounds N` | random key(s) in the range, solve, print K |
| `--dp N` | distinguished-point bits (default: about 8 DPs per kangaroo over the expected run) |
| `--bps N` / `--blocks N` | thread blocks per SM (default 1) / block count |
| `--time S`, `--max K` | stop after S seconds / after K × expected jumps |
| `--check` | verify every kangaroo's point against its distance after each launch (slow: use `--blocks 2` on a GPU) |
| `--groups N`, `--ilp 1\|2`, `--fused 0\|1`, `--inv 1\|0`, `--l2 0\|1`, `--minblocks`, `--blocks`, `--alias N` | kernel variants (defaults: 24, 2, 1, 1, 1): see Measured |
| `--steps N`, `--debug`, `--invtest`, `--hoststart`, `--starttest N` | jumps per launch; per-kernel timing; device inversion/multiplication test; start points on the host; start-kernel bisect |
| `--on-find CMD`, `--agent`, `--out FILE`, `--seed N`, `--report S`, `--gpu N`, `--cpu [--threads N]` | as in `bsgs` |

Pod test script (not in this repository): 0 GPU consistency check · 1–2 random 48/64-bit keys · 3 K over five 70-bit keys ·
4 jumps/s at `--bps 1` and `2`. Ends with `KANGAROO TESTS DONE`.

## Measured

| | |
|---|---|
| CPU, 2 threads, 48 kangaroos | keys found on every selftest 28–50 bits; K averaged 1.66 over six 50-bit solves (heavy-tailed; RCKangaroo quotes 1.15 plus DP overhead over many solves) |
| **RTX 4090, 786k kangaroos (defaults)** | 48-bit key found in the first launch; **7.74 G jumps/s** sustained (2.4 GB VRAM); loops escaped ≈ 1 per 10^6 jumps, as RCKangaroo reports |

How it got from 4.81 G (v1) to 7.74 G, each step measured on the same 4090 (`run_tests.sh` test 4 sweeps the variants):

| Change | jumps/s |
|---|---|
| v1: 24 kangaroos/thread, forward+backward pass, CUDAMath `_ModInv`, EOMUL mul | 4.81 G |
| per-thread flag bitmask instead of per-kangaroo flag loads; DP slot atomic only on a DP; streaming record stores | 5.35 G |
| L2 persistence window over the kangaroo state (`--l2 1`) | 6.56 G |
| fused step: no forward pass, steps alternate direction (`--fused 1`) | 6.67 G |
| safegcd 30-bit-divstep inversion (`modinv30.h`, libsecp256k1 port; 1.96 G inv/s vs 1.29 G) | 6.79 G |
| 32-bit carry-chained reduction for mul/sqr (`fred32.h`: 137/129 SASS vs 165/157), `ModNeg256`, LDS table reads | 7.23 G |
| two kangaroos per iteration, arithmetic interleaved, branch-free jump (`--ilp 2`) | **7.74 G** |

Things that did not help on the 4090: prefix products in shared memory, 2 blocks per SM, more warps with fewer
kangaroos per thread (the inversion is amortised over fewer points), 32/48 kangaroos per thread (state outgrows the
52 MB L2 window). With all state folded into L1 (`--alias 4096`, a probe) the kernel does 8.2 G: memory costs ~6%.

**Where the rest is.** A jump is ~1,130 SASS instructions in the loop plus ~240 for the amortised inversion. RCKangaroo's
CUDA kernels run at about this speed (~8 G); its 14.5 G comes from hand-written SASS: a 112-instruction modmul using
3-operand `IADD3.X` adds with two carry flags (not expressible in PTX; ours is 137) and inversions farmed to dedicated
SMs. From CUDA/PTX the next cuts are the product-scanning part of the multiply (~20 per mul) and a dedicated squaring;
beyond ~9–10 G is assembler territory.

**Inversion note.** CUDAMath's `_ModInv` (VanitySearch's DivStep62) hung the GPU when called from the start-point and
loop-escape kernels while running clean in the walk; both now use a loop-free Fermat inversion (`DevOpsF`), the walk
uses safegcd. `--starttest N` bisects it (modes 3–5 call `_ModInv` and will wedge the GPU: pod restart).

## Pool mode: every GPU in one herd

`--dpstream` prints every distinguished point as a `DP <x:32 hex><distance:48 hex, 192-bit two's complement><type>`
line; `pool/pool_agent.py` batches them to the pool (`POST /api/worker/dps`), which verifies them and keeps the single
herd-wide DP table, detects the first tame/wild or wild/wild collision and derives the key on its own server. Jump
tables are built from a fixed seed, so every worker walks the same graph; start points are random per worker. N GPUs
therefore behave as one N-times-larger herd (total expected work unchanged at 1.15·2^(range/2)). `--species tame|wild`
runs one herd only (the pool assigns it per token) and `--blind R` starts wild kangaroos from a point the pool has
shifted, so no single worker holds both sides of a collision. `--dplocal 0` drops the worker's own table.

## Known limits (prototype)

- One key per run; no save/resume of the tame/wild positions themselves (the pool's DP table is persistent).
- Local DP memory ≈ 64 B per DP; the start-up log prints the expected count. Raise `--dp` for big ranges (or `--dplocal 0`).
- Cycles longer than 10 are not caught (RCKangaroo's estimate: a handful per day per GPU; they only waste that kangaroo).
