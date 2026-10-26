# Kova Pool updates

What changed, when, and what it means for a member. Newest first. Each entry names the commit it came from; the
worker side (solver, agent, Windows app, Docker image) is in this repository, the pool server is not public, so
server-side changes are described here rather than linked.

## 2026-10-26 — share points: every point checked for rarity, duplicates refused (image :6)

Two bugs fixed together on the pool side, and the worker changed to match
([commit](https://github.com/Adz2323/kova-pool/commit/7c7d23e)).

- The pool verified that each point a worker handed in was a genuine point of the walk, but not that it was *rare*
  (distinguished). A worker could have sent every point it visited and been credited for all of them. Now the pool
  checks the rarity criterion on every point and refuses the batch if one fails (HTTP 400 "not a share point").
- Points are deduplicated in the pool's own table: a point handed in twice, by anyone, earns nothing the second time.
- The GPU turbo kernels test rarity on 32 bits of x at most, so the console's "dp 47" was effectively dp 32 on that
  path. Three levels now: the solver runs at dp 32 (candidates), the pool pays for **share points** (candidates whose
  next bits are also zero: dp 40 for GPUs, about one a minute per RTX 5090; dp 32 for the CPU tier), and only
  **collision points** at the console's dp go into the shared table. A share is worth 2^dp jumps at the tier's dp, so
  both tiers pay the same per jump and nobody gains by picking a tier.
- Worker: `pool_agent.py` runs the solver at the job's `cand_dp` and hands in only share points (`share_mask()`);
  `KovaPool` for Windows does the same (`shareMsk` in `native.go`). Old images get 400s on most batches: update to
  `adz23/kova-pool:6`.
- Shares earned before this change were reset (they were the operator's own GPUs; no member work was lost).

## 2026-10-25 — payout script published

`pool/payout_electrum.py` ([commit](https://github.com/Adz2323/kova-pool/commit/4f61228)): turns the share ledger into
an Electrum pay-to-many list and a public ledger JSON, so anyone can check the split. It never signs or broadcasts;
the operator pays by hand from Electrum.

## 2026-10-25 — RTX 50-series hang fixed (image :5)

`kangaroo/rc/kernel_sm120.cubin` declared 0 registers for its kernels in the ELF header where the sm_89 build declares
255. Datacenter Linux drivers worked around it; the GeForce driver on Windows-hosted VMs (Salad, drivers 610–621) gave
the kernel 8 registers and the first launch never returned. Two header bytes fixed; no kernel code changed. RunPod
5090: unchanged at 18.4–19.0 G jumps/s; Salad 5090: hang → 17.9–18.1 G/s. The same cubin ships in the Windows release.
Agent: `KANG_CUBIN_URL` downloads a kernel build at start (`--rc-cubin`) for experiments without a rebuild
([commit](https://github.com/Adz2323/kova-pool/commit/d8d2f41)).

## 2026-10-25 — bad-host watchdog, host reports, Salad reallocate (images :3, :4)

- The agent exits if the solver has not reported within 120 s (`STARTUP_TIMEOUT_S`) or runs under `MIN_GJPS`, logs the
  host's driver / bus / memory / power / clock, reports it to the pool (`worker.host`, `worker.badhost`), and on Salad
  asks for a new node (IMDS reallocate) instead of restarting on the same bad one.
- Solver output is line-buffered (`stdbuf -oL`) so start-up lines show at once; network errors are retried.
  ([commits](https://github.com/Adz2323/kova-pool/commits/main))

## 2026-10-24 — first public release

GPU kangaroo solver (CUDA, with RCKangaroo's turbo kernels for RTX 40/50 under GPLv3), the Linux agent and Docker
image, `KovaPool` for Windows (no Docker needed; pauses when you game), and the GitHub Actions build that produces the
Windows release ([commit](https://github.com/Adz2323/kova-pool/commit/201587f)).
