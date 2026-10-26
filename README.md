# Kova Pool worker

What changed and when: [UPDATES.md](UPDATES.md).

The open-source worker for [pool.kova23.com](https://pool.kova23.com): a pooled Pollard-kangaroo search for the key
behind a Bitcoin puzzle public key, run as one herd across everyone's GPUs. This repository holds everything that runs
on a member's machine; the pool server (accounts, verification, shares) is separate; the payout script is here so the split can be checked.

What's here:

| Path | What |
|---|---|
| `kangaroo/kangaroo.cu` (+ `CUDAMath.h`, `fmul32.h`, `modinv30.h`, `fred32.h`) | the GPU solver: symmetric kangaroo walk, DP streaming, `--species`, `--blind` |
| `kangaroo/rc/` | RetiredCoder's RCKangaroo turbo kernels (GPLv3) used by `--rc`, with their source |
| `pool/pool_agent.py`, `pool/Dockerfile` | the Linux worker: fetches the job, runs the solver, streams DPs, heartbeats. Image `adz23/kova-pool` |
| `pool/windows/` | KovaPool for Windows: tray app, idle-only and gentle modes; runs `kangaroo.exe` natively or via Docker Desktop |
| `pool/payout_electrum.py` | how a find is paid out: the share ledger → an Electrum pay-to-many list plus the public ledger that is published with the txid. The same script the operator runs, so you can check the split |
| `.github/workflows/windows-worker.yml` | builds `kangaroo.exe` + `KovaPool.exe` and publishes them on the release `windows-latest` |

## Run it

Make a token on your account page at pool.kova23.com, then:

- **Linux:** `curl -fsSL https://pool.kova23.com/join.sh | sudo bash -s -- pk_YOURTOKEN`
  (or `docker run -d --gpus all --restart unless-stopped -e POOL_TOKEN=pk_… adz23/kova-pool:5`)
- **Windows:** download `KovaPool.exe` from the latest `windows-latest` release, run it, paste the token.
- **RunPod / Vast.ai:** a pod with image `adz23/kova-pool:5` and env `POOL_TOKEN`.

## What the worker does, and doesn't

It knows a public key, a range and your pool token; nothing else. Each token runs one species (tame or wild), so a
worker never holds both trails of a collision; wild workers start from a point the pool has blinded. Distinguished
points go to the pool, which verifies a sample by re-deriving them, and the key, if found, is derived on the pool's
server. The worker never touches a wallet.

## No GPU? The CPU worker

`kangaroo/cpu/kangaroo_cpu.cpp` is the same walk in plain C++ (no CUDA): it runs the pool's graph on a Mac, a Linux
box or (soon) a phone, and its points collide with the GPUs' points. One line on a Mac or Linux machine:

    curl -fsSL https://pool.kova23.com/cpu.sh | bash -s -- pk_YOURTOKEN        # add a number for fewer cores

or by hand: `cd kangaroo/cpu && c++ -O3 -std=c++17 -pthread -I.. -o kangaroo_cpu kangaroo_cpu.cpp`, then
`KANG_CPU=1 KANG_BIN=$PWD/kangaroo_cpu POOL_TOKEN=pk_... python3 ../../pool/pool_agent.py`.
`./kangaroo_cpu --selftest --range 40` solves a random 40-bit key in under a second.

## Build from source

Linux: `docker build -f pool/Dockerfile -t kova-pool .` from this directory (needs the CUDA 12.8 base images), or
`cd kangaroo && make`. Windows: see `pool/windows/README.md` (the GitHub Actions workflow is the reference build).

## Licence

GPL-3.0 (see `LICENSE`): the solver links RCKangaroo's GPLv3 kernels, so the combined work is distributed under the
same terms. `CUDAMath.h` carries its own notice from CUDACyclone / VanitySearch.
