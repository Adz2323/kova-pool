# KovaPool for Windows

A tray app for people who want to lend a Windows gaming PC to the pool without touching a terminal. Two engines:

- **Built-in solver (default, no Docker):** downloads `kangaroo.exe` (built by the `windows-worker` GitHub Actions
  workflow from the same `kangaroo.cu`, published on the rolling release `windows-latest`) into
  `%LOCALAPPDATA%\KovaPool\solver` and runs it directly. `native.go` is a Go port of `pool_agent.py` (job, DP
  streaming, heartbeat, restart on target change, stop on find); pause/resume use NtSuspendProcess/NtResumeProcess.
- **Docker Desktop (WSL 2):** runs the `adz23/kova-pool` image, same as Linux.

What Windows needs on top:

- a settings page (opens in the browser at `http://127.0.0.1:47716`): paste the token from your account page, Save and
  start, see Docker / GPU status and the worker's log
- tray icon menu: open settings, open your account page, pause / resume, start / stop, quit
- **idle only** (power mode set on the account page): the worker is paused the moment you touch the PC and resumes
  after N idle minutes (default 5, settable)
- **gentle**: a GPU power cap at 60% when KovaPool runs as administrator, otherwise 3 s running / 2 s paused
- start with Windows (registry Run key), single instance (a second launch just opens the settings page)

Mode (pooled / solo) and power (full / gentle / idle) come from the account page: the app asks the pool for the token's
settings every second and applies changes at once.

Needs on the PC: a current NVIDIA driver. Docker Desktop only for the Docker engine.

## Windows build of the solver (CI)

MSVC (nvcc's only host compiler on Windows) has no `__int128`, which `kangaroo.cu` and `CUDAMath.h` use. The workflow
runs `winpatch.py`, which copies the sources into `build/` and substitutes `kp_int128.h`'s emulation (`kp_u128` /
`kp_i128`, host+device) plus shims for `__builtin_clzll`, `__builtin_ctz`, `setenv`. The repo sources are not changed.
The shim is tested on Linux by building the patched copy with nvcc and running the CPU self-test (key found) and a
tame-only DP stream, the same checks the Docker build does; the CI job repeats them on the Windows runner (CPU only,
the driver DLL is delay-loaded so the exe starts without a GPU). The Go agent was tested on Linux against a local pool
with the shim-built solver (job, 35k DPs accepted, pause/resume).

## Build

Standard library only, cross-compiled from anywhere:

```
cd pool/windows && GOOS=windows GOARCH=amd64 go build -ldflags "-H windowsgui -s -w" -o KovaPool.exe .
```

**Not yet run on a real Windows PC** (built and vetted on Linux, 24 Oct 2026). First test: a Windows machine with an
NVIDIA GPU and Docker Desktop, a test token, then check the tray icon, the settings page, start, idle pause/resume
(set the token to `idle` on the account page), and that the GPU shows live on the account page.
Code-signing the exe (SmartScreen) is a launch task.
