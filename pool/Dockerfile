# Kova Pool worker image: the Kangaroo solver + pool_agent.py. Anyone can run it; it joins the pool's herd with a
# POOL_TOKEN from the account page on pool.kova23.com. It runs ONE species (the pool decides) and streams its DPs to
# the pool, which forwards them into the herd's table; the pool's server derives any key. There is nothing secret in
# this image: no console address, no fleet token, no find key.
# Build from the REPO ROOT (needs kangaroo/, CUDAMath.h, fmul32.h, pool/pool_agent.py):
#   docker build -f pool/Dockerfile -t adz23/kova-pool:6 . && docker push adz23/kova-pool:6
# Run:  docker run -d --gpus all --restart unless-stopped -e POOL_TOKEN=pk_... adz23/kova-pool:6
#   (power modes "gentle"/"idle" need --cap-add SYS_ADMIN for the GPU power cap; otherwise the agent uses a duty cycle)
# RCKangaroo's turbo kernels (RetiredCoder, GPLv3) ship in /app/rc with their source, as the licence asks.
ARG CUDA_VERSION=12.8.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
WORKDIR /src
COPY CUDAMath.h fmul32.h /src/
COPY kangaroo/kangaroo.cu kangaroo/modinv30.h kangaroo/fred32.h /src/
COPY kangaroo/rc/ /src/rc/
RUN grep -q -- '--species' kangaroo.cu && grep -q -- '--blind' kangaroo.cu || (echo "kangaroo.cu is older than the pool needs (no --species / --blind)" && exit 1)
# sm_86 RTX 30 · sm_89 RTX 40 / L40 · sm_90 H100/H200 · sm_100 B200 · sm_120 RTX 50
RUN G=$(for a in 86 89 90 100 120; do printf -- "-gencode arch=compute_%s,code=sm_%s " $a $a; done) \
 && nvcc -O3 -std=c++17 -diag-suppress 20014,20011 --ptxas-options=-O3 $G -DCC_EOMUL -Xcompiler -pthread -I. -c kangaroo.cu -o kangaroo.o \
 && nvcc -O3 -std=c++17 -diag-suppress 20014,20011 $G -Irc -c rc/RCGpuCore.cu -o rc/RCGpuCore.o \
 && nvcc -O3 -std=c++17 -diag-suppress 20014,20011 $G -Irc -c rc/rc_glue.cu -o rc/rc_glue.o \
 && nvcc -O3 -std=c++17 $G -Xcompiler -pthread kangaroo.o rc/RCGpuCore.o rc/rc_glue.o -lcuda -o kangaroo \
 && mkdir -p /tmp/stubs && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stubs/libcuda.so.1 \
 && LD_LIBRARY_PATH=/tmp/stubs ./kangaroo --selftest --range 26 --cpu --threads 2 --seed 1 --report 999 | grep "KEY FOUND" \
 && LD_LIBRARY_PATH=/tmp/stubs ./kangaroo --pubkey 030d282cf2ff536d2c42f105d0b8588821a915dc3f9a05bd98bb23af67a2e92a5b --start 0 --range 36 --cpu --threads 2 --species tame --dpstream --dp 6 --time 5 --report 999 | grep -q "^DP " \
 && echo "tame-only streams DPs"

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/kangaroo /app/kangaroo
COPY --from=build /src/rc/kernel_sm89.cubin /src/rc/kernel_sm89_sqr.cubin /src/rc/kernel_sm120.cubin /src/rc/LICENSE.TXT /src/rc/*.asm /src/rc/*.cu /src/rc/*.h /app/rc/
COPY pool/pool_agent.py /app/
ENV PYTHONUNBUFFERED=1
CMD ["python3", "-u", "/app/pool_agent.py"]
