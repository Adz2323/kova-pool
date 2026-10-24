# kangaroo: GPU Pollard kangaroo (prototype). Field maths from ../CUDAMath.h (+ ../fmul32.h with EOMUL=1), as bsgs.
#   make                                   build for the GPU in this machine
#   make GPU_ARCH= SM_ARCHS="86 89 90 100 120"   RTX 30 / 40 / H100 / B200 / RTX 50 in one binary (CUDA 12.8+ for 100/120)
# --rc (RCKangaroo's kernels, GPLv3) links rc/RCGpuCore.cu (CUDA KernelA/B/C/Gen) and loads rc/kernel_sm89.cubin /
# rc/kernel_sm120.cubin at run time: keep the rc/ folder next to the binary (or in /app/rc in the worker image).
TARGET   := kangaroo
GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')
SM_ARCHS ?= $(GPU_ARCH)
EOMUL    ?= 1
# INC: where CUDAMath.h / fmul32.h live (the worker image copies them next to kangaroo.cu and builds with INC=.)
INC      ?= ..
GENCODE  := $(foreach a,$(SM_ARCHS),-gencode arch=compute_$(a),code=sm_$(a))
DEFS     :=
ifeq ($(EOMUL),1)
DEFS += -DCC_EOMUL
endif
NVCC     := nvcc -O3 -std=c++17 -diag-suppress 20014,20011 $(NVCC_EXTRA)

$(TARGET): kangaroo.o rc/RCGpuCore.o rc/rc_glue.o
	$(NVCC) $(GENCODE) -Xcompiler -pthread $^ -lcuda -o $@

kangaroo.o: kangaroo.cu modinv30.h fred32.h $(INC)/CUDAMath.h $(INC)/fmul32.h Makefile
	$(NVCC) --ptxas-options=-O3 $(GENCODE) $(DEFS) -Xcompiler -pthread -I$(INC) -c kangaroo.cu -o $@

rc/RCGpuCore.o: rc/RCGpuCore.cu rc/RCGpuUtils.h rc/defs.h Makefile
	$(NVCC) $(GENCODE) -Irc -c rc/RCGpuCore.cu -o $@

rc/rc_glue.o: rc/rc_glue.cu rc/defs.h Makefile
	$(NVCC) $(GENCODE) -Irc -c rc/rc_glue.cu -o $@

clean:
	rm -f $(TARGET) kangaroo.o rc/RCGpuCore.o rc/rc_glue.o
