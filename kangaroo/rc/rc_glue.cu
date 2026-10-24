// rc_glue.cu: C linkage around RCKangaroo's CUDA kernels (RCGpuCore.cu, GPLv3) so kangaroo.cu can drive them without
// sharing RC's typedefs. The TKparams struct is passed as an opaque blob whose layout kangaroo.cu mirrors (RcParams).
#include "defs.h"
#include <cstring>

extern cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table);
extern void CallGpuKernelGen(TKparams Kparams);
extern void CallGpuKernelA(TKparams Kparams);
extern void CallGpuKernelB(TKparams Kparams);
extern void CallGpuKernelC(TKparams Kparams);

extern "C" {
size_t rc_params_size(void) { return sizeof(TKparams); }
int rc_set_params(const void* p, const unsigned long long* jmp2_table) {
    TKparams k; memcpy(&k, p, sizeof k); return (int)cuSetGpuParams(k, (u64*)jmp2_table);
}
void rc_call_gen(const void* p) { TKparams k; memcpy(&k, p, sizeof k); CallGpuKernelGen(k); }
void rc_call_a(const void* p)   { TKparams k; memcpy(&k, p, sizeof k); CallGpuKernelA(k); }
void rc_call_b(const void* p)   { TKparams k; memcpy(&k, p, sizeof k); CallGpuKernelB(k); }
void rc_call_c(const void* p)   { TKparams k; memcpy(&k, p, sizeof k); CallGpuKernelC(k); }
}
