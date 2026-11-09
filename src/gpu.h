// Interface between the host program and the optional CUDA backend.
// Without WITH_CUDA the functions are stubs that report "no GPU support".
#pragma once
#include "kernel.h"

#define GPU_GRP 32   // kangaroos per GPU thread (they share one inversion)

#ifdef WITH_CUDA
int gpu_device_count();
const char *gpu_device_name(int dev);
// nk must be a multiple of GPU_GRP. Returns false and fills err on failure.
bool gpu_init(int dev, u32 nk, const Fe *jx, const Fe *jy, const u64 *jd, u32 maxOut, char *err, int errLen);
void gpu_upload(const Fe *px, const Fe *py, const u64 *dlo, const u64 *dhi);
// Runs nsteps jumps for every kangaroo; returns the number of DPs written to out.
u32 gpu_run(int nsteps, u64 dpmask, DPRec *out);
void gpu_set(u32 idx, const Fe &x, const Fe &y, u64 dlo, u64 dhi);
void gpu_free();
#else
static inline int gpu_device_count() { return 0; }
static inline const char *gpu_device_name(int) { return ""; }
static inline bool gpu_init(int, u32, const Fe *, const Fe *, const u64 *, u32, char *, int) { return false; }
static inline void gpu_upload(const Fe *, const Fe *, const u64 *, const u64 *) {}
static inline u32 gpu_run(int, u64, DPRec *) { return 0; }
static inline void gpu_set(u32, const Fe &, const Fe &, u64, u64) {}
static inline void gpu_free() {}
#endif
