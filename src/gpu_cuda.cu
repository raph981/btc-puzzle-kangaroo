// CUDA backend. Each GPU thread owns GPU_GRP kangaroos and runs the very same
// kang_step() routine as the CPU backend (see kernel.h).
//
// Build with `make gpu` (needs the CUDA toolkit / nvcc).
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "gpu.h"

static Fe *d_px, *d_py, *d_jx, *d_jy;
static u64 *d_dlo, *d_dhi, *d_jd;
static DPRec *d_out;
static u32 *d_cnt;
static u32 g_nk, g_maxOut;
static char g_name[256];

__global__ void kang_kernel(Fe *px, Fe *py, u64 *dlo, u64 *dhi, int nsteps,
                            const Fe *jx, const Fe *jy, const u64 *jd, u64 dpmask,
                            DPRec *out, u32 *cnt, u32 maxOut, u32 nk) {
    u32 base = (blockIdx.x * blockDim.x + threadIdx.x) * GPU_GRP;
    if (base >= nk) return;
    Fe dx[GPU_GRP], pre[GPU_GRP];
    kang_step(px + base, py + base, dlo + base, dhi + base, GPU_GRP, nsteps,
              jx, jy, jd, dpmask, out, cnt, maxOut, base, dx, pre);
}

int gpu_device_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

const char *gpu_device_name(int dev) {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return "unknown";
    snprintf(g_name, sizeof g_name, "%s (%d MB)", p.name, (int)(p.totalGlobalMem >> 20));
    return g_name;
}

#define CK(call)                                                              \
    do {                                                                      \
        cudaError_t e_ = (call);                                              \
        if (e_ != cudaSuccess) {                                              \
            snprintf(err, errLen, "%s: %s", #call, cudaGetErrorString(e_));   \
            return false;                                                     \
        }                                                                     \
    } while (0)

bool gpu_init(int dev, u32 nk, const Fe *jx, const Fe *jy, const u64 *jd, u32 maxOut, char *err, int errLen) {
    g_nk = nk; g_maxOut = maxOut;
    CK(cudaSetDevice(dev));
    CK(cudaMalloc(&d_px, sizeof(Fe) * nk));
    CK(cudaMalloc(&d_py, sizeof(Fe) * nk));
    CK(cudaMalloc(&d_dlo, sizeof(u64) * nk));
    CK(cudaMalloc(&d_dhi, sizeof(u64) * nk));
    CK(cudaMalloc(&d_jx, sizeof(Fe) * NJ));
    CK(cudaMalloc(&d_jy, sizeof(Fe) * NJ));
    CK(cudaMalloc(&d_jd, sizeof(u64) * NJ));
    CK(cudaMalloc(&d_out, sizeof(DPRec) * maxOut));
    CK(cudaMalloc(&d_cnt, sizeof(u32)));
    CK(cudaMemcpy(d_jx, jx, sizeof(Fe) * NJ, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_jy, jy, sizeof(Fe) * NJ, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_jd, jd, sizeof(u64) * NJ, cudaMemcpyHostToDevice));
    return true;
}

void gpu_upload(const Fe *px, const Fe *py, const u64 *dlo, const u64 *dhi) {
    cudaMemcpy(d_px, px, sizeof(Fe) * g_nk, cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py, sizeof(Fe) * g_nk, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dlo, dlo, sizeof(u64) * g_nk, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dhi, dhi, sizeof(u64) * g_nk, cudaMemcpyHostToDevice);
}

u32 gpu_run(int nsteps, u64 dpmask, DPRec *out) {
    const u32 block = 128;
    u32 threads = g_nk / GPU_GRP;
    u32 grid = (threads + block - 1) / block;
    cudaMemset(d_cnt, 0, sizeof(u32));
    kang_kernel<<<grid, block>>>(d_px, d_py, d_dlo, d_dhi, nsteps, d_jx, d_jy, d_jd,
                                 dpmask, d_out, d_cnt, g_maxOut, g_nk);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(e));
        return 0;
    }
    u32 cnt = 0;
    cudaMemcpy(&cnt, d_cnt, sizeof(u32), cudaMemcpyDeviceToHost);
    if (cnt > g_maxOut) cnt = g_maxOut;
    if (cnt) cudaMemcpy(out, d_out, sizeof(DPRec) * cnt, cudaMemcpyDeviceToHost);
    return cnt;
}

void gpu_set(u32 idx, const Fe &x, const Fe &y, u64 dlo, u64 dhi) {
    cudaMemcpy(d_px + idx, &x, sizeof(Fe), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py + idx, &y, sizeof(Fe), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dlo + idx, &dlo, sizeof(u64), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dhi + idx, &dhi, sizeof(u64), cudaMemcpyHostToDevice);
}

void gpu_free() {
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_dlo); cudaFree(d_dhi);
    cudaFree(d_jx); cudaFree(d_jy); cudaFree(d_jd); cudaFree(d_out); cudaFree(d_cnt);
}
