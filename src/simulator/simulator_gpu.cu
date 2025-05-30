#include "vectorspace.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

namespace
{
    __constant__ int d_Nx, d_Ny, d_Nz;
    __constant__ float d_inv_h2, d_c2_dt2, d_gdt;
}

__global__ void fdtd_kernel(const float*  __restrict pPrev,
                            const float*  __restrict pCurr,
                            float*        __restrict pNext,
                            const uint8_t*__restrict flags)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= d_Nx || y >= d_Ny || z >= d_Nz) return;

    if (x==0 || x>=d_Nx-1 || y==0 || y>=d_Ny-1 || z==0 || z>=d_Nz-1) return;
    std::size_t idx = ( (std::size_t)z * d_Ny + y ) * d_Nx + x;
    //pNext[idx] = 1.0f;

    /* wall voxels stay zero */
    if (flags[idx]) { pNext[idx] = 0.0f; return; }

    auto V = [&](int ix,int iy,int iz)->float {
        std::size_t n = ( (std::size_t)iz * d_Ny + iy ) * d_Nx + ix;
        return flags[n] ? 0.0f : pCurr[n];
    };

    float pc = pCurr[idx];
    float lap = ( V(x+1,y,z) + V(x-1,y,z) +
                  V(x,y+1,z) + V(x,y-1,z) +
                  V(x,y,z+1) + V(x,y,z-1) - 6.0f*pc ) * d_inv_h2;

    pNext[idx] =
          (2.0f - d_gdt) * pc
        - (1.0f - d_gdt) * pPrev[idx]
        + d_c2_dt2 * lap;
}

extern "C"
{
    void fdtd_gpu_step(VectorSpace *space, float h)
    {
        Grid &g = space->getGrid(); // host meta
        //const std::size_t N = g.Nx * g.Ny * g.Nz;

        /* one‑time constant upload (cache inside space) -------------------- */
        static bool constantsUploaded = false;
        int iNx = int(g.Nx);
        int iNy = int(g.Ny);
        int iNz = int(g.Nz);
        if (!constantsUploaded)
        {
            float c = 343.f;
            float dt = 0.5f * h / (c * std::sqrt(3.f));
            float c2_dt2 = c * c * dt * dt;
            float gdt = 5.f * dt;
            float inv_h2 = 1.f / (h * h);
            cudaMemcpyToSymbol(d_Nx, &iNx, sizeof(int));
            cudaMemcpyToSymbol(d_Ny, &iNy, sizeof(int));
            cudaMemcpyToSymbol(d_Nz, &iNz, sizeof(int));
            cudaMemcpyToSymbol(d_inv_h2, &inv_h2, sizeof(float));
            cudaMemcpyToSymbol(d_c2_dt2, &c2_dt2, sizeof(float));
            cudaMemcpyToSymbol(d_gdt, &gdt, sizeof(float));

            constantsUploaded = true;
        }

        /* device pointers held inside VectorSpace -------------------------- */
        float *d_prev = g.d_p_prev;
        float *d_curr = g.d_p_curr;
        float *d_next = g.d_p_next;
        uint8_t *d_flags = g.d_flags;

        /* launch geometry --------------------------------------------------- */
        dim3 B(8, 8, 8);
        dim3 G((iNx + B.x - 1) / B.x,
               (iNy + B.y - 1) / B.y,
               (iNz + B.z - 1) / B.z);

        fdtd_kernel<<<G, B>>>(d_prev, d_curr, d_next, d_flags);

        // CUDA_CHECK(cudaGetLastError());        // macro or manual check
        // CUDA_CHECK(cudaDeviceSynchronize());   // catches bad constants immediately
        cudaDeviceSynchronize();   // ensure completion before we copy

        // /* swap pointers for next iteration ---------------------------------- */
        std::swap(g.d_p_prev, g.d_p_curr);
        std::swap(g.d_p_curr, g.d_p_next);
    }


}