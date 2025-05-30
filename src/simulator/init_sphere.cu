#include <cuda_runtime.h>
#include "vectorspace.h"

__global__ void initSphereKernel(float* __restrict pCurr,
                                 std::size_t Nx, std::size_t Ny, std::size_t Nz,
                                 std::size_t xpos, std::size_t ypos, std::size_t zpos,
                                 std::size_t r2,          // radius^2
                                 float pressure, bool zeroOutside)
{
    std::size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    std::size_t z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= Nx || y >= Ny || z >= Nz) return;

    std::size_t dx = x > xpos ? x - xpos : xpos - x;
    std::size_t dy = y > ypos ? y - ypos : ypos - y;
    std::size_t dz = z > zpos ? z - zpos : zpos - z;

    std::size_t idx = (z * Ny + y) * Nx + x;

    if (dx*dx + dy*dy + dz*dz <= r2)
        pCurr[idx] = pressure;          // inside sphere
    else if (zeroOutside)
        pCurr[idx] = 0.f;               // optional reset
}

extern "C"{
void initPressureSphereGPU(VectorSpace* space,
                               std::size_t xpos, std::size_t ypos, std::size_t zpos,
                               std::size_t radius, float pressure, bool init)
{
    Grid& g = space->getGrid();

    /* ---- launch CUDA kernel ------------------------------------------ */
    dim3 B(8,8,8);
    dim3 G((g.Nx+B.x-1)/B.x,
           (g.Ny+B.y-1)/B.y,
           (g.Nz+B.z-1)/B.z);

    initSphereKernel<<<G,B>>>(
        g.d_p_curr,
        g.Nx, g.Ny, g.Nz,
        xpos, ypos, zpos,
        radius * radius,
        pressure, init);

    cudaDeviceSynchronize();   // ensure completion before we copy

    /* ---- copy to d_p_prev if init flag is true ----------------------- */
    // if (init) {
    //     std::size_t bytes = g.Nx * g.Ny * g.Nz * sizeof(float);
    //     cudaMemcpy(g.d_p_prev, g.d_p_curr,
    //                bytes, cudaMemcpyDeviceToDevice);
    // }
}
}