// sponge_kernel.cu
#include <cuda_runtime.h>
#include "vectorspace.h"

constexpr int N   = 1;                            // absorptionWidth
constexpr float EPS = 1.0f / float(N);            // 1/N

__global__ void spongeKernel(const uint8_t*   __restrict flags,
                             uint8_t*         __restrict flagsOut,
                             float*            __restrict pAbsorbOut,
                             std::size_t Nx, std::size_t Ny, std::size_t Nz)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    /* skip domain border (same as CPU version) */
    if (x<=0 || x>=Nx-1 || y<=0 || y>=Ny-1 || z<=0 || z>=Nz-1) return;

    auto idx = [&] (int ix,int iy,int iz)->std::size_t {
        return ( (std::size_t)iz * Ny + iy) * Nx + ix;
    };

    if( flags[idx(x,y,z)] != 0) { // already occupied
        flagsOut[idx(x,y,z)] = flags[idx(x,y,z)]; // copy flags
        //pAbsorbOut[idx(x,y,z)] = pAbsorb[idx(x,y,z)]; // copy absorbance
        return;
    }

    float minDist2 = 1e30f;        // squared distance
    float absorbValue = 0;
    /* scan neighbourhood cube ------------------------------------------------*/
    for (int dz=-N; dz<=N; ++dz)
    for (int dy=-N; dy<=N; ++dy)
    for (int dx=-N; dx<=N; ++dx)
    {
        if (!dx && !dy && !dz) continue;
        int nx = x+dx, ny=y+dy, nz=z+dz;
        /* skip if out of bounds (rare near edges) */
        if (nx<0||nx>=Nx||ny<0||ny>=Ny||nz<0||nz>=Nz) continue;

        std::size_t n = idx(nx,ny,nz);
        if (flags[n] == 1) {       // occupied wall voxel
            float d2 = float(dx*dx + dy*dy + dz*dz);
            minDist2 = fminf(minDist2, d2);
            //absorbValue = fmaxf(absorbValue, pAbsorb[n]); // max absorbance
        }
    }
    std::size_t id = idx(x,y,z);
    if (minDist2 < float(N*N+1)) {        // at least one wall voxel found
        float minD = sqrtf(minDist2);
        float pAbsorbLayer = fmaxf(0.0f, 1.0f - minD * EPS) * absorbValue; // absorbance based on distance
        //pAbsorbOut[id] = pAbsorbLayer; // set absorbance
        flagsOut[id]   = 2;           // mark as sponge
    }else{
        flagsOut[id] = 0;     // mark as empty
        //pAbsorbOut[id] = 0;   // set absorbance to zero
    }

    // flagsOut[idx(x,y,z)] = flags[idx(x,y,z)]; // copy flags
    // pAbsorbOut[idx(x,y,z)] = pAbsorb[idx(x,y,z)]; // copy absorbance
}


extern "C"
void buildSpongeLayerGPU(VectorSpace* space)
{
    float *outAbsorb;
    uint8_t *outFlags;
    cudaMalloc((void**)&outAbsorb, space->getGrid().size * sizeof(float));
    cudaMalloc((void**)&outFlags, space->getGrid().size * sizeof(uint8_t));
    //cudaMemcpy(outAbsorb, space->getGrid().p_absorb, space->getGrid().size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(outFlags, space->getGrid().flags, space->getGrid().size * sizeof(uint8_t), cudaMemcpyHostToDevice);
    Grid& g = space->getGrid();            // host meta only
    std::size_t Nvox = g.Nx * g.Ny * g.Nz;

    /* device arrays already exist */
    uint8_t* d_flags   = g.d_flags;        // flags on GPU
    //float*   d_absorb  = g.d_p_absorb;     // same size as grid

    /* copy flags and absorbance to device */
    cudaMemcpy(d_flags, g.flags, Nvox * sizeof(uint8_t), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_absorb, g.p_absorb, Nvox * sizeof(float), cudaMemcpyHostToDevice);

    /* launch geometry */
    dim3 B(8,8,8);
    dim3 G( (g.Nx+B.x-1)/B.x,
            (g.Ny+B.y-1)/B.y,
            (g.Nz+B.z-1)/B.z );

    spongeKernel<<<G,B>>>(d_flags, outFlags, outAbsorb, g.Nx, g.Ny, g.Nz);
    cudaDeviceSynchronize();               // wait & catch errors

    // copy results back to host
    //cudaMemcpy(g.p_absorb, outAbsorb, Nvox * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(g.flags, outFlags, Nvox * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(outAbsorb);                   // free temporary arrays
    cudaFree(outFlags);                    // free temporary arrays
}