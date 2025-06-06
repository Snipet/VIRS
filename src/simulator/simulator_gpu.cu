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
    __constant__ float d_inv_h2, d_c2_dt2, d_dt;
    __constant__ float d_source_value;
}

__global__ void fdtd_kernel(const float*  __restrict pPrev,
                            const float*  __restrict pCurr,
                            const float*  __restrict pAbsorb,
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
    if (flags[idx] == 1) { 
        pNext[idx] = 0.0f; 
        return; 
    }

    if(flags[idx] == 3) { // Source voxel
        pNext[idx] = d_source_value; // Set source value
        return;
    }

    float pCenter = pCurr[idx];

    auto V = [&](int ix,int iy,int iz)->float {
        std::size_t n = ( (std::size_t)iz * d_Ny + iy ) * d_Nx + ix;
        if(flags[n] == 1 ) {
            return pCenter; // Wall voxel
        }else{
            return pCurr[n]; // Normal voxel
        }
    };

    float lap = ( V(x+1,y,z) + V(x-1,y,z) +
                  V(x,y+1,z) + V(x,y-1,z) +
                  V(x,y,z+1) + V(x,y,z-1) - 6.0f*pCenter ) * d_inv_h2;

    float local_sigma_dt = pAbsorb[idx] * d_dt;
    pNext[idx] =
          (2.0f) * pCenter
        - (1.0f) * pPrev[idx]
        + d_c2_dt2 * lap;
}


extern "C"
{
    void fdtd_gpu_step(VectorSpace *space, float h, unsigned int step)
    {
        Grid &g = space->getGrid(); // host meta
        //const std::size_t N = g.Nx * g.Ny * g.Nz;

        /* one‑time constant upload (cache inside space) -------------------- */
        static bool constantsUploaded = false;
        int iNx = int(g.Nx);
        int iNy = int(g.Ny);
        int iNz = int(g.Nz);
        float c = 343.f;
        float dt = 0.5f * h / (c * std::sqrt(3.f));
        float c2_dt2 = c * c * dt * dt;
        float inv_h2 = 1.f / (h * h);
        if (!constantsUploaded)
        {
            cudaMemcpyToSymbol(d_Nx, &iNx, sizeof(int));
            cudaMemcpyToSymbol(d_Ny, &iNy, sizeof(int));
            cudaMemcpyToSymbol(d_Nz, &iNz, sizeof(int));
            cudaMemcpyToSymbol(d_inv_h2, &inv_h2, sizeof(float));
            cudaMemcpyToSymbol(d_c2_dt2, &c2_dt2, sizeof(float));
            cudaMemcpyToSymbol(d_dt, &dt, sizeof(float));

            constantsUploaded = true;
        }

        size_t sample_rate = space->audio_file.getSampleRate();
        float simulation_time = (float)step * dt; // in seconds
        float sample_index = simulation_time * (float)sample_rate;
        float source_value = 0.0f;
        if(sample_index + 1.f < (float)space->audio_file.getNumSamplesPerChannel())
        {
            //Simple, linear interpolation for now
            size_t bottom_index = (size_t)sample_index;
            size_t top_index = bottom_index + 1;
            float bottom_value = space->audio_file.samples[0][bottom_index];
            float top_value = space->audio_file.samples[0][top_index];
            float fraction = sample_index - (float)bottom_index;
            source_value = bottom_value + fraction * (top_value - bottom_value);
        }

        if(step < 100){
            //Ramp up the source value for the first 100 steps
            source_value *= (float)step / 100.0f; // Scale the source value
        }
        
        source_value *= 2.f; // Temporary amplification for testing

        /* upload source value to device ------------------------------------ */
        if (step == 0) {
            cudaMemcpyToSymbol(d_source_value, &source_value, sizeof(float));
        }
        else {
            // Update source value for subsequent steps
            cudaMemcpyToSymbol(d_source_value, &source_value, sizeof(float), 0, cudaMemcpyHostToDevice);
        }


        /* device pointers held inside VectorSpace -------------------------- */
        float *d_prev = g.d_p_prev;
        float *d_curr = g.d_p_curr;
        float *d_next = g.d_p_next;
        float *d_absorb = g.d_p_absorb;
        uint8_t *d_flags = g.d_flags;

        /* launch geometry --------------------------------------------------- */
        dim3 B(8, 8, 8);
        dim3 G((iNx + B.x - 1) / B.x,
               (iNy + B.y - 1) / B.y,
               (iNz + B.z - 1) / B.z);

        fdtd_kernel<<<G, B>>>(d_prev, d_curr, d_absorb, d_next, d_flags);

        // CUDA_CHECK(cudaGetLastError());        // macro or manual check
        // CUDA_CHECK(cudaDeviceSynchronize());   // catches bad constants immediately
        cudaDeviceSynchronize();   // ensure completion before we copy

        // /* swap pointers for next iteration ---------------------------------- */
        std::swap(g.d_p_prev, g.d_p_curr);
        std::swap(g.d_p_curr, g.d_p_next);

        size_t read_idx = g.idx(334, 225, 270);

        // Read d_p_curr[read_idx] from device to host
        float pressure_value = 0.0f;
        cudaMemcpy(&pressure_value, g.d_p_curr + read_idx, sizeof(float), cudaMemcpyDeviceToHost);
        g.p_audio_output[step] = pressure_value;
    }


}