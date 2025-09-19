#include "vectorspace.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

namespace
{
    __constant__ int d_Nx, d_Ny, d_Nz;
    __constant__ float d_inv_h2, d_c2_dt2, d_dt, d_c_sound, d_dx;
    __constant__ float d_source_value;
    __constant__ size_t d_num_boundary_indices;

    const uint8_t K_NORMAL_NONE = 0;
    const uint8_t K_NORMAL_POS_X = 1;
    const uint8_t K_NORMAL_NEG_X = 2;
    const uint8_t K_NORMAL_POS_Y = 4;
    const uint8_t K_NORMAL_NEG_Y = 8;
    const uint8_t K_NORMAL_POS_Z = 16;
    const uint8_t K_NORMAL_NEG_Z = 32;

}

__global__ void fdtd_kernel(const float *__restrict pPrev,
                            const float *__restrict pCurr,
                            float *__restrict pNext,
                            const uint8_t *__restrict flags,
                            const uint8_t *__restrict normals)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x == 0 || x >= d_Nx - 1 || y == 0 || y >= d_Ny - 1 || z == 0 || z >= d_Nz - 1)
        return;
    std::size_t idx = ((std::size_t)z * d_Ny + y) * d_Nx + x;

    // Dirichlet boundary condition for walls
    if (flags[idx] == 1)
    {
        pNext[idx] = 0.0f;
        return;
    }

    if (flags[idx] == 3)
    {                                // Source voxel
        pNext[idx] = d_source_value; // Set source value
        return;
    }

    if (flags[idx] == 0 || flags[idx] == 2) //Air cell
    {

        float pCenter = pCurr[idx];

        auto V = [&](int ix, int iy, int iz) -> float
        {
            std::size_t n = ((std::size_t)iz * d_Ny + iy) * d_Nx + ix;
            if (flags[n] == 1)
            {
                return pCenter; // Wall voxel
            }
            else
            {
                return pCurr[n]; // Normal voxel
            }
        };

        float lap = (V(x + 1, y, z) + V(x - 1, y, z) +
                     V(x, y + 1, z) + V(x, y - 1, z) +
                     V(x, y, z + 1) + V(x, y, z - 1) - 6.0f * pCenter) *
                    d_inv_h2;
        pNext[idx] =
            (2.0f) * pCenter - (1.0f) * pPrev[idx] + d_c2_dt2 * lap;


        // float l2 = (d_c_sound * d_dt / d_dx) * (d_c_sound * d_dt / d_dx);
        // float a1 = 2.f - 6.f * l2;
        // float a2 = l2;
        // float partial = a1 * pCurr[idx] - pPrev[idx];
        // partial += a2 * pCurr[idx + 1];
        // partial += a2 * pCurr[idx - 1];
        // partial += a2 * pCurr[idx + d_Nx];
        // partial += a2 * pCurr[idx - d_Nx];
        // partial += a2 * pCurr[idx + (size_t)d_Nx * d_Ny];
        // partial += a2 * pCurr[idx - (size_t)d_Nx * d_Ny];
        // pNext[idx] = partial;
    }
}

__global__ void fdtd_kernel_boundary_biquad(
    const float *__restrict pCurr,
    float *__restrict pNext,
    float *__restrict pPrev,
    float *__restrict filter_coeffs[5],
    const size_t __restrict num_filter_sections,
    float * __restrict filter_states,
    const uint8_t *__restrict flags,
    const uint8_t *__restrict normals,
    const uint32_t *__restrict boundary_indices)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride_z = (size_t)d_Nx * d_Ny;
    if (i < d_num_boundary_indices)
    {
        uint32_t idx = boundary_indices[i];
        int x = idx % d_Nx;
        int y = (idx / d_Nx) % d_Ny;
        int z = (idx / (d_Nx * d_Ny)) % d_Nz;
        size_t material_idx = 0; // Assuming a single material for simplicity (for now)

        float sum = 0.f;
        size_t num_active_components = 0;

        float summed_differences = 0.f;
        float dif = pCurr[idx] - pPrev[idx]; // Temporal derivative at boundary node [idx]
        for (size_t section = 0; section < 1; ++section)
        {
            size_t state_offset = (i * 4);
            float* state = &filter_states[state_offset];
            float input = dif;
            float output = filter_coeffs[0][material_idx] * input + filter_coeffs[1][material_idx] * state[0] + filter_coeffs[2][material_idx] * state[1] - filter_coeffs[3][material_idx] * state[2] - filter_coeffs[4][material_idx] * state[3];
            // Update state
	    //output = 0.f;
            state[1] = state[0];
            state[0] = input;
            state[3] = state[2];
            state[2] = output;
            //summed_differences += output;
        }

        //const float UNKNOWN_CONSTANT = (d_c_sound * d_dt) / d_dx * 0.95f;
        //pNext[idx] = pNext[idx] - UNKNOWN_CONSTANT * (summed_differences * 0.9 + dif * 0.1f) * 0.25f;
    }
}

__global__ void fdtd_kernel_rigid_walls(const float* __restrict pCurr,
                                        float* __restrict pNext,
                                        float* __restrict pPrev,
                                        const uint8_t* __restrict flags,
                                        const uint8_t* __restrict normals,
                                        const uint32_t* __restrict boundary_indices)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < d_num_boundary_indices){
        size_t stride_z = (size_t)d_Nx * d_Ny;
        uint32_t idx = boundary_indices[i];
        uint8_t v = normals[idx];
        uint8_t num_neighbors = 0;
        for(num_neighbors = 0; v; num_neighbors++) v &= v - 1; // Count set bits
        uint8_t adj = normals[idx];
        float l2 = (d_c_sound * d_dt / d_dx) * (d_c_sound * d_dt / d_dx);
        float b1 = (2.f - l2 * (float)num_neighbors);
        float b2 = l2;
        float partial = b1 * pCurr[idx] - pPrev[idx];
        partial += b2 * ((adj & K_NORMAL_POS_X) ? 1.f : 0.f) * pCurr[idx + 1];
        partial += b2 * ((adj & K_NORMAL_NEG_X) ? 1.f : 0.f) * pCurr[idx - 1];
        partial += b2 * ((adj & K_NORMAL_POS_Y) ? 1.f : 0.f) * pCurr[idx + d_Nx];
        partial += b2 * ((adj & K_NORMAL_NEG_Y) ? 1.f : 0.f) * pCurr[idx - d_Nx];
        partial += b2 * ((adj & K_NORMAL_POS_Z) ? 1.f : 0.f) * pCurr[idx + stride_z];
        partial += b2 * ((adj & K_NORMAL_NEG_Z) ? 1.f : 0.f) * pCurr[idx - stride_z];
        pNext[idx] = partial;

    }
}

extern "C"
{
    bool fdtd_gpu_step(VectorSpace *space, float h, unsigned int step)
    {
        Grid &g = space->getGrid(); // host meta
        // const std::size_t N = g.Nx * g.Ny * g.Nz;

        /* one‑time constant upload (cache inside space) -------------------- */
        static bool constantsUploaded = false;
        int iNx = int(g.Nx);
        int iNy = int(g.Ny);
        int iNz = int(g.Nz);
        size_t num_boundary_indices = g.boundary_indices_size;
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
            cudaMemcpyToSymbol(d_c_sound, &c, sizeof(float));
            cudaMemcpyToSymbol(d_dx, &h, sizeof(float));
            cudaMemcpyToSymbol(d_num_boundary_indices, &num_boundary_indices, sizeof(size_t));

            constantsUploaded = true;
        }

        float source_value = 0.0f;
        if (step < 10)
        {
            source_value = 10.f;
        }

        // size_t sample_rate = space->audio_file.getSampleRate();
        // float simulation_time = (float)step * dt; // in seconds
        // float sample_index = simulation_time * (float)sample_rate;
        // float source_value = 0.0f;
        // if(sample_index + 1.f < (float)space->audio_file.getNumSamplesPerChannel())
        // {
        //     //Simple, linear interpolation for now
        //     size_t bottom_index = (size_t)sample_index;
        //     size_t top_index = bottom_index + 1;
        //     float bottom_value = space->audio_file.samples[0][bottom_index];
        //     float top_value = space->audio_file.samples[0][top_index];
        //     float fraction = sample_index - (float)bottom_index;
        //     source_value = bottom_value + fraction * (top_value - bottom_value);
        // }

        // if(step < 100){
        //     //Ramp up the source value for the first 100 steps
        //     source_value *= (float)step / 100.0f; // Scale the source value
        // }
        // constexpr size_t fade_out_steps = 100;
        // // if(step > space->audio_file.getNumSamplesPerChannel() - fade_out_steps){
        // //     //Fade out the source value for the last 100 steps
        // //     size_t fade_out_index = space->audio_file.getNumSamplesPerChannel() - step;
        // //     float fade_out_fraction = (float)fade_out_index / (float)fade_out_steps;
        // //     source_value *= fade_out_fraction; // Scale the source value
        // // }

        // source_value *= 2.f; // Temporary amplification for testing

        /* upload source value to device ------------------------------------ */
        if (step == 0)
        {
            cudaMemcpyToSymbol(d_source_value, &source_value, sizeof(float));
        }
        else
        {
            // Update source value for subsequent steps
            cudaMemcpyToSymbol(d_source_value, &source_value, sizeof(float), 0, cudaMemcpyHostToDevice);
        }

        /* device pointers held inside VectorSpace -------------------------- */
        float *d_prev = g.d_p_prev;
        float *d_curr = g.d_p_curr;
        float *d_next = g.d_p_next;
        uint8_t *d_flags = g.d_flags;
        //float *d_zeta = g.d_pZeta;
        uint8_t *d_normals = g.d_normals;
        uint32_t *d_boundary_indices = g.d_boundary_indices;

        /* launch geometry --------------------------------------------------- */
        dim3 B(16, 4, 4);
        dim3 G((iNx + B.x - 1) / B.x,
               (iNy + B.y - 1) / B.y,
               (iNz + B.z - 1) / B.z);

        size_t threads_per_block_boundary = 256;
        size_t num_blocks_boundary = (g.boundary_indices_size + threads_per_block_boundary - 1) / threads_per_block_boundary;
        fdtd_kernel<<<G, B>>>(d_prev, d_curr, d_next, d_flags, d_normals);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        //fdtd_kernel_rigid_walls<<<num_blocks_boundary, threads_per_block_boundary>>>(d_curr, d_next, d_prev, d_flags, d_normals, d_boundary_indices);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        fdtd_kernel_boundary_biquad<<<num_blocks_boundary, threads_per_block_boundary>>>(d_curr, d_next, d_prev, g.d_biquad_coeffs_ptr, g.num_filter_sections, g.d_biquad_state_ptr, d_flags, d_normals, d_boundary_indices);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // CUDA_CHECK(cudaGetLastError());        // macro or manual check
        // CUDA_CHECK(cudaDeviceSynchronize());   // catches bad constants immediately
        cudaDeviceSynchronize(); // ensure completion before we copy

        // /* swap pointers for next iteration ---------------------------------- */
        std::swap(g.d_p_prev, g.d_p_curr);
        std::swap(g.d_p_curr, g.d_p_next);

        size_t read_idx = g.idx(334, 225, 270);

        // Read d_p_curr[read_idx] from device to host
        float pressure_value = 0.0f;
        cudaMemcpy(&pressure_value, g.d_p_curr + read_idx, sizeof(float), cudaMemcpyDeviceToHost);
        g.p_audio_output[step] = pressure_value;

        constexpr size_t RMS_SIZE = 100;
        constexpr size_t MIN_RMS_CHECK_STEP = 3000;
        if (step > MIN_RMS_CHECK_STEP)
        {
            // Calculate RMS of last 30 steps
            float sum = 0.0f;
            for (size_t i = step - RMS_SIZE; i < step; ++i)
            {
                sum += g.p_audio_output[i] * g.p_audio_output[i];
            }
            float rms = std::sqrt(sum / (float)RMS_SIZE);
            if (rms < 0.0009f)
            {
                return false;
            }
        }
        return true; // continue simulation
    }
}
