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

__global__ void fdtd_kernel(const float*  __restrict pPrev,
                            const float*  __restrict pCurr,
                            float*        __restrict pNext,
                            const uint8_t*__restrict flags)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x==0 || x>=d_Nx-1 || y==0 || y>=d_Ny-1 || z==0 || z>=d_Nz-1) return;
    std::size_t idx = ( (std::size_t)z * d_Ny + y ) * d_Nx + x;

    /* wall voxels stay zero */
    if (flags[idx] == 1) { 
        pNext[idx] = 0.0f; 
        return; 
    }

    if(flags[idx] == 3) { // Source voxel
        pNext[idx] = d_source_value; // Set source value
        return;
    }

    if(flags[idx] == 0){
        
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
        pNext[idx] =
            (2.0f) * pCenter
            - (1.0f) * pPrev[idx]
            + d_c2_dt2 * lap;
        }
}


__global__ void fdtd_kernel_boundary(
                            const float*  __restrict pCurr,
                            const float*  __restrict pZeta,
                            float*        __restrict pNext,
                            const uint8_t*__restrict flags,
                            const uint8_t*__restrict normals,
                            const uint32_t* __restrict boundary_indices)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < d_num_boundary_indices){
        uint32_t idx = boundary_indices[i];
        int x  = idx % d_Nx;
        int y  = (idx / d_Nx) % d_Ny;
        int z  = (idx / (d_Nx * d_Ny)) % d_Nz;
        uint8_t normal_code = normals[idx];
        float zeta_val = pZeta[idx];

        if(normal_code != K_NORMAL_NONE && zeta_val > 0.0f){
            float lambda = d_c_sound * d_dt / d_dx; // Wave speed
            float term_lambda_over_zeta = lambda / zeta_val;
            float denominator = 1.0f + term_lambda_over_zeta;
            if(fabsf(denominator) > 1e-9f){
                float B_coeff = (1.f - term_lambda_over_zeta) / denominator;
                float A_coeff = (2.f * term_lambda_over_zeta) / denominator;
                if(isfinite(A_coeff) && isfinite(B_coeff)){
                    float p_curr_at_idx = pCurr[idx];
                    float accumulated_A_p_air_neighbor = 0.0f;
                    float accumulated_B_p_curr = 0.0f;
                    int active_normal_components = 0;
                    size_t neighbor_idx;

                    // x direction
                    if(normal_code & K_NORMAL_POS_X){
                        active_normal_components++;
                        if(x + 1 < d_Nx){
                            neighbor_idx = idx + 1;
                            accumulated_A_p_air_neighbor += A_coeff * (flags[neighbor_idx] == 0 ? pCurr[neighbor_idx] : p_curr_at_idx);
                        }else{ accumulated_A_p_air_neighbor += A_coeff * p_curr_at_idx; }
                        accumulated_B_p_curr += B_coeff * p_curr_at_idx;
                    }

                    if(normal_code & K_NORMAL_NEG_X){
                        active_normal_components++;
                        if(x - 1 >= 0){
                            neighbor_idx = idx - 1;
                            accumulated_A_p_air_neighbor += A_coeff * (flags[neighbor_idx] == 0 ? pCurr[neighbor_idx] : p_curr_at_idx);
                        }else { accumulated_A_p_air_neighbor += A_coeff * p_curr_at_idx; }
                        accumulated_B_p_curr += B_coeff * p_curr_at_idx;
                    }

                    // y direction
                    if (normal_code & K_NORMAL_POS_Y) {
                        active_normal_components++;
                        if (y + 1 < d_Ny) {
                            neighbor_idx = idx + d_Nx;
                            accumulated_A_p_air_neighbor += A_coeff * (flags[neighbor_idx] == 0 ? pCurr[neighbor_idx] : p_curr_at_idx);
                        } else { accumulated_A_p_air_neighbor += A_coeff * p_curr_at_idx; }
                        accumulated_B_p_curr += B_coeff * p_curr_at_idx;
                    }
                    if (normal_code & K_NORMAL_NEG_Y) {
                        active_normal_components++;
                        if (y - 1 >= 0) {
                            neighbor_idx = idx - d_Nx;
                            accumulated_A_p_air_neighbor += A_coeff * (flags[neighbor_idx] == 0 ? pCurr[neighbor_idx] : p_curr_at_idx);
                        } else { accumulated_A_p_air_neighbor += A_coeff * p_curr_at_idx; }
                        accumulated_B_p_curr += B_coeff * p_curr_at_idx;
                    }

                    // z direction
                    size_t stride_z = (size_t)d_Nx * d_Ny;
                    if (normal_code & K_NORMAL_POS_Z) {
                        active_normal_components++;
                        if (z + 1 < d_Nz) {
                            neighbor_idx = idx + stride_z;
                            accumulated_A_p_air_neighbor += A_coeff * (flags[neighbor_idx] == 0 ? pCurr[neighbor_idx] : p_curr_at_idx);
                        } else { accumulated_A_p_air_neighbor += A_coeff * p_curr_at_idx; }
                        accumulated_B_p_curr += B_coeff * p_curr_at_idx;
                    }
                    if (normal_code & K_NORMAL_NEG_Z) {
                        active_normal_components++;
                        if (z - 1 >= 0) {
                            neighbor_idx = idx - stride_z;
                            accumulated_A_p_air_neighbor += A_coeff * (flags[neighbor_idx] == 0 ? pCurr[neighbor_idx] : p_curr_at_idx);
                        } else { accumulated_A_p_air_neighbor += A_coeff * p_curr_at_idx; }
                        accumulated_B_p_curr += B_coeff * p_curr_at_idx;
                    }

                    if(active_normal_components > 0){
                        pNext[idx] = (accumulated_B_p_curr + accumulated_A_p_air_neighbor) / (float)active_normal_components;
                    }
                }
            }
        }
    }
}

__global__ void fdtd_kernel_boundary_biquad(
                            const float*    __restrict pCurr,
                            float*    __restrict pNext,
                            const float*    __restrict filter_coeffs[5],
                            const size_t    __restrict num_filter_sections,
                            float* __restrict (*filter_states[6])[4],
                            const uint8_t*  __restrict flags,
                            const uint8_t*  __restrict normals,
                            const uint32_t* __restrict boundary_indices)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride_z = (size_t)d_Nx * d_Ny;
    if(i < d_num_boundary_indices){
        uint32_t idx = boundary_indices[i];
        int x  = idx % d_Nx;
        int y  = (idx / d_Nx) % d_Ny;
        int z  = (idx / (d_Nx * d_Ny)) % d_Nz;
        float p_c = pCurr[idx];
        size_t material_idx = 0; // Assuming a single material for simplicity (for now)

        float sum = 0.f;
        size_t num_active_components = 0;
        
        // POSITIVE X
        if(normals[idx] & K_NORMAL_POS_X){
            float inc = (x + 1 < d_Nx && flags[idx + 1] == 0) ? pCurr[idx + 1] : p_c;
            float section_in = inc;
            float section_out = inc;

            for(int s = 0; s < num_filter_sections; ++s){
                size_t section_idx = i * num_filter_sections + s;
                float* x1 = &(*filter_states[0])[0][section_idx];
                float* x2 = &(*filter_states[0])[1][section_idx];
                float* y1 = &(*filter_states[0])[2][section_idx];
                float* y2 = &(*filter_states[0])[3][section_idx];

                section_out = filter_coeffs[material_idx][0] * section_in // b0
                            + filter_coeffs[material_idx][1] * (*x1)      // b1
                            + filter_coeffs[material_idx][2] * (*x2)      // b2
                            - filter_coeffs[material_idx][3] * (*y1)      // a1
                            - filter_coeffs[material_idx][4] * (*y2);     // a2
                
                // Update filter states
                *x2 = *x1;
                *x1 = section_in;
                *y2 = *y1;
                *y1 = section_out;

                section_in = section_out; // Prepare for next section
            }
            sum += section_out;
            num_active_components++;
        }

        // NEGATIVE X
        if(normals[idx] & K_NORMAL_NEG_X){
            float inc = (x - 1 < d_Nx && flags[idx - 1] == 0) ? pCurr[idx - 1] : p_c;
            float section_in = inc;
            float section_out = inc;

            for(int s = 0; s < num_filter_sections; ++s){
                size_t section_idx = i * num_filter_sections + s;
                float* x1 = &(*filter_states[1])[0][section_idx];
                float* x2 = &(*filter_states[1])[1][section_idx];
                float* y1 = &(*filter_states[1])[2][section_idx];
                float* y2 = &(*filter_states[1])[3][section_idx];

                section_out = filter_coeffs[material_idx][0] * section_in // b0
                            + filter_coeffs[material_idx][1] * (*x1)      // b1
                            + filter_coeffs[material_idx][2] * (*x2)      // b2
                            - filter_coeffs[material_idx][3] * (*y1)      // a1
                            - filter_coeffs[material_idx][4] * (*y2);     // a2
                
                // Update filter states
                *x2 = *x1;
                *x1 = section_in;
                *y2 = *y1;
                *y1 = section_out;

                section_in = section_out; // Prepare for next section
            }
            sum += section_out;
            num_active_components++;
        }

        // POSITIVE Y
        if(normals[idx] & K_NORMAL_POS_Y){
            float inc = (y + 1 < d_Ny && flags[idx + d_Nx] == 0) ? pCurr[idx + d_Nx] : p_c;
            float section_in = inc;
            float section_out = inc;

            for(int s = 0; s < num_filter_sections; ++s){
                size_t section_idx = i * num_filter_sections + s;
                float* x1 = &(*filter_states[2])[0][section_idx];
                float* x2 = &(*filter_states[2])[1][section_idx];
                float* y1 = &(*filter_states[2])[2][section_idx];
                float* y2 = &(*filter_states[2])[3][section_idx];

                section_out = filter_coeffs[material_idx][0] * section_in // b0
                            + filter_coeffs[material_idx][1] * (*x1)      // b1
                            + filter_coeffs[material_idx][2] * (*x2)      // b2
                            - filter_coeffs[material_idx][3] * (*y1)      // a1
                            - filter_coeffs[material_idx][4] * (*y2);     // a2

                // Update filter states
                *x2 = *x1;
                *x1 = section_in;
                *y2 = *y1;
                *y1 = section_out;

                section_in = section_out; // Prepare for next section
            }
            sum += section_out;
            num_active_components++;
        }

        // NEGATIVE Y
        if(normals[idx] & K_NORMAL_NEG_Y){
            float inc = (y - 1 >= 0 && flags[idx - d_Nx] == 0) ? pCurr[idx - d_Nx] : p_c;
            float section_in = inc;
            float section_out = inc;

            for(int s = 0; s < num_filter_sections; ++s){
                size_t section_idx = i * num_filter_sections + s;
                float* x1 = &(*filter_states[3])[0][section_idx];
                float* x2 = &(*filter_states[3])[1][section_idx];
                float* y1 = &(*filter_states[3])[2][section_idx];
                float* y2 = &(*filter_states[3])[3][section_idx];

                section_out = filter_coeffs[material_idx][0] * section_in // b0
                            + filter_coeffs[material_idx][1] * (*x1)      // b1
                            + filter_coeffs[material_idx][2] * (*x2)      // b2
                            - filter_coeffs[material_idx][3] * (*y1)      // a1
                            - filter_coeffs[material_idx][4] * (*y2);     // a2
                
                // Update filter states
                *x2 = *x1;
                *x1 = section_in;
                *y2 = *y1;
                *y1 = section_out;

                section_in = section_out; // Prepare for next section
            }
            sum += section_out;
            num_active_components++;
        }

        // POSITIVE Z
        if(normals[idx] & K_NORMAL_POS_Z){
            float inc = (z + 1 < d_Nz && flags[idx + d_Nx * d_Ny] == 0) ? pCurr[idx + d_Nx * d_Ny] : p_c;
            float section_in = inc;
            float section_out = inc;

            for(int s = 0; s < num_filter_sections; ++s){
                size_t section_idx = i * num_filter_sections + s;
                float* x1 = &(*filter_states[4])[0][section_idx];
                float* x2 = &(*filter_states[4])[1][section_idx];
                float* y1 = &(*filter_states[4])[2][section_idx];
                float* y2 = &(*filter_states[4])[3][section_idx];

                section_out = filter_coeffs[material_idx][0] * section_in // b0
                            + filter_coeffs[material_idx][1] * (*x1)      // b1
                            + filter_coeffs[material_idx][2] * (*x2)      // b2
                            - filter_coeffs[material_idx][3] * (*y1)      // a1
                            - filter_coeffs[material_idx][4] * (*y2);     // a2

                // Update filter states
                *x2 = *x1;
                *x1 = section_in;
                *y2 = *y1;
                *y1 = section_out;

                section_in = section_out; // Prepare for next section
            }
            sum += section_out;
            num_active_components++;
        }

        // NEGATIVE Z
        if(normals[idx] & K_NORMAL_NEG_Z){
            float inc = (z - 1 >= 0 && flags[idx - d_Nx * d_Ny] == 0) ? pCurr[idx - d_Nx * d_Ny] : p_c;
            float section_in = inc;
            float section_out = inc;

            for(int s = 0; s < num_filter_sections; ++s){
                size_t section_idx = i * num_filter_sections + s;
                float* x1 = &(*filter_states[5])[0][section_idx];
                float* x2 = &(*filter_states[5])[1][section_idx];
                float* y1 = &(*filter_states[5])[2][section_idx];
                float* y2 = &(*filter_states[5])[3][section_idx];

                section_out = filter_coeffs[material_idx][0] * section_in // b0
                            + filter_coeffs[material_idx][1] * (*x1)      // b1
                            + filter_coeffs[material_idx][2] * (*x2)      // b2
                            - filter_coeffs[material_idx][3] * (*y1)      // a1
                            - filter_coeffs[material_idx][4] * (*y2);     // a2

                // Update filter states
                *x2 = *x1;
                *x1 = section_in;
                *y2 = *y1;
                *y1 = section_out;

                section_in = section_out; // Prepare for next section
            }
            sum += section_out;
            num_active_components++;
        }

        if(num_active_components > 0){
            pNext[idx] = sum / (float)num_active_components; // Average the contributions
        } else {
            pNext[idx] = 0.f; // This should never occur, but just in case.
        }

    }



}

extern "C"
{
    bool fdtd_gpu_step(VectorSpace *space, float h, unsigned int step)
    {
        Grid &g = space->getGrid(); // host meta
        //const std::size_t N = g.Nx * g.Ny * g.Nz;

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
        if(step < 10){
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
        uint8_t *d_flags = g.d_flags;
        float *d_zeta = g.d_pZeta;
        uint8_t *d_normals = g.d_normals;

        /* launch geometry --------------------------------------------------- */
        dim3 B(16, 4, 4);
        dim3 G((iNx + B.x - 1) / B.x,
               (iNy + B.y - 1) / B.y,
               (iNz + B.z - 1) / B.z);

        size_t threads_per_block_boundary = 256;
        size_t num_blocks_boundary = (g.boundary_indices_size + threads_per_block_boundary - 1) / threads_per_block_boundary;
        fdtd_kernel<<<G, B>>>(d_prev, d_curr, d_next, d_flags);
        fdtd_kernel_boundary<<<num_blocks_boundary, threads_per_block_boundary>>>(d_curr, d_zeta, d_next, d_flags, d_normals, g.d_boundary_indices);

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

        constexpr size_t RMS_SIZE = 100;
        constexpr size_t MIN_RMS_CHECK_STEP = 3000;
        if(step > MIN_RMS_CHECK_STEP){
            //Calculate RMS of last 30 steps
            float sum = 0.0f;
            for(size_t i = step - RMS_SIZE; i < step; ++i){
                sum += g.p_audio_output[i] * g.p_audio_output[i];
            }
            float rms = std::sqrt(sum / (float)RMS_SIZE);
            if(rms < 0.004f){
                return false;
            }
        }
         return true; // continue simulation
    }


}