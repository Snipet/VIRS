#include "simulator_gpu.h"

#ifdef VIRS_WITH_CUDA
#include <cuda_runtime.h>
#include <iostream>

void fdtd_gpu_setup(VectorSpace* space) {
    //Set up CPU memory for the grid
    Grid& grid = space->getGrid();

	constexpr size_t ALIGN = 64;
	grid.p_curr = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	grid.p_next = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	grid.p_prev = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	grid.flags = static_cast<uint8_t*>(aligned_alloc(ALIGN, sizeof(uint8_t) * grid.size));
    grid.p_absorb = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
    grid.p_source = space->audio_file.samples[0].data(); // Assuming audio_file is already loaded
    grid.normals = static_cast<uint8_t*>(aligned_alloc(ALIGN, sizeof(uint8_t) * grid.size));
    grid.pZeta = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	
	std::cout << "Initializing VectorSpace with dimensions: "
	          << grid.Nx << " x " << grid.Ny << " x " << grid.Nz
	          << ", total size: " << grid.size << std::endl;
	for (size_t i = 0; i < grid.size; ++i) {
		grid.p_curr[i] = 0.0f;
		grid.p_next[i] = 0.0f;
		grid.p_prev[i] = 0.0f;
		grid.flags[i] = 0; // Initialize flags to zero
        grid.p_absorb[i] = 0.0f; // Initialize p_absorb to zero
        grid.normals[i] = 0;
        grid.pZeta[i] = 0.0f;
	}
	space->resetStopwatch();
	std::cout << "VectorSpace initialized." << std::endl;

    // Reset the stopwatch
    space->resetStopwatch();

    //GPU memory allocation
    float *d_p_curr, *d_p_next, *d_p_prev, *d_p_absorb, *d_p_source, *d_pZeta;
    uint8_t *d_flags, *d_normals;
    cudaError_t err;

    std::cout << "Allocating GPU memory for grid..." << std::endl;
    err = cudaMalloc((void**)&d_p_curr, sizeof(float) * grid.size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating GPU memory for p_curr: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    err = cudaMalloc((void**)&d_p_next, sizeof(float) * grid.size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating GPU memory for p_next: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        return;
    }
    err = cudaMalloc((void**)&d_p_prev, sizeof(float) * grid.size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating GPU memory for p_prev: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        return;
    }
    err = cudaMalloc((void**)&d_flags, sizeof(uint8_t) * grid.size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating GPU memory for flags: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        return;
    }
    err = cudaMalloc((void**)&d_p_absorb, sizeof(float) * grid.size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating GPU memory for p_absorb: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        cudaFree(d_flags);
        return;
    }
    err = cudaMalloc((void**)&d_p_source, sizeof(float) * grid.p_source_size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating GPU memory for p_source: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        cudaFree(d_flags);
        cudaFree(d_p_absorb);
        return;
    }
    err = cudaMalloc((void**)&d_normals, sizeof(uint8_t) * grid.size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating GPU memory for normals: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        cudaFree(d_flags);
        cudaFree(d_p_absorb);
        cudaFree(d_p_source);
        return;
    }
    err = cudaMalloc((void**)&d_pZeta, sizeof(float) * grid.size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating GPU memory for pZeta: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        cudaFree(d_flags);
        cudaFree(d_p_absorb);
        cudaFree(d_p_source);
        cudaFree(d_normals);
        return;
    }

    // Copy initial data from CPU to GPU
    std::cout << "Copying data from CPU to GPU..." << std::endl;
    err = cudaMemcpy(d_p_curr, grid.p_curr, sizeof(float) * grid.size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying p_curr to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        cudaFree(d_flags);
        cudaFree(d_p_absorb);
        cudaFree(d_p_source);
        cudaFree(d_normals);
        cudaFree(d_pZeta);
        return;
    }
    err = cudaMemcpy(d_p_next, grid.p_next, sizeof(float) * grid.size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying p_next to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        cudaFree(d_flags);
        cudaFree(d_p_absorb);
        cudaFree(d_p_source);
        cudaFree(d_normals);
        cudaFree(d_pZeta);
        return;
    }
    err = cudaMemcpy(d_p_prev, grid.p_prev, sizeof(float) * grid.size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying p_prev to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        cudaFree(d_flags);
        cudaFree(d_p_absorb);
        cudaFree(d_p_source);
        cudaFree(d_normals);
        cudaFree(d_pZeta);
        return;
    }
    err = cudaMemcpy(d_flags, grid.flags, sizeof(uint8_t) * grid.size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying flags to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        cudaFree(d_flags);
        cudaFree(d_p_absorb);
        cudaFree(d_p_source);
        cudaFree(d_normals);
        cudaFree(d_pZeta);
        return;
    }
    err = cudaMemcpy(d_p_absorb, grid.p_absorb, sizeof(float) * grid.size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying p_absorb to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        cudaFree(d_flags);
        cudaFree(d_p_absorb);
        cudaFree(d_p_source);
        cudaFree(d_normals);
        cudaFree(d_pZeta);
        return;
    }

    err = cudaMemcpy(d_p_source, grid.p_source, sizeof(float) * grid.p_source_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying p_source to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        cudaFree(d_flags);
        cudaFree(d_p_absorb);
        cudaFree(d_p_source);
        cudaFree(d_normals);
        cudaFree(d_pZeta);
        return;
    }
    err = cudaMemcpy(d_normals, grid.normals, sizeof(uint8_t) * grid.size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying normals to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        cudaFree(d_flags);
        cudaFree(d_p_absorb);
        cudaFree(d_p_source);
        cudaFree(d_normals);
        cudaFree(d_pZeta);
        return;
    }
    err = cudaMemcpy(d_pZeta, grid.pZeta, sizeof(float) * grid.size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying pZeta to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        cudaFree(d_flags);
        cudaFree(d_p_absorb);
        cudaFree(d_p_source);
        cudaFree(d_normals);
        cudaFree(d_pZeta);
        return;
    }

    // boundary_indices are copied later

    std::cout << "GPU memory allocation and data transfer completed successfully." << std::endl;

    // Store the device pointers in the grid
    grid.d_p_curr = d_p_curr;
    grid.d_p_next = d_p_next;
    grid.d_p_prev = d_p_prev;
    grid.d_flags = d_flags;
    grid.d_p_absorb = d_p_absorb;
    grid.d_normals = d_normals;
    grid.d_p_source = d_p_source;
    grid.d_pZeta = d_pZeta;
}

void fdtd_gpu_cleanup(VectorSpace* space) {
    Grid& grid = space->getGrid();

    // Free GPU memory
    cudaFree(grid.d_p_curr);
    cudaFree(grid.d_p_next);
    cudaFree(grid.d_p_prev);
    cudaFree(grid.d_flags);
    cudaFree(grid.d_p_absorb);
    cudaFree(grid.d_p_source);
    cudaFree(grid.d_normals);
    cudaFree(grid.d_pZeta);

    delete[] grid.p_curr;
    delete[] grid.p_next;
    delete[] grid.p_prev;
    delete[] grid.flags;
    delete[] grid.p_absorb;
    delete[] grid.p_audio_output;
    delete[] grid.pZeta;
    delete[] grid.normals;
    grid.p_curr = nullptr;
    grid.p_next = nullptr;
    grid.p_prev = nullptr;
    grid.flags = nullptr;
    grid.p_absorb = nullptr;
    grid.p_source = nullptr;
    grid.pZeta = nullptr;
    grid.normals = nullptr;

    //Clean up filter memory
    if(grid.allocated_filter_memory){
        delete[] grid.biquad_a1;
        delete[] grid.biquad_a2;
        delete[] grid.biquad_b0;
        delete[] grid.biquad_b1;
        delete[] grid.biquad_b2;
        delete[] grid.biquad_state_ptr;

        cudaFree(grid.d_biquad_a1);
        cudaFree(grid.d_biquad_a2);
        cudaFree(grid.d_biquad_b0);
        cudaFree(grid.d_biquad_b1);
        cudaFree(grid.d_biquad_b2);
        cudaFree(grid.d_biquad_state_ptr);

        grid.biquad_a1 = nullptr;
        grid.biquad_a2 = nullptr;
        grid.biquad_b0 = nullptr;
        grid.biquad_b1 = nullptr;
        grid.biquad_b2 = nullptr;
        grid.biquad_state_ptr = nullptr;

        grid.d_biquad_a1 = nullptr;
        grid.d_biquad_a2 = nullptr;
        grid.d_biquad_b0 = nullptr;
        grid.d_biquad_b1 = nullptr;
        grid.d_biquad_b2 = nullptr;
        grid.d_biquad_state_ptr = nullptr;
        
        grid.allocated_filter_memory = false;
    }

    std::cout << "GPU and CPU memory cleaned up successfully." << std::endl;
}
#endif // VIRS_WITH_CUDA