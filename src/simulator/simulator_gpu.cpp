#include "simulator_gpu.h"
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
	
	std::cout << "Initializing VectorSpace with dimensions: "
	          << grid.Nx << " x " << grid.Ny << " x " << grid.Nz
	          << ", total size: " << grid.size << std::endl;
	for (size_t i = 0; i < grid.size; ++i) {
		grid.p_curr[i] = 0.0f;
		grid.p_next[i] = 0.0f;
		grid.p_prev[i] = 0.0f;
		grid.flags[i] = 0; // Initialize flags to zero
	}
	space->resetStopwatch();
	std::cout << "VectorSpace initialized." << std::endl;

    // Reset the stopwatch
    space->resetStopwatch();

    //GPU memory allocation
    float *d_p_curr, *d_p_next, *d_p_prev;
    uint8_t *d_flags;
    cudaError_t err;
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
    // Copy initial data from CPU to GPU
    err = cudaMemcpy(d_p_curr, grid.p_curr, sizeof(float) * grid.size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying p_curr to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        cudaFree(d_flags);
        return;
    }
    err = cudaMemcpy(d_p_next, grid.p_next, sizeof(float) * grid.size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying p_next to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        cudaFree(d_flags);
        return;
    }
    err = cudaMemcpy(d_p_prev, grid.p_prev, sizeof(float) * grid.size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying p_prev to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        cudaFree(d_flags);
        return;
    }
    err = cudaMemcpy(d_flags, grid.flags, sizeof(uint8_t) * grid.size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying flags to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_p_curr);
        cudaFree(d_p_next);
        cudaFree(d_p_prev);
        cudaFree(d_flags);
        return;
    }
    // Store the device pointers in the grid
    grid.d_p_curr = d_p_curr;
    grid.d_p_next = d_p_next;
    grid.d_p_prev = d_p_prev;
    grid.d_flags = d_flags;
}

void fdtd_gpu_step(VectorSpace* space, float h) {
    Grid& grid = space->getGrid();
    const float inv_h2 = 1.f / (h * h);
    const float c = 343.f;
    const float dt = 0.5 * h / (c * std::sqrt(3.f));
    const float c2_dt2 = c * c * dt * dt;
    const float gamma = 5.f;
    const float damp_factor = gamma * dt;

}