#include "simulator_cpu.h"

#ifdef VIRS_WITH_CUDA
#include "simulator_gpu.h"
#include <cuda_runtime.h>
#endif // VIRS_WITH_CUDA

#include "simulator_dispatch.h"
#include <iostream>


void fdtd_setup(VectorSpace* space) {
    #ifdef VIRS_WITH_CUDA
    std::cout << "Using GPU for FDTD simulation." << std::endl;
    fdtd_gpu_setup(space);
    #else
    fdtd_cpu_setup(space);
    #endif
}

void fdtd_step(VectorSpace* space, unsigned int step) {
    #ifdef VIRS_WITH_CUDA
    //std::cout << "Performing FDTD step on GPU." << std::endl;
    fdtd_gpu_step(space, space->h, step);
    #else
    fdtd_cpu_step(space, space->h);
    #endif
}

void updateAllGridFromGPU(VectorSpace* space) {
    #ifdef VIRS_WITH_CUDA
    //std::cout << "Updating all grid data from GPU to CPU." << std::endl;
    cudaMemcpy(space->getGrid().p_curr, space->getGrid().d_p_curr, sizeof(float) * space->getGrid().size, cudaMemcpyDeviceToHost);
    cudaMemcpy(space->getGrid().p_next, space->getGrid().d_p_next, sizeof(float) * space->getGrid().size, cudaMemcpyDeviceToHost);
    cudaMemcpy(space->getGrid().p_prev, space->getGrid().d_p_prev, sizeof(float) * space->getGrid().size, cudaMemcpyDeviceToHost);
    #else
    std::cout << "No GPU support, skipping update." << std::endl;
    #endif
}

void updateCurrentGridFromGPU(VectorSpace* space) {
    #ifdef VIRS_WITH_CUDA
    //std::cout << "Updating current grid data from GPU to CPU." << std::endl;
    cudaMemcpy(space->getGrid().p_curr, space->getGrid().d_p_curr, sizeof(float) * space->getGrid().size, cudaMemcpyDeviceToHost);
    #else
    std::cout << "No GPU support, skipping update." << std::endl;
    #endif
}

void fdtd_cleanup(VectorSpace* space) {
    #ifdef VIRS_WITH_CUDA
    std::cout << "Cleaning up GPU resources." << std::endl;
    fdtd_gpu_cleanup(space);
    #else
    fdtd_cpu_cleanup(space);
    #endif
}

void initPressureSphere(VectorSpace* space, size_t xpos, size_t ypos, size_t zpos, size_t radius, float pressure, bool init) {
    #ifdef VIRS_WITH_CUDA
    //std::cout << "Initializing pressure sphere on GPU." << std::endl;
    initPressureSphereGPU(space, xpos, ypos, zpos, radius, pressure, init);
    #else
    //std::cout << "Initializing pressure sphere on CPU." << std::endl;
    initPressureSphereCPU(space, xpos, ypos, zpos, radius, pressure, init);
    #endif
}

void updateGPUFromGrid(VectorSpace* space) {
    #ifdef VIRS_WITH_CUDA
    std::cout << "Updating GPU from grid data." << std::endl;
    cudaMemcpy(space->getGrid().d_p_curr, space->getGrid().p_curr, sizeof(float) * space->getGrid().size, cudaMemcpyHostToDevice);
    cudaMemcpy(space->getGrid().d_p_next, space->getGrid().p_next, sizeof(float) * space->getGrid().size, cudaMemcpyHostToDevice);
    cudaMemcpy(space->getGrid().d_p_prev, space->getGrid().p_prev, sizeof(float) * space->getGrid().size, cudaMemcpyHostToDevice);
    cudaMemcpy(space->getGrid().d_flags, space->getGrid().flags, sizeof(uint8_t) * space->getGrid().size, cudaMemcpyHostToDevice);
    cudaMemcpy(space->getGrid().d_p_absorb, space->getGrid().p_absorb, sizeof(float) * space->getGrid().size, cudaMemcpyHostToDevice);
    #else
    std::cout << "No GPU support, skipping update." << std::endl;
    #endif
}

void buildSpongeLayer(VectorSpace* space) {
    #ifdef VIRS_WITH_CUDA
    std::cout << "Building sponge layer on GPU." << std::endl;
    buildSpongeLayerGPU(space);
    #else
    std::cout << "Building sponge layer on CPU." << std::endl;
    // Implement CPU version if needed
    #endif
}

void fdtd_start_simulation(VectorSpace* space, size_t steps) {
    #ifdef VIRS_WITH_CUDA
    Grid &grid = space->getGrid();
    grid.p_audio_output_size = steps;
    grid.p_audio_output = new float[grid.p_audio_output_size];
    std::memset(grid.p_audio_output, 0, sizeof(float) * grid.p_audio_output_size);
    #endif // VIRS_WITH_CUDA
}

void uploadNormalsToGPU(VectorSpace* space) {
    #ifdef VIRS_WITH_CUDA
    std::cout << "Uploading normals to GPU." << std::endl;
    cudaMemcpy(space->getGrid().d_normals, space->getGrid().normals, sizeof(uint8_t) * space->getGrid().size, cudaMemcpyHostToDevice);
    #else
    std::cout << "No GPU support, skipping normals upload." << std::endl;
    #endif
}

void uploadPZetaToGPU(VectorSpace* space) {
    #ifdef VIRS_WITH_CUDA
    std::cout << "Uploading pZeta to GPU." << std::endl;
    cudaMemcpy(space->getGrid().d_pZeta, space->getGrid().pZeta, sizeof(float) * space->getGrid().size, cudaMemcpyHostToDevice);
    #else
    std::cout << "No GPU support, skipping pZeta upload." << std::endl;
    #endif
}