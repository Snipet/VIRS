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

bool fdtd_step(VectorSpace* space, unsigned int step) {
    #ifdef VIRS_WITH_CUDA
    //std::cout << "Performing FDTD step on GPU." << std::endl;
    return fdtd_gpu_step(space, space->h, step);
    #else
    fdtd_cpu_step(space, space->h);
    return false;
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
    //cudaMemcpy(space->getGrid().d_p_absorb, space->getGrid().p_absorb, sizeof(float) * space->getGrid().size, cudaMemcpyHostToDevice);
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
    // #ifdef VIRS_WITH_CUDA
    // std::cout << "Uploading pZeta to GPU." << std::endl;
    // cudaMemcpy(space->getGrid().d_pZeta, space->getGrid().pZeta, sizeof(float) * space->getGrid().size, cudaMemcpyHostToDevice);
    // #else
    // std::cout << "No GPU support, skipping pZeta upload." << std::endl;
    // #endif
}

void uploadBoundaryIndicesToGPU(VectorSpace* space) {
    #ifdef VIRS_WITH_CUDA
    std::cout << "Uploading boundary indices to GPU." << std::endl;
    std::cout << "Boundary indices size: " << space->getGrid().boundary_indices_size << std::endl;
    Grid &grid = space->getGrid();
    cudaMalloc((void**)&grid.d_boundary_indices, sizeof(uint32_t) * grid.boundary_indices_size);
    cudaMemcpy(grid.d_boundary_indices, grid.boundary_indices, sizeof(uint32_t) * grid.boundary_indices_size, cudaMemcpyHostToDevice);
    #else
    std::cout << "No GPU support, skipping boundary indices upload." << std::endl;
    #endif
}

void allocFilterStates(VectorSpace* space){
    #ifdef VIRS_WITH_CUDA
    std::cout << "Allocating filter states on GPU." << std::endl;
   
    Grid& grid = space->getGrid();
    size_t num_biquad_sections = grid.num_filter_sections;
    size_t num_boundary_voxels = grid.boundary_indices_size;
    size_t N = num_biquad_sections * num_boundary_voxels;

    // Allocate memory for filter states on CPU
    grid.biquad_state_ptr = new float[N * 4]; // 4 states per section, per voxel
    std::memset(grid.biquad_state_ptr, 0, sizeof(float) * N * 4);
    // Allocate memory for filter states on GPU
    cudaMalloc((void**)&grid.d_biquad_state_ptr, sizeof(float) * N * 4);
    cudaMemset(grid.d_biquad_state_ptr, 0, sizeof(float) * N * 4);



    grid.allocated_filter_memory = true;

    #else
    std::cout << "No GPU support, skipping filter setup." << std::endl;
    #endif
}

void allocFilterCoeffs(VectorSpace* space, const size_t num_materials){
#ifdef VIRS_WITH_CUDA
    std::cout << "Allocating filter coefficients on GPU." << std::endl;

    Grid& grid = space->getGrid();
    grid.num_materials = num_materials;

    // Allocate memory for filter coefficients on CPU
    grid.biquad_a1 = new float[num_materials];
    grid.biquad_a2 = new float[num_materials];
    grid.biquad_b0 = new float[num_materials];
    grid.biquad_b1 = new float[num_materials];
    grid.biquad_b2 = new float[num_materials];

    // Zero out the filter coefficients
    std::memset(grid.biquad_a1, 0, sizeof(float) * num_materials);
    std::memset(grid.biquad_a2, 0, sizeof(float) * num_materials);
    std::memset(grid.biquad_b0, 0, sizeof(float) * num_materials);
    std::memset(grid.biquad_b1, 0, sizeof(float) * num_materials);
    std::memset(grid.biquad_b2, 0, sizeof(float) * num_materials);

    // Allocate memory for filter coefficients on GPU
    cudaMalloc((void**)&grid.d_biquad_a1, sizeof(float) * num_materials);
    cudaMalloc((void**)&grid.d_biquad_a2, sizeof(float) * num_materials);
    cudaMalloc((void**)&grid.d_biquad_b0, sizeof(float) * num_materials);
    cudaMalloc((void**)&grid.d_biquad_b1, sizeof(float) * num_materials);
    cudaMalloc((void**)&grid.d_biquad_b2, sizeof(float) * num_materials);

    // Copy filter coefficients from CPU to GPU
    cudaMemcpy(grid.d_biquad_a1, grid.biquad_a1, sizeof(float) * num_materials, cudaMemcpyHostToDevice);
    cudaMemcpy(grid.d_biquad_a2, grid.biquad_a2, sizeof(float) * num_materials, cudaMemcpyHostToDevice);
    cudaMemcpy(grid.d_biquad_b0, grid.biquad_b0, sizeof(float) * num_materials, cudaMemcpyHostToDevice);
    cudaMemcpy(grid.d_biquad_b1, grid.biquad_b1, sizeof(float) * num_materials, cudaMemcpyHostToDevice);
    cudaMemcpy(grid.d_biquad_b2, grid.biquad_b2, sizeof(float) * num_materials, cudaMemcpyHostToDevice);

    // Set up filter coefficient pointers
    grid.d_biquad_coeffs_ptr[0] = grid.d_biquad_b0;
    grid.d_biquad_coeffs_ptr[1] = grid.d_biquad_b1;
    grid.d_biquad_coeffs_ptr[2] = grid.d_biquad_b2;
    grid.d_biquad_coeffs_ptr[3] = grid.d_biquad_a1;
    grid.d_biquad_coeffs_ptr[4] = grid.d_biquad_a2;

#else
    std::cout << "No GPU support, skipping filter coefficients allocation." << std::endl;
#endif // VIRS_WITH_CUDA
}

void uploadFilterCoeffsToGPU(VectorSpace* space) {
#ifdef VIRS_WITH_CUDA
    std::cout << "Uploading filter coefficients to GPU." << std::endl;

    Grid& grid = space->getGrid();
    cudaMemcpy(grid.d_biquad_a1, grid.biquad_a1, sizeof(float) * grid.num_materials, cudaMemcpyHostToDevice);
    cudaMemcpy(grid.d_biquad_a2, grid.biquad_a2, sizeof(float) * grid.num_materials, cudaMemcpyHostToDevice);
    cudaMemcpy(grid.d_biquad_b0, grid.biquad_b0, sizeof(float) * grid.num_materials, cudaMemcpyHostToDevice);
    cudaMemcpy(grid.d_biquad_b1, grid.biquad_b1, sizeof(float) * grid.num_materials, cudaMemcpyHostToDevice);
    cudaMemcpy(grid.d_biquad_b2, grid.biquad_b2, sizeof(float) * grid.num_materials, cudaMemcpyHostToDevice);
#else
    std::cout << "No GPU support, skipping filter coefficients upload." << std::endl;
#endif // VIRS_WITH_CUDA
}