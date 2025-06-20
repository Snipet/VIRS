#pragma once
#include "vectorspace.h"

void fdtd_setup(VectorSpace* space);
bool fdtd_step(VectorSpace* space, unsigned int step);
void fdtd_cleanup(VectorSpace* space);
void fdtd_start_simulation(VectorSpace* space, size_t steps);
void initPressureSphere(VectorSpace* space, size_t xpos, size_t ypos, size_t zpos, size_t radius, float pressure, bool init);
void buildSpongeLayer(VectorSpace* space);

// If using GPU, this function will update the grid from the GPU memory to the CPU memory.
void updateAllGridFromGPU(VectorSpace* space);

// If using GPU, this function will update the grid from the GPU memory to the CPU memory for a specific space.
void updateCurrentGridFromGPU(VectorSpace* space);

// If using GPU, this function will update the GPU's grid data from the CPU memory.
void updateGPUFromGrid(VectorSpace* space);

void uploadNormalsToGPU(VectorSpace* space);
void uploadPZetaToGPU(VectorSpace* space);
void uploadBoundaryIndicesToGPU(VectorSpace* space);
void allocFilterStates(VectorSpace* space);
void allocFilterCoeffs(VectorSpace* space, const size_t num_materials);
void uploadFilterCoeffsToGPU(VectorSpace* space);

