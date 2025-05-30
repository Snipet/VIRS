#pragma once
#include "vectorspace.h"

void fdtd_setup(VectorSpace* space);
void fdtd_step(VectorSpace* space);

// If using GPU, this function will update the grid from the GPU memory to the CPU memory.
void updateAllGridFromGPU(VectorSpace* space);

// If using GPU, this function will update the grid from the GPU memory to the CPU memory for a specific space.
void updateCurrentGridFromGPU(VectorSpace* space);