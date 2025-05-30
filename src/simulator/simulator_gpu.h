#pragma once
#include "vectorspace.h"

void fdtd_gpu_setup(VectorSpace* space);
void fdtd_gpu_step(VectorSpace* space, float h);