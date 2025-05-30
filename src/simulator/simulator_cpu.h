#pragma once
#include "vectorspace.h"

void fdtd_cpu_setup(VectorSpace* space);
void fdtd_cpu_step(VectorSpace* space, float h);