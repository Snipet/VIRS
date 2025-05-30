#pragma once
#include "vectorspace.h"

void fdtd_cpu_setup(VectorSpace* space);
void fdtd_cpu_step(VectorSpace* space, float h);
void fdtd_cpu_cleanup(VectorSpace* space);
void initPressureSphereCPU(VectorSpace* space, size_t xpos, size_t ypos, size_t zpos, size_t radius, float pressure, bool init);