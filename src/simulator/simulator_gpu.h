#pragma once
#include "vectorspace.h"

void fdtd_gpu_setup(VectorSpace* space);
extern "C"{
    void fdtd_gpu_step(VectorSpace* space, float h);
    void initPressureSphereGPU(VectorSpace* space, size_t xpos, size_t ypos, size_t zpos, size_t radius, float pressure, bool init);
}
void fdtd_gpu_cleanup(VectorSpace* space);