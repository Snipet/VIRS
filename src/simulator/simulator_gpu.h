#pragma once
#include "vectorspace.h"

void fdtd_gpu_setup(VectorSpace* space);
extern "C"{
    bool fdtd_gpu_step(VectorSpace* space, float h, unsigned int step);
    void initPressureSphereGPU(VectorSpace* space, size_t xpos, size_t ypos, size_t zpos, size_t radius, float pressure, bool init);
    void buildSpongeLayerGPU(VectorSpace* space);
}
void fdtd_gpu_cleanup(VectorSpace* space);