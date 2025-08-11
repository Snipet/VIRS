#pragma once
#include <cmath>


/*
* Applies a convolution operation using the kernel on the input data.
*/
void convolute(float* input, float* output, float* kernel, size_t inputSize, size_t kernelSize);