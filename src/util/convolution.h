#pragma once
#include <cmath>


/*
* Creates a brick wall FIR lowpass filter kernel.
*/
float* FIR_Lowpass_Kernel(float sampleRate, float cutoff, size_t numTaps);

/*
* Applies a convolution operation using the kernel on the input data.
*/
void convolute(float* input, float* output, float* kernel, size_t inputSize, size_t kernelSize);