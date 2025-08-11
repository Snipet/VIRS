#include "convolution.h"
#include <cstddef>
#include <cmath>

// Applies y[n] = sum_{k=0}^{kernelSize-1} x[n - k] * h[k]
// For n < 0, x[n] is treated as 0 (zero-padding before the start).
// Writes exactly inputSize samples to output.
void convolute(float* input, float* output, float* kernel, size_t inputSize, size_t kernelSize)
{
    if (!input || !output || !kernel || inputSize == 0 || kernelSize == 0) {
        return;
    }

    for (size_t n = 0; n < inputSize; ++n) {
        double acc = 0.0; // use double accumulator for better precision

        // Only k <= n contributes; for k > n, (n - k) < 0 -> zero due to padding
        const size_t kmax = (n < kernelSize - 1) ? n : (kernelSize - 1);

        for (size_t k = 0; k <= kmax; ++k) {
            acc += static_cast<double>(input[n - k]) * static_cast<double>(kernel[k]);
        }

        output[n] = static_cast<float>(acc);
    }
}
