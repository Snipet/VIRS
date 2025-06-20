#pragma once
#include <cmath>
#include <string>

struct BiquadCoeffs {
    float b0, b1, b2;
    float a1, a2;
};

inline BiquadCoeffs createBiquadCoeffs(float b0, float b1, float b2, float a1, float a2) {
    BiquadCoeffs coeffs;
    coeffs.b0 = b0;
    coeffs.b1 = b1;
    coeffs.b2 = b2;
    coeffs.a1 = a1;
    coeffs.a2 = a2;
    return coeffs;
}

BiquadCoeffs computeLowpassBiquad(float sampleRate, float cutoff, float Q, float peakGain = 1.f);
BiquadCoeffs computeHighpassBiquad(float sampleRate, float cutoff, float Q, float peakGain = 1.f);
float getBiquadGain(const BiquadCoeffs& coeffs, float frequency);
float getBiquadGainDB(const BiquadCoeffs& coeffs, float frequencyHz, float sampleRate);
std::string biquadCoeffsToString(const BiquadCoeffs& coeffs);