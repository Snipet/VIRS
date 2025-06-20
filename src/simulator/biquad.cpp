#include "biquad.h"

BiquadCoeffs computeLowpassBiquad(float sampleRate, float cutoff, float Q, float peakGain) {
    BiquadCoeffs coeffs;

    float omega = 2.f * M_PI * cutoff / sampleRate;
    float alpha = sin(omega) / (2.f * Q);
    float a0 = 1.f + alpha;

    coeffs.b0 = (1.f - cos(omega)) / 2.f / a0;
    coeffs.b1 = (1.f - cos(omega)) / a0;
    coeffs.b2 = coeffs.b0;
    coeffs.a1 = -2.f * cos(omega) / a0;
    coeffs.a2 = (1.f - alpha) / a0;

    // Apply peak gain if specified
    coeffs.b0 *= peakGain;
    coeffs.b1 *= peakGain;
    coeffs.b2 *= peakGain;

    return coeffs;
}

BiquadCoeffs computeHighpassBiquad(float sampleRate, float cutoff, float Q, float peakGain) {
    BiquadCoeffs coeffs;

    float omega = 2.f * M_PI * cutoff / sampleRate;
    float alpha = sin(omega) / (2.f * Q);
    float a0 = 1.f + alpha;

    coeffs.b0 = (1.f + cos(omega)) / 2.f / a0;
    coeffs.b1 = -(1.f + cos(omega)) / a0;
    coeffs.b2 = coeffs.b0;
    coeffs.a1 = -2.f * cos(omega) / a0;
    coeffs.a2 = (1.f - alpha) / a0;

    // Apply peak gain if specified
    coeffs.b0 *= peakGain;
    coeffs.b1 *= peakGain;
    coeffs.b2 *= peakGain;

    return coeffs;
}

float getBiquadGain(const BiquadCoeffs& coeffs, float frequency) {
    float omega = 2.f * M_PI * frequency;
    float s = sin(omega);
    float c = cos(omega);
    float a0 = 1.f + coeffs.a1 * c + coeffs.a2 * c * c;

    float gain = (coeffs.b0 + coeffs.b1 * c + coeffs.b2 * c * c) / a0;
    return gain;
}

float getBiquadGainDB(const BiquadCoeffs& coeffs, float frequencyHz, float sampleRate){
    float frequency = frequencyHz / sampleRate;
    float gain = getBiquadGain(coeffs, frequency);
    float gainDB = 20.f * log10(gain);
    return gainDB;

}

std::string biquadCoeffsToString(const BiquadCoeffs& coeffs){
    return "Biquad Coeffs: b0: " + std::to_string(coeffs.b0) +
           ", b1: " + std::to_string(coeffs.b1) +
           ", b2: " + std::to_string(coeffs.b2) +
           ", a1: " + std::to_string(coeffs.a1) +
           ", a2: " + std::to_string(coeffs.a2);
}