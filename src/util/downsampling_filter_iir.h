#pragma once
#include <vector>
#include <array>
#include <cstddef>

struct Biquad {
    double b0, b1, b2;  // numerator
    double a0, a1, a2;  // denominator (a0 will be normalized to 1)
};

// Design a steep Chebyshev-I low-pass suitable for anti-aliasing before decimation.
// fs_hz: sample rate, fc_hz: passband edge, fstop_hz: stopband edge (> fc_hz, if 0 -> 1.15*fc_hz)
// Ap_dB: passband ripple (e.g., 0.5..1 dB), As_dB: stopband attenuation (e.g., 80–100 dB)
std::vector<Biquad> designChebyshevI_LP(double fs_hz, double fc_hz,
                                        double fstop_hz = 0.0,
                                        double Ap_dB = 0.5,
                                        double As_dB = 90.0);

// -------- Block processing (single-threaded) --------

// Persistent IIR state for cascaded biquads, interleaved audio.
struct IIRState {
    int channels = 0;
    // For DF2T: two delay elements per biquad, per channel.
    // Layout: [ch0 s0:{z1,z2}, ch0 s1:{z1,z2}, ..., ch1 s0:{...}, ...]
    std::vector<std::array<double,2>> z;

    void reset() {
        for (auto& p : z) { p[0] = 0.0; p[1] = 0.0; }
    }
};

// Initialize/resize state for given SOS cascade and channel count.
// Does not modify coefficient values.
void initState(IIRState& st, const std::vector<Biquad>& sos, int channels);

// Process a block of interleaved audio in-place or out-of-place (single thread).
// in/out: interleaved float buffers (size = frames*channels). They may alias (in==out).
void processBlock(const std::vector<Biquad>& sos,
                  IIRState& st,
                  const float* in,
                  float* out,
                  std::size_t frames);
