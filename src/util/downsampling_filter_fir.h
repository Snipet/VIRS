#pragma once
#include <vector>

namespace MinPhaseLPF
{
    // Designs a minimum-phase low-pass FIR kernel.
    // fc_hz: cutoff frequency in Hz.
    // fs_hz: sample rate in Hz.
    // trans_hz: transition width in Hz (if <= 0, defaults to 0.1 * fc_hz).
    // atten_db: stopband attenuation in dB.
    // taps_hint: if >0, use exactly this many taps for the prototype, else auto-estimate.
    std::vector<float> design(float fs_hz, float fc_hz,
                                float trans_hz = -1.0,
                                float atten_db = 80.0,
                                int taps_hint = -1);
}
