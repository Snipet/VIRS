#include "downsampling_filter_fir.h"
#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace MinPhaseLPF
{
    static inline float sinc_pi(float x) {
        if (std::abs(x) < 1e-12) return 1.0;
        return std::sin(M_PI * x) / (M_PI * x);
    }

    static inline float kaiser_beta(float A) {
        if (A <= 21.0) return 0.0;
        if (A <= 50.0) return 0.5842 * std::pow(A - 21.0, 0.4) + 0.07886 * (A - 21.0);
        return 0.1102 * (A - 8.7);
    }

    static inline float I0(float x) {
    #if (__cpp_lib_math_special_functions >= 201603L)
        return std::cyl_bessel_i(0.0, x);
    #else
        float y = x * x / 4.0;
        float sum = 1.0, t = 1.0;
        for (int k = 1; k < 50; ++k) {
            t *= y / (k * k);
            sum += t;
            if (t < 1e-16) break;
        }
        return sum;
    #endif
    }

    static std::vector<std::complex<float>> dft(const std::vector<std::complex<float>>& x) {
        const size_t N = x.size();
        std::vector<std::complex<float>> X(N);
        const float twopi_over_N = -2.0 * M_PI / float(N);
        for (size_t k = 0; k < N; ++k) {
            std::complex<float> acc(0.0, 0.0);
            for (size_t n = 0; n < N; ++n) {
                float ang = twopi_over_N * float(k) * float(n);
                acc += x[n] * std::complex<float>(std::cos(ang), std::sin(ang));
            }
            X[k] = acc;
        }
        return X;
    }

    static std::vector<std::complex<float>> idft(const std::vector<std::complex<float>>& X) {
        const size_t N = X.size();
        std::vector<std::complex<float>> x(N);
        const float twopi_over_N = 2.0 * M_PI / float(N);
        for (size_t n = 0; n < N; ++n) {
            std::complex<float> acc(0.0, 0.0);
            for (size_t k = 0; k < N; ++k) {
                float ang = twopi_over_N * float(k) * float(n);
                acc += X[k] * std::complex<float>(std::cos(ang), std::sin(ang));
            }
            x[n] = acc / float(N);
        }
        return x;
    }

    static std::vector<float> minimum_phase_homomorphic(const std::vector<float>& h_lin) {
        const size_t L = h_lin.size();
        size_t N = 1;
        while (N < 4 * L) N <<= 1;

        std::vector<std::complex<float>> x(N, {0.0, 0.0});
        for (size_t n = 0; n < L; ++n) x[n] = std::complex<float>(h_lin[n], 0.0);
        auto H = dft(x);

        const float eps = 1e-12;
        for (auto& v : H) {
            float mag = std::max(eps, std::abs(v));
            v = std::log(mag);
        }

        auto c = idft(H);
        std::vector<std::complex<float>> cc(N, {0.0, 0.0});
        cc[0] = c[0];
        const bool even = (N % 2 == 0);
        size_t nhalf = N / 2;
        for (size_t n = 1; n < nhalf; ++n) cc[n] = 2.f * c[n];
        if (even) cc[nhalf] = c[nhalf];

        auto LC = dft(cc);
        for (auto& v : LC) v = std::exp(v.real());

        auto h_min_c = idft(LC);
        std::vector<float> h_min(L);
        for (size_t n = 0; n < L; ++n) h_min[n] = h_min_c[n].real();

        return h_min;
    }

    std::vector<float> design(float fs_hz, float fc_hz,
                               float trans_hz,
                               float atten_db,
                               int taps_hint)
    {
        if (fc_hz <= 0 || fs_hz <= 0 || fc_hz >= 0.5 * fs_hz)
            throw std::invalid_argument("fc must be in (0, fs/2)");
        if (trans_hz <= 0) trans_hz = std::max(1.0, 0.1 * fc_hz);

        float beta = kaiser_beta(atten_db);
        float dw = 2.0 * M_PI * (trans_hz / fs_hz);
        int N_est = int(std::ceil((atten_db - 8.0) / (2.285 * dw)));
        int N = (taps_hint > 0 ? taps_hint : std::max(5, N_est));
        if (N % 2 == 0) ++N;

        float fc_norm = fc_hz / fs_hz;
        std::vector<float> h_lin(N);
        int M = N - 1;
        std::vector<float> w(N);
        for (int n = 0; n < N; ++n) {
            float r = (2.0 * n - M) / float(M);
            w[n] = I0(M_PI * beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / I0(M_PI * beta);
        }
        for (int n = 0; n < N; ++n) {
            float x = float(n) - 0.5 * M;
            h_lin[n] = 2.0 * fc_norm * sinc_pi(2.0 * fc_norm * x) * w[n];
        }

        auto h_min = minimum_phase_homomorphic(h_lin);

        float sum = 0.0; for (float v : h_min) sum += v;
        if (std::abs(sum) > 1e-12) for (float& v : h_min) v /= sum;

        return h_min;
    }
}
