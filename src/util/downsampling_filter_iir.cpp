#include "downsampling_filter_iir.h"
#include <complex>
#include <cmath>
#include <stdexcept>
#include <algorithm>

static inline double dB2lin(double dB) { return std::pow(10.0, dB / 20.0); }
static inline double acoshd(double x) { return std::log(x + std::sqrt(x * x - 1.0)); }
static inline double asinhd(double x) { return std::log(x + std::sqrt(x * x + 1.0)); }

static void distributeDCGain(std::vector<Biquad> &sos)
{
    if (sos.empty())
        return;

    // Compute overall H(z=1) = Π ( (b0+b1+b2)/(a0+a1+a2) )
    long double H1 = 1.0L;
    for (const auto &s : sos)
    {
        long double num = (long double)s.b0 + s.b1 + s.b2;
        long double den = (long double)s.a0 + s.a1 + s.a2;
        H1 *= num / den;
    }
    if (!std::isfinite((double)H1) || fabsl(H1) < 1e-300L)
        return; // nothing we can do sanely

    // Target overall DC gain = 1 => multiply numerators by G = 1/H1
    long double G = 1.0L / H1;

    // Spread gain evenly to avoid internal blow-ups:
    // per-section factor g = G^(1/N). Accumulate rounding into the first section.
    const std::size_t N = sos.size();
    long double g = powl(G, 1.0L / (long double)N);

    // Apply to all sections
    for (auto &s : sos)
    {
        s.b0 = (double)((long double)s.b0 * g);
        s.b1 = (double)((long double)s.b1 * g);
        s.b2 = (double)((long double)s.b2 * g);
    }

    // Small correction so exact DC=1 despite rounding
    // Recompute and nudge the first section.
    long double H1b = 1.0L;
    for (const auto &s : sos)
    {
        long double num = (long double)s.b0 + s.b1 + s.b2;
        long double den = (long double)s.a0 + s.a1 + s.a2;
        H1b *= num / den;
    }
    long double c = 1.0L / H1b; // tiny correction close to 1
    sos[0].b0 = (double)((long double)sos[0].b0 * c);
    sos[0].b1 = (double)((long double)sos[0].b1 * c);
    sos[0].b2 = (double)((long double)sos[0].b2 * c);
}

std::vector<Biquad> designChebyshevI_LP(double fs_hz, double fc_hz,
                                        double fstop_hz,
                                        double Ap_dB,
                                        double As_dB)
{
    if (fs_hz <= 0 || fc_hz <= 0 || fc_hz >= 0.5 * fs_hz)
        throw std::invalid_argument("Invalid fs or fc.");
    if (fstop_hz <= 0.0)
        fstop_hz = std::min(0.499 * fs_hz, 1.15 * fc_hz);
    if (fstop_hz <= fc_hz)
        throw std::invalid_argument("fstop_hz must be > fc_hz.");

    // Bilinear prewarping
    const double T = 1.0 / fs_hz;
    const double wp = 2.0 / T * std::tan(M_PI * (fc_hz / fs_hz));    // passband edge (analog)
    const double ws = 2.0 / T * std::tan(M_PI * (fstop_hz / fs_hz)); // stopband edge (analog)
    const double W = 2.0 / T;                                        // bilinear constant

    // Chebyshev-I order from specs
    const double ep = std::sqrt(std::pow(10.0, Ap_dB / 10.0) - 1.0);
    const double num = acoshd(std::sqrt((std::pow(10.0, As_dB / 10.0) - 1.0) /
                                        (std::pow(10.0, Ap_dB / 10.0) - 1.0)));
    const double den = acoshd(ws / wp);
    int N = (int)std::ceil(num / den);
    N = std::max(1, N);

    // Chebyshev-I poles (low-pass, cutoff wp)
    const double a = asinhd(1.0 / ep) / N;
    std::vector<std::complex<double>> poles;
    poles.reserve(N);

    for (int k = 1; k <= N; ++k)
    {
        const double theta = M_PI * (2.0 * k - 1.0) / (2.0 * N);
        const double sigma = -std::sinh(a) * std::sin(theta);
        const double omega = std::cosh(a) * std::cos(theta);
        std::complex<double> sk = std::complex<double>(sigma, omega) * wp;
        if (sk.real() >= 0)
            sk = std::complex<double>(-std::abs(sk.real()), sk.imag());
        poles.push_back(sk);
    }

    // Pair into biquads (complex conjugate), handle odd-order real pole
    std::sort(poles.begin(), poles.end(),
              [](auto A, auto B)
              { return A.imag() < B.imag(); });

    std::vector<Biquad> sos;
    sos.reserve((N + 1) / 2);

    int i = 0;
    while (i < N)
    {
        const auto p = poles[i];
        if (i + 1 == N || std::abs(p.imag()) < 1e-12)
        {
            // First-order section (rare in practice unless N odd)
            std::complex<double> a0c = (W - p);
            std::complex<double> a1c = -(W + p);

            Biquad s{};
            s.b0 = 1.0;
            s.b1 = 1.0;
            s.b2 = 0.0;
            s.a0 = a0c.real();
            s.a1 = a1c.real();
            s.a2 = 0.0;

            const double inva0 = 1.0 / s.a0;
            s.a0 = 1.0;
            s.a1 *= inva0;
            s.a2 *= inva0;
            s.b0 *= inva0;
            s.b1 *= inva0;
            s.b2 *= inva0;

            sos.push_back(s);
            ++i;
        }
        else
        {
            const double are = p.real();
            const double b2 = std::norm(p); // |p|^2
                                            // Bilinear of (s^2 - 2*Re(p)s + |p|^2)
            double A0 = (W * W - 2.0 * are * W + b2);
            double A1 = 2.0 * (b2 - W * W);
            double A2 = (W * W + 2.0 * are * W + b2);

            Biquad s{};
            s.b0 = 1.0;
            s.b1 = 2.0;
            s.b2 = 1.0; // zeros at z=-1
            s.a0 = A0;
            s.a1 = A1;
            s.a2 = A2;

            const double inva0 = 1.0 / s.a0;
            s.a0 = 1.0;
            s.a1 *= inva0;
            s.a2 *= inva0;
            s.b0 *= inva0;
            s.b1 *= inva0;
            s.b2 *= inva0;

            sos.push_back(s);
            i += 2;
        }
    }

    distributeDCGain(sos);
    return sos;
}

// -------- Block processing --------

void initState(IIRState &st, const std::vector<Biquad> &sos, int channels)
{
    if (channels <= 0)
        throw std::invalid_argument("channels must be > 0");
    st.channels = channels;
    st.z.assign(static_cast<std::size_t>(channels) * sos.size(), {0.0, 0.0});
}

void processBlock(const std::vector<Biquad> &sos,
                  IIRState &st,
                  const float *in,
                  float *out,
                  std::size_t frames)
{
    if (st.channels <= 0)
        return;
    const int C = st.channels;
    const std::size_t nSec = sos.size();
    if (nSec == 0)
    {
        // passthrough
        if (out != in)
            std::copy(in, in + frames * C, out);
        return;
    }

    // Single-threaded, interleaved DF2T per section, per channel.
    for (std::size_t n = 0; n < frames; ++n)
    {
        for (int ch = 0; ch < C; ++ch)
        {
            const std::size_t idx = n * C + ch;
            double y = static_cast<double>(in[idx]);

            for (std::size_t s = 0; s < nSec; ++s)
            {
                const Biquad &bq = sos[s];
                auto &zi = st.z[ch * nSec + s]; // zi[0]=z1, zi[1]=z2

                // Standard DF2T:
                // y = b0*x + z1
                // z1' = b1*x - a1*y + z2
                // z2' = b2*x - a2*y
                const double x = y;
                const double yout = bq.b0 * x + zi[0];
                const double z1n = bq.b1 * x - bq.a1 * yout + zi[1];
                const double z2n = bq.b2 * x - bq.a2 * yout;

                zi[0] = z1n;
                zi[1] = z2n;
                y = yout;
            }

            out[idx] = static_cast<float>(y);
        }
    }
}
