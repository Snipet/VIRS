#include "1D_simulation.h"
#include <iostream>
#include <cstring>
#include "../util/image.h"
#include <cmath>

Simulation1D::Simulation1D() : length(512*8)
{
    p_curr = new float[length];
    p_next = new float[length];
    p_prev = new float[length];
    lap = new float[length];
    left_psi = 0.f;
    right_psi = 0.f;

    left_filter_states = new float[4];
    right_filter_states = new float[4];
    for(int i = 0; i < 4; ++i) {
        left_filter_states[i] = 0.f;
        right_filter_states[i] = 0.f;
    }


    dx = 0.007778 / 8.f;
    c = 343.f;
    dt = 0.80f * dx / (c * std::sqrt(1.f)); // Stability condition for wave equation
    simulation_step = 0;
    image_step = 100;

    float l2 = (c * dt / dx) * (c * dt / dx); // Courant number
    std::cout << "Courant number (l2): " << l2 << std::endl;

    float m = dx / (c * 1.225f);
    std::cout << "m: " << m << std::endl;

    lambda = c * dt / dx; // Courant number
    lo2 = l2;
    std::cout << "lo2: " << lo2 << std::endl;

    // Image generation
    width = 800;
    height = 600;

    framebuffer = new unsigned char[width * height * 3]; // RGB format
    memset(framebuffer, 0, width * height * 3);          // Initialize framebuffer to black

    initialize();
    const float fs = 1.f / dt; // Sample rate
    createHighpass(4000, fs);

    //createBoundaryFilter(left_boundary_state, 15000.f, fs);
    //createBoundaryFilter(right_boundary_state, 1000.f, fs);


    std::cout << "Simulation1D initialized with parameters:" << std::endl;
    std::cout << "Length: " << length << ", dx: " << dx << ", dt: " << dt << ", c: " << c << std::endl;;
}

Simulation1D::~Simulation1D()
{
    delete[] p_curr;
    delete[] p_next;
    delete[] p_prev;
}

void Simulation1D::initialize()
{
    for (size_t i = 0; i < length; ++i)
    {
        p_curr[i] = 0.f;
        p_next[i] = 0.f;
        p_prev[i] = 0.f;
        lap[i] = 0.f;
    }
}

void Simulation1D::runStep()
{
    float source_p = 0.f;
    float freq = 2000.f + (float)simulation_step * 6.f;
    float phase = dt * simulation_step * 6000; // 440 Hz tone
    source_p = std::sin(2.f * M_PI * phase);

    float h_phase = (float)simulation_step / 2500.f;
    if (h_phase > 1.f)
        h_phase = 1.f;
    float hann_window = 0.5f * (1.f - std::cos(2.f * M_PI * h_phase));
    source_p *= hann_window;

    size_t center_idx = length / 2;

    if(h_phase < 1.f){
        p_curr[center_idx] = source_p; // Set the source point
    }

    const float c2_dt2 = c * c * dt * dt;
    const float inv_dx2 = 1.f / (dx * dx);


    // float laplacian_val = (p_curr[0] - 2.f * p_curr[1] + p_curr[2]) * inv_dx2;
    // lap[1] = laplacian_val;
    // p_next[1] = 2.f * p_curr[1] - p_prev[1] + c2_dt2 * laplacian_val;

    //  int i = length - 2;
    // laplacian_val = (p_curr[i-1] - 2.f * p_curr[i] + p_curr[i + 1]) * inv_dx2;
    // lap[i] = laplacian_val;
    // p_next[i] = 2.f * p_curr[i] - p_prev[i] + c2_dt2 * laplacian_val;
    float l2 = (c * dt / dx) * (c * dt / dx); // Courant number
    float a1 = 2.f - 2.f * l2;
    float a2 = l2;
    float b1 = 2.f - l2;
    float b2 = a2;

    for (size_t i = 1; i < length - 1; ++i)
    {

        // Update the next pressure value using the wave equation
        //float laplacian_val = (p_curr[i + 1] - 2.f * p_curr[i] + p_curr[i - 1]) * inv_dx2;
        //float laplacian_val = (-p_curr[i + 2] + 16.f * p_curr[i + 1] - 30.f * p_curr[i] + 16.f * p_curr[i - 1] - p_curr[i - 2]) * inv_dx2 / 12.f;
        //lap[i] = laplacian_val;
        //p_next[i] = 2.f * p_curr[i] - p_prev[i] + c2_dt2 * laplacian_val;
        float partial = a1 * p_curr[i] - p_prev[i];
        partial += a2 * p_curr[i-1];
        partial += a2 * p_curr[i+1];
        p_next[i] = partial;
    }

    // Rigid wall boundary conditions
    // Left boundary condition
    p_next[0] = b1 * p_curr[0] - p_prev[0] + b2 * p_curr[1];
    // Right boundary condition
    p_next[length - 1] = b1 * p_curr[length - 1] - p_prev[length - 1] + b2 * p_curr[length - 2];

    const float inv_dx = 1.f / dx;
    const float alpha = dt / (1.225f * dx);
    const float foo = 0.6f;

    // float filtered_p_left = applyFilter(p_curr[0], left_filter_states);
    // float dpdt_left = (p_curr[0] - p_prev[0]) / dt;
    // left_psi += dt * alpha * filtered_p_left;
    //p_next[0] = applyFilter(p_curr[1], left_filter_states);



    //ATTEMPT 3
    // float mix_factor = applyFilter(p_curr[0], left_filter_states);
    // float p_absorb = p_curr[1] + (lambda - 1.f) / (lambda + 1.f) * (p_curr[1] - p_curr[0]);
    // float p_reflect = p_curr[1];
    // mix_factor = std::max(0.f, std::min(1.f, mix_factor));
    // p_next[0] = mix_factor * p_absorb + (1.f - mix_factor) * p_reflect;

    // // Update the right boundary condition
    // mix_factor = applyFilter(p_curr[length - 1], right_filter_states);
    // p_absorb = p_curr[length - 2] + (lambda - 1.f) / (lambda + 1.f) * (p_curr[length - 2] - p_curr[length - 1]);
    // p_reflect = p_curr[length - 2];
    // mix_factor = std::max(0.f, std::min(1.f, mix_factor));
    // p_next[length - 1] = mix_factor * p_absorb + (1.f - mix_factor) * p_reflect;

    //ATTEMPT 4
    // float du = p_next[0] - p_prev[0];
    // float dp_corr = applyBoundaryFilter(left_boundary_state, du);
    // p_next[0] = p_next[0] + dp_corr;

    float du = p_next[0] - p_prev[0];
    p_next[0] -= applyFilter(du, left_filter_states) * 0.4;

    du = p_next[length - 1] - p_prev[length - 1];
    p_next[length - 1] -= applyFilter(du, right_filter_states) * 0.4;

    // du = p_next[length - 1] - p_prev[length - 1];
    // dp_corr = applyBoundaryFilter(right_boundary_state, du);
    // p_next[length - 1] = p_next[length - 1] + dp_corr;



    // float filtered_p_right = applyFilter(p_curr[length - 1], right_filter_states);
    // float dpdt_right = (p_curr[length - 1] - p_prev[length - 1]) / dt;
    // right_psi += dt * alpha * filtered_p_right;
    //p_next[length - 1] = applyFilter(p_curr[length - 2], right_filter_states);

    

    if (simulation_step % image_step == 0)
    {
        renderImageToFile();
    }

    // Swap pointers for the next iteration
    std::swap(p_prev, p_curr);
    std::swap(p_curr, p_next);

    simulation_step++;
}

void Simulation1D::simulate(size_t steps)
{
    initialize();
    simulation_step = 0;
    for (size_t i = 0; i < steps; ++i)
    {
        runStep();
        std::cout << "\rStep " << i + 1 << std::flush;
    }

    std::cout << "\nSimulation completed after " << steps << " steps." << std::endl;
}

void Simulation1D::drawLine(unsigned int x1, unsigned int y1,
                            unsigned int x2, unsigned int y2,
                            unsigned char r, unsigned char g, unsigned char b)
{
    // Work with signed copies to avoid unsigned wrap-around
    int x = static_cast<int>(x1);
    int y = static_cast<int>(y1);
    int xe = static_cast<int>(x2);
    int ye = static_cast<int>(y2);

    int dx = std::abs(xe - x);
    int dy = std::abs(ye - y);
    int sx = (x < xe) ? 1 : -1;
    int sy = (y < ye) ? 1 : -1;
    int err = dx - dy;

    while (true)
    {
        if (x >= 0 && x < static_cast<int>(width) &&
            y >= 0 && y < static_cast<int>(height))
        {
            const std::size_t idx = (static_cast<std::size_t>(y) * width + x) * 3;
            framebuffer[idx] = r;
            framebuffer[idx + 1] = g;
            framebuffer[idx + 2] = b;
        }

        if (x == xe && y == ye)
            break;

        int err2 = err << 1; // 2*err
        if (err2 > -dy)
        {
            err -= dy;
            x += sx;
        }
        if (err2 < dx)
        {
            err += dx;
            y += sy;
        }
    }
}

void Simulation1D::renderImageToFile()
{
    // Clear all pixels to black
    memset(framebuffer, 0, width * height * 3);

    // Draw the current pressure values as a line graph
    unsigned int start_y = 0;
    float width_per_point = (float)width / (float)length;
    unsigned int graph_height = height / 4; // Used for the 4 graphs
    for (size_t i = 0; i < length - 1; ++i)
    {
        unsigned int x1 = i * width_per_point;
        unsigned int y1 = start_y + static_cast<unsigned int>((p_next[i] + 1.f) * graph_height / 2.f);
        unsigned int x2 = (i + 1) * width_per_point;
        unsigned int y2 = start_y + static_cast<unsigned int>((p_next[i + 1] + 1.f) * graph_height / 2.f);

        // Draw the line in red color
        drawLine(x1, y1, x2, y2, 255, 0, 0);
    }

    start_y += graph_height;
    // Draw (p_next - p_curr) as a line graph
    for (size_t i = 0; i < length - 1; ++i)
    {
        unsigned int x1 = i * width_per_point;
        unsigned int y1 = start_y + static_cast<unsigned int>(((p_next[i] - p_curr[i]) + 1.f) * graph_height / 2.f);
        unsigned int x2 = (i + 1) * width_per_point;
        unsigned int y2 = start_y + static_cast<unsigned int>(((p_next[i + 1] - p_curr[i + 1]) + 1.f) * graph_height / 2.f);

        // Draw the line in green color
        drawLine(x1, y1, x2, y2, 0, 255, 0);
    }
    start_y += graph_height;

    // Draw (p_next - p_prev) as a line graph
    for (size_t i = 0; i < length - 1; ++i)
    {
        unsigned int x1 = i * width_per_point;
        unsigned int y1 = start_y + static_cast<unsigned int>(((p_next[i] - p_prev[i]) + 1.f) * graph_height / 2.f);
        unsigned int x2 = (i + 1) * width_per_point;
        unsigned int y2 = start_y + static_cast<unsigned int>(((p_next[i + 1] - p_prev[i + 1]) + 1.f) * graph_height / 2.f);

        // Draw the line in blue color
        drawLine(x1, y1, x2, y2, 0, 0, 255);
    }

    // start_y += graph_height;
    // // Draw the laplacian as a line graph
    // for (size_t i = 0; i < length - 1; ++i) {
    //     unsigned int x1 = i * width_per_point;
    //     unsigned int y1 = start_y + static_cast<unsigned int>((lap[i] + 1.f) * graph_height / 2.f);
    //     unsigned int x2 = (i + 1) * width_per_point;
    //     unsigned int y2 = start_y + static_cast<unsigned int>((lap[i + 1] + 1.f) * graph_height / 2.f);

    //     // Draw the line in yellow color
    //     drawLine(x1, y1, x2, y2, 255, 255, 0);
    //}

    // Save the framebuffer to a PNG file
    std::string filename = getImageFilename(simulation_step / image_step, 1);
    save_png(filename.c_str(), framebuffer, width, height, false);
}


void Simulation1D::createHighpass(float cutoff, float sampleRate)
{
    // const float k = std::tan(M_PI * cutoff / sampleRate);
    // const float inv = 1.f / (1.f + k);

    // b0 = inv;
    // b1 = -inv;
    // b2 = 0.f;
    // a1 = (k - 1.f) * inv;
    // a2 = 0.f;

    const float omega = 2.f * M_PI * cutoff / sampleRate;
    const float sin_omega = std::sin(omega);
    const float cos_omega = std::cos(omega);
    const float Q = sqrt(0.5);
    const float alpha = sin_omega / (2.f * Q);
    const float a0 = 1.f + alpha;
    b0 = (1.f + cos_omega) / 2.f / a0;
    b1 = -(1.f + cos_omega) / a0;
    b2 = b0;
    a1 = -2.f * cos_omega / a0;
    a2 = (1.f - alpha) / a0;

    std::cout << "Highpass filter created with cutoff: " << cutoff << " Hz, sample rate: " << sampleRate << std::endl;
    std::cout << "Filter coefficients: " << std::endl << "b0 = " << b0 << std::endl;
    std::cout << "b1 = " << b1 << std::endl;
    std::cout << "b2 = " << b2 << std::endl;
    std::cout << "a1 = " << a1 << std::endl;
    std::cout << "a2 = " << a2 << std::endl;
}

float Simulation1D::applyFilter(float input, float* filter_states){
        float output = b0 * input + b1 * filter_states[0] + b2 * filter_states[1] - a1 * filter_states[2] - a2 * filter_states[3];
        filter_states[1] = filter_states[0];
        filter_states[0] = input;
        filter_states[3] = filter_states[2];
        filter_states[2] = output;
        return output;
}


void Simulation1D::createBoundaryFilter(BoundaryState& f, float cutoff, float sampleRate){
    float Ts = 1.f / sampleRate;
    float Z0 = 1.225f;
    float wc = 2.f * M_PI * cutoff;

    float D = 2e-3f;
    float E = 151.4f;
    float F = 2.152e7f;

    float Dh = D / Ts;
    float Eh = E;
    float Fh = F * Ts;

    float b = 1.f / (2.f * Dh + Eh + 0.5f * Fh);
    float d = (2.f * Dh - Eh - 0.5f * Fh);

    f.b = b;
    f.bd = b * d;
    f.bFh = b * Fh;
    f.bDh = b * Dh;
    f.beta = b;

    std::cout << "Boundary filter created with cutoff: " << cutoff << " Hz, sample rate: " << sampleRate << std::endl;
    std::cout << "Filter coefficients: " << std::endl;
    std::cout << "b = " << f.b << std::endl;
    std::cout << "bd = " << f.bd << std::endl;
    std::cout << "bDh = " << f.bDh << std::endl;
    std::cout << "bFh = " << f.bFh << std::endl;

    f.vh1 = 0.f;
    f.gh1 = 0.f;
}

float Simulation1D::applyBoundaryFilter(BoundaryState& f, float du) {

    float lo2Kbg = lo2 * dx * f.beta;
    float fac = 2.f * lo2;
    float out = -fac * (2.f * f.bDh * f.vh1 - f.bFh * f.gh1);

    float vh0 = f.b * du + f.bd * f.vh1 - 2.f * f.bFh * f.gh1;

    f.gh1 += 0.5f * (vh0 + f.vh1);
    f.vh1 = vh0;
    return out;
}