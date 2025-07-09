#include "1D_simulation.h"
#include <iostream>
#include <cstring>
#include "../util/image.h"
#include <cmath>

Simulation1D::Simulation1D() : length(512)
{
    p_curr = new float[length];
    p_next = new float[length];
    p_prev = new float[length];
    lap = new float[length];
    dx = 0.007778;
    c = 343.f;
    dt = 0.80f * dx / (c * std::sqrt(1.f)); // Stability condition for wave equation
    simulation_step = 0;
    image_step = 100;

    float m = dx / (c * 1.225f);
    std::cout << "m: " << m << std::endl;

    lambda = c * dt / dx; // Courant number
    float r = 0.0;
    zeta = (1.f + r) / (1.f - r); // Damping factor for boundary conditions

    // Image generation
    width = 800;
    height = 600;

    framebuffer = new unsigned char[width * height * 3]; // RGB format
    memset(framebuffer, 0, width * height * 3);          // Initialize framebuffer to black

    initialize();

    float lambda_over_zeta = lambda / zeta;
    float A_coeff = (1.f - lambda_over_zeta) / (1.f + lambda_over_zeta);
    float B_coeff = 2.f * lambda_over_zeta / (1.f + lambda_over_zeta);

    std::cout << "Simulation1D initialized with parameters:" << std::endl;
    std::cout << "Length: " << length << ", dx: " << dx << ", dt: " << dt << ", c: " << c << std::endl;
    std::cout << "Lambda: " << lambda << ", Zeta: " << zeta << std::endl;
    std::cout << "A_coeff: " << A_coeff << ", B_coeff: " << B_coeff << std::endl;
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

    float phase = dt * simulation_step * 440.f; // 440 Hz tone
    source_p = std::sin(2.f * M_PI * phase);

    float h_phase = (float)simulation_step / 1000.f;
    if (h_phase > 1.f)
        h_phase = 1.f;
    float hann_window = 0.5f * (1.f - std::cos(2.f * M_PI * h_phase));
    source_p *= hann_window;

    size_t center_idx = length / 2;
    p_curr[center_idx] = source_p; // Set the source point

    const float c2_dt2 = c * c * dt * dt;
    const float inv_dx2 = 1.f / (dx * dx);

    // Apply boundary conditions

    float lambda_over_zeta = lambda / zeta;
    float A_coeff = (1.f - lambda_over_zeta) / (1.f + lambda_over_zeta);
    float B_coeff = 2.f * lambda_over_zeta / (1.f + lambda_over_zeta);

    float laplacian_val = (p_curr[0] - 2.f * p_curr[1] + p_curr[2]) * inv_dx2;
    lap[1] = laplacian_val;
    p_next[1] = 2.f * p_curr[1] - p_prev[1] + c2_dt2 * laplacian_val;

     int i = length - 2;
    laplacian_val = (p_curr[i-1] - 2.f * p_curr[i] + p_curr[i + 1]) * inv_dx2;
    lap[i] = laplacian_val;
    p_next[i] = 2.f * p_curr[i] - p_prev[i] + c2_dt2 * laplacian_val;

    p_next[0] = 0.f; // Dirichlet BC
    p_next[length - 1] = 0.f;

    for (size_t i = 2; i < length - 2; ++i)
    {

        // Update the next pressure value using the wave equation
        // float laplacian_val = (p_curr[i + 1] - 2.f * p_curr[i] + p_curr[i - 1]) * inv_dx2;
        float laplacian_val = (-p_curr[i + 2] + 16.f * p_curr[i + 1] - 30.f * p_curr[i] + 16.f * p_curr[i - 1] - p_curr[i - 2]) * inv_dx2 / 12.f;
        lap[i] = laplacian_val;
        p_next[i] = 2.f * p_curr[i] - p_prev[i] + c2_dt2 * laplacian_val;
    }

    p_next[0] = A_coeff * p_curr[1] + B_coeff * p_curr[0];
    p_next[length - 1] = A_coeff * p_curr[length - 2] + B_coeff * p_curr[length - 1];

    if (simulation_step % image_step != 0)
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
