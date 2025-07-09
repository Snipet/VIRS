#pragma once
#include <cstddef>
#include <string>

class Simulation1D {
public:
    Simulation1D();
    ~Simulation1D();

    void initialize();
    void runStep();
    void simulate(size_t steps);
    void renderImageToFile();
    std::string getImageFilename(unsigned int step, unsigned int stepsPerImage) {
        std::string filename = "output/output_step_";
        if (step < 10) {
            filename += "00000" + std::to_string(step);
        } else if (step < 100) {
            filename += "0000" + std::to_string(step);
        } else if (step < 1000) {
            filename += "000" + std::to_string(step);
        } else if (step < 10000) {
            filename += "00" + std::to_string(step);
        } else if (step < 100000) {
            filename += "0" + std::to_string(step);
        } else {
            filename += std::to_string(step);
        }
        filename += ".png";
        return filename;
    }

private:
    size_t length; // Length of the 1D simulation
    float* p_curr;
    float* p_next;
    float* p_prev;
    float* lap;
    float dx; // Spatial step size
    float dt; // Time step size
    float c; // Wave speed
    size_t simulation_step;
    size_t image_step;
    float lambda;
    float zeta;

    //Image generation
    unsigned int width;
    unsigned int height;
    unsigned char* framebuffer;

    void drawLine(unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2, unsigned char r, unsigned char g, unsigned char b);
};