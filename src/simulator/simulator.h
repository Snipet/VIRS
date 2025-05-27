#pragma once
#include <rtcore.h>
#include <rtcore_ray.h>
#include <string>
#include <tiny_obj_loader.h>
#include <memory>
#include "vectorspace.h"


struct BoundingBox {
    Vec3f min;
    Vec3f max;
};

class Simulator {
public:
    Simulator(unsigned int width, unsigned int height);
    ~Simulator();
    std::string toString();
    void loadObj(const std::string& path);
    void renderImageToFile(Vec3f cameraPos, const std::string& output_path);
    void renderImageToMemory(Vec3f cameraPos, unsigned char** out, size_t* out_size);
    void render(Vec3f cameraPos);
    void renderAnimation(std::string path, int frames, float radius);
    

private:
    // Private member variables for the Simulator
    int width;
    int height;
    unsigned char* framebuffer;
    RTCDevice device;
    RTCScene scene;
    std::string object_path;
    float* vector_data;
    BoundingBox bounding_box;
    float vector_box_size; //Size of individual vector box
    std::unique_ptr<VectorSpace> vector_space;
};