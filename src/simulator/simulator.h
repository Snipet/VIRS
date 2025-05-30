#pragma once
#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>
#include <string>
#include <tiny_obj_loader.h>
#include <memory>
#include "vectorspace.h"
#include <cmath>
#include <iostream>
#include <cstring>

// Foo comment

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
    void renderImageToFile(Vec3f cameraPos, const std::string& output_path, bool useGrid = false);
    void renderImageToMemory(Vec3f cameraPos, unsigned char** out, size_t* out_size);
    void render(Vec3f cameraPos, bool useGrid = false);
    void renderAnimation(std::string path, int frames, float radius);
    void doSimulationStep();
    void simulate(size_t steps);
    size_t getGridIdxFromVecPos(const Vec3f& pos) const {
        Vec3f voxel = toVoxel(pos);
        return static_cast<size_t>(voxel.x) + static_cast<size_t>(voxel.y) * vector_space->getGrid().Nx + static_cast<size_t>(voxel.z) * vector_space->getGrid().Nx * vector_space->getGrid().Ny;
    }
    void setOutputLayer(size_t layer) {
        output_layer = layer;
    }
    inline Vec3f toVoxel(const Vec3f& pos) const {
        return {
            (pos.x - bounding_box.min.x) / vector_box_size,
            (pos.y - bounding_box.min.y) / vector_box_size,
            (pos.z - bounding_box.min.z) / vector_box_size
        };
    }

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
    unsigned int simulation_step;
    size_t output_layer;
};
