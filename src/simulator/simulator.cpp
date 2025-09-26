#include "simulator.h"
#include <iostream>
#include "png.h"
#include "../util/image.h"
#include <chrono>
#include "tbb/parallel_for.h"
#include "simulator_dispatch.h"
#include "normals.h"
#include "../util/logger.h"
#include "biquad.h"
#include "kfr/include/kfr/dsp.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_AUDIO_FREQ 22050.f
#define SIMULATION_OVERSAMPLING 2

using namespace kfr;

static Vec3f normalize(const Vec3f &v)
{
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return {v.x / len, v.y / len, v.z / len};
}

static Vec3f cross(const Vec3f &a, const Vec3f &b)
{
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

static float dot(const Vec3f &a, const Vec3f &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec3f abs(const Vec3f &v)
{
    return {std::fabs(v.x), std::fabs(v.y), std::fabs(v.z)};
}

static inline bool triangleBoxOverlap3(const Vec3f &v0w,
                                       const Vec3f &v1w,
                                       const Vec3f &v2w,
                                       const Vec3f &boxMin,
                                       const Vec3f &boxMax)
{
    /* 0. translate triangle so that box centre is at origin */
    Vec3f c = {(boxMin.x + boxMax.x) * 0.5f,
               (boxMin.y + boxMax.y) * 0.5f,
               (boxMin.z + boxMax.z) * 0.5f};

    Vec3f e = {(boxMax.x - boxMin.x) * 0.5f,
               (boxMax.y - boxMin.y) * 0.5f,
               (boxMax.z - boxMin.z) * 0.5f};

    Vec3f v0 = v0w - c;
    Vec3f v1 = v1w - c;
    Vec3f v2 = v2w - c;

    /* 1. edge vectors */
    Vec3f f0 = v1 - v0;
    Vec3f f1 = v2 - v1;
    Vec3f f2 = v0 - v2;

    Vec3f a0 = abs(f0);
    Vec3f a1 = abs(f1);
    Vec3f a2 = abs(f2);

    auto axisTest = [&](float a, float b, float fa, float fb,
                        const Vec3f &p0, const Vec3f &p1, const Vec3f &p2,
                        float ex, float ey) -> bool
    {
        float p0proj = a * p0.y - b * p0.z;
        float p1proj = a * p1.y - b * p1.z;
        float p2proj = a * p2.y - b * p2.z;

        float minP = std::min({p0proj, p1proj, p2proj});
        float maxP = std::max({p0proj, p1proj, p2proj});

        float rad = fa * ey + fb * ex;
        return !(minP > rad || maxP < -rad);
    };

    /* 1.1  edge f0 cross coordinate axes */
    if (!axisTest(f0.z, f0.y, a0.z, a0.y, v0, v1, v2, e.y, e.z))
        return false;
    if (!axisTest(f0.z, f0.x, a0.z, a0.x, v0, v1, v2, e.x, e.z))
        return false;
    if (!axisTest(f0.y, f0.x, a0.y, a0.x, v0, v1, v2, e.x, e.y))
        return false;

    /* 1.2  edge f1 */
    if (!axisTest(f1.z, f1.y, a1.z, a1.y, v0, v1, v2, e.y, e.z))
        return false;
    if (!axisTest(f1.z, f1.x, a1.z, a1.x, v0, v1, v2, e.x, e.z))
        return false;
    if (!axisTest(f1.y, f1.x, a1.y, a1.x, v0, v1, v2, e.x, e.y))
        return false;

    /* 1.3  edge f2 */
    if (!axisTest(f2.z, f2.y, a2.z, a2.y, v0, v1, v2, e.y, e.z))
        return false;
    if (!axisTest(f2.z, f2.x, a2.z, a2.x, v0, v1, v2, e.x, e.z))
        return false;
    if (!axisTest(f2.y, f2.x, a2.y, a2.x, v0, v1, v2, e.x, e.y))
        return false;

    /* 2. test overlap in the (x,y,z) axes — box face planes */
    auto min3 = [](float a, float b, float c)
    { return std::min(a, std::min(b, c)); };
    auto max3 = [](float a, float b, float c)
    { return std::max(a, std::max(b, c)); };

    float min = min3(v0.x, v1.x, v2.x);
    float max = max3(v0.x, v1.x, v2.x);
    if (min > e.x || max < -e.x)
        return false;

    min = min3(v0.y, v1.y, v2.y);
    max = max3(v0.y, v1.y, v2.y);
    if (min > e.y || max < -e.y)
        return false;

    min = min3(v0.z, v1.z, v2.z);
    max = max3(v0.z, v1.z, v2.z);
    if (min > e.z || max < -e.z)
        return false;

    /* 3. triangle plane vs box */
    Vec3f n = cross(f0, f1);
    float d = -dot(n, v0);
    float r = e.x * std::fabs(n.x) + e.y * std::fabs(n.y) + e.z * std::fabs(n.z);
    if (std::fabs(d) > r)
        return false;

    return true; // no separating axis found
}

Simulator::Simulator(unsigned int w, unsigned int h)
{
    // Initialize the Simulator
    width = w;
    height = h;
    simulation_step = 0;
    output_layer = 100;
    framebuffer = new unsigned char[width * height * 3]; // RGB framebuffer
    if (!framebuffer)
    {
        std::cerr << "Failed to allocate memory for Simulator framebuffer" << std::endl;
        return;
    }

    // Initialize the Embree device
    device = rtcNewDevice(nullptr);
    if (!device)
    {
        std::cerr << "Failed to create Embree device" << std::endl;
    }
}

bool Simulator::loadConfig(const std::string &config, bool forSimulation)
{
    // Load configuration from a JSON file
    std::ifstream config_file(config);
    if (!config_file.is_open())
    {
        std::cerr << "Failed to open configuration file: " << config << std::endl;
        return false;
    }

    nlohmann::json j;
    config_file >> j;

    std::cout << "Reading configuration from: " << config << std::endl;

    // Check if the files subkey exists
    if (!j.contains("files"))
    {
        std::cerr << "Configuration file does not contain 'files' key." << std::endl;
        return false;
    }

    // Read meshfile
    std::string meshfilePath = j["files"].value("meshfile", "");
    if (meshfilePath.empty())
    {
        std::cerr << "Configuration file does not contain 'meshfile' key." << std::endl;
        return false;
    }

    
    // Read audio file path
    std::string audioPath = j["files"].value("audiofile", "");
    if (audioPath.empty())
    {
        std::cerr << "Configuration file does not contain 'audiofile' key." << std::endl;
        return false;
    }

    // Read simulation parameters
    if(forSimulation){
        if (!j.contains("simulation"))
        {
            std::cerr << "Configuration file does not contain 'simulation' key." << std::endl;
            return false;
        }

        // Read num simulation steps
        if (!j["simulation"].contains("num_simulation_steps"))
        {
            std::cerr << "Configuration file does not contain 'num_simulation_steps' key." << std::endl;
            return false;
        }
        num_simulation_steps = j["simulation"]["num_simulation_steps"].get<size_t>();
        std::cout << "Number of simulation steps: " << num_simulation_steps << std::endl;

        // Read should save layer images
        should_save_layer_images = false;
        if (j["simulation"].contains("should_save_layer_images"))
        {
            should_save_layer_images = j["simulation"]["should_save_layer_images"].get<bool>();
            // Read image save interval
            if (!j["simulation"].contains("image_save_interval"))
            {
                std::cerr << "Configuration file does not contain 'image_save_interval' key. Defaulting to 10." << std::endl;
            }
            image_save_interval = j["simulation"].value("image_save_interval", 10);
            std::cout << "Image save interval: " << image_save_interval << std::endl;

            // Read output images directory
            if (!j["simulation"].contains("output_images_dir"))
            {
                std::cerr << "Configuration file does not contain 'output_images_dir' key." << std::endl;
                return false;
            }
            else
            {
                output_images_dir = j["simulation"]["output_images_dir"].get<std::string>();
            }
        }

        // Read output layer
        if (!j["simulation"].contains("output_layer"))
        {
            std::cerr << "Configuration file does not contain 'output_layer' key. Defaulting to 100." << std::endl;
        }
        output_layer = j["simulation"].value("output_layer", 100);

        // Load the object file
        setAudioSourcePath(audioPath);
    } // if(forSimulation){

    loadObj(meshfilePath, forSimulation);

    return true;
}

void Simulator::loadObj(const std::string &path, bool forSimulation)
{

    scene = rtcNewScene(device);
    object_path = path;

    // Load the object file using TinyOBJLoader
    Logger::getInstance().log("Loading object file: " + object_path, LOGGER_CRITICAL_INFO);
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    std::string material_dir = object_path.substr(0, object_path.find_last_of('/'));

    Logger::getInstance().log("Material directory: " + material_dir, LOGGER_CRITICAL_INFO);

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, object_path.c_str(), material_dir.c_str()))
    {
        std::cerr << warn << err << std::endl;
    }

    std::vector<float> vertices;             // flat array xyzxyz...
    std::vector<unsigned int> indices;       // index triples
    std::vector<unsigned int> materials_ids; // per-face material IDs

    // Determine bounding box;
    bounding_box.min = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    bounding_box.max = {std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min()};

    for (int i = 0; i < attrib.vertices.size(); i += 3)
    {
        bounding_box.min.x = std::min(bounding_box.min.x, attrib.vertices[i]);
        bounding_box.min.y = std::min(bounding_box.min.y, attrib.vertices[i + 1]);
        bounding_box.min.z = std::min(bounding_box.min.z, attrib.vertices[i + 2]);

        bounding_box.max.x = std::max(bounding_box.max.x, attrib.vertices[i]);
        bounding_box.max.y = std::max(bounding_box.max.y, attrib.vertices[i + 1]);
        bounding_box.max.z = std::max(bounding_box.max.z, attrib.vertices[i + 2]);
    }

    // Nudge the bounding box by 10 cm in each direction to avoid numerical issues
    float nudge = 0.1f; // 10 cm
    bounding_box.min.x -= nudge;
    bounding_box.min.y -= nudge;
    bounding_box.min.z -= nudge;
    bounding_box.max.x += nudge;
    bounding_box.max.y += nudge;
    bounding_box.max.z += nudge;

    // std::cout << "Bounding box: min(" << bounding_box.min.x << ", " << bounding_box.min.y << ", " << bounding_box.min.z
    //           << "), max(" << bounding_box.max.x << ", " << bounding_box.max.y << ", " << bounding_box.max.z << ")" << std::endl;

    Logger::getInstance().log("Bounding box: min(" + std::to_string(bounding_box.min.x) + ", " +
                               std::to_string(bounding_box.min.y) + ", " + std::to_string(bounding_box.min.z) +
                               "), max(" + std::to_string(bounding_box.max.x) + ", " +
                               std::to_string(bounding_box.max.y) + ", " + std::to_string(bounding_box.max.z) + ")",
                               LOGGER_NONCRITIAL_INFO);

    vector_box_size = 343.f / (MAX_AUDIO_FREQ * 2.f * (float)SIMULATION_OVERSAMPLING);
    float spanx = bounding_box.max.x - bounding_box.min.x;
    float spany = bounding_box.max.y - bounding_box.min.y;
    float spanz = bounding_box.max.z - bounding_box.min.z;

    float num_vectors_meter = 1.f / vector_box_size;
    size_t sizex = spanx * num_vectors_meter;
    size_t sizey = spany * num_vectors_meter;
    size_t sizez = spanz * num_vectors_meter;
    std::vector<int> existing_material_ids;
    //std::cout << "Vector box size: " << vector_box_size << "m, size in vectors: (" << sizex << ", " << sizey << ", " << sizez << ")" << std::endl;
    Logger::getInstance().log("Vector box size: " + std::to_string(vector_box_size) + "m, size in vectors: (" +
                               std::to_string(sizex) + ", " + std::to_string(sizey) + ", " + std::to_string(sizez) + ")",
                               LOGGER_NONCRITIAL_INFO);
    size_t memory_usage_bytes = sizex * sizey * sizez * sizeof(float) * 3; // 3 floats per vector

    if(forSimulation){
        vector_space = std::make_unique<VectorSpace>(sizex, sizey, sizez, audio_file, vector_box_size);
        vector_space->getGrid().num_filter_sections = 3;
        vector_space->getGrid().allocated_filter_memory = false;
        fdtd_setup(vector_space.get());
    }

    for (const auto &shape : shapes)
    {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f)
        {
            if (shape.mesh.num_face_vertices[f] != 3)
            {
                // skip non‑triangular faces
                index_offset += shape.mesh.num_face_vertices[f];
                continue;
            }

            for (int v = 0; v < 3; ++v)
            {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                vertices.push_back(attrib.vertices[3 * idx.vertex_index + 0]);
                vertices.push_back(attrib.vertices[3 * idx.vertex_index + 1]);
                vertices.push_back(attrib.vertices[3 * idx.vertex_index + 2]);
                indices.push_back(static_cast<unsigned int>(indices.size()));
            }
            // Store material ID for the face
            if (f < shape.mesh.material_ids.size())
            {
                materials_ids.push_back(shape.mesh.material_ids[f]);
                if (std::find(existing_material_ids.begin(), existing_material_ids.end(), shape.mesh.material_ids[f]) == existing_material_ids.end())
                {
                    existing_material_ids.push_back(shape.mesh.material_ids[f]);
                }
            }
            else
            {
                materials_ids.push_back(-1); // Default material ID if not specified
            }
            index_offset += 3;
        }
    }

    if(forSimulation){
        // Compute cell materials based on the imported mesh
        Grid &grid = vector_space->getGrid();

        //std::cout << "Found " << existing_material_ids.size() << " unique materials." << std::endl;
        Logger::getInstance().log("Found " + std::to_string(existing_material_ids.size()) + " unique materials.", LOGGER_NONCRITIAL_INFO);

        //std::cout << "Material ids:";
        Logger::getInstance().log("Material ids:", LOGGER_NONCRITIAL_INFO);
        for (const auto &id : existing_material_ids)
        {
            //std::cout << " " << id;
            Logger::getInstance().log("  -" + std::to_string(id), LOGGER_NONCRITIAL_INFO);
        }

        // Iterate over every triangle and compute the material for each cell
        for (size_t s = 0; s < indices.size(); s += 3)
        {
            unsigned int idx0 = indices[s];
            unsigned int idx1 = indices[s + 1];
            unsigned int idx2 = indices[s + 2];

            Vec3f v0{vertices[idx0 * 3], vertices[idx0 * 3 + 1], vertices[idx0 * 3 + 2]};
            Vec3f v1{vertices[idx1 * 3], vertices[idx1 * 3 + 1], vertices[idx1 * 3 + 2]};
            Vec3f v2{vertices[idx2 * 3], vertices[idx2 * 3 + 1], vertices[idx2 * 3 + 2]};

            int material_id = materials_ids[s / 3]; // Get the material ID for the triangle

            Vec3f gv0 = toVoxel(v0);
            Vec3f gv1 = toVoxel(v1);
            Vec3f gv2 = toVoxel(v2);

            // Compute AABB for the triangle
            int ixMin = std::clamp(int(std::floor(std::min({gv0.x, gv1.x, gv2.x}))), 0, int(grid.Nx) - 1);
            int iyMin = std::clamp(int(std::floor(std::min({gv0.y, gv1.y, gv2.y}))), 0, int(grid.Ny) - 1);
            int izMin = std::clamp(int(std::floor(std::min({gv0.z, gv1.z, gv2.z}))), 0, int(grid.Nz) - 1);
            int ixMax = std::clamp(int(std::ceil(std::max({gv0.x, gv1.x, gv2.x}))), 0, int(grid.Nx) - 1);
            int iyMax = std::clamp(int(std::ceil(std::max({gv0.y, gv1.y, gv2.y}))), 0, int(grid.Ny) - 1);
            int izMax = std::clamp(int(std::ceil(std::max({gv0.z, gv1.z, gv2.z}))), 0, int(grid.Nz) - 1);

            // std::cout << "Triangle AABB: (" << ixMin << ", " << iyMin << ", " << izMin
            //           << ") to (" << ixMax << ", " << iyMax << ", " << izMax << ")" << std::endl;
            // std::cout << "Material id: " << material_id << std::endl;

            // Sweep through box
            for (int k = izMin; k <= izMax; ++k)
                for (int j = iyMin; j <= iyMax; ++j)
                    for (int i = ixMin; i <= ixMax; ++i)
                    {

                        Vec3f boxMin(i, j, k);
                        Vec3f boxMax(i + 1.f, j + 1.f, k + 1.f);

                        // Check if the triangle intersects the box
                        if (triangleBoxOverlap3(gv0, gv1, gv2, boxMin, boxMax))
                        {
                            size_t idx = grid.idx(i, j, k);
                            // Set the material for the grid cell
                            if (material_id == 0)
                            {
                                // Wall
                                grid.flags[idx] = 1;
                                //grid.p_absorb[idx] = 0.f;
                            }
                            else if (material_id == 1)
                            {
                                // Speaker
                                grid.flags[idx] = 3;
                            }
                        }
                    }
        }

        //std::cout << "Material data computed for the grid." << std::endl;
        Logger::getInstance().log("Material data computed for the grid.", LOGGER_NONCRITIAL_INFO);
        //std::cout << "Computing absorption material..." << std::endl;
        Logger::getInstance().log("Computing absorption material...", LOGGER_NONCRITIAL_INFO);

        uint8_t *tmp_flags = new uint8_t[grid.size];
        std::memcpy(tmp_flags, grid.flags, grid.size * sizeof(uint8_t));

        //float *tmp_p_absorb = new float[grid.size];
        //std::memcpy(tmp_p_absorb, grid.p_absorb, grid.size * sizeof(float));

        const int absorptionWidth = 3;
        const float absorptionEpsilon = 1.f / static_cast<float>(absorptionWidth);

        // Compute absorption material
        buildSpongeLayer(vector_space.get());
        //std::cout << "Absorption material computed." << std::endl;
        Logger::getInstance().log("Absorption material computed.", LOGGER_NONCRITIAL_INFO);
        grid.boundary_indices_size = 0;

        updateGPUFromGrid(vector_space.get()); // Update the GPU grid with the new material data

        // Calculate normals
        //std::cout << "Calculating normals..." << std::endl;
        Logger::getInstance().log("Calculating normals...", LOGGER_NONCRITIAL_INFO);
        int max_neighbors = 0;
        for(int k = 1; k < grid.Nz - 1; ++k)
        {
            for(int j = 1; j < grid.Ny - 1; ++j)
            {
                for(int i = 1; i < grid.Nx - 1; ++i)
                {
                    size_t idx = grid.idx(i, j, k);
                    if(true){
                        uint8_t current_normal_code = static_cast<uint8_t>(ENormals::kNone);


                        int neighbor_count = 0;
                        //Check -x neighbor
                        if(i - 1 >= 0 && grid.flags[grid.idx(i - 1, j, k)] == 0){
                            current_normal_code |= static_cast<uint8_t>(ENormals::kXNegative);
                            neighbor_count++;
                        }

                        //Check +x neighbor
                        if(i + 1 < grid.Nx && grid.flags[grid.idx(i + 1, j, k)] == 0){
                            current_normal_code |= static_cast<uint8_t>(ENormals::kXPositive);
                            neighbor_count++;
                        }

                        //Check -y neighbor
                        if(j - 1 >= 0 && grid.flags[grid.idx(i, j - 1, k)] == 0){
                            current_normal_code |= static_cast<uint8_t>(ENormals::kYNegative);
                            neighbor_count++;
                        }

                        //Check +y neighbor
                        if(j + 1 < grid.Ny && grid.flags[grid.idx(i, j + 1, k)] == 0){
                            current_normal_code |= static_cast<uint8_t>(ENormals::kYPositive);
                            neighbor_count++;
                        }

                        //Check -z neighbor
                        if(k - 1 >= 0 && grid.flags[grid.idx(i, j, k - 1)] == 0){
                            current_normal_code |= static_cast<uint8_t>(ENormals::kZNegative);
                            neighbor_count++;
                        }

                        //Check +z neighbor
                        if(k + 1 < grid.Nz && grid.flags[grid.idx(i, j, k + 1)] == 0){
                            current_normal_code |= static_cast<uint8_t>(ENormals::kZPositive);
                            neighbor_count++;
                        }

                        if(neighbor_count > max_neighbors){
                            max_neighbors = neighbor_count;
                        }
                        grid.normals[idx] = current_normal_code;

                        // // Check for any normal errors
                        // if(grid.normals[idx] == static_cast<uint8_t>(ENormals::kNone))
                        // {
                        //     //std::cout << "Warning: No normals found for bounddary cell at (" << i << ", " << j << ", " << k << ")." << std::endl;
                        //     Logger::getInstance().log("Warning: No normals found for boundary cell at (" + std::to_string(i) + ", " +
                        //                                std::to_string(j) + ", " + std::to_string(k) + ").", LOGGER_WARNING);
                        // }

                        if ((grid.normals[idx] & static_cast<uint8_t>(ENormals::kXPositive)) &&
                            (grid.normals[idx] & static_cast<uint8_t>(ENormals::kXNegative))) {
                            //std::cout << "Warning: Both X normals found for boundary cell at (" << i << ", " << j << ", " << k << ")." << std::endl;
                           // Logger::getInstance().log("Warning: Both X normals found for boundary cell at (" + std::to_string(i) + ", " +
                             //                          std::to_string(j) + ", " + std::to_string(k) + ").", LOGGER_WARNING);
                        }

                        if ((grid.normals[idx] & static_cast<uint8_t>(ENormals::kYPositive)) &&
                            (grid.normals[idx] & static_cast<uint8_t>(ENormals::kYNegative))) {
                            //std::cout << "Warning: Both Y normals found for boundary cell at (" << i << ", " << j << ", " << k << ")." << std::endl;
                           // Logger::getInstance().log("Warning: Both Y normals found for boundary cell at (" + std::to_string(i) + ", " +
                             //                          std::to_string(j) + ", " + std::to_string(k) + ").", LOGGER_WARNING);
                        }

                        if ((grid.normals[idx] & static_cast<uint8_t>(ENormals::kZPositive)) &&
                            (grid.normals[idx] & static_cast<uint8_t>(ENormals::kZNegative))) {
                            //std::cout << "Warning: Both Z normals found for boundary cell at (" << i << ", " << j << ", " << k << ")." << std::endl;
                            //Logger::getInstance().log("Warning: Both Z normals found for boundary cell at (" + std::to_string(i) + ", " +
                              //                         std::to_string(j) + ", " + std::to_string(k) + ").", LOGGER_WARNING);
                        }
                    }
                }
            }
        }
        //std::cout << "Normals calculated." << std::endl;

        std::cout << "Max neighbors for a boundary cell: " << max_neighbors << std::endl;
        Logger::getInstance().log("Normals calculated.", LOGGER_NONCRITIAL_INFO);
        uploadNormalsToGPU(vector_space.get()); // Upload normals to GPU

        // Count number of boundary cells for boundary indices
        uint32_t boundary_count = 0;
        for(int k = 1; k < grid.Nz - 1; ++k)
        {
            for(int j = 1; j < grid.Ny - 1; ++j)
            {
                for(int i = 1; i < grid.Nx - 1; ++i)
                {
                    size_t idx = grid.idx(i, j, k);
                    if(grid.flags[idx] == 2){
                        boundary_count++;
                    }
                }
            }
        }

        grid.boundary_indices_size = boundary_count;

        // set boundary indices
        grid.boundary_indices = new uint32_t[grid.boundary_indices_size];
        size_t boundary_index = 0;

        for(int k = 1; k < grid.Nz - 1; ++k)
        {
            for(int j = 1; j < grid.Ny - 1; ++j)
            {
                for(int i = 1; i < grid.Nx - 1; ++i)
                {
                    size_t idx = grid.idx(i, j, k);
                    if(grid.flags[idx] == 2){
                        grid.boundary_indices[boundary_index++] = idx;
                    }
                }
            }
        }

        //float target_Rp_magnitude = 0.93f;
        //float target_zeta = (1.f + target_Rp_magnitude) / (1.f - target_Rp_magnitude);

        allocFilterStates(vector_space.get());

        //std::cout << "Calculating pZeta..." << std::endl;
        // Logger::getInstance().log("Calculating pZeta...", LOGGER_NONCRITIAL_INFO);
        //     for(int k = 1; k < grid.Nz - 1; ++k)
        // {
        //     for(int j = 1; j < grid.Ny - 1; ++j)
        //     {
        //         for(int i = 1; i < grid.Nx - 1; ++i)
        //         {
        //             size_t idx = grid.idx(i, j, k);
        //             if(grid.flags[idx] == 2){
        //                 grid.pZeta[idx] = target_zeta;
                    

        //             }else{
        //                 grid.pZeta[idx] = 0.f; // No pZeta for non-boundary cells
        //             }
        //         }
        //     }
        // }

        //std::cout << "pZeta calculated." << std::endl;
        //Logger::getInstance().log("pZeta calculated.", LOGGER_NONCRITIAL_INFO);
        //uploadPZetaToGPU(vector_space.get());
        uploadBoundaryIndicesToGPU(vector_space.get());
    }

    const size_t numVerts = vertices.size() / 3;
    const size_t numTriangles = indices.size() / 3;
    //std::cout << "Loaded " << numVerts << " vertices and " << numTriangles << " triangles." << std::endl;
    Logger::getInstance().log("Loaded " + std::to_string(numVerts) + " vertices and " + std::to_string(numTriangles) + " triangles.", LOGGER_CRITICAL_INFO);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    float *vb = static_cast<float *>(rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(float) * 3, numVerts));
    std::memcpy(vb, vertices.data(), vertices.size() * sizeof(float));

    unsigned int *ib = static_cast<unsigned int *>(rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(unsigned int) * 3, numTriangles));
    std::memcpy(ib, indices.data(), indices.size() * sizeof(unsigned int));

    //std::cout << "Geometry created." << std::endl;
    Logger::getInstance().log("Geometry created.", LOGGER_NONCRITIAL_INFO);
    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcCommitScene(scene);

    //std::cout << "Scene committed." << std::endl;
    Logger::getInstance().log("Scene committed.", LOGGER_NONCRITIAL_INFO);
}

void Simulator::renderAnimation(std::string outPath, int frames, float radius)
{
    if (!device || !scene)
    {
        //std::cerr << "Device or scene not initialized." << std::endl;
        Logger::getInstance().log("Device or scene not initialized.", LOGGER_ERROR);
        return;
    }
    if (!framebuffer)
    {
        //std::cerr << "framebuffer not initialized." << std::endl;
        Logger::getInstance().log("Framebuffer not initialized.", LOGGER_ERROR);
        return;
    }
    //std::cout << "Rendering the scene..." << std::endl;
    Logger::getInstance().log("Rendering the scene...", LOGGER_CRITICAL_INFO);

    std::string filename = "output";
    for (int i = 0; i < frames; i++)
    {
        // Format the filename with a number with 3 digits
        std::string num = std::to_string(i);
        if (num.length() == 1)
        {
            num = "00" + num;
        }
        else if (num.length() == 2)
        {
            num = "0" + num;
        }

        std::string full_path = outPath + filename + num + ".png";
        float phase = static_cast<float>(i) / static_cast<float>(frames) * 2.0f * M_PI;
        float x = std::sin(phase) * radius;
        float z = std::cos(phase) * radius;
        float y = 6.0f;
        renderImageToFile({x, y, z}, full_path);
        //std::cout << "Rendering to: " << full_path << std::endl;
        Logger::getInstance().log("Rendering to: " + full_path, LOGGER_CRITICAL_INFO);
    }
}

void Simulator::renderImageToFile(Vec3f cameraPos, const std::string &output_path, bool useGrid)
{
    render(cameraPos, useGrid);
    // Save the image to a file
    //std::cout << "Saving image to: " << output_path << std::endl;
    Logger::getInstance().log("Saving image to: " + output_path, LOGGER_SUCCESS);
    save_png(output_path.c_str(), framebuffer, width, height, false);
}

void Simulator::renderImageToMemory(Vec3f cameraPos, unsigned char **out, size_t *out_size)
{
    render(cameraPos);
    save_png_to_memory(framebuffer, width, height, width * 3, out, out_size);
}

void Simulator::render(Vec3f cameraPos, bool useGrid)
{

    const float fov = 50.0f * M_PI / 180.0f;
    Vec3f eye = cameraPos;
    Vec3f look{0.0f, 2.0f, 0.0f};
    Vec3f dir;
    dir.x = look.x - eye.x;
    dir.y = look.y - eye.y;
    dir.z = look.z - eye.z;
    dir = normalize(dir);

    Vec3f up{0.f, 1.f, 0.f};
    Vec3f right = normalize(cross(dir, up));
    up = normalize(cross(right, dir));
    up.x *= std::tan(fov / 2);
    up.y *= std::tan(fov / 2);
    up.z *= std::tan(fov / 2);
    right.x *= std::tan(fov / 2);
    right.y *= std::tan(fov / 2);
    right.z *= std::tan(fov / 2);

    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float scale = 1.f;

    // std::cout << "Camera position: (" << eye.x << ", " << eye.y << ", " << eye.z << ")" << std::endl;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float u = (2.0f * (x + 0.5f) / width - 1.0f) * scale * aspect;
            float v = (1.0f - 2.0f * (y + 0.5f) / height) * scale;

            // Compute ray direction
            Vec3f rayDir;

            rayDir.x = dir.x + u * right.x + v * up.x;
            rayDir.y = dir.y + u * right.y + v * up.y;
            rayDir.z = dir.z + u * right.z + v * up.z;
            rayDir = normalize(rayDir);

            // Vec3f rayDir = normalize({ -1,0,0});

            struct RTCRayHit rayHit;
            rayHit.ray.org_x = eye.x;
            rayHit.ray.org_y = eye.y;
            rayHit.ray.org_z = eye.z;
            rayHit.ray.dir_x = rayDir.x;
            rayHit.ray.dir_y = rayDir.y;
            rayHit.ray.dir_z = rayDir.z;
            rayHit.ray.tnear = 0.0f;
            rayHit.ray.tfar = std::numeric_limits<float>::infinity();
            rayHit.ray.mask = -1;
            rayHit.ray.flags = 0;
            rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
            bool cont = true;
            int numBounces = 0;
            while (cont)
            {

                rtcIntersect1(scene, &rayHit);
                RTCRayHit16 rayHit16;
                if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
                {
                    // No hit, set background color
                    framebuffer[(y * width + x) * 3 + 0] = 100;
                    framebuffer[(y * width + x) * 3 + 1] = 0;
                    framebuffer[(y * width + x) * 3 + 2] = 0;
                    cont = false;
                }
                else
                {
                    float cameraNormalDot = rayDir.x * rayHit.hit.Ng_x + rayDir.y * rayHit.hit.Ng_y + rayDir.z * rayHit.hit.Ng_z;
                    if (cameraNormalDot > 0 || numBounces > 1)
                    {
                        Vec3f n;
                        n.x = rayHit.hit.Ng_x;
                        n.y = rayHit.hit.Ng_y;
                        n.z = rayHit.hit.Ng_z;
                        n = normalize(n);
                        Vec3f lightDir = rayDir;
                        float shade = 0.7 * std::max(dot(n, lightDir), 0.f) + 0.3f;
                        //float shade = abs(dot(rayDir * -1.f, n));
                        Vec3f shadeColor = {shade, shade, shade};

                        Vec3f rayPos = {rayHit.ray.org_x + rayDir.x * rayHit.ray.tfar,
                                        rayHit.ray.org_y + rayDir.y * rayHit.ray.tfar,
                                        rayHit.ray.org_z + rayDir.z * rayHit.ray.tfar};

                        // Nudge the ray along the normal
                        const float epsilon = vector_box_size; // Amount to "nudge"
                        rayPos.x -= n.x * epsilon;
                        rayPos.y -= n.y * epsilon;
                        rayPos.z -= n.z * epsilon;
                        size_t idx;
                        Vec3f pressureColor = {0.0f, 0.0f, 0.0f};
                        if (useGrid && (idx = getGridIdxFromVecPos(rayPos)) < vector_space->getGrid().size)
                        {
                            float pressure = (vector_space->getGrid().p_curr[idx]);
                            if (pressure > 0.0f)
                            {
                                // Map pressure to color
                                pressureColor.x = 1.f;
                            }
                            else
                            {
                                pressureColor.z = 1.f;
                            }
                            float interpolateVal = std::sqrt(std::abs(pressure * 3.f));
                            interpolateVal = std::min(interpolateVal, 1.f);
                            interpolateVal = std::max(interpolateVal, 0.f);
                            shadeColor.x = pressureColor.x * interpolateVal + shadeColor.x * (1.f - interpolateVal);
                            shadeColor.y = pressureColor.y * interpolateVal + shadeColor.y * (1.f - interpolateVal);
                            shadeColor.z = pressureColor.z * interpolateVal + shadeColor.z * (1.f - interpolateVal);
                        }

                        // Hit detected, color the pixel based on the hit normal
                        unsigned char r = static_cast<unsigned char>(shadeColor.x * 255);
                        unsigned char g = static_cast<unsigned char>(shadeColor.y * 255);
                        unsigned char b = static_cast<unsigned char>(shadeColor.z * 255);

                        r = std::max(r, (unsigned char)0);
                        g = std::max(g, (unsigned char)0);
                        b = std::max(b, (unsigned char)0);
                        r = std::min(r, (unsigned char)255);
                        g = std::min(g, (unsigned char)255);
                        b = std::min(b, (unsigned char)255);
                        framebuffer[(y * width + x) * 3 + 0] = r;
                        framebuffer[(y * width + x) * 3 + 1] = g;
                        framebuffer[(y * width + x) * 3 + 2] = b;
                        cont = false; // We have hit a primitive whose normal faces away from the camera
                    }
                    else
                    {

                        const float epsilon = 0.001f; // Amount to "nudge" the ray to prevent re-intersection
                        Vec3f hitPoint;
                        hitPoint.x = rayHit.ray.org_x + rayDir.x * rayHit.ray.tfar;
                        hitPoint.y = rayHit.ray.org_y + rayDir.y * rayHit.ray.tfar;
                        hitPoint.z = rayHit.ray.org_z + rayDir.z * rayHit.ray.tfar;
                        Vec3f normal;
                        normal.x = rayHit.hit.Ng_x;
                        normal.y = rayHit.hit.Ng_y;
                        normal.z = rayHit.hit.Ng_z;
                        normal = normalize(normal);
                        Vec3f offsetOrigin;
                        offsetOrigin.x = hitPoint.x - normal.x * epsilon;
                        offsetOrigin.y = hitPoint.y - normal.y * epsilon;
                        offsetOrigin.z = hitPoint.z - normal.z * epsilon;
                        rayHit.ray.org_x = offsetOrigin.x;
                        rayHit.ray.org_y = offsetOrigin.y;
                        rayHit.ray.org_z = offsetOrigin.z;
                        rayHit.ray.tnear = 0.0f;
                        rayHit.ray.tfar = std::numeric_limits<float>::infinity();
                    }
                }
                numBounces++;
                // if (numBounces > 10)
                // {
                //     cont = false;
                // }
            }
        }
    }
}

Simulator::~Simulator()
{
    if (framebuffer)
    {
        delete[] framebuffer;
        framebuffer = nullptr;
    }

    // Release the Embree device
    if (device)
    {
        rtcReleaseDevice(device);
        device = nullptr;
    }

    // Release the Embree scene
    if (scene)
    {
        rtcReleaseScene(scene);
        scene = nullptr;
    }
    if(vector_space){
        fdtd_cleanup(vector_space.get());
    }
}

std::string Simulator::toString()
{
    std::string result = "Simulator: \n";
    result += "Width: " + std::to_string(width) + "\n";
    result += "Height: " + std::to_string(height) + "\n";
    return result;
}

bool Simulator::doSimulationStep()
{
    std::string simulation_step_duration = vector_space->stopwatch();
    if(simulation_step % 50 == 0){
    	std::cout << "\rPerforming simulation step " << simulation_step << "... Last step took: " << simulation_step_duration << "; percent to RMS:" << vector_space->getGrid().percent_to_target_RMS * 100.f << std::flush;
    }
    // vector_space->computePressureStage();
    bool shouldContinue = fdtd_step(vector_space.get(), simulation_step);
    simulation_step++;
    return shouldContinue;
}

void Simulator::simulate()
{
    simulation_step = 0;
    Grid &grid = vector_space->getGrid();
    size_t centerx = grid.Nx / 2;
    size_t centery = grid.Ny / 2;
    size_t centerz = grid.Nz / 2;
    double c = 343.0;           // Speed of sound in m/s
    double h = vector_box_size; // Size of the grid cell in meters
    double dt = 0.5 * h / (c * std::sqrt(3.0));
    double c2_dt2 = c * c * dt * dt;
    double gdt = 5.0 * dt;
    double inv_h2 = 1.0 / (h * h);
    //std::cout << "Simulation parameters: c = " << c << " m/s, h = " << h << " m, dt = " << dt << " s, c2_dt2 = " << c2_dt2 << ", gdt = " << gdt << ", inv_h2 = " << inv_h2 << std::endl;
    Logger::getInstance().log("Simulation parameters: c = " + std::to_string(c) + " m/s, h = " + std::to_string(h) +
                               " m, dt = " + std::to_string(dt) + " s, c2_dt2 = " + std::to_string(c2_dt2) +
                               ", gdt = " + std::to_string(gdt) + ", inv_h2 = " + std::to_string(inv_h2),
                               LOGGER_NONCRITIAL_INFO);
    
    const double simulation_fs = 1.f / dt;
    std::cout << "Simulation sample rate: " << simulation_fs << std::endl;
    
    allocFilterCoeffs(vector_space.get(), 1);

    BiquadCoeffs coeffs = computeHighpassBiquad(simulation_fs, 440.f, 0.1f, 1.f);
    grid.biquad_coeffs[0] = coeffs.b0;
    grid.biquad_coeffs[1] = coeffs.b1;
    grid.biquad_coeffs[2] = coeffs.b2;
    grid.biquad_coeffs[3] = coeffs.a1;
    grid.biquad_coeffs[4] = coeffs.a2;

    std::cout << "Filter coefficients: b0 = " << coeffs.b0 << ", b1 = " << coeffs.b1
              << ", b2 = " << coeffs.b2 << ", a1 = " << coeffs.a1 << ", a2 = " << coeffs.a2 << std::endl;

    uploadFilterCoeffsToGPU(vector_space.get());
    

    auto start = std::chrono::high_resolution_clock::now();
    //std::cout << "Starting simulation with " << num_simulation_steps << " steps..." << std::endl;
    Logger::getInstance().log("Starting simulation with " + std::to_string(num_simulation_steps) + " steps...", LOGGER_CRITICAL_INFO);
    fdtd_start_simulation(vector_space.get(), num_simulation_steps);
    // initPressureSphere(vector_space.get(), centerx, centery, centerz, 20, 1.f, true); // Initialize a pressure sphere in the center of the grid
    updateCurrentGridFromGPU(vector_space.get());
    if (true)
    {
        vector_space->layerToImage("output_layer.png", output_layer);
    }
    unsigned int image_step = image_save_interval;
    for (size_t i = 0; i < num_simulation_steps; ++i)
    {

        bool shouldContinue = doSimulationStep();
        if (should_save_layer_images)
        {
            std::string file_out = output_images_dir + "/output_step_";

            if (i % image_step == 0)
            {
                int frame = i / image_step;
                if (frame < 10)
                {
                    file_out += "00000" + std::to_string(frame) + ".png";
                }
                else if (frame < 100)
                {
                    file_out += "0000" + std::to_string(frame) + ".png";
                }
                else if (frame < 1000)
                {
                    file_out += "000" + std::to_string(frame) + ".png";
                }
                else if (frame < 10000)
                {
                    file_out += "00" + std::to_string(frame) + ".png";
                }
                else if (frame < 100000)
                {
                    file_out += "0" + std::to_string(frame) + ".png";
                }
                else
                {
                    file_out += std::to_string(frame) + ".png";
                }
                updateCurrentGridFromGPU(vector_space.get());
                vector_space->layerToImage(file_out, output_layer);
            }
        }
        if(!shouldContinue){
            //std::cout << "RMS threshold reached, stopping simulation." << std::endl;
            Logger::getInstance().log("RMS threshold reached, stopping simulation.", LOGGER_SUCCESS);
            break;
        }
    }
    // initPressureSphere(vector_space.get(), centerx, centery, centerz, 100, 1.f, true);
    updateCurrentGridFromGPU(vector_space.get());

    // We now have the raw pressures from the simulation. However, we must convert them to audio data.
    float *out_audio_data = vector_space->getGrid().p_audio_output;
    AudioFile<float> out_file;
    unsigned int sample_rate = 48000;
    out_file.setSampleRate(sample_rate);
    out_file.setBitDepth(32);
    out_file.setNumChannels(1);
    double length_seconds = (double)num_simulation_steps * dt;
    size_t num_samples = static_cast<size_t>((double)sample_rate * length_seconds);
    out_file.setNumSamplesPerChannel(num_samples);
    //std::cout << "Writing audio file with " << num_samples << " samples, length: " << length_seconds << " seconds." << std::endl;
    Logger::getInstance().log("Writing audio file with " + std::to_string(num_samples) + " samples, length: " + std::to_string(length_seconds) + " seconds.", LOGGER_CRITICAL_INFO);

    double simulation_sample_rate = 1.f / dt;
    // for (int i = 0; i < num_samples; i++)
    // {
    //     float t = (float)i / (float)sample_rate;
    //     float index = t * simulation_sample_rate;
    //     size_t bottom_idx = static_cast<size_t>(std::floor(index));
    //     size_t top_idx = bottom_idx + 1;
    //     float frac = index - (float)bottom_idx;
    //     float bottom_value = out_audio_data[bottom_idx];
    //     float top_value = out_audio_data[top_idx];
    //     float value = bottom_value * (1.f - frac) + top_value * frac; // Linear interpolation
    //     audio_data[0][i] = value;
    // }

    auto r = resampler<float>(resample_quality::high, sample_rate, (size_t)simulation_sample_rate);
    univector<float> xin(out_audio_data, out_audio_data + num_simulation_steps);
    univector<float> xout(num_samples);

    std::vector<std::vector<float>> audio_data;
    audio_data.resize(1); // 1 channel
    audio_data[0].resize(num_samples + r.get_delay(), 0.f); // Initialize with zeros
    r.process(xout, xin);
    for (size_t i = 0; i < num_samples; i++)
    {
        audio_data[0][i] = xout[i];
    }

    float max_amplitude = 0.f;
    for(int i = 0; i < audio_data[0].size(); i++){
	    if(std::abs(audio_data[0][i]) > max_amplitude){
		    max_amplitude = std::abs(audio_data[0][i]);
	    }
    }
    for(int i = 0; i < audio_data[0].size(); i++){
	    audio_data[0][i] /= max_amplitude;
    }
    out_file.setAudioBuffer(audio_data);

    out_file.save("output_audio.wav");
    //std::cout << "Audio data written to output_audio.wav" << std::endl;
    Logger::getInstance().log("Audio data written to output_audio.wav", LOGGER_NONCRITIAL_INFO); 

    // Now p_curr contains the final pressure values, we can render the final image
    // renderImageToFile({7.f, 7.f, 7.f}, "output/output_final.png", true);

    //std::cout << "Simulation completed after " << num_simulation_steps << " steps." << std::endl;
    Logger::getInstance().log("Simulation completed after " + std::to_string(num_simulation_steps) + " steps.", LOGGER_SUCCESS);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //std::cout << "Total simulation time: " << duration << " ms" << std::endl;
    Logger::getInstance().log("Total simulation time: " + std::to_string(duration) + " ms", LOGGER_NONCRITIAL_INFO);
    // renderImageToFile({7.f, 7.f, 7.f}, "output/output_final.png", true);

    if (true)
    {
        vector_space->layerToImage("output_layer_after.png", output_layer);
    }
    // vector_space->gridToImages("output/output_grid_");
}
