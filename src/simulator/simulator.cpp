#include "simulator.h"
#include <iostream>
#include "png.h"
#include "../util/image.h"
#include <chrono>
#include "tbb/parallel_for.h"
#include "simulator_dispatch.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_AUDIO_FREQ 22050.f
#define SIMULATION_OVERSAMPLING 1

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

void Simulator::loadObj(const std::string &path)
{

    scene = rtcNewScene(device);
    object_path = path;

    // Load the object file using TinyOBJLoader
    std::cout << "Loading object file: " << object_path << std::endl;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, object_path.c_str()))
    {
        std::cerr << warn << err << std::endl;
    }

    std::vector<float> vertices;       // flat array xyzxyz...
    std::vector<unsigned int> indices; // index triples

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

    std::cout << "Bounding box: min(" << bounding_box.min.x << ", " << bounding_box.min.y << ", " << bounding_box.min.z
              << "), max(" << bounding_box.max.x << ", " << bounding_box.max.y << ", " << bounding_box.max.z << ")" << std::endl;

    vector_box_size = 343.f / (MAX_AUDIO_FREQ * 2.f * (float)SIMULATION_OVERSAMPLING);
    float spanx = bounding_box.max.x - bounding_box.min.x;
    float spany = bounding_box.max.y - bounding_box.min.y;
    float spanz = bounding_box.max.z - bounding_box.min.z;

    float num_vectors_meter = 1.f / vector_box_size;
    size_t sizex = spanx * num_vectors_meter;
    size_t sizey = spany * num_vectors_meter;
    size_t sizez = spanz * num_vectors_meter;

    std::cout << "Vector box size: " << vector_box_size << "m, size in vectors: (" << sizex << ", " << sizey << ", " << sizez << ")" << std::endl;
    size_t memory_usage_bytes = sizex * sizey * sizez * sizeof(float) * 3; // 3 floats per vector

    vector_space = std::make_unique<VectorSpace>(sizex, sizey, sizez, audio_file.getNumSamplesPerChannel(), vector_box_size);
    fdtd_setup(vector_space.get());

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
            index_offset += 3;
        }
    }

    // Compute cell materials based on the imported mesh
    Grid &grid = vector_space->getGrid();

    // Iterate over every triangle and compute the material for each cell
    for (size_t s = 0; s < indices.size(); s += 3)
    {
        unsigned int idx0 = indices[s];
        unsigned int idx1 = indices[s + 1];
        unsigned int idx2 = indices[s + 2];

        Vec3f v0{vertices[idx0 * 3], vertices[idx0 * 3 + 1], vertices[idx0 * 3 + 2]};
        Vec3f v1{vertices[idx1 * 3], vertices[idx1 * 3 + 1], vertices[idx1 * 3 + 2]};
        Vec3f v2{vertices[idx2 * 3], vertices[idx2 * 3 + 1], vertices[idx2 * 3 + 2]};

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

        std::cout << "Triangle AABB: (" << ixMin << ", " << iyMin << ", " << izMin
                  << ") to (" << ixMax << ", " << iyMax << ", " << izMax << ")" << std::endl;

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
                        grid.flags[idx] = 1; // Mark the cell as occupied
                        grid.p_absorb[idx] = 2000.f;
                    }
                }
    }

    std::cout << "Material data computed for the grid." << std::endl;
    std::cout << "Computing absorption material..." << std::endl;

    uint8_t *tmp_flags = new uint8_t[grid.size];
    std::memcpy(tmp_flags, grid.flags, grid.size * sizeof(uint8_t));

    float *tmp_p_absorb = new float[grid.size];
    std::memcpy(tmp_p_absorb, grid.p_absorb, grid.size * sizeof(float));

    const int absorptionWidth = 3;
    const float absorptionEpsilon = 1.f / static_cast<float>(absorptionWidth);

    // Compute absorption material
    buildSpongeLayer(vector_space.get());
    std::cout << "Absorption material computed." << std::endl;

    updateGPUFromGrid(vector_space.get()); // Update the GPU grid with the new material data

    const size_t numVerts = vertices.size() / 3;
    const size_t numTriangles = indices.size() / 3;
    std::cout << "Loaded " << numVerts << " vertices and " << numTriangles << " triangles." << std::endl;
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    float *vb = static_cast<float *>(rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(float) * 3, numVerts));
    std::memcpy(vb, vertices.data(), vertices.size() * sizeof(float));

    unsigned int *ib = static_cast<unsigned int *>(rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(unsigned int) * 3, numTriangles));
    std::memcpy(ib, indices.data(), indices.size() * sizeof(unsigned int));

    std::cout << "Geometry created." << std::endl;
    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    // rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    std::cout << "Scene committed." << std::endl;
}

void Simulator::renderAnimation(std::string outPath, int frames, float radius)
{
    if (!device || !scene)
    {
        std::cerr << "Device or scene not initialized." << std::endl;
        return;
    }
    if (!framebuffer)
    {
        std::cerr << "framebuffer not initialized." << std::endl;
        return;
    }
    std::cout << "Rendering the scene..." << std::endl;

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
        std::cout << "Rendering to: " << full_path << std::endl;
    }
}

void Simulator::renderImageToFile(Vec3f cameraPos, const std::string &output_path, bool useGrid)
{
    render(cameraPos, useGrid);
    // Save the image to a file
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
                    if (cameraNormalDot > 0)
                    {
                        Vec3f n;
                        n.x = rayHit.hit.Ng_x;
                        n.y = rayHit.hit.Ng_y;
                        n.z = rayHit.hit.Ng_z;
                        n = normalize(n);
                        float shade = abs(dot(rayDir * -1.f, n));
                        Vec3f shadeColor = {shade, shade, shade};

                        Vec3f rayPos = {rayHit.ray.org_x + rayDir.x * rayHit.ray.tfar,
                                        rayHit.ray.org_y + rayDir.y * rayHit.ray.tfar,
                                        rayHit.ray.org_z + rayDir.z * rayHit.ray.tfar};

                        // Nudge the ray along the normal
                        const float epsilon = vector_box_size; // Amount to "nudge"
                        rayPos.x -= n.x * epsilon;
                        rayPos.y -= n.y * epsilon;
                        rayPos.z -= n.z * epsilon;
                        size_t idx = getGridIdxFromVecPos(rayPos);
                        Vec3f pressureColor = {0.0f, 0.0f, 0.0f};
                        if (useGrid && idx < vector_space->getGrid().size)
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
                if (numBounces > 10)
                {
                    cont = false;
                }
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
    fdtd_cleanup(vector_space.get());
}

std::string Simulator::toString()
{
    std::string result = "Simulator: \n";
    result += "Width: " + std::to_string(width) + "\n";
    result += "Height: " + std::to_string(height) + "\n";
    return result;
}

void Simulator::doSimulationStep()
{
    std::cout << "\rPerforming simulation step " << simulation_step << "... Last step took: " << vector_space->stopwatch() << std::flush;
    // vector_space->computePressureStage();
    fdtd_step(vector_space.get());
    simulation_step++;
}

void Simulator::simulate(size_t steps)
{
    simulation_step = 0;
    Grid &grid = vector_space->getGrid();
    size_t centerx = grid.Nx / 2;
    size_t centery = grid.Ny / 2;
    size_t centerz = grid.Nz / 2;
    float c = 343.f;           // Speed of sound in m/s
    float h = vector_box_size; // Size of the grid cell in meters
    float dt = 0.5f * h / (c * std::sqrt(3.f));
    float c2_dt2 = c * c * dt * dt;
    float gdt = 5.f * dt;
    float inv_h2 = 1.f / (h * h);
    std::cout << "Simulation parameters: c = " << c << " m/s, h = " << h << " m, dt = " << dt << " s, c2_dt2 = " << c2_dt2 << ", gdt = " << gdt << ", inv_h2 = " << inv_h2 << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Starting simulation with " << steps << " steps..." << std::endl;
    initPressureSphere(vector_space.get(), centerx, centery, centerz, 20, 1.f, true); // Initialize a pressure sphere in the center of the grid
    updateCurrentGridFromGPU(vector_space.get());
    vector_space->layerToImage("output_layer.png", output_layer);
    for (size_t i = 0; i < steps; ++i)
    {
        float phase = static_cast<float>(simulation_step) / 120.f * 2.f * M_PI;
        float phase2 = static_cast<float>(simulation_step) / 231.f * 2.f * M_PI;
        float pressure = std::sin(phase) * std::sin(phase2) * 0.5f + 0.5f; // Pressure oscillates between 0 and 1
        pressure = pressure * pressure;                                    // Square the pressure to make it more pronounced
        if (simulation_step > 1000)
        {
            pressure = pressure * (1.f - static_cast<float>(std::min(simulation_step - 1000, 200u)) / 200.f); // Pressure decreases over time
        }

        if (simulation_step <= 1200)
        {
            initPressureSphere(vector_space.get(), centerx, centery, centerz, 20, pressure, false);
        }

        doSimulationStep();
        std::string file_out = "output/output_step_";
        if(i % 10 == 0){
            int frame = i / 10;
            if(frame < 10) {
            file_out += "00000" + std::to_string(frame) + ".png";
            } else if (frame < 100) {
                file_out += "0000" + std::to_string(frame) + ".png";
            } else if (frame < 1000) {
                file_out += "000" + std::to_string(frame) + ".png";
            } else if (frame < 10000) {
                file_out += "00" + std::to_string(frame) + ".png";
            } else if (frame < 100000) {
                file_out += "0" + std::to_string(frame) + ".png";
            } else {
                file_out += std::to_string(frame) + ".png";
            }
        updateCurrentGridFromGPU(vector_space.get());
        vector_space->layerToImage(file_out, output_layer);
        //renderImageToFile({7.f, 7.f, 7.f}, file_out, true);
        }
    }
    // initPressureSphere(vector_space.get(), centerx, centery, centerz, 100, 1.f, true);
    updateCurrentGridFromGPU(vector_space.get());

    // Now p_curr contains the final pressure values, we can render the final image
    // renderImageToFile({7.f, 7.f, 7.f}, "output/output_final.png", true);

    std::cout << "Simulation completed after " << steps << " steps." << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Total simulation time: " << duration << " ms" << std::endl;
    // renderImageToFile({7.f, 7.f, 7.f}, "output/output_final.png", true);

    vector_space->layerToImage("output_layer_after.png", output_layer);
    // vector_space->gridToImages("output/output_grid_");
}