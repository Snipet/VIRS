#include "simulator.h"
#include <iostream>
#include "png.h"

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

//Quick, dirty function to save a buffer of RGB(A) pixels to a PNG file.
static void save_png(const char *filename, unsigned char *framebuffer, int width, int height, bool has_alpha)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
        throw std::runtime_error("Failed to open file for writing.");

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png)
        throw std::runtime_error("Failed to create PNG write struct.");

    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        png_destroy_write_struct(&png, nullptr);
        throw std::runtime_error("Failed to create PNG info struct.");
    }

    if (setjmp(png_jmpbuf(png)))
    {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        throw std::runtime_error("Error during PNG creation.");
    }

    png_init_io(png, fp);

    int color_type = has_alpha ? PNG_COLOR_TYPE_RGBA : PNG_COLOR_TYPE_RGB;
    png_set_IHDR(
        png, info, width, height,
        8, color_type,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png, info);

    png_bytep *row_pointers = new png_bytep[height];
    int bytes_per_pixel = has_alpha ? 4 : 3;
    for (int y = 0; y < height; y++)
    {
        row_pointers[y] = framebuffer + y * width * bytes_per_pixel;
    }

    png_write_image(png, row_pointers);
    png_write_end(png, nullptr);

    delete[] row_pointers;
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

static int save_png_to_memory(unsigned char* rgb_buffer, int width, int height, int row_stride, unsigned char** dst, size_t* dst_size) {
    if (!rgb_buffer || width <= 0 || height <= 0 || !dst || !dst_size) {
        return -1;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) return -2;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, nullptr);
        return -3;
    }

    struct PngMemBuffer {
        unsigned char* data = nullptr;
        size_t size = 0;
    } buffer;

    auto png_write_callback = [](png_structp png_ptr, png_bytep data, png_size_t length) {
        PngMemBuffer* buf = reinterpret_cast<PngMemBuffer*>(png_get_io_ptr(png_ptr));
        size_t new_size = buf->size + length;
        buf->data = static_cast<unsigned char*>(realloc(buf->data, new_size));
        if (!buf->data) png_error(png_ptr, "realloc failed");
        memcpy(buf->data + buf->size, data, length);
        buf->size = new_size;
    };

    png_set_write_fn(png_ptr, &buffer, png_write_callback, nullptr);

    // Set image info
    png_set_IHDR(png_ptr, info_ptr, width, height,
                 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png_ptr, info_ptr);

    // Write rows
    for (int y = 0; y < height; ++y) {
        png_write_row(png_ptr, const_cast<png_bytep>(rgb_buffer + y * row_stride));
    }

    png_write_end(png_ptr, nullptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);

    *dst = buffer.data;
    *dst_size = buffer.size;

    return 0;
}

Simulator::Simulator(unsigned int w, unsigned int h)
{
    // Initialize the Simulator
    width = w;
    height = h;
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

void Simulator::loadObj(const std::string& path) {

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

    //Determine bounding box;
    bounding_box.min = { std::numeric_limits<float>::max(),std::numeric_limits<float>::max() ,std::numeric_limits<float>::max() };
    bounding_box.max = { std::numeric_limits<float>::min(),std::numeric_limits<float>::min() ,std::numeric_limits<float>::min() };

    for (int i = 0; i < attrib.vertices.size(); i+=3) {
        bounding_box.min.x = std::min(bounding_box.min.x, attrib.vertices[i]);
        bounding_box.min.y = std::min(bounding_box.min.y, attrib.vertices[i + 1]);
        bounding_box.min.z = std::min(bounding_box.min.z, attrib.vertices[i + 2]);

        bounding_box.max.x = std::max(bounding_box.max.x, attrib.vertices[i]);
        bounding_box.max.y = std::max(bounding_box.max.y, attrib.vertices[i + 1]);
        bounding_box.max.z = std::max(bounding_box.max.z, attrib.vertices[i + 2]);
    }

    std::cout << "Bounding box: min(" << bounding_box.min.x << ", " << bounding_box.min.y << ", " << bounding_box.min.z
              << "), max(" << bounding_box.max.x << ", " << bounding_box.max.y << ", " << bounding_box.max.z << ")" << std::endl;

    //float max_dim_len = std::numeric_limits<float>::min();
    //max_dim_len = std::max(bounding_box.max.x - bounding_box.min.x, max_dim_len);
    //max_dim_len = std::max(bounding_box.max.y - bounding_box.min.y, max_dim_len);
    //max_dim_len = std::max(bounding_box.max.y - bounding_box.min.y, max_dim_len);
    
    vector_box_size = 343.f / (MAX_AUDIO_FREQ * 2.f);
    float spanx = bounding_box.max.x - bounding_box.min.x;
    float spany = bounding_box.max.y - bounding_box.min.y;
    float spanz = bounding_box.max.z - bounding_box.min.z;

    float num_vectors_meter = 1.f / vector_box_size * (float)SIMULATION_OVERSAMPLING;
    size_t sizex = spanx * num_vectors_meter;
    size_t sizey = spany * num_vectors_meter;
    size_t sizez = spanz * num_vectors_meter;

    std::cout << "Vector box size: " << vector_box_size << "m, size in vectors: (" << sizex << ", " << sizey << ", " << sizez << ")" << std::endl;
    size_t memory_usage_bytes = sizex * sizey * sizez * sizeof(float) * 3; // 3 floats per vector

    vector_space = std::make_unique<VectorSpace>(sizex, sizey, sizez);
    
    std::cout << "Vector space created with size: " << sizex << "x" << sizey << "x" << sizez
              << ", memory usage: " << vector_space->getMemoryUsageGB() << " GB." << std::endl;

    for (const auto& shape : shapes)
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

    const size_t numVerts = vertices.size() / 3;
    const size_t numTriangles = indices.size() / 3;
    std::cout << "Loaded " << numVerts << " vertices and " << numTriangles << " triangles." << std::endl;
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    float* vb = static_cast<float*>(rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(float) * 3, numVerts));
    std::memcpy(vb, vertices.data(), vertices.size() * sizeof(float));

    unsigned int* ib = static_cast<unsigned int*>(rtcSetNewGeometryBuffer(
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

void Simulator::renderImageToFile(Vec3f cameraPos, const std::string& output_path) {
    render(cameraPos);
    // Save the image to a file
    save_png(output_path.c_str(), framebuffer, width, height, false);
}

void Simulator::renderImageToMemory(Vec3f cameraPos, unsigned char** out, size_t* out_size) {
    render(cameraPos);
    save_png_to_memory(framebuffer, width, height, width * 3, out, out_size);
}

void Simulator::render(Vec3f cameraPos)
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
                if(rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
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
                        // Hit detected, color the pixel based on the hit normal
                        unsigned char r = static_cast<unsigned char>(abs(n.x) * 255);
                        unsigned char g = static_cast<unsigned char>(abs(n.y) * 255);
                        unsigned char b = static_cast<unsigned char>(abs(n.z) * 255);

                        r = std::max(r, (unsigned char)0);
                        g = std::max(g, (unsigned char)0);
                        b = std::max(b, (unsigned char)0);
                        framebuffer[(y * width + x) * 3 + 0] = r;
                        framebuffer[(y * width + x) * 3 + 1] = g;
                        framebuffer[(y * width + x) * 3 + 2] = b;
                        cont = false; //We have hit a primitive whose normal faces away from the camera
                    }
                    else
                    {

                        const float epsilon = 0.001f; //Amount to "nudge" the ray to prevent re-intersection
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
}

std::string Simulator::toString()
{
    std::string result = "Simulator: \n";
    result += "Width: " + std::to_string(width) + "\n";
    result += "Height: " + std::to_string(height) + "\n";
    return result;
}