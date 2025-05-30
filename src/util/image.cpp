#include "image.h"
#include <stdexcept>
#include <png.h>
#include <string>
#include <cstring>

//Quick, dirty function to save a buffer of RGB(A) pixels to a PNG file.
void save_png(const char *filename, unsigned char *framebuffer, int width, int height, bool has_alpha)
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


int save_png_to_memory(unsigned char* rgb_buffer, int width, int height, int row_stride, unsigned char** dst, size_t* dst_size) {
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