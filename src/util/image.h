#pragma once
#include "png.h"

void save_png(const char *filename, unsigned char *framebuffer, int width, int height, bool has_alpha);

int save_png_to_memory(unsigned char* rgb_buffer, int width, int height, int row_stride, unsigned char** dst, size_t* dst_size);