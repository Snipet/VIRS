#pragma once

#include <float.h>

struct Vec3f { float x, y, z; };

class VectorSpace {
public:
	VectorSpace(size_t x, size_t y, size_t z);
	~VectorSpace();
	float getMemoryUsageGB();

private:
	size_t size_x;
	size_t size_y;
	size_t size_z;
	size_t size;
	float* p_curr;
	float* p_next;
};