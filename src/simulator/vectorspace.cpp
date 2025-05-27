#include "vectorspace.h"

VectorSpace::VectorSpace(size_t x, size_t y, size_t z) {
	size_x = x;
	size_y = y;
	size_z = z;

	size = size_x * size_y * size_z;
	data = new Vec3[size];
}

VectorSpace::~VectorSpace() {
}


float VectorSpace::getMemoryUsageGB() {
	return 0;
}
