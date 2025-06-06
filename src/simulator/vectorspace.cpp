#include "vectorspace.h"
#include <iostream>
#include "tbb/parallel_for.h"
#include <immintrin.h>
#include "../util/image.h"
#include <cstring>

VectorSpace::VectorSpace(size_t x, size_t y, size_t z, const AudioFile<float>& audio_file, float h) : h(h) {
	grid.Nx = x;
	grid.Ny = y;
	grid.Nz = z;
	grid.p_source_size = audio_file.getNumSamplesPerChannel();
	this->audio_file = audio_file;

	grid.size = grid.Nx * grid.Ny * grid.Nz;
	// constexpr size_t ALIGN = 64;
	// grid.p_curr = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	// grid.p_next = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	// grid.p_prev = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	// grid.p_temp = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	// grid.flags = static_cast<uint8_t*>(aligned_alloc(ALIGN, sizeof(uint8_t) * grid.size));
	
	// std::cout << "Initializing VectorSpace with dimensions: "
	//           << grid.Nx << " x " << grid.Ny << " x " << grid.Nz
	//           << ", total size: " << grid.size
	//           << ", memory usage: " << getMemoryUsageGB() << " GB." << std::endl;
	// for (size_t i = 0; i < grid.size; ++i) {
	// 	grid.p_curr[i] = 0.0f;
	// 	grid.p_next[i] = 0.0f;
	// 	grid.p_prev[i] = 0.0f;
	// 	grid.p_temp[i] = 0.0f;
	// 	grid.flags[i] = 0; // Initialize flags to zero
	// }
	// resetStopwatch();
	// std::cout << "VectorSpace initialized." << std::endl;
}

VectorSpace::~VectorSpace() {
	delete[] grid.p_curr;
	delete[] grid.p_next;
	delete[] grid.flags;
}


float VectorSpace::getMemoryUsageGB() {
	return (
		sizeof(float) * grid.size * 4 +
		sizeof(uint8_t) * grid.size
	) / (1000.f * 1000.f * 1000.f);
}


void VectorSpace::computePressureStageParallel(){
	const float inv_h2 = 1.f / (h * h);
	const float c = 343.f;
	const float dt = 0.5 * h / (c * std::sqrt(3.f));
	const float c2_dt2 = c * c * dt * dt;
	const float gamma = 5.f;
	const float damp_factor = gamma * dt;
	//std::cout << "dt: " << dt << std::endl;

	tbb::parallel_for(tbb::blocked_range<size_t>(1, grid.Nz - 1), [&](const tbb::blocked_range<size_t>& r) {
		for (size_t k = r.begin(); k < r.end(); ++k) {
			for (size_t j = 1; j < grid.Ny - 1; ++j) {
				for (size_t i = 1; i < grid.Nx - 1; ++i) {
					size_t idx = grid.idx(i, j, k);
					grid.p_next[idx] =  ((2.f - damp_factor) * grid.p_curr[idx] - (1.f - damp_factor) * grid.p_prev[idx] + c2_dt2 * Grid::laplacian7(grid.p_curr, grid.flags, i, j, k, grid.Nx, grid.Ny, grid.Nz, inv_h2));
				}
			}
		}
	});
	std::swap(grid.p_prev, grid.p_curr);
	std::swap(grid.p_curr, grid.p_next);
}


void VectorSpace::layerToImage(const std::string& out, std::size_t layer)
{
    if (layer >= grid.Ny)
        throw std::out_of_range("layerToImage: layer index out of range");

    std::size_t width   = grid.Nx;      // horizontal X
    std::size_t height  = grid.Nz;      // vertical in image = world Z
    const std::size_t xyStride = grid.Nx;     // stride in y
    const std::size_t zStride  = grid.Nx * grid.Ny;

	size_t output_width = width;
	size_t output_height = height;
	if(width % 2 != 0){
		output_width += 1; // Ensure width is even for better image processing
	}
	if (height % 2 != 0) {
		output_height += 1; // Ensure height is even for better image processing
	}

    float scale = 1500.f;

    std::vector<unsigned char> pixels(output_width * output_height * 3, 0);

    for (std::size_t k = 0; k < output_height; ++k) {          // z  (image row)
        const std::size_t rowOff = k * zStride + layer * xyStride;
        for (std::size_t i = 0; i < output_width; ++i) {       // x  (image col)
			if(i >= width || k >= height) {
				// Fill with black if out of bounds
				pixels[(k * output_width + i) * 3 + 0] = 0; // R
				pixels[(k * output_width + i) * 3 + 1] = 0; // G
				pixels[(k * output_width + i) * 3 + 2] = 0; // B
				continue;
        	}else{
				float p = grid.p_curr[rowOff + i];
				uint8_t flag = grid.flags[rowOff + i];
				float absorb = grid.p_absorb[rowOff + i];

				unsigned char red  = 0;
				unsigned char blue = 0;
				unsigned char green = 0;
				if(flag == 1 || flag == 3) {
					green = 255;
				}
				// }else if (flag == 2) {
				// 	// green = static_cast<unsigned char>(
				// 	// 		std::clamp(absorb * 255.f, 0.0f, 255.0f));
				// 	green = 128;
				// }

				if (p >= 0.0f)
					red  = static_cast<unsigned char>(
							std::clamp(p * scale, 0.0f, 255.0f));
				else
					blue = static_cast<unsigned char>(
							std::clamp(-p * scale, 0.0f, 255.0f));

				std::size_t pixIdx = (k * width + i) * 3;
				pixels[pixIdx + 0] = red;      // R
				pixels[pixIdx + 1] = green;        // G
				pixels[pixIdx + 2] = blue;     // B
			}
   		}
	}

    save_png(out.c_str(), pixels.data(),
             static_cast<int>(output_width),
             static_cast<int>(output_height),
             /*has_alpha=*/false);
}

//On XZ plane, centered at (Nx/2, Ny/2)
void VectorSpace::initPressureCircleOnLayer(std::size_t layer,
                                            std::size_t radius,
                                            float       pressure,
                                            bool        init)
{
    if (layer >= grid.Ny)
        throw std::out_of_range("layer index out of range");


    if (radius > grid.Nx / 2 || radius > grid.Nz / 2)
        throw std::out_of_range("radius exceeds half of grid dimensions");

    const std::size_t centreX = grid.Nx / 2;
    const std::size_t centreZ = grid.Nz / 2;

    
    const std::size_t xyStride = grid.Nx;          
    const std::size_t zStride  = grid.Nx * grid.Ny;

    for (std::size_t k = 0; k < grid.Nz; ++k) {            // z
        for (std::size_t i = 0; i < grid.Nx; ++i) {        // x
            const std::size_t dx = i - centreX;
            const std::size_t dz = k - centreZ;

            std::size_t idx = k * zStride + layer * xyStride + i;

            if (dx * dx + dz * dz <= radius * radius) {
                grid.p_curr[idx] = pressure;               // inside circle
            } else if (init) {
                grid.p_curr[idx] = 0.0f;                   // outside circle
            }
        }
    }

    /* initialise p_prev if requested */
    if (init) {
        std::memcpy(grid.p_prev, grid.p_curr,
                    sizeof(float) * grid.size);
    }
}


void VectorSpace::initPressureSphere(size_t xpos, size_t ypos, size_t zpos, size_t radius, float pressure, bool init)
{
	if (xpos >= grid.Nx || ypos >= grid.Ny || zpos >= grid.Nz)
		throw std::out_of_range("initPressureSphere: position out of range");

	if (radius > grid.Nx / 2 || radius > grid.Ny / 2 || radius > grid.Nz / 2)
		throw std::out_of_range("initPressureSphere: radius exceeds half of grid dimensions");

	const std::size_t xyStride = grid.Nx;          
	const std::size_t zStride  = grid.Nx * grid.Ny;

	tbb::parallel_for(tbb::blocked_range<size_t>(0, grid.Nz), [&](const tbb::blocked_range<size_t>& r) {
		for (size_t k = r.begin(); k < r.end(); ++k) {            // z
			for (size_t j = 0; j < grid.Ny; ++j) {                // y
				for (size_t i = 0; i < grid.Nx; ++i) {            // x
					const size_t dx = i - xpos;
					const size_t dy = j - ypos;
					const size_t dz = k - zpos;
					size_t idx = k * zStride + j * xyStride + i;
					if (dx * dx + dy * dy + dz * dz <= radius * radius) {
						grid.p_curr[idx] = pressure;               // inside sphere
					} else if (init) {
						grid.p_curr[idx] = 0.0f;                   // outside sphere
					}
				}
			}
		}
	});

	/* initialise p_prev if requested */
	if (init) {
		std::memcpy(grid.p_prev, grid.p_curr,
					sizeof(float) * grid.size);
	}
}