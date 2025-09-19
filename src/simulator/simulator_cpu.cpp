#include "simulator_cpu.h"
#include "tbb/parallel_for.h"
#include <iostream>
#include <cstring>

void fdtd_cpu_setup(VectorSpace* space) {
    Grid& grid = space->getGrid();

	constexpr size_t ALIGN = 64;
	grid.p_curr = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	grid.p_next = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	grid.p_prev = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	grid.p_temp = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	grid.flags = static_cast<uint8_t*>(aligned_alloc(ALIGN, sizeof(uint8_t) * grid.size));
	//grid.p_absorb = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	
	std::cout << "Initializing VectorSpace with dimensions: "
	          << grid.Nx << " x " << grid.Ny << " x " << grid.Nz
	          << ", total size: " << grid.size << std::endl;
	for (size_t i = 0; i < grid.size; ++i) {
		grid.p_curr[i] = 0.0f;
		grid.p_next[i] = 0.0f;
		grid.p_prev[i] = 0.0f;
		grid.p_temp[i] = 0.0f;
		grid.flags[i] = 0; // Initialize flags to zero
		//grid.p_absorb[i] = 0.0f; // Initialize p_absorb to zero
	}
	space->resetStopwatch();
	std::cout << "VectorSpace initialized." << std::endl;

    // Reset the stopwatch
    space->resetStopwatch();
}

void fdtd_cpu_cleanup(VectorSpace* space) {
	Grid& grid = space->getGrid();

	// Free CPU memory
	delete[] grid.p_curr;
	delete[] grid.p_next;
	delete[] grid.p_prev;
	delete[] grid.p_temp;
	delete[] grid.flags;
	//delete[] grid.p_absorb;
	grid.p_curr = nullptr;
	grid.p_next = nullptr;
	grid.p_prev = nullptr;
	grid.p_temp = nullptr;
	grid.flags = nullptr;
	//grid.p_absorb = nullptr;

	std::cout << "CPU memory cleaned up successfully." << std::endl;
}

void fdtd_cpu_step(VectorSpace* space, float h) {
    Grid& grid = space->getGrid();
    const float inv_h2 = 1.f / (h * h);
    const float c = 343.f;
	const float dt = 0.5 * h / (c * std::sqrt(3.f));
	const float c2_dt2 = c * c * dt * dt;
	const float gamma = 5.f;
	const float damp_factor = gamma * dt;
    const auto Nx = grid.Nx, Ny = grid.Ny, Nz = grid.Nz;
    auto idx = [=](int i,int j,int k) { return (k*Ny + j)*Nx + i; };

 	tbb::parallel_for(tbb::blocked_range<size_t>(1, grid.Nz - 1), [&](const tbb::blocked_range<size_t>& r) {
		for (size_t k = r.begin(); k < r.end(); ++k) {
			for (size_t j = 1; j < grid.Ny - 1; ++j) {
				for (size_t i = 1; i < grid.Nx - 1; ++i) {
					size_t idx = grid.idx(i, j, k);
                    float lap = Grid::laplacian7(grid.p_curr, grid.flags, i, j, k, grid.Nx, grid.Ny, grid.Nz, inv_h2);
					grid.p_next[idx] =  (2.f - damp_factor) * grid.p_curr[idx] - (1.f - damp_factor) * grid.p_prev[idx] + c2_dt2 * lap;
				}
			}
		}
	});   


    std::swap(grid.p_prev, grid.p_curr);
    std::swap(grid.p_curr, grid.p_next);
}

void initPressureSphereCPU(VectorSpace* space, size_t xpos, size_t ypos, size_t zpos, size_t radius, float pressure, bool init) {
	Grid& grid = space->getGrid();

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
		std::memcpy(grid.p_prev, grid.p_curr, sizeof(float) * grid.size);
	}
}