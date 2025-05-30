#include "simulator_cpu.h"
#include "tbb/parallel_for.h"
#include <iostream>

void fdtd_cpu_setup(VectorSpace* space) {
    Grid& grid = space->getGrid();

	constexpr size_t ALIGN = 64;
	grid.p_curr = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	grid.p_next = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	grid.p_prev = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	grid.p_temp = static_cast<float*>(aligned_alloc(ALIGN, sizeof(float) * grid.size));
	grid.flags = static_cast<uint8_t*>(aligned_alloc(ALIGN, sizeof(uint8_t) * grid.size));
	
	std::cout << "Initializing VectorSpace with dimensions: "
	          << grid.Nx << " x " << grid.Ny << " x " << grid.Nz
	          << ", total size: " << grid.size << std::endl;
	for (size_t i = 0; i < grid.size; ++i) {
		grid.p_curr[i] = 0.0f;
		grid.p_next[i] = 0.0f;
		grid.p_prev[i] = 0.0f;
		grid.p_temp[i] = 0.0f;
		grid.flags[i] = 0; // Initialize flags to zero
	}
	space->resetStopwatch();
	std::cout << "VectorSpace initialized." << std::endl;

    // Reset the stopwatch
    space->resetStopwatch();
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