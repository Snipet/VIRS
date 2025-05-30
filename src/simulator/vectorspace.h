#pragma once

#include <float.h>
#include <iosfwd>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <string>

struct Vec3f
{
	float x, y, z;
	Vec3f(float x = 0.0f, float y = 0.0f, float z = 0.0f) : x(x), y(y), z(z) {}
	Vec3f operator-(const Vec3f &other) const
	{
		return {x - other.x, y - other.y, z - other.z};
	}
	Vec3f operator+(const Vec3f &other) const
	{
		return {x + other.x, y + other.y, z + other.z};
	}
	Vec3f operator*(float scalar) const
	{
		return {x * scalar, y * scalar, z * scalar};
	}
	Vec3f operator/(float scalar) const
	{
		return {x / scalar, y / scalar, z / scalar};
	}
	Vec3f& operator+=(const Vec3f &other)
	{
		x += other.x; y += other.y; z += other.z;
		return *this;
	}
	Vec3f& operator-=(const Vec3f &other)
	{
		x -= other.x; y -= other.y; z -= other.z;
		return *this;
	}

	
};

struct Grid
{
	float *p_curr;
	float *p_next;
	float *p_prev;
	float *p_temp;

	uint8_t *flags;

	std::size_t Nx;
	std::size_t Ny;
	std::size_t Nz;
	std::size_t size;

	// Flattened index: z‑major for unit‑stride in innermost loop
	inline size_t idx(size_t x, size_t y, size_t z)
	{
		return z * Ny * Nx + y * Nx + x;
	}

	static inline float laplacian7(const float *__restrict p, const uint8_t *__restrict flags,
								   std::size_t i, std::size_t j, std::size_t k,
								   std::size_t Nx, std::size_t Ny, std::size_t Nz,
								   float inv_h2)
	{
		if (i == 0 || i == Nx - 1 ||
			j == 0 || j == Ny - 1 ||
			k == 0 || k == Nz - 1)
		{
			return 0.0f; // Boundary condition
		}
		const size_t xyStride = Nx;
		const size_t zStride = Nx * Ny;
		const size_t idx = k * zStride + j * xyStride + i;
		
		if(flags[idx] == 1)
			return 0.0f; // Skip if the cell is not occupied

		float p_center = p[idx];
		float p_xp = p[idx + 1];
		float p_xm = p[idx - 1];
		float p_yp = p[idx + xyStride];
		float p_ym = p[idx - xyStride];
		float p_zp = p[idx + zStride];
		float p_zm = p[idx - zStride];

		return (p_xp + p_xm + p_yp + p_ym + p_zp + p_zm - 6.0f * p_center) * inv_h2;
	}
	static inline float laplacian19(const float *__restrict p,
									std::size_t i, std::size_t j, std::size_t k,
									std::size_t Nx, std::size_t Ny, std::size_t Nz,
									float inv_h2)
	{
		if (i < 2 || i > Nx - 3 ||
			j < 2 || j > Ny - 3 ||
			k < 2 || k > Nz - 3)
			return 0.0f; // Dirichlet boundary


		const std::size_t xy = Nx;
		const std::size_t z = Nx * Ny;
		const std::size_t idx = k * z + j * xy + i;

		const float xp1 = p[idx + 1];
		const float xm1 = p[idx - 1];
		const float yp1 = p[idx + xy];
		const float ym1 = p[idx - xy];
		const float zp1 = p[idx + z];
		const float zm1 = p[idx - z];

		const float xp2 = p[idx + 2];
		const float xm2 = p[idx - 2];
		const float yp2 = p[idx + 2 * xy];
		const float ym2 = p[idx - 2 * xy];
		const float zp2 = p[idx + 2 * z];
		const float zm2 = p[idx - 2 * z];

		const float centre = p[idx];

		// 4th‑order finite‑difference Laplacian
		constexpr float w1 = 16.0f / 12.0f;	 //  1.333 333 333
		constexpr float w2 = -1.0f / 12.0f;	 // −0.083 333 333
		constexpr float wc = -90.0f / 12.0f; // −7.5

		float lap =
			w1 * (xp1 + xm1 + yp1 + ym1 + zp1 + zm1) + w2 * (xp2 + xm2 + yp2 + ym2 + zp2 + zm2) + wc * centre;

		return lap * inv_h2;
	}
};

class VectorSpace
{
public:
	VectorSpace(std::size_t x, std::size_t y, std::size_t z, float h);
	~VectorSpace();
	float getMemoryUsageGB();
	void computePressureStage();
	void computePressureStageParallel();
	void layerToImage(const std::string &out, size_t layer);
	void initPressureCircleOnLayer(size_t layer, size_t radius, float pressure, bool init = true);
	void initPressureSphere(size_t xpos, size_t ypos, size_t zpos, size_t radius, float pressure, bool init = true);
	Grid& getGrid()
	{
		return grid;
	}
	void gridToImages(const std::string &out_prefix)
	{
		for (size_t layer = 0; layer < grid.Nz; ++layer)
		{
			// Pad the layer number with leading zeros to ensure consistent file naming
			if (layer < 10)
			{
				layerToImage(out_prefix + "00" + std::to_string(layer) + ".png", layer);
			}
			else if (layer < 100)
			{
				layerToImage(out_prefix + "0" + std::to_string(layer) + ".png", layer);
			}
			else
			{
				layerToImage(out_prefix + std::to_string(layer) + ".png", layer);
			}
		}
	}
	void resetStopwatch()
	{
		last_update_time = std::chrono::high_resolution_clock::now();
	}
	std::string stopwatch()
	{
		auto now = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update_time).count();
		last_update_time = now;
		return std::to_string(duration) + " ms";
	}

private:
	Grid grid;
	float h;
	float current_pressure;

	std::chrono::time_point<std::chrono::high_resolution_clock> last_update_time;
};