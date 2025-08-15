# VIRS (Virtual Impulse Response Synthesizer)

VIRS runs acoustics simulations to generate room impulse responses for auralization.

## How It Works
VIRS uses a Finite-Difference Time-Domain scheme for wave acoustics simulation. Each simulation step works as follows:
Each finite cell (referred to as cells) contains a byte flag; 0 represents air, 1 represents a wall, and 2 represents a boundary (next to walls). First, the air and boundary cells are updated. This is done using a Neumann boundary condition for the walls, and each cell uses a Laplacian stencil. That update code looks like this: 
```C++
        float pCenter = pCurr[idx];

        auto V = [&](int ix, int iy, int iz) -> float
        {
            std::size_t n = ((std::size_t)iz * d_Ny + iy) * d_Nx + ix;
            if (flags[n] == 1)
            {
                return pCenter; // Wall voxel
            }
            else
            {
                return pCurr[n]; // Normal voxel
            }
        };

        float lap = (V(x + 1, y, z) + V(x - 1, y, z) +
                     V(x, y + 1, z) + V(x, y - 1, z) +
                     V(x, y, z + 1) + V(x, y, z - 1) - 6.0f * pCenter) *
                    d_inv_h2;
        pNext[idx] =
            (2.0f) * pCenter - (1.0f) * pPrev[idx] + d_c2_dt2 * lap;

```

All incoming waves against a boundary are *reflected perfectly and preserving phase*, per the Neumann boundary condition. However, acoustic materials never behave this way in real life; they exhibit frequency-dependent qualities. To accomplish this, another step is added: Frequency-Dependent Correction. This method is applied to *only* the boundary cells. To do this, the difference between time steps is taken (simulating a time derivative of the cell pressure). If this difference is subtracted (while also multiplied by a scaling constant) from the current next pressure, virtually no reflections occur (the wave is absorbed). If we apply a discrete filter to that difference, we can dictate what frequencies can be reflected and absorbed. For example, if the difference is passed through a high-pass filter, this simulates high frequencies being absorbed into the boundary. I am multiplying the difference by the Courant number, but this is also scaled by a constant to prevent phase inversion (why?). Here is the code for the frequency-dependent method:
```C++
       uint32_t idx = boundary_indices[i];
        int x = idx % d_Nx;
        int y = (idx / d_Nx) % d_Ny;
        int z = (idx / (d_Nx * d_Ny)) % d_Nz;
        size_t material_idx = 0; // Assuming a single material for simplicity (for now)

        float sum = 0.f;
        size_t num_active_components = 0;

        float summed_differences = 0.f;
        float dif = pCurr[idx] - pPrev[idx]; // Temporal derivative at boundary node [idx]
        for (size_t section = 0; section < 1; ++section)
        {
            size_t state_offset = (i * 4);
            float* state = &filter_states[state_offset];
            float input = dif;
            float output = filter_coeffs[0][material_idx] * input + filter_coeffs[1][material_idx] * state[0] + filter_coeffs[2][material_idx] * state[1] - filter_coeffs[3][material_idx] * state[2] - filter_coeffs[4][material_idx] * state[3];
            // Update state
            state[1] = state[0];
            state[0] = input;
            state[3] = state[2];
            state[2] = output;
            summed_differences += output;
        }

        const float UNKNOWN_CONSTANT = (d_c_sound * d_dt) / d_dx * 0.95f;
        pNext[idx] = pNext[idx] - UNKNOWN_CONSTANT * summed_differences;
```

