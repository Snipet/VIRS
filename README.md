# Virtual Impulse Response Synthesis
Optimized acoustics simulator for virtual spaces

## Building (Linux)
1. After cloning this repository and its submodules to your machine, ensure TBB (Thread Building Blocks), Libpng, and Zlib are installed on your machine. These can usually be installed with your Linux distro package manager.
2. (Optional) build and install TBB to your machine. Using the package manager has been problematic occasionally.
3. You will need to build Embree (raytracing & BVH library). Create a build folder ```embree/build``` and run ```cmake .. -DCMAKE_BUILD_TYPE=Release -DEMBREE_TUTORIALS=OFF -DEMBREE_ISPC_SUPPORT=OFF```.
4. If you are using a CUDA-capable GPU, you can build in ```VIRS/build``` with ```cmake ..  -DCMAKE_BUILD_TYPE=Release```. If you wish to use the CPU, build with ```cmake ..  -DCMAKE_BUILD_TYPE=Release -DVIRS_WITH_CUDA=OFF```.

## Commands
- Run simulation for 5000 steps using room9.obj(```VIRS/assets```):
```./VIRS --run-simulation -i ../assets/room9.obj -f 5000```
- Output room mesh (faces with normals facing the camera will be omitted): 
```./VIRS --render-scene -i ../assets/room9.obj -o outrender.png```