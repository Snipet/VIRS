# Virtual Impulse Response Synthesis
Optimized acoustics simulator for virtual spaces

## Building (Linux)
1. After cloning this repository and its submodules to your machine, ensure Libpng and Zlib are installed on your machine. These can usually be installed with your Linux distro package manager.
2. Build and install TBB to your machine
3. You will need to build Embree (raytracing & BVH library). Create a build folder ```embree/build``` and run ```cmake .. -DCMAKE_BUILD_TYPE=Release -DEMBREE_STATIC_LIB=ON -DEMBREE_TUTORIALS=OFF -DEMBREE_ISPC_SUPPORT=OFF```. Embree *must* be built static.
4. If you are using a CUDA-capable GPU, you can build in ```VIRS/build``` with ```cmake ..  -DCMAKE_BUILD_TYPE=Release```. If you wish to use the CPU, build with ```cmake ..  -DCMAKE_BUILD_TYPE=Release -DVIRS_WITH_CUDA=OFF```. It is recommended to use the GPU; my tests using GPU and CPU builds showed that CPU simulations are ~20x slower than GPU simulations. In addition, building the Sponge Layer on the CPU will sometimes take upwards of 10 minutes (g4dn.xlarge EC2 instance with 4 VCPU).

## Commands
- Run simulation for 5000 steps using room9.obj(```VIRS/assets```):
```./VIRS --run-simulation -i ../assets/room9.obj -f 5000```
- Output room mesh (faces with normals facing the camera will be omitted): 
```./VIRS --render-scene -i ../assets/room9.obj -o outrender.png```