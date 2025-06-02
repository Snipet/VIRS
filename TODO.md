## Optimizations:
- **Simulation caching**
    - After the sponge layer is build according to the material parameters, store it in a cache file. When the user wishes to re-run the simulation and has made no changes to the input mesh (.obj), use the cached grid. An algorithm compute an store a hash on the input mesh file and check if is has changed. If it has changed, recompute the sponge layers.
- **Better GPU Practices**: Instead of "launching" the processes on the CUDA cores each simulation step, would it be more efficient to do one launch somehow. How would we keep the cores synchronized? How inefficient is launching all of the CUDA cores each simulation step?

## Audio file implementation
The simulator's audio sources should be able to play multi-channel audio files. The audio files would be loaded in to the simulator. At a given step, the current value of the audio file will be read by using (which?) interpolation. The spot in the audio file that is read should be determined by the current time of the simulation (dt * step; in microseconds). 

## Material indexing
Mesh primitives will have different acoustical properties. Some mesh primitives will need to generate acoustic waves, and some will interact with them. A material indexing scheme is needed to tell which material has which property. Multiple audio files with N channels should be used, and an arbitrary amount of materials will be used in the simulation. Therefore, a JSON configuration file should be passed into the simulation. This configuration file should detail which mesh material index correlates with which material or audio file channel. The configuration file should contain the file path for the .obj file, the paths for input audio files, and material data for each material index.