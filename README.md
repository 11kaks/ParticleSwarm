# ParticleSwarm

Parallellizised (CUDA) particle swarm single objective optimizer

## Compiling

Minimum language standard required: C++11

You might need to modify the arch option in makefile depending
on your Nvidia GPU architecture (if compiled in Linux).

Visual Studio solution works with VS 2017 at least. CUDA 10.1.

## Test runs

Testing done with Windows 10 using Nvidia GTX1070.

Paralellization useful when problem dimension is high. Making 
memory copies for CUDA at every iteration kills the performance
gain by calculating position and velocity updates on the GPU.