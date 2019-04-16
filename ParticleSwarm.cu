// ParticleSwarm.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "OP.h"
#include "Rastriging.h"
#include "Particle.h"
#include "Swarm.h"

/*
To set CUDA working with Visual Studio 2017:
- right click on project
- build dependencies -> build customization
- tick on some installed CUDA version
- OK
and it should build.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// Clock
#include <chrono>

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h> 
#include <math.h>


static void printRun(OP &op, Swarm &s, bool CUDAPosVel, int runtimeMicS) {

	std::string mics = " us";

	printf("-- %s -- \n", op.name.c_str());
	printf("Running the swarm took: %.4f s \n", runtimeMicS / 1000000.0);
	printf(" of which \n");
	std::cout << "initialization                         " << (int)s.durInit.count() << mics << std::endl;
	std::cout << "updating best value                    " << (int)s.durUBest.count() << mics << std::endl;
	std::cout << "updating positions and velocities      " << (int)s.durPPosVel.count() << mics << std::endl;
	std::cout << "updating function values               " << (int)s.durPFun.count() << mics << std::endl;

	if(CUDAPosVel) {
		std::cout << "making memory copies               " << (int)s.durMemcpy.count() << mics << std::endl;
	}
}

static void prinrRunCSV(std::string upgrade, float accuracy, Swarm *s, bool CUDAPosVel, int runtimeMicS, int generations, int size) {
	printf("%s;%d;%d;%.6f;%d;%.6f;%d;%d;%d;%d;%d   \n"
		, upgrade.c_str()
		, generations
		, size
		, accuracy
		, runtimeMicS
		, (float)runtimeMicS / 1000000.f
		, (int)s->durInit.count()
		, (int)s->durUBest.count()
		, (int)s->durPPosVel.count()
		, (int)s->durPFun.count()
		, (int)s->durMemcpy.count()
	);
}

static float distanceFromOptimum(OP &op, std::vector<float> result) {
	std::vector<float> known = op.getKnownOptimumPoint();
	float dist = 0.f;
	for(size_t i = 0; i < op.getDecDimension(); ++i) {
		dist += (result[i] - known[i]) * (result[i] - known[i]);
	}
	return (float)sqrt(dist);
}

static void runSwarm(std::string upgrade, OP &op, const bool CUDAposvel, const int size, const int iterations, const int maxThreads) {

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	int dim = op.getDecDimension();
	Swarm *swarm = new Swarm(size, dim, op, CUDAposvel);

	/*
	Calculate gridsize and blocksize. These are not needed if CUDA is not in use.
	Block width is the same as problem dimension, and all blocks are on 
	top of each other to allow easy indexing in kernel program. Blocks are created 
	so that there is minimal amount of unused threads. A block will have as many threads
	as the device allows.

	Consider using grid stride loops?
	https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
	*/
	int blockX = dim;
	// floored by int trunktuation
	int blockY = maxThreads / blockX;
	int blockCount = (int)(ceil((float)size / (float)blockY));

	dim3 gridSize(1, blockCount);
	dim3 blockSize(blockX, blockY);

	// Run the swarm
	for(size_t i = 0; i < iterations; ++i) {
		swarm->updateParticlePositions(gridSize, blockSize);
	}
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	int runtime = (int)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	float accuracy = distanceFromOptimum(op, swarm->bestParticle->xBest);

	//printRun(op, swarm, CUDAposvel, runtime);
	prinrRunCSV(upgrade, accuracy, swarm, CUDAposvel, runtime, iterations, size);

	delete swarm;

}

int main() {
	// Print some general iformation about the CUDA devices used.
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for(int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);

		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
		printf("  Warp size : %d\n", prop.warpSize);
		printf("  Max threads per block : %d\n\n",
			prop.maxThreadsPerBlock);
	}

	srand(145623);
	Rastriging problem(20);

	int sizeLevels = 3;
	int iterationLevels = 2;

	// Few words how this version is different from base version.
	std::string upgrade = "Base";
	bool CUDAposvel = false;
	int size = 32;
	int iterations = 100;

	// Size loop
	for(size_t i = 0; i < sizeLevels; ++i) {
		// Iteration loop
		for(size_t j = 0; j < iterationLevels; ++j) {
			runSwarm(upgrade, problem, CUDAposvel, size, iterations, 0);
			iterations *= 10;
		}
		iterations = 100; // reset iterations
		size *= 10;
	}


	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int maxThreads = prop.maxThreadsPerBlock;

	// Reset and run with CUDA
	upgrade = "CUDA pos vel";
	CUDAposvel = true;
	size = 32;
	iterations = 100;

	// Size loop
	for(size_t i = 0; i < sizeLevels; ++i) {
		// Iteration loop
		for(size_t j = 0; j < iterationLevels; ++j) {
			runSwarm(upgrade, problem, CUDAposvel, size, iterations, maxThreads);
			iterations *= 10;
		}
		iterations = 100; // reset iterations
		size *= 10;
	}

	

	// Reset and run with CUDA
	upgrade = "CUDA small block";
	maxThreads = 256;
	CUDAposvel = true;
	size = 32;
	iterations = 100;

	// Size loop
	for(size_t i = 0; i < sizeLevels; ++i) {
		// Iteration loop
		for(size_t j = 0; j < iterationLevels; ++j) {
			runSwarm(upgrade, problem, CUDAposvel, size, iterations, maxThreads);
			iterations *= 10;
		}
		iterations = 100; // reset iterations
		size *= 10;
	}

}

// Print code for debugging the particle arrays
/*for(int i = 0; i < size; i++) {
			for(size_t j = 1; j < dim; ++j) {
				std::vector<float> x = { swarm.xx[i*(j - 1) + j - 1], swarm.xx[i*j + j] };
				std::vector<float> xb = { swarm.xb[i*(j - 1) + j - 1], swarm.xb[i*j + j] };
				std::cout << "(" << swarm.xx[i*j - 1 + j] << "," << swarm.xx[i*j + j] << ") V ("
					<< op.evaluate(xb) << ") -> ("
					<< swarm.vv[i*j - 1 + j] << "," << swarm.vv[i*j + j] << ") B ("
					<< swarm.xb[i*j - 1 + j] << "," << swarm.xb[i*j + j] << ") BV ("
					<< op.evaluate(xb) << ")"
					<< std::endl;
			}
		}*/
