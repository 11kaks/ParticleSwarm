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


static void prinRun(OP &op, Swarm &s, bool CUDAPosVel, int runtimeMicS) {

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


static void prinRunCSV(OP &op, Swarm &s, bool CUDAPosVel, int runtimeMicS, int generations, int size) {
	// Few words how this version is different from base version.
	std::string upgrade = "Base";
	printf("%s;%d;%d;%d;%d;%d;%d;%d;%d   \n"
		, upgrade.c_str()
		, generations
		, size
		, runtimeMicS
		, (int)s.durInit.count()
		, (int)s.durUBest.count()
		, (int)s.durPPosVel.count()
		, (int)s.durPFun.count()
		, (int)s.durMemcpy.count()
	);
}

static void testSwarm(OP &op) {
	bool CUDAposvel = false;
	// TODO: Create particles based on warp size?
	const int size = 20;
	const int dim = 2; // OP decision space dimension
	Swarm swarm(size, dim, op);
	int generations = 1000;

	

	if(CUDAposvel) {
		std::cout << "First generation:" << std::endl;
		swarm.print();
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
	} else {
		std::cout << "First generation:" << std::endl;
		swarm.print();
		//swarm.printParticles();
	}

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	for(int i = 0; i < generations; ++i) {
		swarm.updateParticlePositions(CUDAposvel);
	}
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	int runtime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	if(CUDAposvel) {
		std::cout << "Last generation:" << std::endl;
		swarm.print();
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
	} else {
		std::cout << "Last generation:" << std::endl;
		swarm.print();
		//swarm.printParticles(); 
	}

	prinRun(op, swarm, CUDAposvel, runtime);
	prinRunCSV(op, swarm, CUDAposvel, runtime, generations, size);

}

int main() {
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
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
	}

	srand(145623);
	Rastriging problem;
	testSwarm(problem);
}
