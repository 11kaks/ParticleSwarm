// ParticleSwarm.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "OP.h"
#include "Rastriging.h"
#include "Particle.h"
#include "Swarm.h"
#include "XM.h"

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

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h> 


static void testRastriging(OP &op) {
	std::vector<std::vector<float>> range = op.getSearchRange();
	std::cout << "Search range:" << std::endl;

	for(int i = 0; i < range.size(); i++) {
		std::cout << range[i][0] << " <= " << "x" << i + 1 << " <= " << range[i][1] << std::endl;
	}
	std::vector<float> point = { 0.0f,0.0f };
	std::cout << "f(" << point[0] << "," << point[1] << ") = " << op.evaluate(point) << std::endl;
}

static void testParticle(OP &op) {
	std::vector<float> x = { 0.f,0.f }; // val should be 0 at (0,0)
	std::vector<float> v = { 0.2f,0.1f };
	Particle part(x, v, op);
	// Print in initial position
	part.print();
	// Move the particle around for a few times
	int maxRounds = 2000;
	// Print every now and then
	int printEvery = 100;
	// Guide the particle towards origin at all times
	std::vector<float> dir = { 0.f,0.f };

	/*for(int i = 0; i < maxRounds; i++) {
		part.update(dir);
		if(i % printEvery == 0) {
			part.print();
		}
	}*/

	part.print();
}


static void testSwarm(OP &op) {
	bool useCuda = true;
	XM xm;
	// TODO: Create particles based on warp size?
	const int size = 20;
	const int dim = 2; // OP decision space dimension
	Swarm swarm(size, dim, op);
	int generations = 1000;

	std::cout << "First generation:" << std::endl;
	swarm.print();


	xm.startSwarm(std::chrono::high_resolution_clock::now());
	//if(useCuda){
	for(int i = 0; i < generations; ++i) {
		cudaError_t cudaStatus = cudaGetLastError();
		if(cudaStatus != cudaSuccess) {
			fprintf(stderr, "PS before update failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		//swarm.updateParticlePositions<<<1,1>>>();
		swarm.updateParticlePositions();
		cudaStatus = cudaGetLastError();
		if(cudaStatus != cudaSuccess) {
			fprintf(stderr, "PS after update failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		// Wait for GPU to finish before accessing on host
		cudaDeviceSynchronize();
	}
	/*} else {
		swarm.updateParticlePositions();
	}*/
	xm.endSwarm(std::chrono::high_resolution_clock::now());
	swarm.end();


	for(int i = 0; i < size; i++) {
		for(size_t j = 1; j < dim; ++j) {
			std::cout << "(" << swarm.xx[i*j - 1 + j] << "," << swarm.xx[i*j + j] << ")" << std::endl;
		}
	}

	std::cout << "Last generation:" << std::endl;
	swarm.print();


	std::cout << "-- " << op.name << " problem --" << std::endl;
	std::cout << "Running the swarm took: " << xm.swarmDuration << " s." << std::endl;
	std::cout << "of which " << std::endl;
	std::cout << "initialization      " << swarm.initTimeMicS << " micro seconds." << std::endl;
	std::cout << "updating best value " << swarm.updateBestTimeMicS << " micro seconds." << std::endl;
	std::cout << "updating particles  " << swarm.updateParticlesTimeMicS << " micro seconds." << std::endl;
	std::cout << "of which" << std::endl;
	std::cout << "updating positions       " << swarm.updatePosTimeMicS << " micro seconds." << std::endl;
	std::cout << "updating velocities      " << swarm.updateVelTimeMicS << " micro seconds." << std::endl;
	std::cout << "updating function values " << swarm.updateFunTimeMicS << " micro seconds." << std::endl;
	std::cout << "Total function evaluations " << swarm.fEvals << std::endl;

	// Print out csv-style to be pasted in Excel
	char sep = ';';
	// Few words how this version is different from base version.
	std::string upgrade = "Base";
	std::cout 
		<< std::fixed 
		<< upgrade << sep 
		<< generations << sep
		<< size << sep
		<< xm.swarmDuration * 1000000 << sep
		<< swarm.initTimeMicS << sep
		<< swarm.updateBestTimeMicS << sep
		<< swarm.updateParticlesTimeMicS << sep
		<< swarm.updatePosTimeMicS << sep
		<< swarm.updateVelTimeMicS << sep
		<< swarm.updateFunTimeMicS << sep
		<< swarm.fEvals 
		<< std::endl;
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
