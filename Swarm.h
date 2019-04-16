#pragma once

#include <iostream>
#include <vector>
#include <stdlib.h>     /* srand, rand */

#include "Particle.h"

// Clock
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// For RNG
#include <cuda.h>
#include <curand_kernel.h>

class Swarm
{
public:

	std::vector<Particle*> particles;
	Particle* bestParticle;
	/* Particle positions as flattened 2D array. Particle i's position
	along axis j is at index i*j + j.*/
	float *xx;
	/* Particle velocities as flattened 2D array. Particle i's velocity
	along axis j is at index i*j + j.*/
	float *vv;
	/* Particle's best personal position. Particle i's best position
	along axis j is at index i*j + j.*/
	float *xb;
	/* The global best position of the whole swarm. A single position, not a 
	list of positions so it's a 1D array. */
	float *xbg;

	curandState_t *RNGstate;

	const size_t decDim;
	const size_t size;

	std::chrono::microseconds durInit = std::chrono::microseconds();
	std::chrono::microseconds durPPosVel = std::chrono::microseconds();
	std::chrono::microseconds durPFun = std::chrono::microseconds();
	std::chrono::microseconds durMemcpy = std::chrono::microseconds();
	std::chrono::microseconds durUBest = std::chrono::microseconds();

	
	/* Best value of current generation. */
	float bestVal = 100000;

	Swarm(std::size_t size, const std::size_t dim, OP &problem);
	~Swarm();

	/**
	Print best value of the swarm.
	*/
	__host__ void print();

	/**
	Print particles of the swarm.
	*/
	__host__ void printParticles();

	/**
	 * Update all particles in the list using global or local paradigm.
	 */
	__host__ void updateParticlePositions(bool CUDAposvel, dim3 gridSize, dim3 blockSize);

	void particlesToArrays();
	void arraysToParticles();


private:
	/* Optimization problem. */
	OP &op;
	/* Global or local paradigm. */
	const bool useGlobal = true;


	/**
	* Update current generation's best value to the swarm.
	*/
	__host__ void updateBest();

	/**
	Random number between M and N.
	*/
	float randMToN(float M, float N);

};

