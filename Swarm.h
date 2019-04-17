#pragma once

#include <iostream>
#include <vector>
// For non-CUDA random numbers
#include <stdlib.h>

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
	/* List of particles. */
	std::vector<Particle*> particles;
	/* Particle with best value in current generation. */
	Particle* bestParticle;
	/* Best value of current generation. Initialized to some big number. */
	float bestVal = 100000;

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
	list of positions so it's a true 1D array with decDim values. */
	float *xbg;

	/* Store curand random value states. */
	curandState_t *RNGstate;
	/* Problem's decision space dimension. */
	const size_t decDim;
	/* Amount of particles. */
	const size_t size;

	/* Duration of initialization. */
	std::chrono::microseconds durInit = std::chrono::microseconds();
	/* Duration of position and velocity updates. */
	std::chrono::microseconds durPPosVel = std::chrono::microseconds();
	/* Duration of function evaluations. */
	std::chrono::microseconds durPFun = std::chrono::microseconds();
	/* Duration of making memory copys. Doesn't apply if CUDA not used. */
	std::chrono::microseconds durMemcpy = std::chrono::microseconds();
	/* Duration of searching the best value of current generation. */
	std::chrono::microseconds durUBest = std::chrono::microseconds();

	/**
	Initializes a new swarm. Creates size amount of particles to random
	positions with zero velocity.

	@param size Amount of particles.
	@param dim  Problem dimension.
	@param problem Problem to be optimized. Passed to all particles.
	@param CUDAposvel Should CUDA be used to particles' position and velocity
	updates. Causes initialization of curand RNG which takes a lot of time.
	*/
	__host__ Swarm(std::size_t size, const std::size_t dim, OP &problem, bool CUDAposvel);
	__host__ ~Swarm();

	/**
	 Updates all particles (position, velocity, function value) in the swarm.
	 If CUDAposvel = true, it uses CUDA for position and velocity updates and
	 normal CPU otherwise. Function evaluations are always done on the CPU.

	 @param gridSize  Specify gridsize to be used is CUDAposvel = true.
	 Grid should be shaped like dim3(1, y, 1) i.e. one block wide because of
	 how thread indicies are used in the kernel code.

	 @param blockSize Specify block size of shape dim3(dim, y, 1) where dim is
	 the dimension of the problem. Thread's x coordinate is used as an index for
	 position and velocity.
	 Using blocks with less or equal to 256 threads may somethimes yield better
	 accuracy because of how curand RNG works. I'm not sure about it though.
	 */
	__host__ void updateParticles(dim3 gridSize, dim3 blockSize);
	/**
	Copy particles to arrays for CUDA. Executed only if CUDAposvel = true.
	*/
	__host__ void particlesToArrays();
	/**
	Copy data back from arrays to particles for other calculations. Executed only if CUDAposvel = true.
	*/
	__host__ void arraysToParticles();
	/**
	Print best value of the swarm.
	*/
	__host__ void print();
	/**
	Print particles of the swarm.
	*/
	__host__ void printParticles();


private:
	/* Use CUDA for position and velocity calculations. */
	const bool CUDAposvel;

	/**
	* Update current generation's best value to the swarm.
	*/
	__host__ void updateBest();

	/**
	Random number between M and N. (For non-CUDA.)
	*/
	__host__ float randMToN(float M, float N);
};

