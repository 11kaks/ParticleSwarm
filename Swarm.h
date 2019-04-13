#pragma once

#include <iostream>
#include <vector>
#include <stdlib.h>     /* srand, rand */

#include "Particle.h"


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Swarm
{
public:

	std::vector<Particle*> particles;

	float *xx;

	const size_t decDim;
	const size_t size;

	int initTimeMicS = 0;
	int updateBestTimeMicS = 0;
	int updateParticlesTimeMicS = 0;
	int updateVelTimeMicS = 0;
	int updatePosTimeMicS = 0;
	int updateFunTimeMicS = 0;

	/* Best value of current generation. */
	float bestVal = 100000;
	/* Index of the best particle. */
	int bestParticleIdx = 0;

	int fEvals = 0;

	Swarm(std::size_t size, const std::size_t dim, OP &problem);
	~Swarm();

	/**
	Print best value of the swarm.
	*/
	void print();

	/**
	Print particles of the swarm.
	*/
	void printParticles();

	/**
	 * Update all particles in the list using global or local paradigm.
	 */
	__host__ void updateParticlePositions();

	void posToList();


	/**
	Update execution times from particles.
	Call after optimization complete.
	*/
	void end() {
		for(Particle *p : particles) {
			updateVelTimeMicS += p->updateVelTimeMicS;
			updatePosTimeMicS += p->updatePosTimeMicS;
			updateFunTimeMicS += p->updateFunTimeMicS;
		}
	}


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

