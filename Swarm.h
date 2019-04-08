#pragma once

#include <iostream>
#include <vector>
#include <stdlib.h>     /* srand, rand */

#include "Particle.h"

class Swarm
{
public:

	std::vector<Particle*> particles;

	/* Best value of current generation. */
	float bestVal = 100000;
	/* Index of the best particle. */
	int bestParticleIdx = 0;

	Swarm(std::size_t size, OP &problem);
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
	void updateParticlePositions();


private:
	/* Optimization problem. */
	OP &op;
	/* Global or local paradigm. */
	const bool useGlobal = true;


	/**
	* Update current generation's best value to the swarm.
	*/
	void updateBest();

	/**
	Random number between M and N.
	*/
	float randMToN(float M, float N);

};

