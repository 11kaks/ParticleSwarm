#pragma once

#include "OP.h"

#include <iostream>
#include <vector>
#include <stdlib.h>     /* srand, rand */

/**
Particle with position and velocity.
*/
class Particle
{
public:
	/* Current position of the particle. */
	std::vector<float> x;
	/* Current velocity of the particle. */
	std::vector<float> v;
	/* Current value of the function. */
	float fVal;
	/* Particle's best position ever encountered. */
	std::vector<float> xBest;
	/* Particle's best value corresponding to xBest.*/
	float fValBest;

	/**
	Create a new particle.

	@param initialX Initial position of the particle.
	@param initialV Initial velocity of the particle.
	@param problem  Optimization problem to be solved.
	*/
	Particle(std::vector<float> initialX, std::vector<float> initialV, OP &problem);

	~Particle();

	/**
	  Update velocity clamped to maxVel in any axis.
	 */
	void updateVelPos(std::vector<float> direction);	
	/**
	  Update function value and set xBest and fValBest if needed.
	 */
	void updateFuncValue();

	/**
	Print something to standard stream. 
	*/
	void print();

private:
	/* Cognitive coefficient. Bigger value guides the particle towards 
	it's best position found previously.*/
	const float c1 = 0.3f;
	/* Social coefficient. Bigger value guides the particle towards 
	swarm's best position. */
	const float c2 = 0.3f;
	/* Inertia coefficient. Bigger value makes the particle want 
	to stay nearer to previous position. */
	const float w = 0.8f;
	/* Maximum velocity along any coordinate axis. */
	const float maxVel = 1.0f;

	/* Optimization problem. */
	OP &op;

	/**
	Random number in range [0,1[
	*/
	float rnd01();
};

