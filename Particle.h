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
	/* Old position of the particle. */
	std::vector<float> xOld;
	/* Old velocity of the particle. */
	std::vector<float> vOld;
	/* Current value of the function. */
	float fVal;
	/* Particle's best position. */
	std::vector<float> xBest;
	/* Particle's best value corresponding to xBest.*/
	float valBest;
	/* Optimization problem. */
	OP &op;

	// Execution metrics
	int fEvals = 0;

	int updateVelTimeMicS = 0;
	int updatePosTimeMicS = 0;
	int updateFunTimeMicS = 0;

	/**
	Create a new particle.

	@param initialX Initial position of the particle.
	@param initialV Initial velocity of the particle.
	@param problem  Optimization problem to be solved.
	*/
	Particle(std::vector<float> initialX, std::vector<float> initialV, OP &problem);

	~Particle();

	/**
	  Update particle's position and velociy.

	  @param direction Best local or global position towards which the particle
					 should accelerate.
	 */
	void update(std::vector<float> direction);

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

	/**
	  Update velocity clamped to maxVel in any axis.
	 */
	void updateVelocity(std::vector<float> direction);

	/**
	  Update particle to a new position based on current velocity.
	  The new velocity must be calculated before calling this.
	 */
	void updatePosition();

	/**
	  Update function value and set best x and fVal if needed.
	 */
	void updateFuncValue();

	/**
	Random number in range [0,1[
	*/
	float rnd01();
};

