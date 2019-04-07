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


	/**
	Create a new particle.

	@param initialX Initial position of the particle.
	@param initialV Initial velocity of the particle.
	@param problem  Optimization problem to be solved.
	*/
	Particle(std::vector<float> initialX, std::vector<float> initialV, OP &problem);
	~Particle();

	/**
	 * Update particle's position and velociy.
	 *
	 * @param towards Best local or global position towards which the particle
	 *                should accelerate.
	 * @param rnd     Random number generator.
	 */
	void update(std::vector<float> direction) {
		updateVelocity(direction);
		updatePosition();
		updateFuncValue();
	}

	/**
	 * Update velocity clamped to maxVel in any axis.
	 */
	void updateVelocity(std::vector<float> direction) {
		vOld = v;
		for(int i = 0; i < x.size(); i++) {
			v[i] = w * vOld[i] + c1 * rnd01() * (xBest[i] - x[i]) + c2 * rnd01() * (direction[i] - x[i]);
			if(v[i] > maxVel) {
				v[i] = maxVel;
			} else if(v[i] < -maxVel) {
				v[i] = -maxVel;
			}
		}
	}

	/**
	 * Update particle to a new position based on current velocity.
	 * The velocity must be calculated before calling this.
	 */
	void updatePosition() {
		xOld = x;
		for(int i = 0; i < x.size(); i++) {
			x[i] = xOld[i] + v[i];
		}
	}

	/**
	 * Update function value and set best if needed.
	 *
	 * This is the hard coded minimization problem, which takes into account
	 * the lower and upper bounds through a penalty function.
	 */
	void updateFuncValue() {
		fVal = op.evaluate(x);
		if(fVal < valBest) {
			valBest = fVal;
			xBest = x;
		}
	}

	/**
	Random number in range [0,1[
	*/
	float rnd01() {
		return rand() / (RAND_MAX + 1.f);
	}

	/*Print something to standard stream. */
	void print() {
		std::cout << "Curr pos (" << x[0] << "," << x[1] << ")" << std::endl;
		std::cout << "Curr vel (" << v[0] << "," << v[1] << ")" << std::endl;
		std::cout << "Curr val " << op.evaluate(x) << std::endl;
	}

private:
	/* Cognitive coefficient. */
	const float c1 = 2.0f;
	/* Social coefficient. */
	const float c2 = 2.0f;
	/* Inertia coefficient. */
	const float w = 1.0f;
	/* Maximum velocity along any coordinate axis. */
	const float maxVel = 3.0f;

};

