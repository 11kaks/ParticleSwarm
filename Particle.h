#pragma once

#include "OP.h"

#include <iostream>
#include <vector>

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

	Particle(std::vector<float> initialX, std::vector<float> initialV, OP &problem);
	~Particle();

	/*Print something to standard stream. */
	void print() {
		std::cout << "Curr pos (" << x[0] << "," << x[1] << ")" << std::endl;
		std::cout << "Curr vel (" << v[0] << "," << v[1] << ")" << std::endl;
		std::cout << "Curr val " << op.evaluate(x) << std::endl;
	}

};

