#include "Particle.h"

#include <iostream>
#include <vector>
#include <stdlib.h>

Particle::Particle(std::vector<float> initialX, std::vector<float> initialV, OP &problem) :
	x(initialX),
	xBest(initialX),
	v(initialV),
	op(problem)
{
	fVal = problem.evaluate(initialX);
	fValBest = fVal;
}


Particle::~Particle() {
}

void Particle::updateVelPos(std::vector<float> direction) {
	for(int i = 0; i < x.size(); i++) {
		v[i] = w * v[i] + c1 * rnd01() * (xBest[i] - x[i]) + c2 * rnd01() * (direction[i] - x[i]);
		if(v[i] > maxVel) {
			v[i] = maxVel;
		} else if(v[i] < -maxVel) {
			v[i] = -maxVel;
		}
		x[i] = x[i] + v[i];
	}
}

void Particle::updateFuncValue() {

	fVal = op.evaluate(x);

	if(fVal < fValBest) {
		fValBest = fVal;
		xBest = x;
	}
}

float Particle::rnd01() {
	return rand() / (RAND_MAX + 1.f);
}

void Particle::print() {
	std::cout << "Curr pos (" << x[0] << "," << x[1] << ")" << std::endl;
	std::cout << "Curr vel (" << v[0] << "," << v[1] << ")" << std::endl;
	std::cout << "Curr val " << op.evaluate(x) << std::endl;
}