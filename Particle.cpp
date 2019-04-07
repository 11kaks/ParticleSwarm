#include "pch.h"
#include "Particle.h"


Particle::Particle(std::vector<float> initialX, std::vector<float> initialV, OP &problem) :
	x(initialX),
	xOld(initialX),
	xBest(initialX),
	v(initialV),
	vOld(initialV),
	op(problem)
{
	fVal = problem.evaluate(initialX);
	valBest = fVal;
}


Particle::~Particle() {
}
