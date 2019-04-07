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
	
	 /* initialize random 
	 TODO: different seed for every particle
	 */
	srand(11354);
}


Particle::~Particle() {
}
