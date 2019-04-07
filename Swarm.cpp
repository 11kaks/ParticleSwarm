#include "pch.h"
#include "Swarm.h"
#include "Particle.h"

#include <vector>
#include <stdlib.h>


Swarm::Swarm(std::size_t size, OP &problem) :
	op(problem) {
	srand(145623);
	particles.resize(size);
	std::vector<std::vector<float>> range = problem.getSearchRange();

	int dim = problem.getDecDimension();
	std::vector<float> x(dim);
	std::vector<float> v(x.size());
	for(int k = 0; k < size; k++) {
		for(int i = 0; i < x.size(); ++i) {
			// TODO: Rand keeps oscillating between two numbers why?????
			float a = range[i][0];
			float b = range[i][1];
			float diff = b - a;
			int r = rand();
			float randval = a + (r / (RAND_MAX / diff));
			x[i] = randval;
			v[i] = 0.f;
		}
		// Can I do this without pointers?
		particles[k] = new Particle(x, v, problem);
	}
	updateBest();
}


Swarm::~Swarm() {
	for(Particle *p : particles) {
		delete p;
	}
}