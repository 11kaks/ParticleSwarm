#include "pch.h"
#include "Swarm.h"
#include "Particle.h"

#include <iostream>
#include <vector>
#include <stdlib.h>


Swarm::Swarm(std::size_t size, OP &problem) :
	op(problem) {
	particles.resize(size);
	std::vector<std::vector<float>> range = problem.getSearchRange();

	int dim = problem.getDecDimension();
	std::vector<float> x(dim);
	std::vector<float> v(x.size());
	// TODO: this could be parallellized. Propably not a big speedup.
	for(int k = 0; k < size; k++) {
		for(int i = 0; i < x.size(); ++i) {
			float a = range[i][0];
			float b = range[i][1];
			float randval = randMToN(a, b);
			x[i] = randval;
			v[i] = 0.f;
		}
		particles[k] = new Particle(x, v, problem);
	}
	updateBest();
}


Swarm::~Swarm() {
	for(Particle *p : particles) {
		delete p;
	}
}

void Swarm::print() {
	std::vector<float> x = particles.at(bestParticleIdx)->x;
	float best = particles.at(bestParticleIdx)->fVal;
	std::cout << "Swarm's best f(" << x[0] << "," << x[1] << ") = " << best << std::endl;
	std::cout << "---------------------------------------------" << std::endl;
}

void Swarm::printParticles() {
	for(Particle *p : particles) {
		p->print();
	}
}

void Swarm::updateParticlePositions() {
	// Guide all particles towards the current best position.
	Particle *best= particles.at(bestParticleIdx);
	// TODO: As particles are not related to each other
	// this update should be parallellized.
	for(Particle *p : particles) {
		p->update(best->x);
	}
	updateBest();
}

void Swarm::updateBest() {
	// TODO: can this be parallelized?
	// Anyways, need to wait for all particles to be updated.
	for(int i = 0; i < particles.size(); i++) {
		float val = particles.at(i)->fVal;
		if(val < bestVal) {
			bestParticleIdx = i;
			bestVal = val;
		}
	}
}

float Swarm::randMToN(float M, float N) {
	return M + (rand() / (RAND_MAX / (N - M)));
}
