#include "pch.h"
#include "Swarm.h"
#include "Particle.h"

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <chrono>


Swarm::Swarm(std::size_t size, OP &problem) :
	op(problem) 
{
	particles.resize(size);
	std::vector<std::vector<float>> range = problem.getSearchRange();

	int dim = problem.getDecDimension();
	std::vector<float> x(dim);
	std::vector<float> v(x.size());

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
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
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	initTimeMicS = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	updateBest();
}


Swarm::~Swarm() {
	for(Particle *p : particles) {
		delete p;
	}
}

void Swarm::updateParticlePositions() {

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	// Guide all particles towards the current best position.
	Particle *best= particles.at(bestParticleIdx);

	// TODO: As particles are not related to each other
	// this update should be parallellized.
	for(Particle *p : particles) {
		p->update(best->x);
	}
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	updateParticlesTimeMicS += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	updateBest();
}

void Swarm::updateBest() {
	// TODO: can this be parallelized?
	// Anyways, need to wait for all particles to be updated.

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	for(int i = 0; i < particles.size(); i++) {
		Particle *p = particles.at(i);
		float val = p->fVal;
		fEvals += p->fEvals;
		if(val < bestVal) {
			bestParticleIdx = i;
			bestVal = val;
		}
	}
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	updateBestTimeMicS += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

float Swarm::randMToN(float M, float N) {
	return M + (rand() / (RAND_MAX / (N - M)));
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