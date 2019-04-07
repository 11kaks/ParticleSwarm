#pragma once

#include <iostream>
#include <vector>
#include <stdlib.h>     /* srand, rand */

#include "Particle.h"

class Swarm
{
public:

	std::vector<Particle*> particles;

	/* Best value of current generation. */
	float bestVal = FLT_MAX;
	/* Index of the best particle. */
	int bestParticleIdx = 0;

	Swarm(std::size_t size, OP &problem);
	~Swarm();

	void print() {
		for(Particle *p : particles) {
			p->print();
		}
		std::vector<float> x = particles.at(bestParticleIdx)->x;
		float best = particles.at(bestParticleIdx)->fVal;
		std::cout << "Swarm's best f(" << x[0] << "," << x[1] << ") = " << best << std::endl;
		std::cout << "---------------------------------------------" << std::endl;
	}

	/**
	 * Update all particles in the list using global or local paradigm.
	 */
	void updateParticlePositions() {
		//if(useGlobal) {
		for(Particle *p : particles) {
			p->update(particles.at(bestParticleIdx)->x);
		}
		//} 
		/*else {
			for(int i = 0; i < particles.size(); i++) {
				particles.get(i).update(particles.get(localBestIdx(i, 2)).getX());
			}
		}*/
		updateBest();
	}


private:
	/* Optimization problem. */
	OP &op;
	/* Global or local paradigm. */
	const bool useGlobal = true;


	/**
	* Update current generation's best value.
	*/
	void updateBest() {
		for(int i = 0; i < particles.size(); i++) {
			float val = particles.at(i)->fVal;
			if(val < bestVal) {
				bestParticleIdx = i;
				bestVal = val;
			}
		}
	}

	float randMToN(float M, float N) {
		return M + (rand() / (RAND_MAX / (N - M)));
	}

};

