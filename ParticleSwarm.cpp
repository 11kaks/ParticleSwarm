// ParticleSwarm.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "OP.h"
#include "Rastriging.h"
#include "Particle.h"
#include "Swarm.h"
#include "XM.h"

#include <iostream>
#include <vector>


static void testRastriging(OP &op) {
	std::vector<std::vector<float>> range = op.getSearchRange();	
	std::cout << "Search range:" << std::endl;

	for(int i = 0; i < range.size(); i++) {
		std::cout << range[i][0]  << " <= " << "x" << i+1 << " <= " << range[i][1] << std::endl;		
	}
	std::vector<float> point = { 0.0f,0.0f };
	std::cout << "f(" << point[0] << "," << point[1] << ") = " << op.evaluate(point) << std::endl;
}

static void testParticle(OP &op) {
	std::vector<float> x = { 0.f,0.f }; // val should be 0 at (0,0)
	std::vector<float> v = { 0.2f,0.1f };
	Particle part(x, v, op);
	// Print in initial position
	part.print();
	// Move the particle around for a few times
	int maxRounds = 2000;
	// Print every now and then
	int printEvery = 100;
	// Guide the particle towards origin at all times
	std::vector<float> dir = { 0.f,0.f };

	for(int i = 0; i < maxRounds; i++) {
		part.update(dir);
		if(i % printEvery == 0) {
			part.print();
		}
	}

	part.print();
}

static void testSwarm(OP &op) {
	XM xm;
	// TODO: Create particles based on warp size?
	int amountParticles = 20;
	Swarm swarm(amountParticles, op);
	int generations = 1000;

	std::cout << "First generation:" << std::endl;
	swarm.print();

	xm.startSwarm(std::chrono::high_resolution_clock::now());
	for(int i = 0; i < generations; ++i) {
		swarm.updateParticlePositions();
	}
	xm.endSwarm(std::chrono::high_resolution_clock::now());
	swarm.end();

	std::cout << "Last generation:" << std::endl;
	swarm.print();

	std::cout << "Running the swarm took: " << xm.swarmDuration << " s." << std::endl;
	std::cout << "of which " << std::endl;
	std::cout << "initialization      " << swarm.initTimeMicS << " micro seconds." << std::endl;
	std::cout << "updating best value " << swarm.updateBestTimeMicS << " micro seconds." << std::endl;
	std::cout << "updating particles  " << swarm.updateParticlesTimeMicS << " micro seconds." << std::endl;
	std::cout << "of which" << std::endl;
	std::cout << "updating positions       " << swarm.updatePosTimeMicS << " micro seconds." << std::endl;
	std::cout << "updating velocities      " << swarm.updateVelTimeMicS << " micro seconds." << std::endl;
	std::cout << "updating function values " << swarm.updateFunTimeMicS << " micro seconds." << std::endl;
	std::cout << "Total function evaluations " << swarm.fEvals << std::endl;
}

int main()
{
	srand(145623);
	Rastriging problem;
	testSwarm(problem);
}
