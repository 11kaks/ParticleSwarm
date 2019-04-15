#include "Particle.h"

#include <iostream>
#include <vector>
#include <stdlib.h> 
#include <chrono>

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

//void Particle::update(std::vector<float> direction) {
//	updateVelocity(direction);
//	updatePosition();
//	updateFuncValue();
//}

void Particle::updateVelocity(std::vector<float> direction) {
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	//vOld = v;
	for(int i = 0; i < x.size(); i++) {
		v[i] = w * v[i] + c1 * rnd01() * (xBest[i] - x[i]) + c2 * rnd01() * (direction[i] - x[i]);
		if(v[i] > maxVel) {
			v[i] = maxVel;
		} else if(v[i] < -maxVel) {
			v[i] = -maxVel;
		}
	}
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	updateVelTimeMicS += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

void Particle::updatePosition() {
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	//xOld = x;
	for(int i = 0; i < x.size(); i++) {
		x[i] = x[i] + v[i];
	}
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	updatePosTimeMicS += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

void Particle::updateFuncValue() {
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	// If penaltys are included, call to op.evaluate() will cause
	// more than one function evaluation, so this is not actually 
	// correct.
	fVal = op.evaluate(x);
	fEvals++;

	if(fVal < valBest) {
		valBest = fVal;
		xBest = x;
	}
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	updateFunTimeMicS += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

float Particle::rnd01() {
	return rand() / (RAND_MAX + 1.f);
}

void Particle::print() {
	std::cout << "Curr pos (" << x[0] << "," << x[1] << ")" << std::endl;
	std::cout << "Curr vel (" << v[0] << "," << v[1] << ")" << std::endl;
	std::cout << "Curr val " << op.evaluate(x) << std::endl;
}