#include "pch.h"
#include "Particle.h"

#include <iostream>
#include <vector>
#include <stdlib.h> 

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

void Particle::update(std::vector<float> direction) {
	updateVelocity(direction);
	updatePosition();
	updateFuncValue();
}

void Particle::updateVelocity(std::vector<float> direction) {
	vOld = v;
	for(int i = 0; i < x.size(); i++) {
		v[i] = w * vOld[i] + c1 * rnd01() * (xBest[i] - x[i]) + c2 * rnd01() * (direction[i] - x[i]);
		if(v[i] > maxVel) {
			v[i] = maxVel;
		} else if(v[i] < -maxVel) {
			v[i] = -maxVel;
		}
	}
}

void Particle::updatePosition() {
	xOld = x;
	for(int i = 0; i < x.size(); i++) {
		x[i] = xOld[i] + v[i];
	}
}

void Particle::updateFuncValue() {
	fVal = op.evaluate(x);
	if(fVal < valBest) {
		valBest = fVal;
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