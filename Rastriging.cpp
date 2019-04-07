#include "pch.h"
#include "Rastriging.h"

#include <vector>
#include <math.h>


Rastriging::Rastriging() {
}


Rastriging::~Rastriging() {
}

std::vector<std::vector<float>> Rastriging::getSearchRange() {
	//std::cout << "Call for getSearchRange() in Rastriging-class" << std::endl;

	std::vector<std::vector<float>> res(decDim);
	float lim = 5.12f;

	for(int i = 0; i < decDim; i++) {
		std::vector<float> upLow(2);
		upLow[0] = -lim;
		upLow[1] = lim;
		res[i] = upLow;
	}

	return res;
}

/*
Definition := A*n + sum(x_i^2 - A cos(2*PI*x_i)) where A = 10
and n is the dimension of objective space.
*/
float  Rastriging::evaluateOriginalObjective(std::vector<float> &point) {
	int A = 10;
	float res = A * point.size();
	for(const auto& xi : point) {
		res += xi * xi - A * cos(2 * PI * xi);
	}
	return res;
}

/*
Rastring is unconstrained.
*/
float  Rastriging::evaluatePenalty(std::vector<float> &point) {
	return 0.f;
}
