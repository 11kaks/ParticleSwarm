#pragma once

#include <vector>
#include <math.h>

#include "OP.h"

/*
Rastrigin function from: 
https://en.wikipedia.org/wiki/Test_functions_for_optimization

For testing f(0,0,...,0) = 0

*/
class Rastriging : public OP
{
public:
	Rastriging();
	/**
	Special constructor for Rastriging as it is defined in any-dimensional
	decision space.
	*/
	Rastriging(const int decDim);
	~Rastriging();

	std::vector<std::vector<float>> getSearchRange();

	std::vector<float> getKnownOptimumPoint();

	float evaluateOriginalObjective(std::vector<float> &point);

	float evaluatePenalty(std::vector<float> &point);

};

