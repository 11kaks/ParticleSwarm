#pragma once

#include <iostream>
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
	~Rastriging();

	std::vector<std::vector<float>> getSearchRange();

	float evaluateOriginalObjective(std::vector<float> &point);

	float evaluatePenalty(std::vector<float> &point);

private:
	int objDim = 1;
	int decDim = 2;
};

