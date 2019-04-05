#pragma once

#include <iostream>

#include "OP.h"

/*
Rastrigin function
*/
class Rastriging : public OP
{
public:
	Rastriging();
	~Rastriging();

	std::vector<std::vector<float>> getSearchRange();
};

