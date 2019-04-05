#pragma once

#include <vector>

/*
Optimization Problem
*/
class OP
{
public:
	OP() {};
	~OP() {};

	/**
	  Search range of the problem. Used to generate initial random 
	  population inside the range. Does not imply that the range 
	  should be feasible.

	 
	  @return Two dimensional vector of doubles. Access double[i][j]
	          where i means i'th coordinate of the space and j = 0 or 1. j = 0
	          is the lower bound and j = 1 is the upper bound on i'th axis.
	 */
	virtual std::vector<std::vector<float>> getSearchRange()=0;

};

