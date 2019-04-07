#pragma once

#include <vector>

#define PI 3.14159265

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
	virtual std::vector<std::vector<float>> getSearchRange() = 0;

	/**
	  Dimension of the decision space.
	 */
	int getDecDimension();

	/**
	  Dimension of the objective space.
	 */
	int getObjDimension();

	/**
	  Evaluate objective functions at point x.
	 
	  @param x Point in decision space.
	 
	  @return Objective function value at given point.
	 */
	virtual float evaluateOriginalObjective(std::vector<float> &point) = 0;

	/**
	  Evaluate the amount of penalty at given point.
	 
	  @param x Point in desicion space.
	 
	  @return Amount of penalty at given point.
	 */
	virtual float evaluatePenalty(std::vector<float> &point) = 0;

	/**
	  Evaluates the objective function value at given point.
	  Implementing class desides if there is any penalty included in the
	  evaluation. For evaluation of just the original objective, use
	  evaluateOriginalObjectives().
	 
	  @param x Point in decision space.
	 
	  @return Objective function value + possible penalty at given point.
	 */
	float evaluate(std::vector<float> &point);

private:
	/* Decision space dimension. */
	int decDim;
	/* Objective space dimension. */
	int objDim = 1;
};

