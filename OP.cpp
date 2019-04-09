#include "OP.h"

#include <vector>

int OP::getDecDimension() {
	return decDim;
}

int OP::getObjDimension() {
	return objDim;
}

float OP::evaluate(std::vector<float> &point) {
	float fVal = evaluateOriginalObjective(point);
	float penalty = evaluatePenalty(point);
	return fVal + penalty;
}
