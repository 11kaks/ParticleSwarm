#include "pch.h"
#include "Rastriging.h"


Rastriging::Rastriging() {
}


Rastriging::~Rastriging() {
}

std::vector<std::vector<float>> Rastriging::getSearchRange() {
	std::cout << "Call for getSearchRange() in Rastriging-class" << std::endl;
	std::vector<std::vector<float>> res = { {1.1f,2.2f}, {3.3f,4.4f} };
	return res;
}
