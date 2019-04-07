// ParticleSwarm.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "OP.h"
#include "Rastriging.h"

#include <iostream>
#include <vector>



static void test(OP &op) {
	std::vector<std::vector<float>> range = op.getSearchRange();	
	std::cout << "Search range:" << std::endl;

	for(int i = 0; i < range.size(); i++) {
		std::cout << range[i][0]  << " <= " << "x" << i+1 << " <= " << range[i][1] << std::endl;		
	}
	std::vector<float> point = { 0.0f,0.0f };
	std::cout << "f(" << point[0] << "," << point[1] << ") = " << op.evaluate(point) << std::endl;
}

int main()
{
	Rastriging problem;
	test(problem);
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
