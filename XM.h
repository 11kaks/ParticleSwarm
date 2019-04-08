#pragma once

#include <chrono>

/**
Execution Monitor

Save execution times and other performance info like 
amount of function calls.
*/
class XM
{
public:

	std::chrono::steady_clock::time_point swarmStart;
	float swarmDuration;

	XM();
	~XM();

	void startSwarm(std::chrono::steady_clock::time_point tp) {
		swarmStart = tp;
	}

	void endSwarm(std::chrono::steady_clock::time_point tp) {
		swarmDuration = std::chrono::duration_cast<std::chrono::microseconds>(tp - swarmStart).count() / 1000000.f;
	}

};

