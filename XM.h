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

	XM();
	~XM();
	/* 
	 Define float seconds. From:
	 https://stackoverflow.com/questions/14391327/how-to-get-duration-as-int-millis-and-float-seconds-from-chrono
	 Usage: 
	 float durSecs = std::chrono::duration_cast<fsec>(tp1 - tp2).count()
	*/
	typedef std::chrono::duration<float> fsec;

	std::chrono::high_resolution_clock::time_point swarmStart;
	float swarmDuration;


	void startSwarm(std::chrono::high_resolution_clock::time_point tp) {
		swarmStart = tp;
	}

	void endSwarm(std::chrono::high_resolution_clock::time_point tp) {
		swarmDuration = std::chrono::duration_cast<fsec>(tp - swarmStart).count();
	}

};

