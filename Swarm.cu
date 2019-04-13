#include "Swarm.h"
#include "Particle.h"

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <chrono>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCKSIZE_x 32
#define BLOCKSIZE_y 32

Swarm::Swarm(const std::size_t size, const std::size_t dim, OP &problem) :
	size(size),
	decDim(dim),
	op(problem)
{
	xx = new float[size*dim];
	/*for(int i = 0; i < size; ++i) {
		xx[i] = new float[dim];
	}*/

	particles.resize(size);
	std::vector<std::vector<float>> range = problem.getSearchRange();
	std::vector<std::vector<float>> particV(size);
	std::vector<std::vector<float>> particVOld(size);

	int probdim = problem.getDecDimension();
	std::vector<float> x(probdim);
	std::vector<float> v(x.size());

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	// TODO: this could be parallellized. Propably not a big speedup.
	for(int k = 0; k < size; k++) {
		for(int i = 0; i < x.size(); ++i) {
			float a = range[i][0];
			float b = range[i][1];
			float randval = randMToN(a, b);
			x[i] = randval;
			v[i] = 0.f;
		}
		particles[k] = new Particle(x, v, problem);
		
	}

	posToList();

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	initTimeMicS = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	for(int i = 0; i < size; i++) {
		for(size_t j = 1; j < dim; ++j) {
			std::cout << "(" << xx[i*j-1 + j] << "," << xx[i*j + j] << ")" << std::endl;
		}
	}

	updateBest();
}


Swarm::~Swarm() {
	for(Particle *p : particles) {
		delete p;
	}
	/*for(size_t i = 0; i < size; ++i) {
		delete[] xx[i];
	}*/
	delete[] xx;
}

int iDivUp(int hostPtr, int b) { 
	return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); 
}

void Swarm::posToList() {
	for(int i = 0; i < size; i++) {
		Particle *p = particles[i];
		for(size_t j = 0; j < decDim; ++j) {
			xx[i*j + j] = p->x[j];
		}
	}
}

__global__ void updateParticlePositionsKernel(float *buffer, size_t pitch, size_t size, size_t dim) {

	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;

	if((tidx < dim) && (tidy < size)) {
		float *row_a = (float *)((char*)buffer + tidy * pitch);
		row_a[tidx] = 15.f;// row_a[tidx] * tidx * tidy;
	}
}

/*
https://stackoverflow.com/questions/39006348/accessing-class-data-members-from-within-cuda-kernel-how-to-design-proper-host
*/
__host__ void Swarm::updateParticlePositions() {

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	cudaError_t cudaStatus;

	

	// pointer to particles vector
	//std::vector<Particle*> * pParticles = &particles;
	//std::vector<Particle*> * tempParticles = &pParticles[0];

	float * devxx;
	size_t pitch;
	size_t sSize = (size_t)size;

	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "before failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMallocPitch(&devxx, &pitch, decDim * sizeof(float), sSize);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocPitch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMemcpy2D(devxx, pitch, xx, decDim * sizeof(float), decDim * sizeof(float), size, cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Guide all particles towards the current best position.
	Particle *best= particles.at(bestParticleIdx);

	dim3 gridSize(iDivUp(decDim, BLOCKSIZE_x), iDivUp(size, BLOCKSIZE_y));
	dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "2  failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	updateParticlePositionsKernel << <gridSize, blockSize >> > (devxx, pitch, size, decDim);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "updateParticlePositionsKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching updateParticlePositionsKernel!\n", cudaStatus);
		goto Error;
	}


	// Copy output vector from GPU buffer to host memory.
	//cudaStatus = cudaMemcpy(xx, devxx, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy2D(xx, decDim * sizeof(float), devxx, pitch, decDim * sizeof(float), size, cudaMemcpyDeviceToHost);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy back to host failed!");
		goto Error;
	}

	// TODO: As particles are not related to each other
	// this update should be parallellized.
	for(Particle *p : particles) {
		p->update(best->x);
	}

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	updateParticlesTimeMicS += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();


	updateBest();

Error:
	cudaFree(devxx);

	return;
}

__host__ void Swarm::updateBest() {
	// TODO: can this be parallelized?
	// Anyways, need to wait for all particles to be updated.

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	for(int i = 0; i < particles.size(); i++) {
		Particle *p = particles.at(i);
		float val = p->fVal;
		fEvals += p->fEvals;
		if(val < bestVal) {
			bestParticleIdx = i;
			bestVal = val;
		}
	}
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	updateBestTimeMicS += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

float Swarm::randMToN(float M, float N) {
	return M + (rand() / (RAND_MAX / (N - M)));
}

void Swarm::print() {
	std::vector<float> x = particles.at(bestParticleIdx)->x;
	float best = particles.at(bestParticleIdx)->fVal;
	std::cout << "Swarm's best f(" << x[0] << "," << x[1] << ") = " << best << std::endl;
	std::cout << "---------------------------------------------" << std::endl;
}

void Swarm::printParticles() {
	for(Particle *p : particles) {
		p->print();
	}
}