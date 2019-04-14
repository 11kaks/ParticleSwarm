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
	op(problem) {
	xx = new float[size*dim];
	vv = new float[size*dim];
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

	/* It is enough to copy the particle velocities to the array once. They will be ok after that. */
	particlesToArrays();

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	initTimeMicS = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	updateBest();
}


Swarm::~Swarm() {
	for(Particle *p : particles) {
		delete p;
	}
	delete[] xx;
	delete[] vv;
}

int iDivUp(int hostPtr, int b) {
	return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b);
}

void Swarm::particlesToArrays() {
	for(int i = 0; i < size; i++) {
		Particle *p = particles[i];
		for(size_t j = 0; j < decDim; ++j) {
			xx[i*j + j] = p->x[j];
			vv[i*j + j] = p->v[j];
		}
	}
}

void Swarm::arraysToParticles() {
	for(int i = 0; i < size; i++) {
		Particle *p = particles[i];
		for(size_t j = 0; j < decDim; ++j) {
			p->x[j] = xx[i*j + j];
			p->v[j] = vv[i*j + j];
		}
	}
}


__global__ void updateParticlePositionsKernel(float *vel, float *pos, size_t pitch, size_t size, size_t dim) {

	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;

	if((tidx < dim) && (tidy < size)) {
		float *posRow = (float *)((char*)pos + tidy * pitch);
		posRow[tidx] = 15.f;
		float *velRow = (float *)((char*)vel + tidy * pitch);
		velRow[tidx] = 0.5f;
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
	float * devvv;
	size_t pitch;
	size_t sSize = (size_t)size;

	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "before failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	// Allocate device position array
	cudaStatus = cudaMallocPitch(&devxx, &pitch, decDim * sizeof(float), sSize);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocPitch position failed!");
		goto Error;
	}
	// Allocate device velocity array
	cudaStatus = cudaMallocPitch(&devvv, &pitch, decDim * sizeof(float), sSize);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocPitch velocity failed!");
		goto Error;
	}

	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocPitch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	// Copy positions to device array
	cudaStatus = cudaMemcpy2D(devxx, pitch, xx, decDim * sizeof(float), decDim * sizeof(float), size, cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	// Copy velocities to device array
	cudaStatus = cudaMemcpy2D(devvv, pitch, vv, decDim * sizeof(float), decDim * sizeof(float), size, cudaMemcpyHostToDevice);
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
	Particle *best = particles.at(bestParticleIdx);

	dim3 gridSize(iDivUp(decDim, BLOCKSIZE_x), iDivUp(size, BLOCKSIZE_y));
	dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "2  failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Kernel call
	updateParticlePositionsKernel << <gridSize, blockSize >> > (devvv, devxx, pitch, size, decDim);

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


	// Copy position array from GPU buffer to host memory.
	cudaStatus = cudaMemcpy2D(xx, decDim * sizeof(float), devxx, pitch, decDim * sizeof(float), size, cudaMemcpyDeviceToHost);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy back to host failed!");
		goto Error;
	}

	// Copy velocity array from GPU buffer to host memory.
	cudaStatus = cudaMemcpy2D(vv, decDim * sizeof(float), devvv, pitch, decDim * sizeof(float), size, cudaMemcpyDeviceToHost);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy back to host failed!");
		goto Error;
	}

	/* The particles have to be updated after every time that there has been changes to
	positions so that the function evaluation can be done on CPU-side.*/
	arraysToParticles();

	for(Particle *p : particles) {
		p->update(best->x);
	}

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	updateParticlesTimeMicS += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	updateBest();

Error:
	cudaFree(devxx);
	cudaFree(devvv);

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