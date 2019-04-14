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
	xb = new float[size*dim];
	vv = new float[size*dim];
	xbg = new float[dim];

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
	delete[] xb;
	delete[] xbg;
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
			xb[i*j + j] = p->xBest[j];
			vv[i*j + j] = p->v[j];
		}
	}
}

void Swarm::arraysToParticles() {
	for(int i = 0; i < size; i++) {
		Particle *p = particles[i];
		for(size_t j = 0; j < decDim; ++j) {
			p->x[j] = xx[i*j + j];
			p->xBest[j] = xb[i*j + j];
			p->v[j] = vv[i*j + j];
		}
	}
}

#define C1 0.3
#define C2 0.3
#define W 0.8

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__global__ void updateParticleVelocityKernel(float *vel, float *pos, float *best, float *gb, size_t pitch, size_t size, size_t dim) {

	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;

	if((tidx < dim) && (tidy < size)) {
		float *posRow = (float *)((char*)pos + tidy * pitch);
		//posRow[tidx] = 15.f;
		float *bestRow = (float *)((char*)best + tidy * pitch);
		//bestRow[tidx] = 1.f;
		float *gbRow = (float *)((char*)best + 1 * pitch);
		//gbRow[tidx] = 1.f;
		float *velRow = (float *)((char*)vel + tidy * pitch);
		//velRow[tidx] = 0.5f;

		//
		//v[i] = w * vOld[i] + c1 * rnd01() * (xBest[i] - x[i]) + c2 * rnd01() * (direction[i] - x[i]);
		// placeholder for a random number
		float rnd = 1.1;
		velRow[tidx] = W * velRow[tidx] + C1 * rnd * (bestRow[tidx] - posRow[tidx]) + C2 * rnd * (gbRow[tidx] - posRow[tidx]);
		// update position
		posRow[tidx] = posRow[tidx] + velRow[tidx];
		//
	}
}

/*
https://stackoverflow.com/questions/39006348/accessing-class-data-members-from-within-cuda-kernel-how-to-design-proper-host
*/
__host__ void Swarm::updateParticlePositions(bool CUDAposvel) {

	if(CUDAposvel) {
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		cudaError_t cudaStatus;

		float * devxx;
		float * devxb;
		float * devxbg;
		float * devvv;
		size_t pitch;
		size_t sSize = (size_t)size;

		// Guide all particles towards the current best position.
		//Particle *best = particles.at(bestParticleIdx);

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
		// Allocate device best position array
		cudaStatus = cudaMallocPitch(&devxb, &pitch, decDim * sizeof(float), sSize);
		if(cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMallocPitch position failed!");
			goto Error;
		}
		// Allocate device global best position 
		cudaStatus = cudaMallocPitch(&devxbg, &pitch, decDim * sizeof(float), 1);
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
		// Copy best positions to device array
		cudaStatus = cudaMemcpy2D(devxb, pitch, xb, decDim * sizeof(float), decDim * sizeof(float), size, cudaMemcpyHostToDevice);
		if(cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		// Copy global best position to device
		cudaStatus = cudaMemcpy2D(devxbg, pitch, xbg, decDim * sizeof(float), decDim * sizeof(float), 1, cudaMemcpyHostToDevice);
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

		dim3 gridSize(iDivUp(decDim, BLOCKSIZE_x), iDivUp(size, BLOCKSIZE_y));
		dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if(cudaStatus != cudaSuccess) {
			fprintf(stderr, "2  failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// Kernel call
		updateParticleVelocityKernel << <gridSize, blockSize >> > (devvv, devxx, devxb, devxbg, pitch, size, decDim);

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
		// Copy best position array from GPU buffer to host memory.
		cudaStatus = cudaMemcpy2D(xb, decDim * sizeof(float), devxb, pitch, decDim * sizeof(float), size, cudaMemcpyDeviceToHost);
		if(cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy back to host failed!");
			goto Error;
		}
		// Copy global best position from GPU buffer to host memory.
		cudaStatus = cudaMemcpy2D(xbg, decDim * sizeof(float), devxbg, pitch, decDim * sizeof(float), 1, cudaMemcpyDeviceToHost);
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

	Error:
		cudaFree(devxx);
		cudaFree(devxb);
		cudaFree(devvv);

		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		updateParticlesTimeMicS += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

		updateBest();

		return;

	// Without CUDA
	} else {

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

		// Guide all particles towards the current best position.
		Particle *best = particles.at(bestParticleIdx);

		for(Particle *p : particles) {
			//p->update(best->x);
			p->updateVelocity(best->x);
			p->updatePosition();
			p->updateFuncValue();
		}

		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		updateParticlesTimeMicS += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

		updateBest();
	}


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

	// This update is for CUDA kernel methods.
	Particle *bestParticle = particles[bestParticleIdx];
	for(size_t i = 0; i < decDim; ++i) {
		xbg[i] = bestParticle->x[i];
	}

	//std::cout << "update best (" << xbg[0] << "," << xbg[1] <<")" << std::endl;

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