#include "Swarm.h"
#include "Particle.h"

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand_kernel.h>

// Constants for velocity calculations. See Particle class.
#define C1 0.3
#define C2 1.3
#define W 0.8
#define MAX_VEL 1.0

/* Store curand random value states. */
curandState_t *RNGstate;

/**
CUDA kernel for particle position and velocity updates.
*/
__global__ void updateParticlePosVelKernel(float *vel, float *pos, float *best, float *gb, size_t pitch, size_t size, size_t dim, curandState *RNGstate);
/*
CUDA kernel to setup random number generation.
*/
__global__ void setupRNG(curandState *state);

__host__ Swarm::Swarm(const std::size_t size, const std::size_t dim, OP &problem, bool CUDAposvel)
	:
	size(size),
	decDim(dim),
	CUDAposvel(CUDAposvel) {

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	if(CUDAposvel) {
		// Allocate host side arrays for particle data.
		xx = new float[size*dim];
		xb = new float[size*dim];
		vv = new float[size*dim];
		xbg = new float[dim];

		cudaError_t cudaStatus;

		cudaStatus = cudaMalloc((void **)&RNGstate, size * dim * sizeof(curandState));
		if(cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc RNG state failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		setupRNG << <100, 256 >> > (RNGstate);

		cudaStatus = cudaGetLastError();
		if(cudaStatus != cudaSuccess) {
			fprintf(stderr, "setup_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
	}

	// Initialize particles regardless if using CUDA or not.
	particles.resize(size);
	std::vector<std::vector<float>> range = problem.getSearchRange();

	std::vector<float> x(dim);
	std::vector<float> v(x.size());

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
	durInit = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	updateBest();
}

__global__ void setupRNG(curandState *state) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	/* Each thread gets same seed, a different sequence number, no offset */
	curand_init(1234, id, 0, &state[id]);
}

__host__ Swarm::~Swarm() {
	for(Particle *p : particles) {
		delete p;
	}
	if(CUDAposvel) {
		delete[] xx;
		delete[] xb;
		delete[] xbg;
		delete[] vv;
		cudaFree(RNGstate);
	}
}

__host__ void Swarm::particlesToArrays() {
	if(CUDAposvel) {
		for(int i = 0; i < size; i++) {
			Particle *p = particles[i];
			for(size_t j = 0; j < decDim; ++j) {
				xx[i*j + j] = p->x[j];
				xb[i*j + j] = p->x[j];
				vv[i*j + j] = p->v[j];
			}
		}
	}
}

__host__ void Swarm::arraysToParticles() {
	if(CUDAposvel) {
		for(int i = 0; i < size; i++) {
			Particle *p = particles[i];
			for(size_t j = 0; j < decDim; ++j) {
				p->x[j] = xx[i*j + j];
				// updateFunctionValue() does this if needed
				//p->xBest[j] = xb[i*j + j];
				p->v[j] = vv[i*j + j];
			}
		}
	}
}

__host__ int Swarm::updateParticles(dim3 gridSize, dim3 blockSize) {

	if(CUDAposvel) {
		std::chrono::high_resolution_clock::time_point startMemcpy1 = std::chrono::high_resolution_clock::now();
		cudaError_t cudaStatus;

		// Device arrays
		float * devxx;
		float * devxb;
		float * devxbg;
		float * devvv;
		// Same pitch for all arrays. Is this OK?
		size_t pitch;
		size_t sSize = (size_t)size;

		try {
			cudaStatus = cudaGetLastError();
			if(cudaStatus != cudaSuccess) {
				throw std::runtime_error(std::string("some old unchecked error before allocating memory:") + cudaGetErrorString(cudaStatus));
			}
			// Allocate device position array
			cudaStatus = cudaMallocPitch(&devxx, &pitch, decDim * sizeof(float), sSize);
			if(cudaStatus != cudaSuccess) {
				throw std::runtime_error(std::string("cudaMallocPitch position failed!") + cudaGetErrorString(cudaStatus));
			}
			// Allocate device best position array
			cudaStatus = cudaMallocPitch(&devxb, &pitch, decDim * sizeof(float), sSize);
			if(cudaStatus != cudaSuccess) {
				throw std::runtime_error(std::string("cudaMallocPitch best position failed!") + cudaGetErrorString(cudaStatus));
			}
			// Allocate device global best position 
			cudaStatus = cudaMallocPitch(&devxbg, &pitch, decDim * sizeof(float), 1);
			if(cudaStatus != cudaSuccess) {
				throw std::runtime_error(std::string("cudaMallocPitch global best position failed!") + cudaGetErrorString(cudaStatus));
			}
			// Allocate device velocity array
			cudaStatus = cudaMallocPitch(&devvv, &pitch, decDim * sizeof(float), sSize);
			if(cudaStatus != cudaSuccess) {
				throw std::runtime_error(std::string("cudaMallocPitch velocity failed!") + cudaGetErrorString(cudaStatus));
			}

			// Copy positions to device array
			cudaStatus = cudaMemcpy2D(devxx, pitch, xx, decDim * sizeof(float), decDim * sizeof(float), size, cudaMemcpyHostToDevice);
			if(cudaStatus != cudaSuccess) {
				throw std::runtime_error(std::string("cudaMemcpy positions failed!") + cudaGetErrorString(cudaStatus));
			}
			// Copy best positions to device array
			cudaStatus = cudaMemcpy2D(devxb, pitch, xb, decDim * sizeof(float), decDim * sizeof(float), size, cudaMemcpyHostToDevice);
			if(cudaStatus != cudaSuccess) {
				throw std::runtime_error(std::string("cudaMemcpy best positions failed!") + cudaGetErrorString(cudaStatus));
			}
			// Copy global best position to device
			cudaStatus = cudaMemcpy2D(devxbg, pitch, xbg, decDim * sizeof(float), decDim * sizeof(float), 1, cudaMemcpyHostToDevice);
			if(cudaStatus != cudaSuccess) {
				throw std::runtime_error(std::string("cudaMemcpy global best position failed!") + cudaGetErrorString(cudaStatus));
			}
			// Copy velocities to device array
			cudaStatus = cudaMemcpy2D(devvv, pitch, vv, decDim * sizeof(float), decDim * sizeof(float), size, cudaMemcpyHostToDevice);
			if(cudaStatus != cudaSuccess) {
				throw std::runtime_error(std::string("cudaMemcpy velocities failed!") + cudaGetErrorString(cudaStatus));
			}

			std::chrono::high_resolution_clock::time_point endMemcpy1 = std::chrono::high_resolution_clock::now();
			std::chrono::microseconds durMemcpy1 = std::chrono::duration_cast<std::chrono::microseconds>(endMemcpy1 - startMemcpy1);

			std::chrono::high_resolution_clock::time_point startPosVel = std::chrono::high_resolution_clock::now();
			
			// Kernel call
			updateParticlePosVelKernel << <gridSize, blockSize >> > (devvv, devxx, devxb, devxbg, pitch, size, decDim, RNGstate);

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if(cudaStatus != cudaSuccess) {
				throw std::runtime_error(std::string("updateParticlePosVelKernel launch failed:") + cudaGetErrorString(cudaStatus));
			}

			// synchronize before continuing
			cudaStatus = cudaDeviceSynchronize();
			if(cudaStatus != cudaSuccess) {
				throw std::runtime_error(std::string("cudaDeviceSynchronize failed") + cudaGetErrorString(cudaStatus));
			}

			std::chrono::high_resolution_clock::time_point endPosVel = std::chrono::high_resolution_clock::now();
			durPPosVel += std::chrono::duration_cast<std::chrono::microseconds>(endPosVel - startPosVel);

			std::chrono::high_resolution_clock::time_point startMemcpy2 = std::chrono::high_resolution_clock::now();

			// Copy position array from GPU buffer to host memory.
			cudaStatus = cudaMemcpy2D(xx, decDim * sizeof(float), devxx, pitch, decDim * sizeof(float), size, cudaMemcpyDeviceToHost);
			if(cudaStatus != cudaSuccess) {
				throw std::runtime_error(std::string("cudaMemcpy position back to host failed!") + cudaGetErrorString(cudaStatus));
			}
			// Copy best position array from GPU buffer to host memory.
			cudaStatus = cudaMemcpy2D(xb, decDim * sizeof(float), devxb, pitch, decDim * sizeof(float), size, cudaMemcpyDeviceToHost);
			if(cudaStatus != cudaSuccess) {
				throw std::runtime_error(std::string("cudaMemcpy best position back to host failed!") + cudaGetErrorString(cudaStatus));
			}
			// Copy global best position from GPU buffer to host memory.
			cudaStatus = cudaMemcpy2D(xbg, decDim * sizeof(float), devxbg, pitch, decDim * sizeof(float), 1, cudaMemcpyDeviceToHost);
			if(cudaStatus != cudaSuccess) {
				throw std::runtime_error(std::string("cudaMemcpy lobal best position back to host failed!") + cudaGetErrorString(cudaStatus));
			}
			// Copy velocity array from GPU buffer to host memory.
			cudaStatus = cudaMemcpy2D(vv, decDim * sizeof(float), devvv, pitch, decDim * sizeof(float), size, cudaMemcpyDeviceToHost);
			if(cudaStatus != cudaSuccess) {
				throw std::runtime_error(std::string("cudaMemcpy velocity back to host failed!") + cudaGetErrorString(cudaStatus));
			}

			/* The particles have to be updated after every time that there has been changes to
			positions so that the function evaluation can be done on CPU-side.*/
			arraysToParticles();

			std::chrono::high_resolution_clock::time_point endMemcpy2 = std::chrono::high_resolution_clock::now();
			durMemcpy += std::chrono::duration_cast<std::chrono::microseconds>(endMemcpy2 - startMemcpy2) + durMemcpy1;

		} catch(const std::runtime_error &e) {
			// Free memory if we got an exception.
			fprintf(stderr, "CUDA things failed: %s\n ", e.what());
			cudaFree(devxx);
			cudaFree(devxb);
			cudaFree(devxbg);
			cudaFree(devvv);
			return 1;
		}
		// Free memory if success.
		cudaFree(devxx);
		cudaFree(devxb);
		cudaFree(devxbg);
		cudaFree(devvv);

		std::chrono::high_resolution_clock::time_point startFun = std::chrono::high_resolution_clock::now();

		// Funtion value still needs to be updated on CPU side
		for(Particle *p : particles) {
			p->updateFuncValue();
		}

		std::chrono::high_resolution_clock::time_point endFun = std::chrono::high_resolution_clock::now();
		durPFun += std::chrono::duration_cast<std::chrono::microseconds>(endFun - startFun);

		updateBest();

		// Without CUDA
	} else {
		for(Particle *p : particles) {
			std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
			p->updateVelPos(bestParticle->x);
			std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
			durPPosVel += std::chrono::duration_cast<std::chrono::microseconds>(end - start);

			start = std::chrono::high_resolution_clock::now();
			p->updateFuncValue();
			end = std::chrono::high_resolution_clock::now();
			durPFun += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		}
		updateBest();
	}
	return 0;
}

__global__ void updateParticlePosVelKernel(float *vel, float *pos, float *best, float *gb, size_t pitch, size_t size, size_t dim, curandState *RNGstate) {

	// Blocks width is based solely on problem dimension.
	int tidx = /*blockIdx.x * blockDim.x +*/ threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	if((tidx < dim) && (tidy < size)) {
		float *posRow = (float *)((char*)pos + tidy * pitch);
		float *bestRow = (float *)((char*)best + tidy * pitch);
		float *gbRow = (float *)((char*)gb + 1 * pitch);
		float *velRow = (float *)((char*)vel + tidy * pitch);

		float rnd1 = curand_uniform(RNGstate + tidx * tidy);
		float rnd2 = curand_uniform(RNGstate + tidx * tidy);

		/*if(tidx == 0){
			printf("rnd1 %f\n", rnd1);
			printf("rnd2 %f\n", rnd2);
		}*/

		velRow[tidx] = W * velRow[tidx] + C1 * rnd1 * (bestRow[tidx] - posRow[tidx]) + C2 * rnd2 * (gbRow[tidx] - posRow[tidx]);

		if(velRow[tidx] > MAX_VEL) {
			velRow[tidx] = MAX_VEL;
		} else if(velRow[tidx] < -MAX_VEL) {
			velRow[tidx] = -MAX_VEL;
		}

		// update position
		posRow[tidx] = posRow[tidx] + velRow[tidx];
	}
}

__host__ void Swarm::updateBest() {
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	for(int i = 0; i < particles.size(); ++i) {
		Particle *p = particles.at(i);
		float val = p->fVal;
		if(val < bestVal) {

			// See the convergence
			//printf("%.20f < %.20f \n", val, bestVal);

			bestVal = val;
			bestParticle = p;

			if(CUDAposvel) {
				// This update is for CUDA kernel methods.
				for(size_t k = 0; k < decDim; ++k) {
					xbg[k] = p->x[k];
				}
			}
		}
	}

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	durUBest += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

__host__ float Swarm::randMToN(float M, float N) {
	return M + (rand() / (RAND_MAX / (N - M)));
}

__host__ void Swarm::print() {
	printf("Swarm's best f(%.6f,%.6f) =  %.6f \n", bestParticle->xBest[0], bestParticle->xBest[1], bestParticle->fValBest);
	std::cout << "---------------------------------------------" << std::endl;
}

__host__ void Swarm::printParticles() {
	for(Particle *p : particles) {
		p->print();
	}
}