#include "Swarm.h"
#include "Particle.h"

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <chrono>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCKSIZE_x 32
#define BLOCKSIZE_y 32



int iDivUp(int hostPtr, int b) {
	return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b);
}


__global__ void setup_kernel(curandState *state) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	/* Each thread gets same seed, a different sequence
	   number, no offset */
	curand_init(1234, id, 0, &state[id]);
}

Swarm::Swarm(const std::size_t size, const std::size_t dim, OP &problem)
	:
	size(size),
	decDim(dim),
	op(problem) {

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	xx = new float[size*dim];
	xb = new float[size*dim];
	vv = new float[size*dim];
	xbg = new float[dim];

	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void **)&RNGstate, BLOCKSIZE_x * sizeof(curandState));
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc RNG state failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	dim3 gridSize(iDivUp(decDim, BLOCKSIZE_x), iDivUp(size, BLOCKSIZE_y));
	dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);
	setup_kernel << <1, 1 >> > (RNGstate);

	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "setup_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	particles.resize(size);
	std::vector<std::vector<float>> range = problem.getSearchRange();
	std::vector<std::vector<float>> particV(size);
	std::vector<std::vector<float>> particVOld(size);

	int probdim = problem.getDecDimension();
	std::vector<float> x(probdim);
	std::vector<float> v(x.size());

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
	durInit = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

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
	cudaFree(RNGstate);
}


void Swarm::particlesToArrays() {
	for(int i = 0; i < size; i++) {
		Particle *p = particles[i];
		for(size_t j = 0; j < decDim; ++j) {
			xx[i*j + j] = p->x[j];
			xb[i*j + j] = p->x[j];
			vv[i*j + j] = p->v[j];
		}
	}
}

void Swarm::arraysToParticles() {
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

#define C1 0.3
#define C2 1.3
#define W 0.8
#define MAX_VEL 1.0

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__global__ void updateParticleVelocityKernel(float *vel, float *pos, float *best, float *gb, size_t pitch, size_t size, size_t dim, curandState *RNGstate) {

	int tidx = /*blockIdx.x * blockDim.x +*/ threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;


	//printf("access: (%d,%d)\n", tidx, tidy);


	if((tidx < dim) && (tidy < size)) {
		float *posRow = (float *)((char*)pos + tidy * pitch);
		float *bestRow = (float *)((char*)best + tidy * pitch);
		float *gbRow = (float *)((char*)gb + 1 * pitch);
		float *velRow = (float *)((char*)vel + tidy * pitch);
		//velRow[tidx] = 0.5f;

	/*printf("global best in kernel: (%f, %f)\n", gbRow[0], gbRow[1]);
	printf("own best in kernel: (%f, %f)\n", bestRow[0], bestRow[1]);*/

	//curandState tstate = RNGstate[tidx];

	//
	//v[i] = w * vOld[i] + c1 * rnd01() * (xBest[i] - x[i]) + c2 * rnd01() * (direction[i] - x[i]);
	// placeholder for a random number

		float rnd1 = curand_uniform(RNGstate + tidx);
		float rnd2 = curand_uniform(RNGstate + tidx);

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
		//
		//RNGstate[tidx] = tstate;
	}
	/*else {
		printf("loose thread: (%d,%d)\n", tidx, tidy);
	}*/
}

/*
https://stackoverflow.com/questions/39006348/accessing-class-data-members-from-within-cuda-kernel-how-to-design-proper-host
*/
__host__ void Swarm::updateParticlePositions(bool CUDAposvel, dim3 gridSize, dim3 blockSize) {

	if(CUDAposvel) {
		std::chrono::high_resolution_clock::time_point startMemcpy1 = std::chrono::high_resolution_clock::now();
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


		std::chrono::high_resolution_clock::time_point endMemcpy1 = std::chrono::high_resolution_clock::now();
		std::chrono::microseconds durMemcpy1 = std::chrono::duration_cast<std::chrono::microseconds>(endMemcpy1 - startMemcpy1);

		std::chrono::high_resolution_clock::time_point startPosVel = std::chrono::high_resolution_clock::now();
		
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if(cudaStatus != cudaSuccess) {
			fprintf(stderr, "2  failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// Kernel call
		//updateParticleVelocityKernel << <gridSize, blockSize >> > (devvv, devxx, devxb, devxbg, pitch, size, decDim, RNGstate);
		updateParticleVelocityKernel << <gridSize, blockSize >> > (devvv, devxx, devxb, devxbg, pitch, size, decDim, RNGstate);

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

		std::chrono::high_resolution_clock::time_point endPosVel = std::chrono::high_resolution_clock::now();
		durPPosVel += std::chrono::duration_cast<std::chrono::microseconds>(endPosVel - startPosVel);

		std::chrono::high_resolution_clock::time_point startMemcpy2 = std::chrono::high_resolution_clock::now();

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



	Error:
		cudaFree(devxx);
		cudaFree(devxb);
		cudaFree(devxbg);
		cudaFree(devvv);

		/* The particles have to be updated after every time that there has been changes to
		positions so that the function evaluation can be done on CPU-side.*/
		arraysToParticles();

		std::chrono::high_resolution_clock::time_point endMemcpy2 = std::chrono::high_resolution_clock::now();
		durMemcpy += std::chrono::duration_cast<std::chrono::microseconds>(endMemcpy2 - startMemcpy2) + durMemcpy1;

		std::chrono::high_resolution_clock::time_point startFun = std::chrono::high_resolution_clock::now();

		// Funtion value still needs to be updated on CPU side
		for(Particle *p : particles) {
			p->updateFuncValue();
		}

		std::chrono::high_resolution_clock::time_point endFun = std::chrono::high_resolution_clock::now();
		durPFun += std::chrono::duration_cast<std::chrono::microseconds>(endFun - startFun);

		updateBest();

		return;

		// Without CUDA
	} else {

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

		// Guide all particles towards the current best position.

		for(Particle *p : particles) {
			p->updateVelPos(bestParticle->x);
			p->updateFuncValue();
		}

		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		durPPosVel += std::chrono::duration_cast<std::chrono::microseconds>(end - start);

		updateBest();
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
			// This update is for CUDA kernel methods.
			for(size_t k = 0; k < decDim; ++k) {
				xbg[k] = p->x[k];
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
	printf("Swarm's best f(%.6f,%.6f) =  %.20f \n", bestParticle->xBest[0], bestParticle->xBest[1], bestParticle->valBest);
	//std::cout << "Swarm's best f(" << bestParticle->xBest[0] << "," << bestParticle->xBest[1] << ") = " << bestParticle->valBest << std::endl;
	std::cout << "---------------------------------------------" << std::endl;
}

__host__ void Swarm::printParticles() {
	for(Particle *p : particles) {
		p->print();
	}
}