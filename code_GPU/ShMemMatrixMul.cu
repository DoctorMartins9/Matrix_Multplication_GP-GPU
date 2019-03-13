#define DIM 4096
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <cuda.h>
//#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>
#include <chrono>

#define TILE_WIDTH 32


__global__
void MatrixMulKernelTiled(double *M, double *N, double *P, int size) {
	__shared__ double Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ double Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	int Pvalue = 0;

	for (int ph = 0; ph < (int)ceil(size / (double)TILE_WIDTH); ++ph) {
		if ((Row < size) && ((ph*TILE_WIDTH + tx) < size)) {
			Mds[ty][tx] = M[Row * size + ph * TILE_WIDTH + tx];
		}
		if (((ph * TILE_WIDTH + ty) < size) && (Col < size)) {
			Nds[ty][tx] = N[(ph*TILE_WIDTH + ty) * size + Col];
		}
		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; ++k) {
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();
	}

	if ((Row < size) && (Col < size)) {
		P[Row * size + Col] = Pvalue;
	}
}

void LaunchKernel(double *M, double *N, double *P, int size) {


	double *d_A, *d_B, *d_C;

	int spazio_tot = (size * size) * sizeof(double);
	cudaMalloc((void **)&d_A, spazio_tot);
	cudaMalloc((void **)&d_B, spazio_tot);
	cudaMalloc((void **)&d_C, spazio_tot);

	cudaMemcpy(d_A, M, spazio_tot, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, N, spazio_tot, cudaMemcpyHostToDevice);


	dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 grid(ceil((double)DIM / TILE_WIDTH), ceil((double)DIM / TILE_WIDTH), 1);

	//MatrixMulKernel << <grid, block >> > (d_A, d_B, d_C, size);
	MatrixMulKernelTiled << <grid, block >> > (d_A, d_B, d_C, size);

	cudaMemcpy(P, d_C, spazio_tot, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

}

void MatrixMulHost(double(*A)[DIM], double(*B)[DIM], double(*C)[DIM]) {
	for (int c = 0; c < DIM; c++) {
		for (int d = 0; d < DIM; d++) {

			int Pvalue = 0;
			for (int k = 0; k < DIM; k++) {
				Pvalue += A[c][k] * B[k][d];
			}

			C[c][d] = Pvalue;

		}
	}
}

int main() {

	double *A =(double *)malloc(DIM*DIM*sizeof(double));
	double *B=(double *)malloc(DIM*DIM*sizeof(double));
	double *C=(double *)malloc(DIM*DIM*sizeof(double));

	//riempio le matrici con dei valori arbitrari
	for (int i = 0; i < DIM; i++) {
		for (int j = 0; j < DIM; j++) {
			A[i*DIM+j] = 1.0;
			B[i*DIM+j] = 1.0;
		}
	}
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	LaunchKernel(&A[0], &B[0], &C[0], DIM);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	double tempo = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();

	printf("%f\n",tempo);

	free(A);
	free(B);
	free(C);
}

