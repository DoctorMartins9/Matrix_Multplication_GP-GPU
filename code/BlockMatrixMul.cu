#define DIM 64
char[] name = "BlockMatrixMul";

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <chrono>

__global__
void matrixMul(double *A, double *B, double *C, int size) {

	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	double Cvalue = 0.0;
	for (int k = 0; k < size; k++) {
		Cvalue += A[size * Row + k] * B[size * k + Row];
	}

	C[Row * size + Col] = Cvalue;

}

void addKernel(double *h_A, double *h_B, double *h_C, int size) {
	int size_tot = size * size * sizeof(double);
	double *d_A, *d_B, *d_C;
	
	cudaMalloc((void **)&d_A, size_tot);
	cudaMalloc((void **)&d_B, size_tot);
	cudaMalloc((void **)&d_C, size_tot);

	cudaMemcpy(d_A, h_A, size_tot, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size_tot, cudaMemcpyHostToDevice);

	dim3 block(16, 16, 1);
	dim3 grid((int)ceil((double)DIM / 16), (int)ceil((double)DIM / 16), 1);

	matrixMul << <grid, block >> > (d_A, d_B, d_C, size);

	cudaMemcpy(h_C, d_C, size_tot, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

//FUNZIONE CHE RIEMPIE LA MATRICE DI NUMERI double CASUALI
void populateMatrix(double M[DIM][DIM]) {
	for (int i = 0; i < DIM; i++) {
		for (int j = 0; j < DIM; j++) {
			M[i][j] = 1.0;
		}
	}
}

//FUNZIONE PER STAMPARE UNA MATRICE
void printMatrix(double M[DIM][DIM]) {
	for (int i = 0; i < DIM; i++) {
		for (int j = 0; j < DIM; j++) {
			printf("%f	", M[i][j]);
		}
		printf("\n");
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
	addKernel(&A[0], &B[0], &C[0], DIM);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	double tempo = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();

	printf("%.3f\n",tempo);
	
	free(A);
	free(B);
	free(C);
	return 0;
}
