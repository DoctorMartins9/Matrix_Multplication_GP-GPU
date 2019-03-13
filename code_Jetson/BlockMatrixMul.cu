#define DIM 4096
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <chrono>

#include "jetson_tx2_power.h"
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

	start_thread();
	matrixMul << <grid, block >> > (d_A, d_B, d_C, size);
	cudaDeviceSynchronize();
	stop_thread();
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

void MatrixMulHost(double *A, double *B, double *C) {
	int c, d, k;
	for (c = 0; c < DIM; c++) {
		for (d = 0; d < DIM; d++) {

			int Pvalue = 0;
			for (k = 0; k < DIM; k++) {
				Pvalue += A[c * DIM + k] * B[k * DIM + d];
			}

			C[c *DIM + d] = Pvalue;

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

void checkResult(double *D, double *H){
	int length = DIM * DIM;
	

	for(int i = 0; i < length; i++){
		if(D[i] != H[i]){
			printf("Errore!!!!!!!\n");
			break;
		}
	}
}

int main() {

	double *A =(double *)malloc(DIM*DIM*sizeof(double));
	double *B=(double *)malloc(DIM*DIM*sizeof(double));
	double *C=(double *)malloc(DIM*DIM*sizeof(double));
	double *C_H = (double *)malloc(DIM*DIM *sizeof(double));

	//riempio le matrici con dei valori arbitrari
	for (int i = 0; i < DIM; i++) {
		for (int j = 0; j < DIM; j++) {
			A[i*DIM+j] = 1.0;
			B[i*DIM+j] = 1.0;
		}
	}

	//MatrixMulHost(A, B, C_H);

	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	addKernel(&A[0], &B[0], &C[0], DIM);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	double tempo = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();

	if(C[DIM * DIM -1] == DIM)
		printf("%f\n",tempo);
	else
		printf("error");
	

	//checkResult(C, C_H);
	free(A);
	free(B);
	free(C);
	return 0;
}
