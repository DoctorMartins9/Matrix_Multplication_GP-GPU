#define DIM 64
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <chrono>


#define TILE_DIM 32


__global__
void MatrixMulKernel(double *M, double *N, double *P, int Width) {
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	if ((Row < Width) && (Col < Width)) {
		double Pvalue = 0;

		for (int k = 0; k < Width; k++) {
			Pvalue += M[Row*Width + k] * N[k*Width + Col];

		}
		P[Row*Width + Col] = Pvalue;
	}
}

//FUNZIONE CHE RIEMPIE LA MATRICE DI NUMERI double CASUALI
void populateMatrix(double *M) {
	srand(time(NULL));
	for (int i = 0; i < DIM; i++) {
		for (int j = 0; j < DIM; j++) {
			//M[i * DIM + j] = (double)(1.0);
			M[i * DIM + j] = (double)((rand() % 10000) /(double)DIM);
		}
	}
}



__global__ void MatMul(double* A, double* B, double* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {
    double CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ double As[TILE_DIM][TILE_DIM];
    __shared__ double Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

         if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
             As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)
             Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n)
             CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}

void LaunchKernel(double *M, double *N, double *P, int size) {
	
	//Creazione Streams
	int n_stream = 4;
	cudaStream_t stream[n_stream];

	for(int i = 0; i < n_stream; i++){
		cudaStreamCreate(&stream[i]);
	}

	// Divisione della matrice in 2 sottomatrici
	double *N_3 = (double *)malloc(DIM * DIM/2 *(sizeof(double)));
	double *N_4 = (double *)malloc(DIM * DIM/2 *(sizeof(double)));

	for(int i = 0; i < DIM; i++){
		for(int j = 0; j < DIM; j++){
			if(j < DIM/2)
				N_3[i*DIM/2 + j]=N[i*DIM+j]; 
			else
				N_4[i*DIM/2 + (j-DIM/2)] = N[i*DIM + j];
		}
	}

	//Allocazione dei segmenti di memoria
	size_t slice = DIM * (DIM/2);

	double *d_M0, *d_N0, *d_P0;
	double *d_M1, *d_N1, *d_P1;
	double *d_M2, *d_N2, *d_P2;
	double *d_M3, *d_N3, *d_P3;

	cudaMalloc((void **)&d_M0, DIM * (DIM/2) * sizeof(double));
	cudaMalloc((void **)&d_N0, DIM * (DIM/2) * sizeof(double));
	cudaMalloc((void **)&d_P0, (DIM/2)* (DIM/2) * sizeof(double));

	cudaMalloc((void **)&d_M1, DIM * (DIM/2) * sizeof(double));
	cudaMalloc((void **)&d_N1, DIM * (DIM/2) * sizeof(double));
	cudaMalloc((void **)&d_P1, (DIM/2) * (DIM/2) * sizeof(double));

	cudaMalloc((void **)&d_M2, DIM * (DIM/2) * sizeof(double));
	cudaMalloc((void **)&d_N2, DIM * (DIM/2) * sizeof(double));
	cudaMalloc((void **)&d_P2, (DIM/2) * (DIM/2) * sizeof(double));

	cudaMalloc((void **)&d_M3, DIM * (DIM/2) * sizeof(double));
	cudaMalloc((void **)&d_N3, DIM * (DIM/2) * sizeof(double));
	cudaMalloc((void **)&d_P3, (DIM/2) * (DIM/2) * sizeof(double));


	// DICHIARAZIONE blockDim e gridDim   
	dim3 block(TILE_DIM, TILE_DIM, 1);
	dim3 grid(ceil((double)DIM / block.x), ceil(((double)DIM/2)/block.y) , 1);

	// Esecuzione del kernel

	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	cudaMemcpyAsync(d_M0, M, slice * sizeof (double), cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyAsync(d_N0, N_3, slice * sizeof (double), cudaMemcpyHostToDevice, stream[0]);
	double *ker0 = (double *)malloc(DIM/2 * DIM/2 * sizeof(double));
	
	cudaMemcpyAsync(d_M1, M, slice * sizeof (double), cudaMemcpyHostToDevice, stream[1]);
	cudaMemcpyAsync(d_N1, N_4, slice * sizeof (double), cudaMemcpyHostToDevice, stream[1]);
	double *ker1 = (double *)malloc(DIM/2 * DIM/2 * sizeof(double));

	cudaMemcpyAsync(d_M2, M + slice, slice * sizeof (double), cudaMemcpyHostToDevice, stream[2]);
	cudaMemcpyAsync(d_N2, N_3, slice * sizeof (double), cudaMemcpyHostToDevice, stream[2]);
	double *ker2 = (double *)malloc(DIM/2 * DIM/2 * sizeof(double));

	cudaMemcpyAsync(d_M3, M + slice, slice * sizeof (double), cudaMemcpyHostToDevice, stream[3]);
	cudaMemcpyAsync(d_N3, N_4, slice * sizeof (double), cudaMemcpyHostToDevice, stream[3]);
	double *ker3 = (double *)malloc(DIM/2 * DIM/2 * sizeof(double));

	MatMul<<<grid, block, block.x * block.y * sizeof(double), stream[0]>>>(d_M0, d_N0, d_P0, DIM/2, DIM, DIM, DIM/2, DIM/2, DIM/2);
	MatMul<<<grid, block, block.x * block.y * sizeof(double), stream[1]>>>(d_M1, d_N1, d_P1, DIM/2, DIM, DIM, DIM/2, DIM/2, DIM/2);
	MatMul<<<grid, block, block.x * block.y * sizeof(double), stream[2]>>>(d_M2, d_N2, d_P2, DIM/2, DIM, DIM, DIM/2, DIM/2, DIM/2);
	MatMul<<<grid, block, block.x * block.y * sizeof(double), stream[3]>>>(d_M3, d_N3, d_P3, DIM/2, DIM, DIM, DIM/2, DIM/2, DIM/2);
	
	cudaDeviceSynchronize();

	cudaMemcpyAsync(ker0, d_P0, DIM/2 * DIM/2 * sizeof (double), cudaMemcpyDeviceToHost, stream[0]);
	cudaMemcpyAsync(ker1, d_P1, DIM/2 * DIM/2 * sizeof (double), cudaMemcpyDeviceToHost, stream[1]);
	cudaMemcpyAsync(ker2, d_P2, DIM/2 * DIM/2 * sizeof (double), cudaMemcpyDeviceToHost, stream[2]);
	cudaMemcpyAsync(ker3, d_P3, DIM/2 * DIM/2 * sizeof (double), cudaMemcpyDeviceToHost, stream[3]);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	double tempo = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
	printf("%lf\n", tempo);

	// Copio le sottomatrici nella matrixce finale
	for(int i = 0; i < DIM; i++){
		for(int j = 0; j<DIM ; j++){
			if(i < DIM/2 && j < DIM/2)
				P[i * DIM + j ] = ker0[i * DIM/2 + j];
			else if(i < DIM/2 && j >= DIM/2)
				P[i * DIM + j ] = ker1[i * DIM/2 + (j-DIM/2)];
			else if(i >= DIM/2 && j < DIM/2)
				P[i * DIM + j ] = ker2[(i-DIM/2) * DIM/2 + j];
			else if(i >= DIM/2 && j >= DIM/2)
				P[i * DIM + j ] = ker3[(i-DIM/2) * DIM/2 + (j-DIM/2)];
		}
	}

	cudaFree(d_M0);
	cudaFree(d_N0);
	cudaFree(d_P0);

	cudaFree(d_M1);
	cudaFree(d_N1);
	cudaFree(d_P1);

	cudaFree(d_M2);
	cudaFree(d_N2);
	cudaFree(d_P2);

	cudaFree(d_M3);
	cudaFree(d_N3);
	cudaFree(d_P3);

}


void MatrixMulHost(double *A, double *B, double *C) {
	//std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	int c, d, k;
	for (c = 0; c < DIM; c++) {
		for (d = 0; d < DIM; d++) {

			double Pvalue = 0;
			for (k = 0; k < DIM; k++) {
				Pvalue += A[c * DIM + k] * B[k * DIM + d];
			}

			C[c * DIM + d] = Pvalue;

		}
	}
	//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	//double tempo = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
	//printf("TEMPO ELABORAZIONE SU HOST: %lf\n", tempo);
}



int main() {

	double *A = (double *)malloc(DIM * DIM * sizeof(double));
	double *B = (double *)malloc(DIM * DIM * sizeof(double));
	double *C = (double *)malloc(DIM * DIM * sizeof(double));
	double *C_H = (double *)malloc(DIM * DIM * sizeof(double));
	populateMatrix(A);
	populateMatrix(B);
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	LaunchKernel(&A[0], &B[0], &C[0], DIM);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	double tempo = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
}

