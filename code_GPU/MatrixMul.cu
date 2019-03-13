#define DIM 1024
#include<stdio.h>
#include<stdlib.h>
#include <chrono>

//FUNZIONE PER STAMPARE UNA MATRICE
void printMatrix(double *M) {
	int i, j;
	for (i = 0; i < DIM; i++) {
		for (j = 0; j < DIM; j++) {
			printf("%f	", M[i * DIM + j]);
		}
		printf("\n");
	}
}

//FUNZIONE CHE RIEMPIE LA MATRICE DI NUMERI double CASUALI
void populateMatrix(double *M) {
	int i, j;
	for (i = 0; i < DIM; i++) {
		for (j = 0; j < DIM; j++) {
			M[i * DIM + j] = (double)(i + j);
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


int main(){
    double *A = (double *)malloc(DIM * DIM * sizeof(double));
	double *B = (double *)malloc(DIM * DIM * sizeof(double));
	double *C = (double *)malloc(DIM * DIM * sizeof(double));

    populateMatrix(A);
    populateMatrix(B);
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	MatrixMulHost(A, B, C);
		
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	double tempo = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();

    printf("%f\n", tempo);
   // printMatrix(C);

}
