#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cublas_v2.h>
#include "TimingGPU.cuh"

using namespace std;

#define N					256
#define Matrix_Block_Size	32

#define THRESHOLD_PIVOT		1e-20

/***********************/
/* CUDA ERROR CHECKING */
/***********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
/**********************/
/* cuBLAS ERROR CHECK */
/**********************/
#ifndef cublasSafeCall
#define cublasSafeCall(err)     __cublasSafeCall(err, __FILE__, __LINE__)
#endif

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
    if( CUBLAS_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n",__FILE__, __LINE__,err);
		cudaDeviceReset(); assert(0);
    }
}
/********************************/
/* MATRIX RANDOM INITIALIZATION */
/********************************/
void init_matrices(double* A, double* B) {

	srand (time(NULL));

	for (int i=0; i<N; i++) {
		B[i] = rand();
		for (int j=0; j<N; j++)
			A[i*N+j] = rand();
	}

}

/*****************/
/* MATRIX SAVING */
/*****************/
void save_matrix(double* A, int len, char* filenameA) {

	ofstream outfile;
	outfile.open(filenameA);
	for(int i=0; i<len; i++)  outfile << A[i] << "\n";
	outfile.close();

}

/******************************************/
/* ELIMINATION KERNEL iB - iB - NO SHARED */
/******************************************/
__global__ void kernel_elim_block_iB_iB(int* pokpiv, double* A, double* B, double* R, int i0, int i1) {

	for (int i = i0; i < i1; i++) {

		if (fabs(A[i*N+i]) < THRESHOLD_PIVOT) *pokpiv = 0;
		else {
			*pokpiv = 1;
			for (int j = i+1; j < i1; j++) {
				R[j*Matrix_Block_Size+(i-i0)] = A[j*N+i] / A[i*N+i];
				//printf("Matrix kernel1 %i %i %f\n",j,i-i0,R[j*Matrix_Block_Size+(i-i0)]);
				B[j] -= R[j*Matrix_Block_Size+(i-i0)] * B[i];
				for (int k = i; k < i1; k++) {
					A[j*N+k] -= R[j*Matrix_Block_Size+(i-i0)] * A[i*N+k];
					//printf("kernel_elim_block_iB_iB %i %i %f\n",j,k,A[j*N+k]);
				}
			}
		}
	}
}

/******************************************/
/* ELIMINATION KERNEL iB - kB - NO SHARED */
/******************************************/
__global__ void kernel_elim_block_iB_kB(double* A, double* B, double* R, int Num_Blocks, int iB, int i0, int i1) {

	int kB = (iB + 1) + blockIdx.x;

	int k0 = kB * N / Num_Blocks;
	int k1 = (kB + 1) * N / Num_Blocks;
	int k = k0 + threadIdx.x;

	if (k < k1) {
		for (int i = i0; i < i1 ; i++)
			for (int j = i + 1; j < i1 ; j++) {
				A[j*N+k] -= R[j*Matrix_Block_Size+(i-i0)] * A[i*N+k];
				//printf("kernel_elim_block_iB_kB %i %i %f\n",j,k,A[j*N+k]);
				//printf("Matrix kernel2 %i %i %f\n",j,i-i0,R[j*Matrix_Block_Size+(i-i0)]);
				}
	}
}

/******************************************/
/* ELIMINATION KERNEL jB - iB - NO SHARED */
/******************************************/
__global__ void kernel_elim_block_jB_iB(double* A, double* B, double* R, int Num_Blocks, int iB, int i0, int i1) {

	int jB = (iB + 1) + blockIdx.x;

	int j0 = jB * N / Num_Blocks;
	int j1 = (jB + 1) * N / Num_Blocks;
	int j = j0 + threadIdx.x;

	if (j < j1)
		for (int i = i0 ; i < i1 ; i++) {
			R[j*Matrix_Block_Size+(i-i0)] = A[j*N+i] / A[i*N+i];
			B[j] -= R[j*Matrix_Block_Size+(i-i0)] * B[i];
			for (int k = i ; k < i1 ; k++) {
				A[j*N+k] -= R[j*Matrix_Block_Size+(i-i0)] * A[i*N+k];
				//printf("kernel_elim_block_jB_iB %i %i %f\n",j,k,A[j*N+k]);
				}
		}
}

/******************************************/
/* ELIMINATION KERNEL jB - kB  */
/******************************************/
__global__ void kernel_elim_block_jB_kB(double* A, double* B, double* R, int Num_Blocks, int iB, int i0, int i1) {

	int jB = (iB + 1) + blockIdx.x;
	int j0 = jB * N / Num_Blocks;
	int j1 = (jB + 1) * N / Num_Blocks;
	int j = j0 + threadIdx.x;

	int kB = (iB + 1) + blockIdx.y;
	int k0 = kB * N / Num_Blocks;
	int k1 = (kB + 1) * N / Num_Blocks;
	int k = k0 + threadIdx.y;

	if (j < j1 && k < k1)
		for (int i = i0; i < i1; i++) {
			A[j*N+k] -= R[j*Matrix_Block_Size+(i-i0)] * A[i*N+k];
			//printf("kernel_elim_block_jB_kB %i %i %f\n",j,k,A[j*N+k]);
		}
}


/***************************************/
/* FORWARD ELIMINATION GPU */
/***************************************/
int gauss_elimination_GPU_tiling(double* A, double* B) {

	int Num_Blocks = N / Matrix_Block_Size + (N % Matrix_Block_Size ? 1 : 0);

	// --- GPU memory allocations
	double *d_A;
	size_t sA = sizeof(double) * N * N;
	gpuErrchk(cudaMalloc((void**)&d_A,sA));
	double *d_B; size_t sB = sizeof(double) * N;						gpuErrchk(cudaMalloc((void**)&d_B,sB));
	double *d_R; size_t sR = sizeof(double) * N * Matrix_Block_Size;	gpuErrchk(cudaMalloc((void**)&d_R,sR));

	int ok_pivoting, *d_ok_pivoting;									gpuErrchk(cudaMalloc((void**)&d_ok_pivoting,sizeof(int)));

	// --- CPU->GPU matrix copies
	gpuErrchk(cudaMemcpy(d_A,A,sA,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_B,B,sB,cudaMemcpyHostToDevice));

	for (int iB=0; iB<Num_Blocks; iB++) {

		int i0 = iB * N / Num_Blocks;
		int i1 = (iB+1) * N / Num_Blocks;

		kernel_elim_block_iB_iB<<<1,1>>>(d_ok_pivoting,d_A,d_B,d_R,i0,i1);
		gpuErrchk(cudaThreadSynchronize());

		gpuErrchk(cudaMemcpy(&ok_pivoting, d_ok_pivoting, sizeof(int), cudaMemcpyDeviceToHost));
		if (!ok_pivoting) return 0;

		kernel_elim_block_iB_kB<<<Num_Blocks-(iB+1),Matrix_Block_Size>>>(d_A,d_B,d_R,Num_Blocks,iB,i0,i1);
		gpuErrchk(cudaThreadSynchronize());

		if (iB < Num_Blocks-1) {

			kernel_elim_block_jB_iB<<<Num_Blocks-(iB+1),Matrix_Block_Size>>>(d_A,d_B,d_R,Num_Blocks,iB,i0,i1);
			gpuErrchk(cudaThreadSynchronize());

			dim3 blocks(Num_Blocks-(iB+1),Num_Blocks-(iB+1));
			dim3 threads(Matrix_Block_Size,Matrix_Block_Size);

			 kernel_elim_block_jB_kB <<<blocks,threads>>>(d_A,d_B,d_R,Num_Blocks,iB,i0,i1);
			gpuErrchk(cudaThreadSynchronize());

		}
	}

	gpuErrchk(cudaMemcpy(A,d_A,sA,cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(B,d_B,sB,cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(d_A)); gpuErrchk(cudaFree(d_B));
	gpuErrchk(cudaFree(d_R)); gpuErrchk(cudaFree(d_ok_pivoting));

	return 1;

}
/*************************************************/
/* SOLUTION OF AN UPPER TRINAGULAR SYSTEM ON GPU */
/*************************************************/
// using cuBLAS.
int solution_of_a_triangular_system_GPU(double* A, double* B, double* x) {

	double Aii, alpha;

	int Num_Blocks = N / Matrix_Block_Size + (N % Matrix_Block_Size ? 1 : 0);

	cublasHandle_t handle;

	double *d_A; size_t sA = sizeof(double) * N * N;							gpuErrchk(cudaMalloc((void**)&d_A,sA));
	double *d_B; size_t sB = sizeof(double) * N;								gpuErrchk(cudaMalloc((void**)&d_B,sB));

	cublasSetMatrix(N, N, sizeof(double), A, N, d_A, N);
	cublasSetVector(N, sizeof(double), B, 1, d_B, 1);

	gpuErrchk(cudaMemcpy(&Aii, d_A+(N-1)+(N-1)*N, sizeof(double), cudaMemcpyDeviceToHost));

	if (fabs(Aii) < THRESHOLD_PIVOT) return 0;

	cublasSafeCall(cublasCreate(&handle));

	alpha = 1.;
	cublasSafeCall(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, 1, &alpha, d_A, N, d_B, N));

	cublasSafeCall(cublasGetVector(N, sizeof(double), d_B, 1, x, 1));

	return 1;
}

/********/
/* MAIN */
/********/
int main() {

	TimingGPU timerGPU;

	double* A = (double*)malloc(N*N*sizeof(double));
	double* B = (double*)malloc(N*sizeof(double));
	double* x = (double*)malloc(N*sizeof(double));

	// --- Matrix initialization
	init_matrices(A,B);

	// --- Saving the original matrices to disk
	save_matrix(A,N*N,"A.txt");
	save_matrix(B,N,"B.txt");


	// --- Running the GPU procedures
	timerGPU.StartCounter();
	gauss_elimination_GPU_tiling(A,B);
	solution_of_a_triangular_system_GPU(A,B,x);
	printf("%f\n",timerGPU.GetCounter());

	// --- Saving the data
	save_matrix(A,N*N,"A_prime.txt");
	save_matrix(B,N,"B_prime.txt");
	save_matrix(x,N,"x_prime.txt");

	printf("Done\n");

	return 0;

}
