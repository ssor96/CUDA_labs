#include <stdio.h>

#define CSC(call) do { \
	cudaError_t res = call;	\
	if (res != cudaSuccess) { \
		fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
		exit(0); \
	} \
} while (0)

__global__ void kernel(double* da, double* db, int n) {
	int offset = blockDim.x * gridDim.x;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += offset) {
		da[idx] -= db[idx];
	}
}

int main() {
	int n;
    scanf("%d", &n);
	double* a = (double *)malloc(sizeof(double) * n);
	double* b = (double *)malloc(sizeof(double) * n);
	double* da;
	double* db;
	CSC(cudaMalloc(&da, sizeof(double) * n));
	CSC(cudaMalloc(&db, sizeof(double) * n));
	for(int i = 0; i < n; ++i) {
	    scanf("%lf", a + i);
    }
    for(int i = 0; i < n; ++i) {
        scanf("%lf", b + i);
    }
	CSC(cudaMemcpy(da, a, sizeof(double) * n, cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(db, b, sizeof(double) * n, cudaMemcpyHostToDevice));
	
	kernel<<<256, 256>>> (da, db, n);

	CSC(cudaMemcpy(a, da, sizeof(double) * n, cudaMemcpyDeviceToHost));
	
	CSC(cudaFree(da));
	CSC(cudaFree(db));
	for(int i = 0; i < n; ++i) {
		printf("%.10lf ", a[i]);
    }
	printf("\n");
	free(a);
	free(b);
	return 0;
}