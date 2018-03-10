#include <iostream>
#include <iomanip>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

using namespace std;

struct comparator {
	__host__ __device__ bool operator()(double a, double b)
	{
		return fabs(a) < fabs(b);
	}
};

#define CSC(call) do { \
	cudaError_t res = call;	\
	if (res != cudaSuccess) { \
		fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
		exit(0); \
	} \
} while (0)

__global__ void swapStr(double* da, double* res, int n, int from, int to) {
	int offset = gridDim.x * blockDim.x;
	double tmp;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += offset) {
		tmp = da[idx * n + from];
		da[idx * n + from] = da[idx * n + to];
		da[idx * n + to] = tmp;
	}
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += offset) {
		tmp = res[idx * n + from];
		res[idx * n + from] = res[idx * n + to];
		res[idx * n + to] = tmp;
	}
}

__global__ void zeroColumn(double* da, double* res, int n, int cur) {
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	int idxx = blockDim.x * blockIdx.x + threadIdx.x;
	int idxy = blockDim.y * blockIdx.y + threadIdx.y;
	for(y = idxy; y < n; y += offsety) {
		for(x = idxx + cur + 1; x < n; x += offsetx) {
//			if (fabs(da[cur * n + x]) > 1e-7) {
				res[y * n + x] = res[y * n + x] - da[cur * n + x] * res[y * n + cur] / da[cur * n + cur];
//			}
		}
	}
	for(y = idxy + cur + 1; y < n; y += offsety) {
		for(x = idxx + cur + 1; x < n; x += offsetx) {
			da[y * n + x] = da[y * n + x] - da[cur * n + x] * da[y * n + cur] / da[cur * n + cur];
		}
	}
}

__global__ void zeroColumnUp(double* da, double* res, int n, int cur) {
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	int idxx = blockDim.x * blockIdx.x + threadIdx.x;
	int idxy = blockDim.y * blockIdx.y + threadIdx.y;
	for(y = idxy; y < n; y += offsety) {
		for(x = idxx; x < cur; x += offsetx) {
//			if (fabs(da[cur * n + x]) > 1e-7)
				res[y * n + x] = res[y * n + x] - res[y * n + cur] * da[cur * n + x] / da[cur * n + cur];
		}
	}
}

__global__ void divide(double* da, double* res, int n) {
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	int idxx = blockDim.x * blockIdx.x + threadIdx.x;
	int idxy = blockDim.y * blockIdx.y + threadIdx.y;
	for(y = idxy; y < n; y += offsety) {
		for(x = idxx; x < n; x += offsetx) {
			res[y * n + x] = res[y * n + x] / da[x * n + x];
		}
	}
}

int main() {
	ios_base::sync_with_stdio(false);
	comparator comp = comparator();
	int n;
	cin >> n;
	double* data = (double*)malloc(sizeof(double*) * n * n);
	
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			data[j * n + i] = (i == j);
		}
	}

	double* dev_res;
	cudaMalloc(&dev_res, sizeof(double) * n * n);
	cudaMemcpy(dev_res, data, sizeof(double) * n * n, cudaMemcpyHostToDevice);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			cin >> data[j * n + i];
		}
	}
	
	double* dev_arr;
	cudaMalloc(&dev_arr, sizeof(double) * n * n);
	cudaMemcpy(dev_arr, data, sizeof(double) * n * n, cudaMemcpyHostToDevice);

	thrust::device_ptr<double> p_arr = thrust::device_pointer_cast(dev_arr);
	thrust::device_ptr<double> res;

	for (int i = 0; i < n - 1; ++i) {
		res = thrust::max_element(p_arr + i * n + i, p_arr + i * n + n, comp);
		if ((int)(res - p_arr) % n > i) swapStr<<<256, 256>>>(dev_arr, dev_res, n, i, (int)(res - p_arr) % n);
		zeroColumn<<<dim3(16, 16), dim3(16, 16)>>>(dev_arr, dev_res, n, i);
	}

	for (int i = n - 1; i > 0; --i) {
		zeroColumnUp<<<dim3(16, 16), dim3(16, 16)>>>(dev_arr, dev_res, n, i);	
	}

	divide<<<dim3(16, 16), dim3(16, 16)>>>(dev_arr, dev_res, n);

	cudaMemcpy(data, dev_res, sizeof(double) * n * n, cudaMemcpyDeviceToHost);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			cout << fixed << setprecision(10) << data[j * n + i] << " ";
		}
		cout << endl;
	}

	cudaFree(dev_arr);
	cudaFree(dev_res);
	free(data);
	return 0;
}