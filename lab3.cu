#include <stdio.h>
#include <stdlib.h>
#include <cstring>

#define CSC(call) do { \
	cudaError_t res = call;	\
	if (res != cudaSuccess) { \
		fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
		exit(1); \
	} \
} while (0)

#define mult(a, b) ((data[y[j] * w + x[j]].a - avg[i].a) * (data[y[j] * w + x[j]].b - avg[i].b))
#define calc(a) ((av[i].x - p.x) * cov[i][0][a] + (av[i].y - p.y) * cov[i][1][a] + (av[i].z - p.z) * cov[i][2][a])

__constant__ double cov[32][3][3];
__constant__ double av[32][3];

__global__ void kernel(uchar4 *dst, int w, int h, int nc) {
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	for(x = blockDim.x * blockIdx.x + threadIdx.x; x < w; x += offsetx) {
		for(y = blockDim.y * blockIdx.y + threadIdx.y; y < h; y += offsety) {
			double mn;
			int id = -1;
			uchar4 p = dst[y * w + x];
			for (int i = 0; i < nc; ++i) {
				double cur[3] = {p.x - av[i][0], p.y - av[i][1], p.z - av[i][2]};
				double val[3] = {0, 0, 0};
				for (int j = 0; j < 3; ++j) {
					for (int k = 0; k < 3; ++k) {
						val[j] += cur[k] * cov[i][k][j];
					}
				}
				double vval = 0;
				for (int j = 0; j < 3; ++j) {
					vval += val[j] * cur[j];
				}
				//double cur = calc(0) * (av[i].x - p.x) + calc(1) * (av[i].y - p.y) + calc(2) * (av[i].z - p.z);
				if (vval < mn || id == -1) {
					mn = vval;
					id = i;
				}
			}
			dst[y * w + x].w = id;
		}
	}
}

int main() {
	int w, h;
	size_t sz = 256;
	char* name = (char*) malloc(sizeof(char) * sz);
	name[getline(&name, &sz, stdin) - 1] = '\0';
	FILE *in = fopen(name, "rb");
	fread(&w, sizeof(int), 1 , in);
	fread(&h, sizeof(int), 1 , in);
	uchar4 *data = (uchar4*)malloc(sizeof(uchar4) * h * w);
	fread(data, sizeof(uchar4), h * w, in);
	fclose(in);

	name[getline(&name, &sz, stdin) - 1] = '\0';
	
	int nc;
	scanf("%d", &nc);
	int x[1 << 19];
	int y[1 << 19];
	double matr[32][3][3];
	double inv[32][3][3];
	double avg[32][3];
	for (int i = 0; i < nc; ++i) {
		int np;
		scanf("%d", &np);
		int avgt[3] = {0, 0, 0};
		
		for (int j = 0; j < np; ++j) {
			scanf("%d %d", x + j, y + j);
			avgt[0] += data[y[j] * w + x[j]].x;
			avgt[1] += data[y[j] * w + x[j]].y;
			avgt[2] += data[y[j] * w + x[j]].z;
		}


		for (int j = 0; j < 3; ++j) {
			avg[i][j] = avgt[j] / double(np);
		}

		for (int k = 0; k < 3; ++k) {
			for (int l = 0; l < 3; ++l) {
				matr[i][k][l] = 0;
			}
		}

		for (int j = 0; j < np; ++j) {
			double cur[3] = {avg[i][0] - data[y[j] * w + x[j]].x, avg[i][1] - data[y[j] * w + x[j]].y, avg[i][2] - data[y[j] * w + x[j]].z};
			for (int k = 0; k < 3; ++k) {
				for (int l = 0; l < 3; ++l) {
					matr[i][l][k] += cur[k] * cur[l];
				}
			}
		}
		for (int k = 0; k < 3; ++k) {
			for (int l = 0; l < 3; ++l) {
				matr[i][k][l] /= np - 1;
			}
		}
		double det = matr[i][0][0] * (matr[i][1][1] * matr[i][2][2] - matr[i][1][2] * matr[i][2][1]) -
					 matr[i][0][1] * (matr[i][1][0] * matr[i][2][2] - matr[i][1][2] * matr[i][2][0]) + 
					 matr[i][0][2] * (matr[i][1][0] * matr[i][2][1] - matr[i][1][1] * matr[i][2][0]);

		inv[i][0][0] = (matr[i][1][1] * matr[i][2][2] - matr[i][1][2] * matr[i][2][1]) / det;
		inv[i][1][0] = -(matr[i][1][0] * matr[i][2][2] - matr[i][1][2] * matr[i][2][0]) / det;
		inv[i][2][0] = (matr[i][1][0] * matr[i][2][1] - matr[i][1][1] * matr[i][2][0]) / det;

		inv[i][0][1] = -(matr[i][0][1] * matr[i][2][2] - matr[i][0][2] * matr[i][2][1]) / det;
		inv[i][1][1] = (matr[i][0][0] * matr[i][2][2] - matr[i][0][2] * matr[i][2][0]) / det;
		inv[i][2][1] = -(matr[i][0][0] * matr[i][2][1] - matr[i][0][1] * matr[i][2][0]) / det;

		inv[i][0][2] = (matr[i][0][1] * matr[i][1][2] - matr[i][0][2] * matr[i][1][1]) / det;
		inv[i][1][2] = -(matr[i][0][0] * matr[i][1][2] - matr[i][0][2] * matr[i][1][0]) / det;
		inv[i][2][2] = (matr[i][0][0] * matr[i][1][1] - matr[i][0][1] * matr[i][1][0]) / det;
	}

	CSC(cudaMemcpyToSymbol(cov, inv, sizeof(double) * nc * 9));
	CSC(cudaMemcpyToSymbol(av, avg, sizeof(double) * nc * 3));

	uchar4 *dev_data;
	CSC(cudaMalloc(&dev_data, sizeof(uchar4) * h * w));
	CSC(cudaMemcpy(dev_data, data, sizeof(uchar4) * h * w, cudaMemcpyHostToDevice));
	kernel<<<dim3(16, 16), dim3(16, 16)>>>(dev_data, w, h, nc);
	CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * h * w, cudaMemcpyDeviceToHost));
	FILE *out = fopen(name, "wb");
	fwrite(&w, sizeof(int), 1, out);
	fwrite(&h, sizeof(int), 1, out);
	fwrite(data, sizeof(uchar4), w * h, out);
	fclose(out);

	CSC(cudaFree(dev_data));
	free(data);
	free(name);
	return 0;
}