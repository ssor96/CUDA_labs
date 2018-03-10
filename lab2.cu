#include <stdio.h>
#include <stdlib.h>
#include <cstring>

texture<uchar4, 2, cudaReadModeElementType> tex;

#define CSC(call) do { \
	cudaError_t res = call;	\
	if (res != cudaSuccess) { \
		fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
		exit(1); \
	} \
} while (0)

#define max(a, b) (a > b? a: b)
#define min(a, b) (a > b? b: a)

#define calc(val) for (i = 0; i < 256; ++i) cnt[i] = 0; \
			for (int dx = max(-r, -x); dx <= r && x + dx < w; ++dx) { \
				for (int dy = max(-r, -y); dy <= r && y + dy < h; ++dy) { \
					cnt[tex2D(tex, x + dx, y + dy).val]++; \
				} \
			} \
			cur = 0; \
			for (i = 0; i < 255 && 2 * cur <= all; ++i) { \
				cur += cnt[i]; \
			} \
			dst[y * w + x].val = i;

__global__ void kernel(uchar4 *dst, int w, int h, int r) {
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	ushort cnt[256];
	ushort cur;
	for(x = blockDim.x * blockIdx.x + threadIdx.x; x < w; x += offsetx) {
		for(y = blockDim.y * blockIdx.y + threadIdx.y; y < h; y += offsety) {
			int i;
			ushort all = (min(x + r, w - 1) - max(x - r, 0) + 1) * (min(y + r, h - 1) - max(y - r, 0) + 1);
			calc(x);
			calc(y);
			calc(z);
			dst[y * w + x].w = tex2D(tex, x, y).w;
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
	int r;
	scanf("%d", &r);
	cudaArray *arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	cudaMallocArray(&arr, &ch, w, h);
	cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * h * w, cudaMemcpyHostToDevice);

	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false; 

	cudaBindTextureToArray(tex, arr, ch);
	uchar4 *dev_data;
	cudaMalloc(&dev_data, sizeof(uchar4) * h * w);


	cudaEvent_t start, stop;
	float t;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&stop));

	CSC(cudaEventRecord(start, 0));
	
	kernel<<<dim3(8, 8), dim3(8, 8)>>>(dev_data, w, h, r);
	CSC(cudaGetLastError());
	
	CSC(cudaEventRecord(stop, 0));
	CSC(cudaEventSynchronize(stop));
	CSC(cudaEventElapsedTime(&t, start, stop));	
	printf("time = %f\n", t);
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(stop));

	cudaMemcpy(data, dev_data, sizeof(uchar4) * h * w, cudaMemcpyDeviceToHost);
	FILE *out = fopen(name, "wb");
	fwrite(&w, sizeof(int), 1, out);
	fwrite(&h, sizeof(int), 1, out);
	fwrite(data, sizeof(uchar4), w * h, out);
	fclose(out);

	cudaUnbindTexture(tex);
	cudaFreeArray(arr);
	cudaFree(dev_data);
	free(data);
	free(name);
	return 0;
}