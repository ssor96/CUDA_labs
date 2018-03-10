#include <stdio.h>

#define LOG_BANK_SIZE 5
#define BLOCK_SIZE 256
#define LOG_BLOCK_SIZE 8
#define BLOCK_NUM 8
#define IDX(n) (n + ((n) >> LOG_BANK_SIZE))

#define CSC(call) do { \
    cudaError_t res = call; \
    if (res != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(1); \
    } \
} while (0)

int getNearPow2(int n) {
    int m = BLOCK_SIZE;
    while (m < n) {
        m <<= 1;
    }
    return m;
}

__global__ void makeKey(uint* in, uint* key, int n, int shift) {
    int offset = gridDim.x * blockDim.x;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += offset) {
        key[i] = (in[i] >> shift) & 1;
    }
}
 
__global__ void scan(uint* in, uint* out, uint* s, int gn) {  
    volatile extern __shared__ uint temp[];
    int tid = threadIdx.x;
    int shift = BLOCK_SIZE * blockIdx.x;
    int n = BLOCK_SIZE;
    while (shift < gn) {
        int offset = 1;
        int ai = tid;
        int bi = tid + (n / 2);
        temp[IDX(ai)] = in[ai + shift];
        temp[IDX(bi)] = in[bi + shift];
        for (int d = n >> 1; d > 0; d >>= 1) {
            __syncthreads();
            if (tid < d) {
                int ai = offset * (2 * tid + 1) - 1;
                int bi = offset * (2 * tid + 2) - 1;
                temp[IDX(bi)] += temp[IDX(ai)];
            }
            offset *= 2;
        }
        if (tid == 0) {
            s[shift / BLOCK_SIZE] = temp[IDX(n - 1)];
            temp[IDX(n - 1)] = 0;
        }
        for (int d = 1; d < n; d <<= 1) {
            offset >>= 1;
            __syncthreads();
            if (tid < d) {
                int ai = offset * (2 * tid + 1) - 1;
                int bi = offset * (2 * tid + 2) - 1;
                uint t = temp[IDX(ai)];
                temp[IDX(ai)] = temp[IDX(bi)];
                temp[IDX(bi)] += t;
            }
        }
        __syncthreads();

        out[ai + shift] = temp[IDX(ai)];
        out[bi + shift] = temp[IDX(bi)];
        shift += gridDim.x * BLOCK_SIZE;
    }
}

__global__ void sum(uint* in, uint* s, int n) {
    int offset = gridDim.x * blockDim.x;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += offset) {
        in[i] += s[i >> LOG_BLOCK_SIZE];
    }
}

__global__ void swap(uint* in, uint* out, uint* s, int n, uint mask) {
    int beg = n - (s[n - 1] + !!(in[n - 1] & mask));
    int offset = gridDim.x * blockDim.x;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += offset) {
        if ((in[i] & mask) == 0) {
            out[i - s[i]] = in[i];
        }
        else {
            out[beg + s[i]] = in[i];
        }
    }
}

uint** scanArr;

void largeScan(uint* dev_in, uint* dev_s, int n, int k) {
    int sz = getNearPow2(n >> LOG_BLOCK_SIZE);
    scan<<<BLOCK_NUM, BLOCK_SIZE / 2, sizeof(uint) * BLOCK_NUM * (BLOCK_SIZE + (BLOCK_SIZE >> LOG_BANK_SIZE)) >>> (dev_in, dev_s, scanArr[k], n);
    if (n > BLOCK_SIZE) {
        largeScan(scanArr[k], scanArr[k + 1], sz, k + 2);
        sum<<<32, 32>>>(dev_s, scanArr[k + 1], n);
    }
}

void SortBit(uint* dev_in, uint* dev_out, uint* dev_s, int n, int i) {
    makeKey<<<32, 32>>>(dev_in, dev_out, n, i);
    largeScan(dev_out, dev_s, n, 0);
    swap<<<32, 32>>>(dev_in, dev_out, dev_s, n, 1u << i);
}

int main() {
    int n;
    fread(&n, sizeof(int), 1, stdin);
    if (n == 0) return 0;
    int sz = getNearPow2(n);
    uint* ar = (uint*)malloc(sz * sizeof(uint));
    fread(ar, sizeof(uint), n, stdin);

    for (int i = n; i < sz; ++i) {
        ar[i] = 0u - 1;
    }

    int tmp;
    int cnt = 0;
    for (tmp = sz; tmp > BLOCK_SIZE;) {
        cnt++;
        tmp = getNearPow2(tmp / BLOCK_SIZE);
    }
    
    scanArr = (uint**)malloc((2 * cnt + 1) * sizeof(uint*));
    tmp = sz;
    for (int i = 0; tmp > BLOCK_SIZE; i += 2) {
        tmp = getNearPow2(tmp / BLOCK_SIZE);
        CSC(cudaMalloc(&(scanArr[i]), sizeof(uint) * tmp));
        CSC(cudaMalloc(&(scanArr[i + 1]), sizeof(uint) * tmp));
    }
    CSC(cudaMalloc(&(scanArr[2 * cnt]), sizeof(uint) * BLOCK_SIZE));

    uint* dev_in;
    uint* dev_out;
    uint* dev_s;
    CSC(cudaMalloc(&dev_in, sizeof(uint) * sz));
    CSC(cudaMalloc(&dev_out, sizeof(uint) * sz));
    CSC(cudaMalloc(&dev_s, sizeof(uint) * sz));
    CSC(cudaMemcpy(dev_in, ar, sizeof(uint) * sz, cudaMemcpyHostToDevice));
    for (int i = 0; i < 32; i += 2) {
        SortBit(dev_in, dev_out, dev_s, sz, i);
        SortBit(dev_out, dev_in, dev_s, sz, i + 1);
    }

    CSC(cudaMemcpy(ar, dev_in, sizeof(uint) * n, cudaMemcpyDeviceToHost));
    for (int i = 0; i <= 2 * cnt; ++i) {
        CSC(cudaFree(scanArr[i]));
    }
    CSC(cudaFree(dev_in));
    CSC(cudaFree(dev_out));
    CSC(cudaFree(dev_s));
    free(scanArr);

    fwrite(ar, sizeof(uint), n, stdout);

    free(ar);
}