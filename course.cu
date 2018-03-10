#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

const int w = 1024;
const int h = 680;
const int n = 500;
const float dt = 0.025;
const int prime_sz = 13577;
const float inert = 0.8;//0.9345 * 0.7;
const float loc_w = 0.3;
const float glob_w = 0.7;
const float force_w = 1.5;
const float eps = 1e-4;
const int pow_deg = 3;
const float pointR = 0.1;
const float sz = 2.1;//10 * pointR;

dim3 blocks(32, 32), threads(16,16);
dim3 blocksP(64), threadsP(32);

#define SQRTT(a) sqrtf(a)//(1 + ((a) / 2) - (((a) * (a)) / 8) + (a) * (a) * (a) / 16)
#define TOI(a) int((((a) - dev_xc) / dev_sx + 1) * (w - 1) / 2.0)
#define TOJ(a) int((((a) - dev_yc) / dev_sy + 1) * (h - 1) / 2.0)

#define TOOI(a) int((((a) - xc) / sx + 1) * (w - 1) / 2.0)
#define TOOJ(a) int((((a) - yc) / -sy + 1) * (h - 1) / 2.0)

#define CSC(call) {                         \
    cudaError err = call;                       \
    if(err != cudaSuccess) {                        \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",    \
            __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(1);                            \
    }                                   \
} while (0)

float xc = 0.0f, yc = 0.0f, sx = 100.0f, sy = sx * h / w, minf = 0, maxf = 1e+6;
int prime_ptr = 0;

float3 *dev_points;
float2 *dev_v, *dev_loc;
float *dev_f_val, *dev_prime;

__constant__ float dev_xc, dev_yc, dev_sx, dev_sy, dev_minf, dev_maxf, dev_p;
__constant__ float2 dev_glob;

__constant__ int blDimx, blDimy;
__constant__ float minx, miny;

__device__ float fun(float x, float y) {
    return (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
}

__device__ float fun(int i, int j)  {
    float x = 2.0f * i / (float)(w - 1) - 1.0f;
    float y = 2.0f * j / (float)(h - 1) - 1.0f; 
    return fun(x * dev_sx + dev_xc, -y * dev_sy + dev_yc);   
}

__device__ float fun(float2 p) {
    return fun(p.x, p.y);
}

__device__ float fun3(float3 p) {
    return fun(p.x, p.y);
}

__device__ int blId(float3 p) {
    return int((p.y - miny) / sz) * blDimx + int((p.x - minx) / sz);
}

struct cmp {
    __device__  bool operator()(float2 p1, float2 p2) {
        return fun(p1) < fun(p2);
    }
};

struct cmpx {
    __device__  bool operator()(float3 p1, float3 p2) {
        return p1.x < p2.x;
    }
};

struct cmpy {
    __device__  bool operator()(float3 p1, float3 p2) {
        return p1.y < p2.y;
    }
};

struct cmpBl {
    __device__  bool operator()(float3 p1, float3 p2) {
        return blId(p1) < blId(p2);
    }
};

__device__ float3 operator+(float3 p1, float3 p2) {
    return make_float3(p1.x + p2.x, p1.y + p2.y, 0);
}

__global__ void calcMap(float* f_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int i, j;
    for (i = idx; i < w; i += offsetx) {
        for (j = idy; j < h; j += offsety) {
            f_val[j * w + i] = fun(i, j);
        }
    }
}

__global__ void drawMap(uchar4* data, float* f_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int i, j;
    for (i = idx; i < w; i += offsetx) {
        for (j = idy; j < h; j += offsety) {
            float f = ((f_val[j * w + i] - dev_minf) / (dev_maxf - dev_minf));
            data[j * w + i] = make_uchar4(int(SQRTT(SQRTT(f)) * 255), 0, int((1 - SQRTT(SQRTT(f))) * 255), 255);
        }
    }
}

__global__ void drawPoints(uchar4* data, float3* points, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    for (;idx <= n; idx += offset) {
        if (idx == n) {
            for (int i = max(0, TOI(dev_glob.x - 2 * pointR) + 1); i < min(w, TOI(dev_glob.x + 2 * pointR)); ++i) {
                for (int j = max(0, TOJ(dev_glob.y - 2 * pointR) + 1); j < min(h, TOJ(dev_glob.y + 2 * pointR)); ++j) {
                    data[j * w + i] = make_uchar4(0, 255, 0, 255);
                }
            }
            break;
        }
        for (int i = max(0, TOI(points[idx].x - pointR) + 1); i < min(w, TOI(points[idx].x + pointR)); ++i) {
            for (int j = max(0, TOJ(points[idx].y - pointR) + 1); j < min(h, TOJ(points[idx].y + pointR)); ++j) {
                data[j * w + i] = make_uchar4(255, 255, 255, 255);
            }
        }
    }
}

// __global__ void updateV(float3* points, float2* v, float2* loc, float* prime, int n, float dt, int prime_ptr) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int offset = gridDim.x * blockDim.x;
//     int lidx;
//     for (;idx < n; idx += offset) {
//         lidx = int(points[idx].z);
//         v[idx].x = v[idx].x * inert 
//                 + (loc_w * prime[(prime_ptr + idx) % prime_sz] * (loc[lidx].x - points[idx].x) 
//                 + glob_w * prime[(prime_ptr + idx + 1) % prime_sz] * (dev_glob.x - points[idx].x)) * dt;
//         v[idx].y = v[idx].y * inert 
//                 + (loc_w * prime[(prime_ptr + idx + 2) % prime_sz] * (loc[lidx].y - points[idx].y) 
//                 + glob_w * prime[(prime_ptr + idx + 3) % prime_sz] * (dev_glob.y - points[idx].y)) * dt;

//         float2 force = make_float2(0, 0);

//         for (int i = 0; i < n; ++i) {
//             float d = sqrtf(powf(points[idx].x - points[i].x, 2) + powf(points[idx].y - points[i].y, 2));
//             if (d > eps) {
//                 force.x += (points[idx].x - points[i].x) * force_w / powf(d, pow_deg + 1);
//                 force.y += (points[idx].y - points[i].y) * force_w / powf(d, pow_deg + 1);
//             }
//         }
//         v[idx].x += force.x * dt;
//         v[idx].y += force.y * dt;
//     }
// }

__device__ int find(float3* points, int n, int beg, int bid) {
    int l = beg;
    int r = n - 1;
    while (l < r - 1) {
        int mid = (l + r) >> 1;
        if (blId(points[mid]) < bid) {
            l = mid;
        }
        else {
            r = mid;
        }
    }
    return l;
}

__global__ void updateV(float3* points, float2* v, float2* loc, float* prime, int n, float dt, int prime_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lidx;
    for (;idx < n; idx += gridDim.x * blockDim.x) {
        lidx = int(points[idx].z);
        v[idx].x = v[idx].x * inert 
                + (loc_w * prime[(prime_ptr + idx) % prime_sz] * (loc[lidx].x - points[idx].x) 
                + glob_w * prime[(prime_ptr + idx + 1) % prime_sz] * (dev_glob.x - points[idx].x)) * dt;
        v[idx].y = v[idx].y * inert 
                + (loc_w * prime[(prime_ptr + idx + 2) % prime_sz] * (loc[lidx].y - points[idx].y) 
                + glob_w * prime[(prime_ptr + idx + 3) % prime_sz] * (dev_glob.y - points[idx].y)) * dt;

        float2 force = make_float2(0, 0);

        int bid = blId(points[idx]) - blDimx;
        int cur = 0;
        for (int j = -1; j < 2; ++j) {
            cur = find(points, n, cur, bid - 1);
            int end = bid + 1;
            while (blId(points[cur]) <= end && cur < n) {
                float d = sqrtf(powf(points[idx].x - points[cur].x, 2) + powf(points[idx].y - points[cur].y, 2));
                if (d > eps) {
                    force.x += (points[idx].x - points[cur].x) * force_w / powf(d, pow_deg + 1);
                    force.y += (points[idx].y - points[cur].y) * force_w / powf(d, pow_deg + 1);
                }
                cur++;
            }
            bid += blDimx;
        }
        v[idx].x += force.x * dt;
        v[idx].y += force.y * dt;
    }
}

__global__ void movePoints(float3* points, float2* v, int n, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    for (;idx < n; idx += offset) {
        points[idx].x += v[idx].x * dt;
        points[idx].y += v[idx].y * dt;
        // points[idx].x = max(-dev_sx + dev_xc, min(dev_sx + dev_xc, points[idx].x));
        // points[idx].y = max(-dev_sy + dev_yc, min(dev_sy + dev_yc, points[idx].y));
    }
}

__global__ void updateLocalMin(float3* points, float2* loc, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    int lidx;
    for (;idx < n; idx += offset) {
        lidx = int(points[idx].z);
        if (fun3(points[idx]) < fun(loc[lidx])) {
            loc[lidx].x = points[idx].x;
            loc[lidx].y = points[idx].y;
        }
    }
}

void updateGlobalMin() {
    thrust::device_ptr<float2> dev_ptr = thrust::device_pointer_cast(dev_loc);
    float2 mn = thrust::min_element(dev_ptr, dev_ptr + n, cmp())[0];
    // printf("%f %f -> %f\n", mn.x, mn.y, ffun(mn));
    cudaMemcpyToSymbol(dev_glob, &mn, sizeof(float2));
}

struct cudaGraphicsResource *res;

void updateMinMax() {
	thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(dev_f_val);
	float mn = thrust::min_element(dev_ptr, dev_ptr + w * h)[0];
	float mx = thrust::max_element(dev_ptr, dev_ptr + w * h)[0];
	CSC(cudaMemcpyToSymbol(dev_minf, &mn, sizeof(float)));
    CSC(cudaMemcpyToSymbol(dev_maxf, &mx, sizeof(float)));
}

void updateCenter() {
    thrust::device_ptr<float3> dev_ptr = thrust::device_pointer_cast(dev_points);
    float3 sum = thrust::reduce(dev_ptr, dev_ptr + n, make_float3(0, 0, 0), thrust::plus<float3>());
    xc += (sum.x / n - xc) * dt;
    yc += (sum.y / n - yc) * dt;
    // float2 glob;
    // CSC(cudaMemcpyFromSymbol(&glob, dev_glob, sizeof(float2)));
    // xc = glob.x;
    // yc = glob.y;
    CSC(cudaMemcpyToSymbol(dev_xc, &xc, sizeof(float)));
    CSC(cudaMemcpyToSymbol(dev_yc, &yc, sizeof(float)));
}

void sortPoints(float3* points, int n) {
    thrust::device_ptr<float3> dev_ptr = thrust::device_pointer_cast(dev_points);
    float3 tmp = thrust::min_element(dev_ptr, dev_ptr + n, cmpx())[0];
    float mnx = tmp.x;
    tmp = thrust::max_element(dev_ptr, dev_ptr + n, cmpx())[0];
    float mxx = tmp.x;
    tmp = thrust::min_element(dev_ptr, dev_ptr + n, cmpy())[0];
    float mny = tmp.y;
    tmp = thrust::max_element(dev_ptr, dev_ptr + n, cmpy())[0];
    float mxy = tmp.y;
    int hblDimx = (mxx - mnx) / sz + 1;
    int hblDimy = (mxy - mny) / sz + 1;
    cudaMemcpyToSymbol(minx, &mnx, sizeof(float));
    cudaMemcpyToSymbol(miny, &mny, sizeof(float));
    cudaMemcpyToSymbol(blDimx, &hblDimx, sizeof(int));
    cudaMemcpyToSymbol(blDimy, &hblDimy, sizeof(int));
    thrust::sort(dev_ptr, dev_ptr + n, cmpBl());
}

int cnt = 0;
int fps = 0;
clock_t beg;
clock_t fbeg = clock();

void update() {
	// if (cnt == 0) {
	// 	fbeg = beg = clock();
 //        cnt = 1;
	// }
    // printf("%f %f\n", xc, yc);
    uchar4* dev_data;
    size_t size;
    CSC(cudaGraphicsMapResources(1, &res, 0));
    CSC(cudaGraphicsResourceGetMappedPointer((void**) &dev_data, &size, res));
    
    updateGlobalMin();
    sortPoints(dev_points, n);
    updateV<<<blocksP, threadsP>>> (dev_points, dev_v, dev_loc, dev_prime, n, dt, prime_ptr);
    movePoints<<<blocksP, threadsP>>> (dev_points, dev_v, n, dt);
    updateLocalMin<<<blocksP, threadsP>>> (dev_points, dev_loc, n);
    prime_ptr = (prime_ptr + 4 * n) % prime_sz;
    CSC(cudaDeviceSynchronize());
    updateCenter();
    calcMap<<<blocks, threads>>>(dev_f_val);
    updateMinMax();
    drawMap<<<blocks, threads>>>(dev_data, dev_f_val);
    drawPoints<<<blocksP, threadsP>>> (dev_data, dev_points, n);

    CSC(cudaGraphicsUnmapResources(1, &res, 0));

    glutPostRedisplay();
    fps++;
    if ((clock() - fbeg >= CLOCKS_PER_SEC)) {
        printf("fps = %lf\n", fps * CLOCKS_PER_SEC / double(clock() - fbeg));
        fbeg = clock();
        fps = 0;
    }
    // cnt++;
    // if (cnt % 10 == 0) {
    // 	printf("time = %lf\n", (clock() - beg) / float(CLOCKS_PER_SEC));
    // }
}

void display() {
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, 0);   
    glutSwapBuffers();
}

void keyFunc(unsigned char key, int x, int y) {
    float val = 0.1;
    switch (key) {
        case '-':
            sx += 2;
        case '+':
        case '=':
            sx = max(1.0f, sx - 1);
            sy = sx * h / w;
            CSC(cudaMemcpyToSymbol(dev_sy, &sy, sizeof(float)));
            CSC(cudaMemcpyToSymbol(dev_sx, &sx, sizeof(float)));
            return;
        case 'W':
            val *= 10;
        case 'w':
            yc += val;
            break;
        case 'A':
            val *= 10;
        case 'a':
            xc -= val;
            break;
        case 'S':
            val *= 10;
        case 's':
            yc -= val;
            break;
        case 'D':
            val *= 10;
        case 'd':
            xc += val;
            break;
        case 'c':
            xc = 1;
            yc = 1;
            break;
        case 't':
            sx = 25;
            sy = sx * h / w;
            CSC(cudaMemcpyToSymbol(dev_sy, &sy, sizeof(float)));
            CSC(cudaMemcpyToSymbol(dev_sx, &sx, sizeof(float)));
            return;
    }
    CSC(cudaMemcpyToSymbol(dev_xc, &xc, sizeof(float)));
    CSC(cudaMemcpyToSymbol(dev_yc, &yc, sizeof(float)));
}

int main(int argc, char** argv) {
    srand(time(NULL));
    CSC(cudaMalloc(&dev_points, n * sizeof(float3)));
    CSC(cudaMalloc(&dev_f_val, w * h * sizeof(float)));
    CSC(cudaMalloc(&dev_loc, n * sizeof(float2)));
    float3* ar = (float3*)malloc(n * sizeof(float3));
    float2* ar2 = (float2*)malloc(n * sizeof(float2));
    for (int i = 0; i < n; ++i) {
        ar2[i].x = ar[i].x = sx * (2 * (rand() / float(RAND_MAX)) - 1) + xc;
        ar2[i].x = ar[i].y = sy * (2 * (rand() / float(RAND_MAX)) - 1) + yc;
        ar[i].z = i + 0.5;
    }
    CSC(cudaMemcpy(dev_points, ar, n * sizeof(float3), cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(dev_loc, ar2, n * sizeof(float2), cudaMemcpyHostToDevice));
    CSC(cudaMalloc(&dev_v, n * sizeof(float2)));
    for (int i = 0; i < n; ++i) {
        ar2[i].x = 1000 * (2 * (rand() / float(RAND_MAX)) - 1);
        ar2[i].y = 1000 * (2 * (rand() / float(RAND_MAX)) - 1);
    }
    CSC(cudaMemcpy(dev_v, ar2, n * sizeof(float2), cudaMemcpyHostToDevice));
    free(ar);
    free(ar2);
    CSC(cudaMalloc(&dev_prime, prime_sz * sizeof(float)));
    float* prime = (float*)malloc(prime_sz * sizeof(float));
    for (int i = 0; i < prime_sz; ++i) {
        prime[i] = rand() / float(RAND_MAX);
    }
    CSC(cudaMemcpy(dev_prime, prime, prime_sz * sizeof(float), cudaMemcpyHostToDevice));
    free(prime);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(w, h);
    glutCreateWindow("Sorokin S A 80-408 awesssome course project");
    
    glutIdleFunc(update);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyFunc);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, (GLfloat) w, 0.0, (GLfloat) h);

    glewInit();
    
    CSC(cudaMemcpyToSymbol(dev_sx, &sx, sizeof(float)));
        CSC(cudaMemcpyToSymbol(dev_sy, &sy, sizeof(float)));
        CSC(cudaMemcpyToSymbol(dev_xc, &xc, sizeof(float)));
        CSC(cudaMemcpyToSymbol(dev_yc, &yc, sizeof(float)));
        CSC(cudaMemcpyToSymbol(dev_minf, &minf, sizeof(float)));

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);

    CSC(cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard));

    glutMainLoop();

    CSC(cudaGraphicsUnregisterResource(res));

    glBindBuffer(1, vbo);
    glDeleteBuffers(1, &vbo);
    CSC(cudaFree(dev_points));
    CSC(cudaFree(dev_prime));
    CSC(cudaFree(dev_f_val));
    return 0;
}
