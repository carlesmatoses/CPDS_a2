#include <math.h>
#include <float.h>
#include <cuda.h>

__global__ void gpu_Heat (float *h, float *g, int N) {

	// TODO: kernel computation
	int np = N;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i > 0 && i < np-1 && j > 0 && j < np-1) {
		g[i*np+j] = 0.25 * (h[(i-1)*np+j] + h[(i+1)*np+j] + h[i*np+(j-1)] + h[i*np+(j+1)]);
	}
}
