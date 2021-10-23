#ifndef converge_H
#define converge_H

#include "cuda.h"
#include "view.h"

__device__ int windSizeX;
__device__ int windSizeY;
__device__ int ite;
__device__ unsigned char* colors;
__device__ bool *rgb;
__global__ void init(int nbIte, int hostFen, bool red, bool green, bool blue);
__global__ void convergence(unsigned char *device_pixels, const int N, const view v);

#endif
