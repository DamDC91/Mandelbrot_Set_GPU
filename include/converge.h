#ifndef converge_H
#define converge_H

#include "cuda.h"
#include "view.h"

__device__ int windSizeX;
__device__ int windSizeY;
__device__ int ite;
__device__ double *rgb;
__global__ void init(int nbIte, int windowSizeX, int windowSizeY, double red, double green, double blue, unsigned char *colors);
__global__ void convergence(unsigned char *device_pixels, const int N, const view v, unsigned char *colors);

#endif
