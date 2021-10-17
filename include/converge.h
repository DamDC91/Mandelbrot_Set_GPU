#ifndef converge_H
#define converge_H

#include "cuda.h"
#include "view.h"

__device__ const int fen=1000;
__device__ const int ite=400;
__global__
void convergence(unsigned char *device_pixels, const int N, const view v);

#endif
