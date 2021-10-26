#include "cuda.h"
#include "view.h"
#include "converge.h"
#include <stdio.h>

__global__
void init(int nbIte, int windowSizeX, int windowSizeY, double red, double green, double blue, unsigned char *colors)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i == 0)
    {
        ite = nbIte;
        windSizeX = windowSizeX;
        windSizeY = windowSizeY;
        rgb = new double[3];
        rgb[0] = red;
        rgb[1] = green;
        rgb[2] = blue;
    }
    __syncthreads();
    if (i < nbIte)
        colors[i]= (unsigned char)(255.0 * cbrt((float)i/(float)ite));
}

__global__
void convergence(unsigned char *device_pixels, const int N, const view v, unsigned char *colors)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<N)
    {
        int y = idx / windSizeX;
        int x = idx - y * windSizeX; 
        double cx = (x * (v.Xmax -v.Xmin) / double(windSizeX) + v.Xmin);
        double cy = (y * (v.Ymin - v.Ymax) / double(windSizeY) + v.Ymax);
        double zn_r = 0.0;
        double zn_i = 0.0;
        int n = 0;

        while ( zn_r * zn_r + zn_i * zn_i < 4.0 && n < ite)
        {
            double tmp = zn_r *zn_r - zn_i * zn_i + cx;
            zn_i = 2 * zn_i * zn_r + cy;
            zn_r = tmp;
            ++n;
        }

        if(n == ite)
        {
            device_pixels[idx*4]=0;
            device_pixels[(idx*4)+1] = 0;
            device_pixels[(idx*4)+2] = 0;
            device_pixels[(idx*4)+3] = 255;
        }
        else
        {
            device_pixels[idx*4] = colors[n] * rgb[0];
            device_pixels[(idx*4)+1] = colors[n] * rgb[1];
            device_pixels[(idx*4)+2] = colors[n] * rgb[2];
            device_pixels[(idx*4)+3] = 255;

        }

    }
}
