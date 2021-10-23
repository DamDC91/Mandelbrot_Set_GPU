#include "cuda.h"
#include "view.h"
#include "converge.h"
#include <stdio.h>

__global__
void init(int nbIte, int hostFen, bool red, bool green, bool blue)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i == 0)
    {
        ite = nbIte;
        fen = hostFen;
        colors = new unsigned char[nbIte];
        rgb = new bool[3];
        rgb[0] = red;
        rgb[1] = green;
        rgb[2] = blue;
    }
    __syncthreads();
    if (i < nbIte)
        colors[i]= (unsigned char)(255.0 * cbrt((float)i/(float)ite));
}

__global__
void convergence(unsigned char *device_pixels, const int N, const view v)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<N)
    {
        int x= idx % fen;
        int y = idx / fen;
        double cx = (x * (v.Xmax -v.Xmin) / double(fen) + v.Xmin);
        double cy = (y * (v.Ymin - v.Ymax) / double(fen) + v.Ymax);
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
        int x2 = x*4;
        int y2=y*fen*4;

        if(n==ite)
        {
            device_pixels[x2+y2]=0;
            device_pixels[(x2+y2)+1] = 0;
            device_pixels[(x2+y2)+2] = 0;
            device_pixels[(x2+y2)+3] = 255;
        }
        else
        {
            device_pixels[x2 + y2] = colors[n] * rgb[0];
            device_pixels[(x2+y2)+1] = colors[n] * rgb[1];
            device_pixels[(x2+y2)+2] = colors[n] * rgb[2];
            device_pixels[(x2+y2)+3] = 255;

        }

    }
}
