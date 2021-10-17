#include "cuda.h"
#include "view.h"
#include "converge.h"

__global__
void convergence(unsigned char *device_pixels, const int N, const view v)
{
    const int ite=500;
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
            n++;
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
            device_pixels[x2+y2]= (unsigned char)(255.0 * cbrt((float)n/(float)ite));
            device_pixels[(x2+y2)+1] = 0;
            device_pixels[(x2+y2)+2] = 0;
            device_pixels[(x2+y2)+3] = 255;

        }

    }
}
