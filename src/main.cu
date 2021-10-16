#include <SFML/Graphics.hpp>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>
#include <cuda.h>

const int fenetre=1000;
__device__ const int fen=1000;

__device__ const unsigned int COLOR_TABLE[] = {
    0xf7df, 0xff5a, 0x07ff, 0x7ffa, 0xf7ff, 0xf7bb, 0xff38, 0xff59, 0x001f, 0x895c, 
    0xa145, 0xddd0, 0x5cf4, 0x7fe0, 0xd343, 0xfbea, 0x64bd, 0xffdb, 0xd8a7, 0x07ff, 
    0x0011, 0x0451, 0xbc21, 0xad55, 0x0320, 0xbdad, 0x8811, 0x5345, 0xfc60, 0x9999, 
    0x8800, 0xecaf, 0x8df1, 0x49f1, 0x2a69, 0x067a, 0x901a, 0xf8b2, 0x05ff, 0x6b4d, 
    0x1c9f, 0xd48e, 0xb104, 0xffde, 0x2444, 0xf81f, 0xdefb, 0xffdf, 0xfea0, 0xdd24, 
    0x8410, 0x0400, 0xafe5, 0xf7fe, 0xfb56, 0xcaeb, 0x4810, 0xfffe, 0xf731, 0xe73f, 
    0xff9e, 0x7fe0, 0xffd9, 0xaedc, 0xf410, 0xe7ff, 0xffda, 0xd69a, 0x9772, 0xfdb8, 
    0xfd0f, 0x2595, 0x867f, 0x839f, 0x7453, 0xb63b, 0xfffc, 0x07e0, 0x3666, 0xff9c, 
    0xf81f, 0x8000, 0x6675, 0x0019, 0xbaba, 0x939b, 0x3d8e, 0x7b5d, 0x07d3, 0x4e99, 
    0xc0b0, 0x18ce, 0xf7ff, 0xff3c, 0xff36, 0xfef5, 0x0010, 0xffbc, 0x8400, 0x6c64, 
    0xfd20, 0xfa20, 0xdb9a, 0xef55, 0x9fd3, 0xaf7d, 0xdb92, 0xff7a, 0xfed7, 0xcc27, 
    0xfe19, 0xdd1b, 0xb71c, 0x8010, 0xf800, 0xbc71, 0x435c, 0x8a22, 0xfc0e, 0xf52c, 
    0x2c4a, 0xffbd, 0xa285, 0xc618, 0x867d, 0x6ad9, 0x7412, 0xffdf, 0x07ef, 0x4416, 
    0xd5b1, 0x0410, 0xddfb, 0xfb08, 0x471a, 0xec1d, 0xd112, 0xf6f6, 0xffff, 0xf7be, 
    0xffe0, 0x9e66, 0x0000
};


struct view {
    double Xmax;
    double Xmin;
    double Ymax;
    double Ymin;
};

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
            unsigned int color = COLOR_TABLE[n];                
            device_pixels[x2+y2]=((color >> 11) & 0x1F) <<3;
            device_pixels[(x2+y2)+1] = ((color >>5) & 0x3F) << 2;
            device_pixels[(x2+y2)+2] = ((color & 0x1F) << 3);
            device_pixels[(x2+y2)+3] = 255;

        }

    }
}



int main()
{

    view current_view;
    current_view.Xmin = -1.0;
    current_view.Xmax = 1.0;
    current_view.Ymin = -1.0;
    current_view.Ymax = 1.0;

    sf::RenderWindow window(sf::VideoMode(fenetre, fenetre),"ensemble de Mandelbrot");
    sf::Image image;
    image.create(fenetre, fenetre, sf::Color(0, 0, 0));
    sf::Texture texture;
    texture.create(fenetre,fenetre);
    unsigned char *pixels =(unsigned char *) malloc(fenetre * fenetre * 4 *sizeof(char));

    unsigned char *device_pixels;
    cudaError_t error= cudaMalloc((void **)&device_pixels, fenetre * fenetre * 4 *sizeof(char));
    window.setFramerateLimit(30);



    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
        {
            double d=std::abs(current_view.Xmax-current_view.Xmin)*0.02;
            current_view.Xmax-=d;
            current_view.Xmin-=d;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
        {
            double d=std::abs(current_view.Xmax-current_view.Xmin)*0.02;
            current_view.Xmax+=d;
            current_view.Xmin+=d;
        }
         if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))
        {
            double d=std::abs(current_view.Ymax-current_view.Ymin)*0.02;
            current_view.Ymax+=d;
            current_view.Ymin+=d;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))
        {
            double d=std::abs(current_view.Ymax-current_view.Ymin)*0.02;
            current_view.Ymax-=d;
            current_view.Ymin-=d;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Add))
        {
            double dY= std::abs(current_view.Ymax-current_view.Ymin)*0.04;
            double dX= std::abs(current_view.Xmax-current_view.Xmin)*0.04;
            current_view.Ymax-=dY;
            current_view.Ymin+=dY;
            current_view.Xmax-=dX;
            current_view.Xmin+=dX;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Subtract))
        {
            double dY= std::abs(current_view.Ymax-current_view.Ymin)*0.04;
            double dX= std::abs(current_view.Xmax-current_view.Xmin)*0.04;
            current_view.Ymax+=dY;
            current_view.Ymin-=dY;
            current_view.Xmax+=dX;
            current_view.Xmin-=dX;
        }



    int block = 1024;
    int numblock = fenetre*fenetre / block;
    convergence<<<numblock,block>>>(device_pixels, fenetre*fenetre, current_view);
    error = cudaMemcpy(pixels,device_pixels,fenetre*fenetre*4*sizeof(char),cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        fprintf(stderr,"GPUassert: %s \n", cudaGetErrorString(error));
    texture.update((sf::Uint8 *)pixels);
    sf::Sprite sprite(texture);

    window.clear(sf::Color::White);
    window.draw(sprite);
    window.display();
    }
    return 0;
}

