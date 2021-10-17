#include <SFML/Graphics.hpp>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>
#include <cuda.h>
#include "converge.h"
#include "view.h"

const int fenetre=1000;

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

