#include <SFML/Graphics.hpp>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>
#include <cuda.h>
#include <converge.h>
#include <view.h>
#include <chrono>
#include <unistd.h>
#include <configuration.h>


int main()
{
    conf c = loadConfiguration("conf.yaml");


    view current_view;
    current_view.Xmin = -1.0;
    current_view.Xmax = 1.0;
    current_view.Ymin = -1.0;
    current_view.Ymax = 1.0;


    sf::RenderWindow window(sf::VideoMode(c.windowSizeX, c.windowSizeY),"Mandelbrot set");
    sf::Texture texture;
    texture.create(c.windowSizeX, c.windowSizeY);
    unsigned char *pixels = (unsigned char *) malloc(c.windowSizeX * c.windowSizeY * 4 * sizeof(char));

    unsigned char *device_pixels;
    cudaError_t error= cudaMalloc((void **)&device_pixels, c.windowSizeX * c.windowSizeY * 4 * sizeof(char));

    window.setFramerateLimit(c.frameRate);

    init<<<1,c.iteration>>>(c.iteration, c.windowSizeX, c.red, c.green, c.blue);



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
            double d=std::abs(current_view.Xmax-current_view.Xmin)*c.moveStep;
            move(&current_view, -d, 0.0);  
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
        {
            double d=std::abs(current_view.Xmax-current_view.Xmin)*c.moveStep;
            move(&current_view, d, 0.0);  
        }
         if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))
        {
            double d=std::abs(current_view.Ymax-current_view.Ymin)*c.moveStep;
            move(&current_view, 0.0, d);   
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))
        {
            double d=std::abs(current_view.Ymax-current_view.Ymin)*c.moveStep;
            move(&current_view, 0.0, -d);
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Add))
        {
            zoom(&current_view, c.zoomStep);
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Subtract))
        {
            zoom(&current_view, -c.zoomStep);
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
        {
            window.close();
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
        {
            save(&current_view, "view.bin");
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::L))
        {
            view *l = load("view.bin");
            if (l != nullptr)
                current_view = *l;
        }
        
        int numBlock = c.windowSizeX * c.windowSizeY / c.GPUblock;
        convergence<<<numBlock,c.GPUblock>>>(device_pixels, c.windowSizeX * c.windowSizeY, current_view);
        error = cudaMemcpy(pixels,device_pixels, c.windowSizeX * c.windowSizeY * 4 * sizeof(char), cudaMemcpyDeviceToHost);

        if (error != cudaSuccess)
            std::cerr<<"GPUassert : " << cudaGetErrorString(error) << std::endl;

        texture.update((sf::Uint8 *)pixels);
        sf::Sprite sprite(texture);
        window.clear(sf::Color::White);
        window.draw(sprite);
        window.display();
    }
    return 0;
}

