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
    current_view.Xmin = -1.5 * (1 + c.windowSizeX/c.windowSizeY * (c.windowSizeX > c.windowSizeY));
    current_view.Xmax = 0.5 * (1 + c.windowSizeX/c.windowSizeY * (c.windowSizeX > c.windowSizeY));
    current_view.Ymin = -1.0 * (1 +c.windowSizeY/c.windowSizeX* (c.windowSizeX < c.windowSizeY));
    current_view.Ymax = 1.0 * (1 + c.windowSizeY/c.windowSizeX* (c.windowSizeX < c.windowSizeY));


    sf::RenderWindow window(sf::VideoMode(c.windowSizeX, c.windowSizeY), "Mandelbrot set");
    sf::Texture texture;
    texture.create(c.windowSizeX, c.windowSizeY);
    unsigned char *pixels = new unsigned char[c.windowSizeX * c.windowSizeY * 4];

    unsigned char *device_pixels;
    cudaError_t error= cudaMalloc((void **) &device_pixels, c.windowSizeX * c.windowSizeY * 4 * sizeof(char));
    if (error != cudaSuccess)
        std::cerr<<"GPUassert : " << cudaGetErrorString(error) << std::endl;

    unsigned char *device_colors;
    error= cudaMalloc((void **)&device_colors, c.iteration);
    if (error != cudaSuccess)
        std::cerr<<"GPUassert : " << cudaGetErrorString(error) << std::endl;

    window.setFramerateLimit(c.frameRate);

    int numBlock = (float)c.iteration / (float)c.GPUblock + 1;
    init<<<numBlock,c.GPUblock>>>(c.iteration, c.windowSizeX, c.windowSizeY, c.red, c.green, c.blue, device_colors);
    cudaDeviceSynchronize();


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
            save(&current_view, ".view.bin");
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::L))
        {
            view *l = load(".view.bin");
            if (l != nullptr)
                current_view = *l;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::C))
        {
            capture(pixels, c.windowSizeX, c.windowSizeY, "capture.ppm");
        }

        
        numBlock = c.windowSizeX * c.windowSizeY / c.GPUblock + 1;
        convergence<<<numBlock,c.GPUblock>>>(device_pixels, c.windowSizeX * c.windowSizeY, current_view, device_colors);
        error = cudaMemcpy(pixels,device_pixels, c.windowSizeX * c.windowSizeY * 4 * sizeof(char), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
            std::cerr<<"GPUassert : " << cudaGetErrorString(error) << std::endl;

        texture.update((sf::Uint8 *)pixels);
        sf::Sprite sprite(texture);
        window.clear(sf::Color::White);
        window.draw(sprite);
        window.display();
    }
    cudaFree(device_colors);
    cudaFree(device_pixels);
    delete pixels;
    return 0;
}

