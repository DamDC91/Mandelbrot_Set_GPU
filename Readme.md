#  Mandelbrot set visualization
## Overview
The purpose of this project is to have a nice and interactive view of the [Mandelbrot fractal](https://en.wikipedia.org/wiki/Mandelbrot_set).

I first realized this project using C++ and SFML but I was not satisfied with the weak performances that I had on CPU, so I implemented it on my GPU with CUDA.
I am interested in learning GPU programming and it was a good opportunity because it was easy to parallelize the image generation. 
I am now highly satisfied of the performance reached even on my old graphic card.

demo

![demo](./media/media.gif) 

## Features
The software allows us to move around the fractal using the `arrows key` and zoom in/out using the `+ key`/`- key`. 
You can save the current view by pressing `S` and load it later using `L`, you can even capture the view into an image file (`capture.ppm`) using the letter `C`.

The software is configurable through a YAML file named `conf.yaml` . 
This file allows to set almost all variables used by the software such as window size, window frame rate, color gradient,  etc... But also variables that can heavily impact performances like the number of GPU thread per block or the number of loop iteration for each pixel.

## Suggested improvements

Currently I am sending the computed image from the device memory to the host memory and display it through SFML. A way of improvement would be to not send the computed image from the GPU to the CPU and directly print it to the screen from the graphic memory using openGL.

## Setup
You need a Nvidia graphic card and the package `nvidia-cuda-toolkit`
* Clone this projet
```bash
git clone https://github.com/DamDC91/Mandelbrot_Set_GPU
```
* Compile it using make
```bash
make
```
If your graphic is not detected it may be a driver issue.

