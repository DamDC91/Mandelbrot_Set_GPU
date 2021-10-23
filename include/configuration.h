#ifndef configuration_H
#define configuration_H
#include <string>

struct conf {
    int windowSizeX;
    int windowSizeY;    
    int iteration;
    double moveStep;
    double zoomStep; 
    int frameRate;
    int GPUblock;
    bool red;
    bool green;
    bool blue;
};

const conf defaultConf = {800, 800, 400, 0.02, 0.04, 30, 1024, true, false, false};

conf loadConfiguration(std::string filename);

#endif