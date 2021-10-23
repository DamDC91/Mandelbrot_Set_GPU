#include <view.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

void move(view *v, double dx, double dy)
{
    v->Xmax+=dx;
    v->Xmin+=dx;
    v->Ymax+=dy;
    v->Ymin+=dy;
}

void zoom(view *v, double d)
{
    double dY= std::abs(v->Ymax-v->Ymin)*d;
    double dX= std::abs(v->Xmax-v->Xmin)*d;
    v->Ymax-=dY;
    v->Ymin+=dY;
    v->Xmax-=dX;
    v->Xmin+=dX;
}

void save(view *v, std::string fileName)
{
    std::ofstream wf(fileName, std::ofstream::out | std::ofstream::binary);
    if(!wf) 
    {
      std::cerr << "Cannot open file!" << std::endl;
      return;
    }
    wf.write((char *)v , sizeof(*v));
    wf.close();
}

view *load(std::string fileName)
{
    std::ifstream rf(fileName, std::ofstream::in | std::ifstream::binary);
    if(!rf) 
    {
      std::cerr << "Cannot open file!" << std::endl;
      return nullptr;
    }
    rf.seekg (0, rf.end);
    int length = rf.tellg();
    if (length != sizeof(view))
        std::cerr << "invalid file " << fileName << std::endl;
    rf.seekg (0, rf.beg);
    char *buffer = new char[length];
    rf.read(buffer , length);
    rf.close();
    return (view *) buffer;
}
