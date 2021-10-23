#ifndef VIEW_H
#define VIEW_H
#include <string>

struct view {
    double Xmax;
    double Xmin;
    double Ymax;
    double Ymin;
};

void move(view *v, double dx, double dy);

void zoom(view *v, double d);

void save(view *v, std::string fileName);

view *load(std::string fileName);

#endif
