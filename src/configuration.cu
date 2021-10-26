#include <configuration.h>
#include <string>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <iostream>


template<typename T>
void readField(const YAML::Node &doc, T *Field, std::string FieldName) 
{
    if(doc[FieldName])
    {
        try {
            *Field = doc[FieldName].as<T>();
        }
        catch (const YAML::TypedBadConversion<T> e) {
            std::cerr << "invalid field " << FieldName << " : " << e.what() << std::endl;
        }
    }
}

void check(double *f) {
    if (*f < 0.0)
        *f = 0.0;
    if (*f > 1.0)
        *f = 1.0;
}

conf loadConfiguration(std::string filename)
{
    conf c = defaultConf;
    std::ifstream fin(filename);
    YAML::Node doc = YAML::Load(fin);
    readField<int>(doc, &c.windowSizeX, "windowSizeX");
    readField<int>(doc, &c.windowSizeY, "windowSizeY");
    readField<int>(doc, &c.iteration, "iteration"); 
    readField<double>(doc, &c.moveStep, "moveStep");
    readField<double>(doc, &c.zoomStep, "zoomStep");
    readField<int>(doc, &c.frameRate, "frameRate");
    readField<int>(doc, &c.GPUblock, "GPUblock");
    readField<double>(doc, &c.red, "red");
    readField<double>(doc, &c.green, "green");
    readField<double>(doc, &c.blue, "blue");
    check(&c.red);
    check(&c.green);
    check(&c.blue);

    return c;
    
}