#include <configuration.h>
#include <string>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <iostream>

void readIntField(const YAML::Node &doc, int *field, std::string fieldName)
{
    if(doc[fieldName])
    {
        try {
            *field = doc[fieldName].as<int>();
        }
        catch (const YAML::TypedBadConversion<int> e) {
            std::cerr << "invalid field " << fieldName << " : " << e.what() << std::endl;
        }
    }
}

void readDoubleField(const YAML::Node &doc, double *field, std::string fieldName)
{
    if(doc[fieldName])
    {
        try {
            *field = doc[fieldName].as<double>();
        }
        catch (const YAML::TypedBadConversion<double> e) {
            std::cerr << "invalid field " << fieldName << " : " << e.what() << std::endl;
        }
    }
}

void readBoolField(const YAML::Node &doc, bool *field, std::string fieldName)
{
    if(doc[fieldName])
    {
        try {
            *field = doc[fieldName].as<bool>();
        }
        catch (const YAML::TypedBadConversion<bool> e) {
            std::cerr << "invalid field " << fieldName << " : " << e.what() << std::endl;
        }
    }
}


conf loadConfiguration(std::string filename)
{
    conf c = defaultConf;
    std::ifstream fin(filename);
    YAML::Node doc = YAML::Load(fin);
    readIntField(doc, &c.windowSizeX, "windowSizeX");
    readIntField(doc, &c.windowSizeY, "windowSizey");
    readDoubleField(doc, &c.moveStep, "moveStep");
    readDoubleField(doc, &c.zoomStep, "zoomStep");
    readIntField(doc, &c.frameRate, "frameRate");
    readIntField(doc, &c.GPUblock, "GPUblock");
    readBoolField(doc, &c.red, "red");
    readBoolField(doc, &c.blue, "green");
    readBoolField(doc, &c.green, "blue");
    return c;
    
}