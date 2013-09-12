#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

class HogFeatures
{
public:
    void init();
    void processOptions(int argc, char **argv);

    int argc;
    char **argv;

    char *imageFile;
    char *imageListFile;
    char *outputFile;

    vector<string> images;
};
