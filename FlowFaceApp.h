#include <stdlib.h>
#include <string>
#include "CVOpticalFlow.h"

using namespace std;
using namespace cv;

class FlowFaceApp
{
public:
    void processOptions(int argc, char **argv);
    void loadAllImagesAndComputeFlow(int argc, char** argv);

    void init();
    void loadFaceImages();
    void computeFlow();

    int argc;
    char **argv;
    
    char *image1File;
    char *image2File;
    char *flowFile;

    Mat im1;
    Mat im2;
};
