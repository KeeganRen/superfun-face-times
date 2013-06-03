#include <stdlib.h>
#include <string>
#include "CVOpticalFlow.h"

using namespace std;
using namespace cv;

class MorphApp
{
public:
    void init();
    void loadFaceImages();
    void computeFlow();
    void saveAsF(char *filename, float i, Mat m);

    int argc;
    char **argv;
    
    char *a_file;
    char *b_file;
    char *c_file;
    char *d_file;

    char *out_file;

    Mat a, b, c, d;
};
