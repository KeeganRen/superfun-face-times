#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

class AverageFaces
{
public:
    void init();
    void loadAndMakeAverage(char *file_list, char *output_image);

    int argc;
    char **argv;
};
