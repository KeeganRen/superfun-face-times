#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

class ComputeDuplicates
{
public:
    void init();
    void processOptions(int argc, char **argv);
    void loadAllImages();
    void match();
    void matchBuildList();
    void saveMatchList();

    int argc;
    char **argv;

    char *imageListFile;
    char *outputFile;

    vector<string> imageNames;
    vector<Mat> images;

    map<int, vector<int> > matches;

};
