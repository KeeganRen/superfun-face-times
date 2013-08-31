#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "ANN/ANN.h"

using namespace std;
using namespace cv;

class PhotoBioApp
{
public:
    void init();
    void load(char *listFile);
    void loadFeaturesAndDoStuff(char* outfile);
    int countFeatures(string feature_file);
    void readFeatures(int *features, string feature_file);

    int argc;
    char **argv;

    vector<string> featureFiles;
    vector<int> imageIds;
    vector<int> imageCategories;

    int num_faces, num_features;

    ANNpointArray pts;
    ANNkd_tree *tree;
    vector<int> seq;

    void findNextPic(int idx);

};
