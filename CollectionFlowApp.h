#include <stdlib.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "CVOpticalFlow.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

using namespace std;

class CollectionFlowApp
{
public:
    void processOptions(int argc, char **argv);
    void init();
    void loadFacesFromList();
    void findImageSizeFromFirstImage();
    void openImages();
    void convertImages();
    void buildMatrixAndRunPca();
    void saveAs(char* filename, Mat m);
    
    void gslVecToMat(gsl_vector *vec, Mat &m);
    void matToGslVec(Mat &m, gsl_vector *vec);

    int argc;
    char **argv;
    
    char inputFile[1024];
    char outputDir[512];
    bool visualize;
    bool gray;
    vector<string> faceList;
    int w, h, d;
    vector<Mat> faceImages;
};
