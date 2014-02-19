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
    Mat computeImageHistogram(Mat img, Mat img2);
    void convertImages();
    void buildMatrixAndRunPca();
    void saveAs(char* filename, Mat m);
    void saveAsCropBorder(char* filename, Mat m);
    void saveBinaryEigenface(char* filename, gsl_vector *f);
    
    void gslVecToMat(gsl_vector *vec, Mat &m);
    void gslVecToMatWithBorder(gsl_vector *vec, Mat &m);
    void matToGslVec(Mat &m, gsl_vector *vec);
    void matToGslVecWithBorder(Mat &m, gsl_vector *vec);

    const char* faceFileName(int i);

    int argc;
    char **argv;
    
    char* inputFile;
    char* outputDir;
    char* maskFile;
    bool visualize;
    bool gray;
    vector<string> faceList;
    int w, h, d;
    vector<Mat> faceImages;

    float scale;

    const static int borderSize = 20;
};
