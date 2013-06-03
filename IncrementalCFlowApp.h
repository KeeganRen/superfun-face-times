#include <stdlib.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "CVOpticalFlow.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

using namespace std;

class IncrementalCFlowApp
{
public:
    void processOptions(int argc, char **argv);
    void init();
    void loadFaces();
    void loadEigenfaces();
    void findImageSizeFromFirstImage();
    void projectNewFace();
    void warpNewFace();
    void makeNewAvg();
    void buildMatrixAndRunPca();
    void saveAs(char* filename, Mat m);
    void saveBinaryEigenface(char* filename, gsl_vector *f);
    
    void gslVecToMat(gsl_vector *vec, Mat &m);
    void matToGslVec(Mat &m, gsl_vector *vec);

    const char* faceFileName(char* f);

    int argc;
    char **argv;
    
    char newFaceFile[1024];
    char avgFaceFile[1024];
    char eigenfacesFile[1024];
    bool visualize;

    char outputDir[1024];

    Mat newFace;
    Mat avgFace;
    Mat warped;
    int numFaces;

    int w, h, d;
    gsl_matrix *m_gsl_eigenfaces;
    gsl_vector *m_gsl_meanface;
    gsl_vector *m_gsl_newface;
    gsl_vector *m_gsl_projectednewface;

    int num_pixels;
    int num_eigenfaces;

    vector<Mat> faceImages;
};
