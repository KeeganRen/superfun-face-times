#include <stdlib.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "CVOpticalFlow.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

using namespace std;

class ProjectFace
{
public:
    void processOptions(int argc, char **argv);
    void init();
    void loadFaces();
    void findImageSizeFromFirstImage();
    void loadEigenfaces();
    double projectAndWarpFace(Mat face);

    void projectNewFace();
    double warpNewFace();
    void saveAs(const char* filename, Mat m);
    
    void gslVecToMat(gsl_vector *vec, Mat &m);
    void matToGslVec(Mat &m, gsl_vector *vec);

    const char* faceFileName(char* f);

    int argc;
    char **argv;

    vector<string> faceList;
    vector<Mat> faceImages;
    char *eigenfacesDir;

    char *extension;

    char *facesFile;
    bool visualize;

    char outputDir[1024];

    Mat newFace;
    Mat avgFace;
    Mat warped;
    Mat lowRank;
    Mat flowField;
    Mat flow_vx, flow_vy;
    Mat reverse_flow_vx, reverse_flow_vy;
    int numFaces;

    int w, h, d;
    gsl_matrix *m_gsl_eigenfaces;
    gsl_vector *m_gsl_meanface;
    gsl_vector *m_gsl_newface;
    gsl_vector *m_gsl_projectednewface;

    int num_pixels;
    int num_eigenfaces;

};
