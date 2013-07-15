#include <stdlib.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "CVOpticalFlow.h"

#include <gsl/gsl_linalg.h>


using namespace std;
using namespace cv;

class CrossClusterMorphApp
{
public:
    void processOptions(int argc, char **argv);
    void init();
    void loadFace();
    void loadEigenfaces();
    void projectNewFace();

    int argc;
    char **argv;

    int w, h, d;
    int num_pixels;
    int num_eigenfaces;
    
    char *faceFile;
    char *avgFaceFile;
    char *eigenfacesFile;

    bool visualize;

    Mat faceImage;
    Mat avgFace;
    gsl_vector *m_gsl_face;
    gsl_matrix *m_gsl_eigenfaces;
    gsl_vector *m_gsl_meanface;

};
