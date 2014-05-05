#include <stdlib.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "CVOpticalFlow.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

using namespace std;

class TransferExpression
{
public:
    void processOptions(int argc, char **argv);
    void init();
    void load();
    void loadEigenfaces(char* eigenfacesDir, gsl_matrix **eigenfaces, gsl_vector **meanface);
    void loadFlow(char* eigenfacesDir, Mat &vx, Mat &vy, bool reverse);
    void step1();
    Mat projectFace(Mat &face, gsl_matrix* m_gsl_eigenfaces, gsl_vector* m_gsl_meanface);
    Mat warpFaceToMatch(Mat &faceToWarp, Mat &source, Mat &target);
    void computeFlow(Mat &source, Mat &target, Mat &vx, Mat &vy);

    void gslVecToMat(gsl_vector *vec, Mat &m);
    void matToGslVec(Mat &m, gsl_vector *vec);

    void saveAs(const char* filename, Mat m);

    int argc;
    char **argv;

    bool visualize;

    
    char *faceFile;
    char *faceListFile;
    char *sourceDir;
    char *referenceDir;
    char *targetDir; 
    char *saveExtension;

    vector<string> faceList;
    Mat face;

    gsl_matrix *m_gsl_source_eigenfaces;
    gsl_vector *m_gsl_source_meanface;

    gsl_matrix *m_gsl_reference_eigenfaces;
    gsl_vector *m_gsl_reference_meanface;

    gsl_matrix *m_gsl_target_eigenfaces;
    gsl_vector *m_gsl_target_meanface;

    Mat flow_source_ref_vx;
    Mat flow_source_ref_vy;

    Mat flow_ref_target_vx;
    Mat flow_ref_target_vy;



    int w, h, d;
    int num_pixels;
    int num_eigenfaces;

};
