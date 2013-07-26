#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

#include "CVOpticalFlow.h"


using namespace std;
using namespace cv;

class SwapApp
{
public:
    void init();
    void load(char *faceAFile, char *faceBFile, char *landmarkAFile, char *landmarkBFile, char *clusterPath);
    Mat makeFullFaceMask(Mat &frame, Mat &transform);
    Mat makeCroppedFaceMask(Mat &frame, Mat &transform);
    void swap();
    void animateBlend();
    Point2f applyXform(Point2f pt, Mat &xform);
    Point2f applyInverseXform(Point2f pt, Mat &xform);

    int argc;
    char **argv;
    char *outPath;

    Mat A, B, A_mask, B_mask;
    Mat BBlendedToA;
    vector<Point2f> landmarkA;
    vector<Point2f> landmarkB;

    Mat maskImage;
    vector<Point2f> templatePoints2D;

    Mat A_xform;
    Mat B_xform;

    int w, h, d;
    int num_eigenfaces, num_pixels;
    gsl_matrix *m_gsl_eigenfaces;
    gsl_vector *m_gsl_meanface;

    Mat A_vx, A_vy;
    Mat B_vx, B_vy;
    Mat vx, vy;

};
