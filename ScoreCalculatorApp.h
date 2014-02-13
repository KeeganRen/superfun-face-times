#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_statistics.h>

#include "CVOpticalFlow.h"


using namespace std;
using namespace cv;

enum ScoreMetric { HOG_SCORE, FIDUCIAL_SCORE };

class ScoreCalculatorApp
{
public:
    void processOptions(int argc, char **argv);
    void init();

    void loadAndTransformFacePoints();

    Point2f applyXform(Point2f pt, Mat &xform);
    Point2f applyInverseXform(Point2f pt, Mat &xform);


    int argc;
    char **argv;
    char *outPath;

    Mat A, B, A_mask, B_mask, A_color_corrected, B_color_corrected;
    Mat BBlendedToA, ABlendedToB;
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

    // scoring stuff
    char* faceToScoreFile;
    char* facesToCompareFile;
    char* basePath;
    char *templatePointFile;
    char *outFile;

    char *scoringMetricWord;
    ScoreMetric metric;

    int faceIdToScore;
    vector<int> faceIdsToCompare;

    gsl_matrix *gsl_landmarks;

};
