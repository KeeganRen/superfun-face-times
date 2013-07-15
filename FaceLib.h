//#include <opencv/highgui.h>
//#include "opencv2/imgproc/imgproc.hpp"
#include <istream>
#include <opencv2/opencv.hpp>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

using namespace cv;

class FaceLib {
    public:
        // magic optical flow variables
        const static double alpha = 0.02;
        const static double ratio = 0.85;
        const static int minWidth = 20;
        const static int nOuterFPIterations = 4;
        const static int nInnerFPIterations = 1;
        const static int nSORIterations = 40;

        const static int borderSize = 20;

        static void gslVecToMat(gsl_vector *orig, Mat &im, int d, int w, int h);
        static void gslVecToMatWithBorder(gsl_vector *orig, Mat &im, int d, int w, int h);
        static void matToGslVec(Mat &im, gsl_vector *vec, int d, int w, int h);
        static void matToGslVecWithBorder(Mat &im, gsl_vector *vec, int d, int w, int h);
        static Mat  addBorder(Mat im);
        static Mat  removeBorder(Mat im);
        static void saveAs(char* filename, Mat m);
        static Mat computeFlowAndWarp(Mat &face1, Mat &face2);
        static Mat computeFlowAndApplyFlow(Mat &face2, Mat &face1, Mat &faceToWarp);
};

