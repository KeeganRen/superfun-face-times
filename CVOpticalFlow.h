/* This code is from Ce Liu's Optical Flow code: http://people.csail.mit.edu/celiu/OpticalFlow/ */
/* This particular file is from Supasorn Aek Suwajanakorn with some additions by Kathleen Tuite. */

#include "project.h"
#include "Image.h"
#include "OpticalFlow.h"

using namespace cv;

class CVOpticalFlow {
  public:
    // Move im2 to make it look like im1
    static void findFlow(Mat &vx, Mat &vy, Mat &warp, Mat &im1, Mat &im2, double alpha, double ratio, int minWidth, int nOuterFPIterations, int nInnerFPIterations, int nSORIterations);
    static Mat showFlow(Mat &vx, Mat &vy);
    static void warp(Mat &out, Mat &im, Mat &vx, Mat &vy);
    static void warpInterpolation(Mat &out, Mat &im1, Mat &im2, Mat &vx, Mat &vy, float dt);
    static void bilinear(double *out, Mat &im, double r, double c, int channels);
    static double bilinearFlow(Mat &im, double r, double c);
    static void compositeFlow(Mat &ax, Mat &ay, Mat &bx, Mat &by, Mat &outx, Mat &outy);
  private:
    static inline void clip(int &a, int lo, int hi);

};


