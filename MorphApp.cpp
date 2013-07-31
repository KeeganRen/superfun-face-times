/* MorphApp.cpp */

#include "MorphApp.h"

#include "getopt.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <opencv/highgui.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <istream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;
#define e3 at<Vec3b>

void error(char *msg) {
    perror(msg);
    exit(1);
}

void PrintUsage() 
{
    printf("Usage:  \n" \
            "   \tto morph from image a to b via intermediate neutral expression...\n" \
            "   \t<image 1> <low rank 1> <low rank 2> <image 2> <output-pattern>\n" \
            "\n");
}

void MorphApp::init(){
    printf("[init] Running program %s\n", argv[0]);
    
    if (argc < 5 ){
        PrintUsage();
        exit(0);
    }
    
    a_file = argv[1];
    b_file = argv[2];
    c_file = argv[3];
    d_file = argv[4];
    out_file = argv[5];

    loadFaceImages();
    computeFlow();
   
}

void MorphApp::loadFaceImages(){
    printf("[loadFaceImages] loading image of face from files %s, %s, %s, %s\n", a_file, b_file, c_file, d_file);
    a = imread(a_file);
    b = imread(b_file);
    c = imread(c_file);
    d = imread(d_file);

    a.convertTo(a, CV_64FC3, 1.0/255, 0);
    b.convertTo(b, CV_64FC3, 1.0/255, 0);
    c.convertTo(c, CV_64FC3, 1.0/255, 0);
    d.convertTo(d, CV_64FC3, 1.0/255, 0);
}

void MorphApp::computeFlow(){
    printf("[computeFlow]\n");
    // magic variables
    double alpha = 0.03;
    double ratio = 0.85;
    int minWidth = 20;
    int nOuterFPIterations = 4;
    int nInnerFPIterations = 1;
    int nSORIterations = 40;

    bool visualize = false;

    if (visualize) { 
        imshow("a", a);
        imshow("b", b);
        imshow("c", c);
        imshow("d", d);
    }

    Mat vx_ab, vy_ab, warp_ab;
    
    CVOpticalFlow::findFlow(vx_ab, vy_ab, warp_ab, a, b, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
    if (visualize) imshow("warp_ab", warp_ab);
    if (visualize) imshow("flow_ab", CVOpticalFlow::showFlow(vx_ab, vy_ab));


    Mat vx_cd, vy_cd, warp_cd;
    
    CVOpticalFlow::findFlow(vx_cd, vy_cd, warp_cd, c, d, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
    if (visualize) imshow("warp_cd", warp_cd);
    if (visualize) imshow("flow_cd", CVOpticalFlow::showFlow(vx_cd, vy_cd));


    Mat out_x = Mat(vx_ab.size(), CV_64F);
    Mat out_y = Mat(vy_ab.size(), CV_64F);
    CVOpticalFlow::compositeFlow(vx_ab, vy_ab, vx_cd, vy_cd, out_x, out_y);

    if (visualize) imshow("flow_out", CVOpticalFlow::showFlow(out_x, out_y));

    char new_filename[512];
    sprintf(new_filename, "%s_flow.png", out_file);

    Mat composedFlow = CVOpticalFlow::showFlow(out_x, out_y);
    Mat new_m = Mat(composedFlow.size(), CV_8UC3);
    composedFlow.convertTo(new_m, CV_8UC3, 1.0*255, 0);
    imwrite(new_filename, new_m);


    Mat vx_ad, vy_ad, warp_ad;
    
    /*
    CVOpticalFlow::findFlow(vx_ad, vy_ad, warp_ad, a, d, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
    if (visualize) {
        imshow("warp_ad", warp_ad);
        imshow("flow_ad", CVOpticalFlow::showFlow(vx_ad, vy_ad));
        waitKey(0);
    }
    */

    Mat morph(a);
    //ImageProcessing::warpImage(morph, im1, im2, vx, vy, im1.rows, im1.cols, 3);
    for (float dt = 0; dt <= 1.1; dt += .1){
        CVOpticalFlow::warpInterpolation(morph, a, d, out_x, out_y, dt);
        if (visualize) imshow("morph", morph);
        saveAsF(out_file, dt, morph);

        /*
        CVOpticalFlow::warpInterpolation(morph, a, b, vx_ab, vy_ab, dt);
        imshow("morph_ab", morph);
        saveAsF("morph_ab", dt, morph);

        CVOpticalFlow::warpInterpolation(morph, c, d, vx_cd, vy_cd, dt);
        imshow("morph_cd", morph);
        saveAsF("morph_cd", dt, morph);

        CVOpticalFlow::warpInterpolation(morph, a, d, vx_ad, vy_ad, dt);
        if (visualize) imshow("morph_ad", morph);
        saveAsF("morph_ad", dt, morph);
        */
    }
    
}

void MorphApp::saveAsF(char *filename, float i, Mat m){
    char new_filename[512];
    sprintf(new_filename, "%s_%f.jpg", filename, i);
    Mat new_m = Mat(m.size(), CV_8UC3);
    m.convertTo(new_m, CV_8UC3, 1.0*255, 0);
    imwrite(new_filename, new_m);
}

static MorphApp *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new MorphApp();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

