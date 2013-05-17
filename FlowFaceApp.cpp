/* FlowFaceApp.cpp */

#include "FlowFaceApp.h"

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
            "   \t--image1 [first image]\n" \
            "   \t--image2 [second image]\n" \
            "\n");
}

void FlowFaceApp::processOptions(int argc, char **argv){
    while (1){
        static struct option long_options[] = {
            {"help",        0, 0, 'h'},
            {"image1",       1, 0, 400},
            {"image2",      1, 0, 401},
            {0,0,0,0} 
        };

        int option_index;;
        int c = getopt_long(argc, argv, "f:do:a:i:x",
                long_options, &option_index);

        if (c == -1)
            break;

        switch (c) {
            case 'h':
                PrintUsage();
                exit(0);
                break;
                
            case 400:
                sprintf(image1File, "%s", optarg);
                break;

            case 401:
                sprintf(image2File, "%s", optarg);
                break;

            default: 
                printf("Unrecognized option %d\n", c);
                break;
        }
    }
}

void FlowFaceApp::init(){
    printf("[init] Running program %s\n", argv[0]);
    
    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }
    
    processOptions(argc, argv);

    loadFaceImages();
    computeFlow();
   
}

void FlowFaceApp::loadFaceImages(){
    printf("[loadFaceImages] loading image of face from files %s and %s\n", image1File, image2File);
    im1 = imread(image1File);
    im2 = imread(image2File);
    
    im1.convertTo(im1, CV_64FC3, 1.0/255, 0);
    im2.convertTo(im2, CV_64FC3, 1.0/255, 0);
}

void FlowFaceApp::computeFlow(){
    printf("[computeFlow]\n");
    // magic variables
    double alpha = 0.03;
    double ratio = 0.85;
    int minWidth = 20;
    int nOuterFPIterations = 4;
    int nInnerFPIterations = 1;
    int nSORIterations = 40;
    
    Mat vx, vy, warp;
    
    CVOpticalFlow::findFlow(vx, vy, warp, im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
    imshow("warp", warp);
    imshow("flow", CVOpticalFlow::showFlow(vx, vy));

    imshow("im1", im1);
    imshow("im2", im2);

    Mat morph(im1);
    //ImageProcessing::warpImage(morph, im1, im2, vx, vy, im1.rows, im1.cols, 3);
    for (float dt = 0; dt <= 1.1; dt += .1){
        CVOpticalFlow::warpInterpolation(morph, im1, im2, vx, vy, dt);
        imshow("morph", morph);
        cvWaitKey(0);
    }
}


static FlowFaceApp *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new FlowFaceApp();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

