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
                image1File = strdup(optarg);
                break;

            case 401:
                image2File = strdup(optarg); 
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
    
    flowFile = NULL;

    processOptions(argc, argv);

    //loadFaceImages();
    //computeFlow();

    loadAllImagesAndComputeFlow(argc, argv);
   
}

void FlowFaceApp::loadFaceImages(){
    printf("[loadFaceImages] loading image of face from files %s and %s\n", image1File, image2File);
    im1 = imread(image1File);
    im2 = imread(image2File);

    //resize(im1, im1, Size(), 0.5, 0.5);
    //resize(im2, im2, Size(), 0.5, 0.5);

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
    
    Mat vx, vy, warp, flow;
    
    // one direction: 1 -> 2
    CVOpticalFlow::findFlow(vx, vy, warp, im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
    imshow("warp", warp);
    imshow("flow", CVOpticalFlow::showFlow(vx, vy));

    flow = CVOpticalFlow::showFlow(vx, vy);
    flow.convertTo(flow, CV_8UC3, 1.0*255, 0);
    
    char *fileFlow = "faces/flowAB.png";
    char *fileWarp = "faces/warpAB.jpg";
    imwrite(fileFlow, flow);
    imwrite(fileWarp, warp);

    // other direction: 2->1
    CVOpticalFlow::findFlow(vx, vy, warp, im2, im1, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
    imshow("warp", warp);
    imshow("flow", CVOpticalFlow::showFlow(vx, vy));

    flow = CVOpticalFlow::showFlow(vx, vy);
    flow.convertTo(flow, CV_8UC3, 1.0*255, 0);
    
    char *fileFlowBA = "faces/flowBA.png";
    char *fileWarpBA = "faces/warpBA.jpg";
    imwrite(fileFlowBA, flow);
    imwrite(fileWarpBA, warp);

    char *fileImA = "faces/A.jpg";
    char *fileImB = "faces/B.jpg";


    printf("%s %s %s\n", fileImA, fileImB, fileFlow);
    im1.convertTo(im1, CV_8UC3, 1.0*255, 0);
    im2.convertTo(im2, CV_8UC3, 1.0*255, 0);
    imwrite(fileImA, im1);
    imwrite(fileImB, im2);

    if (0){
        imshow("im1", im1);
        imshow("im2", im2);

        Mat morph(im1);
        //ImageProcessing::warpImage(morph, im1, im2, vx, vy, im1.rows, im1.cols, 3);
        for (float dt = 0; dt <= 1.1; dt += .1){
            CVOpticalFlow::warpInterpolation(morph, im1, im2, vx, vy, dt);
            imshow("morph", morph);
            cvWaitKey(100);
        }
    }
}

void FlowFaceApp::loadAllImagesAndComputeFlow(int argc, char** argv){
    vector<Mat> images; 
    for (int i = 1; i < argc; i++){
        Mat img = imread(argv[i]);
        //resize(img, img, Size(), .4, .4);
        imshow("image", img);
        waitKey(1000);
        img.convertTo(img, CV_64FC3, 1.0/255, 0);
        images.push_back(img);
    }    

    char *fileFlowAB = "faces/flowAB.png";
    char *fileFlowBA = "faces/flowBA.png";
    char *fileWarpAB = "faces/warpAB.jpg";
    char *fileWarpBA = "faces/warpBA.jpg";
    char *fileImA = "faces/A.jpg";
    char *fileImB = "faces/B.jpg";




    double alpha = 0.03;
    double ratio = 0.85;
    int minWidth = 20;
    int nOuterFPIterations = 4;
    int nInnerFPIterations = 1;
    int nSORIterations = 40;
    
    Mat final_vx, final_vy;
    vector<Mat> vec_vx;
    vector<Mat> vec_vy;
    Mat warped;

    // one direction: 1 -> 2
    for (int i = 0; i < images.size() - 1; i++){
        Mat vx, vy, warp, flow;
        CVOpticalFlow::findFlow(vx, vy, warp, images[i], images[i+1], alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
        vec_vx.push_back(vx);
        vec_vy.push_back(vy);
        imshow("image 1:", images[i]);
        imshow("image 2:", images[i+1]);
        imshow("warp", warp);
        imshow("flow", CVOpticalFlow::showFlow(vx, vy));
        waitKey(100);
        
    }

    final_vx = vec_vx[0].clone();
    final_vy = vec_vy[0].clone();

    for (int i = 1; i < vec_vx.size(); i++){
        CVOpticalFlow::compositeFlow(final_vx, final_vy, vec_vx[i], vec_vy[i], final_vx, final_vy);
    }

    imshow("composite flow a->b", CVOpticalFlow::showFlow(final_vx, final_vy));

    warped = images[images.size() - 1].clone();
    CVOpticalFlow::warp(warped, images[images.size() - 1], final_vx, final_vy);
    imshow("final warped image (a->b)", warped);

    Mat flow = CVOpticalFlow::showFlow(final_vx, final_vy);
    flow.convertTo(flow, CV_8UC3, 1.0*255, 0);
    warped.convertTo(warped, CV_8UC3, 1.0*255, 0);
    imwrite(fileFlowAB, flow);
    imwrite(fileWarpAB, warped);

    vec_vx.clear();
    vec_vy.clear();


    // OTHER direction: 1 -> 2
    for (int i = images.size() - 1; i > 0; i--){
        Mat vx, vy, warp, flow;
        CVOpticalFlow::findFlow(vx, vy, warp, images[i], images[i-1], alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
        vec_vx.push_back(vx);
        vec_vy.push_back(vy);
        imshow("image 1:", images[i]);
        imshow("image 2:", images[i-1]);
        imshow("warp", warp);
        imshow("flow", CVOpticalFlow::showFlow(vx, vy));
        waitKey(100);
        
    }

    final_vx = vec_vx[0].clone();
    final_vy = vec_vy[0].clone();

    for (int i = 1; i < vec_vx.size(); i++){
        CVOpticalFlow::compositeFlow(final_vx, final_vy, vec_vx[i], vec_vy[i], final_vx, final_vy);
    }

    imshow("composite flow b->a", CVOpticalFlow::showFlow(final_vx, final_vy));

    warped = images[0].clone();
    CVOpticalFlow::warp(warped, images[0], final_vx, final_vy);
    imshow("final warped image b->a", warped);

    Mat flow2 = CVOpticalFlow::showFlow(final_vx, final_vy);
    flow2.convertTo(flow2, CV_8UC3, 1.0*255, 0);
    warped.convertTo(warped, CV_8UC3, 1.0*255, 0);
    imwrite(fileFlowBA, flow2);
    imwrite(fileWarpBA, warped);


    Mat A = images[0];
    Mat B = images[images.size() - 1];
    A.convertTo(A, CV_8UC3, 1.0*255, 0);
    B.convertTo(B, CV_8UC3, 1.0*255, 0);
    imwrite(fileImA, A);
    imwrite(fileImB, B);

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

