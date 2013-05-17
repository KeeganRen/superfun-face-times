/* FlowFaceApp.cpp */

#include "DetectFaceApp.h"

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
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

using namespace std;
using namespace cv;

void error(char *msg) {
    perror(msg);
    exit(1);
}

void PrintUsage() 
{
    printf("Usage:  \n" \
            "   \t--input <file> [file containing list of images]\n" \
            "   \t--output <directory path> [directory to save detected, cropped files to]\n" \
            "   \t--noalign [do NOT align the face because maybe it's already aligned]\n" \
            "   \t--mask [apply a face-shaped mask to the image]\n" \
            "   \t--small [make the face half-size]\n" \
            "   \t--visualize [see popups of the images]\n" \
            "\n");
}

void DetectFaceApp::processOptions(int argc, char **argv){
    while (1){
        static struct option long_options[] = {
            {"help",        0, 0, 'h'},
            {"input",       1, 0, 400},
            {"output",      1, 0, 401},
            {"noalign",     0, 0, 402},
            {"mask",        0, 0, 403},
            {"visualize",   0, 0, 404},
            {"small",       0, 0, 405},
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
                sprintf(inputFile, "%s", optarg);
                break;
            
            case 401:
                sprintf(outputDir, "%s", optarg);
                break;

            case 402:
                align = false;
                break;

            case 403:
                mask = true;
                break;

            case 404:
                visualize = true;
                break;

            case 405:
                small = true;
                break;

            default: 
                printf("Unrecognized option %d\n", c);
                break;
        }
    }
}

void DetectFaceApp::init(){
    printf("[init] Running program %s\n", argv[0]);
    
    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }

    ftFile  = (char*)"model/face2.tracker";
    triFile = (char*)"model/face.tri";
    conFile = (char*)"model/face.con";
    canonicalFace = (char*)"model/canonical-k-2.txt";
    maskFile = (char*)"model/ovalmask.png";
    align = true;
    mask = false;
    visualize = false;
    small = false;
    sprintf(outputDir, ".");
    
    processOptions(argc, argv);

    loadCanonicalFace();
    loadFacesFromList();
    startFaceTracker();
    
    for (int i = 0; i < faceImages.size(); i++){
        detectFace(faceImages[i], i);
    }
}

void DetectFaceApp::loadCanonicalFace(){
    printf("[loadCanonicalFace] loading from file %s\n", canonicalFace);

    FILE *f1 = fopen(canonicalFace, "r");
    double x,y;
    while (fscanf(f1, "%lf %lf", &x, &y) != EOF){
        canonicalPoints.push_back(Point2f(x,y));
    }
    
    printf("Size of canonical points vector: %d\n", int(canonicalPoints.size()));
    maskImage = imread(maskFile, CV_LOAD_IMAGE_GRAYSCALE);
}

void DetectFaceApp::startFaceTracker(){
    printf("[startFaceTracker] %s\n", ftFile);
    model = new FACETRACKER::Tracker(ftFile);
    printf("[startFaceTracker] tracker model loaded\n");
    Mat tri=FACETRACKER::IO::LoadTri(triFile);
    printf("[startFaceTracker] triangles loaded\n");
    Mat con=FACETRACKER::IO::LoadCon(conFile);
    printf("[startFaceTracker] connections loaded\n");
}

void DetectFaceApp::loadFacesFromList(){
    printf("[loadFacesFromList] Loading faces from list: %s\n", inputFile);
    FILE *file = fopen ( inputFile, "r" );
    if ( file != NULL ) {
        char line [ 256 ]; 
        while( fscanf(file, "%s\n", line) > 0 ) {
            Mat face = imread(line, CV_LOAD_IMAGE_COLOR);
            if (face.data==NULL) {
                printf("Error loading image %s\n", line);
            } 
            else {
                faceList.push_back(string(line));
                faceImages.push_back(face);
            }
        }
        fclose (file);
    }
    else {
        perror (inputFile);
    }
    printf("[loadFacesFromList] Read %d files and loaded %d images!\n", (int)faceList.size(), (int)faceImages.size());
}


void DetectFaceApp::detectFace(Mat &faceImage, int i){
    printf("[detectFace] detectng and planning to save to directory: %s\n", outputDir);

    Mat im = faceImage;
    Mat gray; 

    visualizeFrame("detecting face", im);
    
    cvtColor(im,gray,CV_BGR2GRAY);

    //set other tracking parameters
    vector<int> wSize1(1); wSize1[0] = 7;
    vector<int> wSize2(3); wSize2[0] = 11; wSize2[1] = 9; wSize2[2] = 7;
    int nIter = 10; 
    double clamp=3,fTol=0.01; 
    int fpd = -1; 
    bool fcheck = false;
    bool success = false;

    //track this image
    vector<int> wSize; 
    wSize = wSize2;
    if (model->Track(gray, wSize, fpd, nIter, clamp, fTol, fcheck) == 0){
        int idx = model->_clm.GetViewIdx();
        alignFace(im, model->_shape, model->_clm._visi[idx]);
        success = true;
    }

    if (success){
        char filename[512];
        sprintf(filename, "%s/cropped-%02d.jpg", outputDir, i);
        printf("[detectFace] saving new face to file: %s\n", filename);
        imwrite(filename, im);
    }
    else {
        printf("[detectFace] face not detected :(\n");
    }

    model->FrameReset();
}

void DetectFaceApp::alignFace(Mat &frame, Mat &shape, Mat &visi){
    printf("[alignFace] \n");
    vector<Point2f> facePoints;

    double x,y;
    int i, n;
    n = canonicalPoints.size();
    
    for(i = 0; i < n; i++){    
        if(visi.at<int>(i,0) == 0)continue;
        x = shape.at<double>(i,0);
        y = shape.at<double>(i+n,0);
        facePoints.push_back(Point2f(x,y));
    }

    if (align){
        Mat canonicalMat = Mat(canonicalPoints);
        Mat faceMat = Mat(facePoints);
        Mat transform = estimateRigidTransform(facePoints,canonicalPoints, false);
        cout << "transform = "<< endl << " "  << transform << endl << endl;

        Mat warped = Mat::zeros(500, 500, frame.type() );
        warpAffine(frame, warped, transform, warped.size() );
        frame = warped;
        visualizeFrame("warped", frame);
    }
    
    bool equalize = true;
    if (equalize){
        Mat ycrcb;

        cvtColor(frame,ycrcb,CV_BGR2YCrCb);

        vector<Mat> channels;
        split(ycrcb,channels);

        equalizeHist(channels[0], channels[0]);

        merge(channels,ycrcb);

        cvtColor(ycrcb,frame,CV_YCrCb2BGR);
    }

    if (mask) {
        Mat masked = Mat::zeros(500, 500, frame.type() );
        frame.copyTo(masked, maskImage);
        frame = masked;
        visualizeFrame("masked", frame);
    }

    Rect cropROI(144, 95, 210, 310);
    frame = frame(cropROI);
    visualizeFrame("cropped", frame);

    if (small) {
        Mat smaller;
        resize(frame,smaller,Size(frame.cols/2,frame.rows/2),0,0,INTER_LINEAR);
        frame = smaller;
        visualizeFrame("small", frame);
    }
    
}

void DetectFaceApp::visualizeFrame(string title, Mat &frame){
    if (visualize){
        imshow(title, frame);
        waitKey(0);
    }
}

static DetectFaceApp *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new DetectFaceApp();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

