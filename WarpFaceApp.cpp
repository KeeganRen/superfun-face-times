/* WarpFaceApp.cpp */

#include "WarpFaceApp.h"
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

using namespace cv;

void error(char *msg) {
    perror(msg);
    exit(1);
}

void PrintUsage() 
{
    printf("Usage:  \n" \
            "   \t--faceImage [image with of the face]\n" \
            "   \t--facePoints [text file with detected face points]\n" \
            "   \t--faceMesh [text file with list of thousands of points to use for coloring]\n" \
            "   \t--output [image to save warped face to]\n" \
            "\n");
}

void WarpFaceApp::processOptions(int argc, char **argv){
    while (1){
        static struct option long_options[] = {
            {"help",        0, 0, 'h'},
            {"faceImage",       1, 0, 400},
            {"facePoints",      1, 0, 401},
            {"faceMesh",        1, 0, 402},
            {"output",          1, 0, 403},
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
                sprintf(faceImageFile, "%s", optarg);
                break;

            case 401:
                sprintf(facePointsFile, "%s", optarg);
                break;
                
            case 402:
                sprintf(faceMeshFile, "%s", optarg);
                break;

            case 403:
                sprintf(outFaceFile, "%s", optarg);
                break;

            default: 
                printf("Unrecognized option %d\n", c);
                break;
        }
    }
}

void WarpFaceApp::init(){
    printf("[init] Running program %s\n", argv[0]);
    
    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }
    
    processOptions(argc, argv);

    loadFaceMesh();
    loadFaceImage();
    loadFacePoints();

    populateModelPoints();
    findTransformation();
    
    makeNewFace();
   
}

void WarpFaceApp::loadFaceImage(){
    printf("[loadFaceImage] loading image of face from file %s\n", faceImageFile);
    faceImage = cvLoadImage(faceImageFile, CV_LOAD_IMAGE_COLOR);
    if (!faceImage){
        printf("Error loading image %s\n", faceImageFile);
        exit(1);
    }
}

void WarpFaceApp::loadFacePoints(){
    printf("[loadFaceImage] loading pre-detected 10 points of face from file %s\n", facePointsFile);
    FILE *file = fopen ( facePointsFile, "r" );
    if ( file != NULL ) {
        float x, y;
        while( fscanf(file, "%f %f\n", &x, &y) > 0 ) {
            facePoints.push_back(Point2f(x,y));
        }
        fclose (file);
    }
    else {
        perror (facePointsFile);
    }
}


void WarpFaceApp::loadFaceMesh(){
    printf("[loadFaceMesh] loading lots of points on a 3D face model file %s\n", faceMeshFile);
    FILE *file = fopen ( faceMeshFile, "r" );
    if ( file != NULL ) {
        float x, y, z;
        while( fscanf(file, "%f %f %f\n", &x, &y, &z) > 0 ) {
            faceMesh.push_back(Point3f(x,y,z));
        }
        fclose (file);
    }
    else {
        perror (faceMeshFile);
    }
    printf("[loadFaceMesh] %d points loaded\n", (int)faceMesh.size());
}

void WarpFaceApp::populateModelPoints(){
    modelPoints.push_back(Point3f(90.0, 36.0, 68.0));
    pointLabels.push_back("left eye (left)");
    modelPoints.push_back(Point3f(90.0, 65.0, 71.0));
    pointLabels.push_back("left eye (right");
    modelPoints.push_back(Point3f(156.0, 58.0, 72.0));
    pointLabels.push_back("mouth corner (left)");
    modelPoints.push_back(Point3f(156.0, 116.0, 72.0));
    pointLabels.push_back("mouth corner (right)");
    modelPoints.push_back(Point3f(167.0, 87.0, 87.0));
    pointLabels.push_back("mouth bottom");
    modelPoints.push_back(Point3f(150.0, 87.0, 92.0));
    pointLabels.push_back("mouth top");
    modelPoints.push_back(Point3f(90.0, 106.0, 71.0));
    pointLabels.push_back("right eye (left)");
    modelPoints.push_back(Point3f(90.0, 135.0, 68.0));
    pointLabels.push_back("right eye (right)");
    modelPoints.push_back(Point3f(134.0, 71.0, 86.0));
    pointLabels.push_back("nose base (left)");
    modelPoints.push_back(Point3f(134.0, 102.0, 83.0));
    pointLabels.push_back("nose base (right)");

    /*
    for (int i = 0; i < facePoints.size(); i++){
        printf("\t%s \t2D: %f %f \t3D: %f %f %f \n", 
            pointLabels[i].c_str(), 
            facePoints[i].x, facePoints[i].y, 
            modelPoints[i].x, modelPoints[i].y, modelPoints[i].z);
    }
    */
}

void WarpFaceApp::findTransformation(){
    printf("[findTransformation]\n");

    float fx = 40000;
    cameraMatrix = Matx33f(fx, 0, 0, 0, fx, 0, 0, 0, 1);
    vector<double> rv(3), tv(3);
    distortion_coefficients = Mat(5,1,CV_64FC1);
    rvec = Mat(3, 1, CV_64FC1);
    tvec = Mat(3, 1,CV_64FC1); 
    bool b = solvePnP(modelPoints, facePoints, cameraMatrix, distortion_coefficients, rvec, tvec, false, CV_ITERATIVE);
    /*
    printf("what happened with solvepnp? %d\n", b);
    cout << "cameraMatrix: " << cameraMatrix << endl;
    cout << "rvec: " << rvec << endl;
    cout << "tvec: " << tvec << endl;
    cout << "distortion_coefficients: " << distortion_coefficients << endl;
    */
        
    /*
    vector<Point2f> newPoints;
    projectPoints(modelPoints, rvec, tvec, cameraMatrix, distortion_coefficients, newPoints);

    Mat drawable = faceImage;
    for (int i = 0; i < facePoints.size(); i++){
        printf("\t%f %f \tvs \t%f %f\n", facePoints[i].x, facePoints[i].y, newPoints[i].x, newPoints[i].y);
        circle(drawable, facePoints[i], 4, CV_RGB(255, 0, 0), 2, 8, 0);
        circle(drawable, newPoints[i], 4, CV_RGB(0, 100, 255), 2, 8, 0);
    }
    imshow("projected points", drawable);
    cvWaitKey(0);
    */
}

void WarpFaceApp::makeNewFace(){
    printf("[makeNewFace]\n");
    
    vector<Point3f> mesh = faceMesh;

    vector<Point2f> newPoints;
    projectPoints(mesh, rvec, tvec, cameraMatrix, distortion_coefficients, newPoints);

    printf("[makeNewFace] done projecting %d points\n", (int)mesh.size());
    IplImage* new_face = cvCreateImage(cvSize(170, 220), IPL_DEPTH_8U, 3);
    cvZero(new_face);
    Mat new_face_mat = new_face;
    Mat face_image_mat = faceImage;
    for (int i = 0; i < mesh.size(); i++){
        float new_x = mesh[i].x;
        float new_y = mesh[i].y;
        float photo_x = newPoints[i].x;
        float photo_y =newPoints[i].y;
        new_face_mat.at<Vec3b>(new_x, new_y) = face_image_mat.at<Vec3b>(photo_y, photo_x);
    }
    
    Mat drawable = faceImage;
    for (int i = 0; i < mesh.size(); i++){
        circle(drawable, newPoints[i], 1, CV_RGB(0, 100, 255), 2, 8, 0);
    }
    imshow("projected face", drawable);
    cvWaitKey(0);

    imwrite(outFaceFile, new_face_mat);
    //imshow("new face", new_face_mat);
    //cvWaitKey(0);
    
}


static WarpFaceApp *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new WarpFaceApp();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

