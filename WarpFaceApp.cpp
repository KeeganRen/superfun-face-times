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
            "   \t--facePoints [text file with detected fiducial points]\n" \
            "   \t--templateMesh [template 3D point cloud]\n" \
            "   \t--templatePoints [the 3d canonical points]\n" \
            "   \t--output [image to save warped face to]\n" \
            "\n\n" \
            "   \t /warpFace --faceImage test/108.jpg --facePoints test/108.txt "\
            "--templateMesh model/igor.txt --templatePoints model/igor-canonical.txt "\
            "--canonicalPoints model/canonical_faceforest.txt" \
            "\n");
}

void WarpFaceApp::processOptions(int argc, char **argv){
    while (1){
        static struct option long_options[] = {
            {"help",        0, 0, 'h'},
            {"faceImage",       1, 0, 400},
            {"facePoints",      1, 0, 401},
            {"templateMesh",        1, 0, 402},
            {"output",          1, 0, 403},
            {"templatePoints",        1, 0, 404},
            {"canonicalPoints",        1, 0, 405},
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
                sprintf(templateMeshFile, "%s", optarg);
                break;

            case 403:
                sprintf(outFaceFile, "%s", optarg);
                break;

            case 404:
                sprintf(templatePointsFile, "%s", optarg);
                break;

            case 405:
                sprintf(canonicalPointsFile, "%s", optarg);
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

    loadTemplateFiles();
    loadFaceSpecificFiles();

    getColorFromImage();
    //findTransformation();
    
    //makeNewFace();
   
}

void WarpFaceApp::loadFaceSpecificFiles(){
    printf("[loadFaceSpecificFiles] loading image of face from file %s\n", faceImageFile);
    faceImage = imread(faceImageFile, CV_LOAD_IMAGE_COLOR);
    if (faceImage.data == NULL){
        printf("Error loading image %s\n", faceImageFile);
        exit(1);
    }
    faceImage.convertTo(faceImage, CV_64FC3, 1.0/255, 0);

    printf("[loadFaceSpecificFiles] loading pre-detected 10 points of face from file %s\n", facePointsFile);
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

void WarpFaceApp::loadTemplateFiles(){
    printf("[loadTemplateFiles] loading canonical face points file %s\n", canonicalPointsFile);
    FILE *file;

    file = fopen ( canonicalPointsFile, "r" );
    if ( file != NULL ) {
        float x, y;
        while( fscanf(file, "%f %f\n", &x, &y) > 0 ) {
            canonicalPoints.push_back(Point2f(x,y));
        }
        fclose (file);
    }
    else {
        perror (canonicalPointsFile);
    }
    printf("[loadTemplateFiles] sweet, {{%d}} canonical (2d) points loaded\n", (int)canonicalPoints.size());

    printf("[loadTemplateFiles] loading template face mesh from file %s\n", templateMeshFile);
    file = fopen ( templateMeshFile, "r" );
    if ( file != NULL ) {
        float x, y, z, nx, ny, nz;
        int r, g, b, a;
        while( fscanf(file, "%f %f %f %f %f %f %d %d %d %d \n", &x, &y, &z, &nx, &ny, &nz, &r, &g, &b, &a) > 0 ) {
            templateMesh.push_back(Point3f(x,y,z));
            templateNormals.push_back(Point3f(nx,ny,nz));
            templateColors.push_back(Point3f(r,g,b));
        }
        fclose (file);
    }
    else {
        perror (templateMeshFile);
    }
    printf("[loadTemplateFiles] yay, {{%d}} points loaded into templateMesh\n", (int)templateMesh.size());


    printf("[loadTemplateFiles] loading 3d canonical points corresponding to face mesh from file %s\n", templatePointsFile);
    file = fopen ( templatePointsFile, "r" );
    if ( file != NULL ) {
        float x, y, z;
        while( fscanf(file, "%f,%f,%f\n", &x, &y, &z) > 0 ) {
            templatePoints.push_back(Point3f(x,y,z));
        }
        fclose (file);
    }
    else {
        perror (templatePointsFile);
    }

    printf("[loadTemplateFiles] huzzah, {{%d}} points loaded into templatePoints\n", (int)templatePoints.size());
}

void WarpFaceApp::findTransformation(){
    printf("[findTransformation]\n");

    fx = 4000;
    cameraMatrix = Matx33f(fx, 0, 0, 0, fx, 0, 0, 0, 1);
    vector<double> rv(3), tv(3);
    distortion_coefficients = Mat(5,1,CV_64FC1);
    rvec = Mat(3, 1, CV_64FC1);
    tvec = Mat(3, 1,CV_64FC1); 

    vector<Point2f> new_points;
   
    // transform between template and canonical front-of-face view
    solvePnP(templatePoints, canonicalPoints, cameraMatrix, distortion_coefficients, rvec, tvec, false, CV_ITERATIVE);

    cout << "cameraMatrix: " << cameraMatrix << endl;
    cout << "rvec: " << rvec << endl;
    cout << "tvec: " << tvec << endl;
    cout << "distortion_coefficients: " << distortion_coefficients << endl;

    // project onto canonical view
    projectPoints(templateMesh, rvec, tvec, cameraMatrix, distortion_coefficients, new_points);

    Mat templateImage = Mat::zeros(500, 500, CV_8UC3);
    for (int i = 0; i < templateMesh.size(); i++){
        //Point3d color = templateColors[i];
        Point3f normal = templateNormals[i];
        Point3d color;
        float f = 100.0;
        color.x = normal.x * f;
        color.y = normal.y * f;
        color.z = normal.z * f;
        circle(templateImage, new_points[i], .7, CV_RGB(color.x, color.y, color.z), 2, 8, 0);
    }
    Rect cropROI(128, 80, 250, 310);
    templateImage = templateImage(cropROI);
    imshow("all points drawn on canonical thinggy", templateImage);
    cvWaitKey(0);

    // transformation between 3D mesh points and image
    solvePnP(templatePoints, facePoints, cameraMatrix, distortion_coefficients, rvec, tvec, false, CV_ITERATIVE);

    cout << "cameraMatrix: " << cameraMatrix << endl;
    cout << "rvec: " << rvec << endl;
    cout << "tvec: " << tvec << endl;
    cout << "distortion_coefficients: " << distortion_coefficients << endl;
    
    // project template canonical points onto image
    projectPoints(templatePoints, rvec, tvec, cameraMatrix, distortion_coefficients, new_points);

    Mat drawable = faceImage;
    for (int i = 0; i < facePoints.size(); i++){
        printf("\t%f %f \tvs \t%f %f\n", facePoints[i].x, facePoints[i].y, new_points[i].x, new_points[i].y);
        circle(drawable, facePoints[i], 4, CV_RGB(255, 0, 0), 2, 8, 0);
        circle(drawable, new_points[i], 4, CV_RGB(0, 100, 255), 2, 8, 0);
    }
    imshow("projected points", drawable);
    cvWaitKey(0);

    // project all template points onto image
    projectPoints(templateMesh, rvec, tvec, cameraMatrix, distortion_coefficients, new_points);

    drawable = faceImage;
    for (int i = 0; i < templateMesh.size(); i++){
        Point3f color = templateColors[i];
        circle(drawable, new_points[i], .7, CV_RGB(color.x, color.y, color.z), 2, 8, 0);
    }
    imshow("all points", drawable);
    cvWaitKey(0);
    
}

void WarpFaceApp::getColorFromImage(){
    printf("[getColorFromImage]\n");

    fx = 4000;
    cameraMatrix = Matx33f(fx, 0, 0, 0, fx, 0, 0, 0, 1);
    vector<double> rv(3), tv(3);
    distortion_coefficients = Mat(5,1,CV_64FC1);
    rvec = Mat(3, 1, CV_64FC1);
    tvec = Mat(3, 1,CV_64FC1); 

    vector<Point2f> new_points;
   
    // transform between template and canonical front-of-face view
    solvePnP(templatePoints, facePoints, cameraMatrix, distortion_coefficients, rvec, tvec, false, CV_ITERATIVE);

    cout << "cameraMatrix: " << cameraMatrix << endl;
    cout << "rvec: " << rvec << endl;
    cout << "tvec: " << tvec << endl;
    cout << "distortion_coefficients: " << distortion_coefficients << endl;

    // project onto canonical view
    projectPoints(templateMesh, rvec, tvec, cameraMatrix, distortion_coefficients, new_points);

    vector<Point3f> new_colors;

    for (int i = 0; i < new_points.size(); i++){
        double c[3];
        bilinear(c, faceImage, new_points[i].x, new_points[i].y);
        //new_colors.push_back(Point3d(122, 44, 39));
        new_colors.push_back(Point3d(c[0], c[1], c[2]));
    }

    // transform between template and canonical front-of-face view
    solvePnP(templatePoints, canonicalPoints, cameraMatrix, distortion_coefficients, rvec, tvec, false, CV_ITERATIVE);

    cout << "cameraMatrix: " << cameraMatrix << endl;
    cout << "rvec: " << rvec << endl;
    cout << "tvec: " << tvec << endl;
    cout << "distortion_coefficients: " << distortion_coefficients << endl;

    // project onto canonical view
    projectPoints(templateMesh, rvec, tvec, cameraMatrix, distortion_coefficients, new_points);

    Mat drawable = Mat::zeros(500, 500, CV_32FC3);
    for (int i = 0; i < templateMesh.size(); i++){
        Point3f color = new_colors[i];
        circle(drawable, new_points[i], 4, CV_RGB(color.z, color.y, color.x), 0, 8, 0);
    }
    for (int i = 0; i < templateMesh.size(); i++){
        Point3f color = new_colors[i];
        circle(drawable, new_points[i], 2, CV_RGB(color.z, color.y, color.x), 0, 8, 0);
    }
    for (int i = 0; i < templateMesh.size(); i++){
        Point3f color = new_colors[i];
        circle(drawable, new_points[i], 1, CV_RGB(color.z, color.y, color.x), 0, 8, 0);
    }
    imshow("all points", drawable);
    cvWaitKey(0);

}

void WarpFaceApp::clip(int &a, int lo, int hi) {
    a = (a < lo) ? lo : (a>=hi ? hi-1: a);
}

void WarpFaceApp::bilinear(double *out, Mat im, float c, float r){
    int r0 = r, r1 = r+1;
    int c0 = c, c1 = c+1;
    clip(r0, 0, im.rows);
    clip(r1, 0, im.rows);
    clip(c0, 0, im.cols);
    clip(c1, 0, im.cols);

    double tr = r - r0;
    double tc = c - c0;
    for (int i=0; i<3; i++) {
        double ptr00 = im.at<Vec3d>(r0, c0)[i];
        double ptr01 = im.at<Vec3d>(r0, c1)[i];
        double ptr10 = im.at<Vec3d>(r1, c0)[i];
        double ptr11 = im.at<Vec3d>(r1, c1)[i];
        out[i] = ((1-tr) * (tc * ptr01 + (1-tc) * ptr00) + tr * (tc * ptr11 + (1-tc) * ptr10));

    }
}

void WarpFaceApp::makeNewFace(){
    printf("[makeNewFace]\n");
    
    vector<Point3f> mesh = templateMesh;

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

