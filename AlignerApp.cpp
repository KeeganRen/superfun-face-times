/* AlignerApp.cpp */

#include "AlignerApp.h"

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
#include "opencv2/video/tracking.hpp"

using namespace std;
using namespace cv;
#define e3 at<Vec3b>

void error(char *msg) {
    perror(msg);
    exit(1);
}

void PrintUsage() 
{
    printf("\nAlign and crop an image! Produces a warped (500x500 image), a cropped head and a cropped face.\n");
    printf("Usage:  \n" \
            "   \t<face image> <face points> <output base>\n" \
            "   \t or\n" \
            "   \t<list of [input image, input points, output mask]>\n" \
            "\n");
}

void AlignerApp::init(){
    printf("[init] Running program %s\n", argv[0]);
    
    visualize = false;

    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }
    
    if (argc == 4){
        face_file = argv[1];
        face_points_file = argv[2];
        output_base = argv[3];

        load();
        test1();
    }
    else if (argc == 2){
        list_file = argv[1];
        loadFromList();
        for (int i = 0; i < (int)imageFiles.size(); i++){
            dealWithImage(imageFiles[i], imagePointFiles[i], outputImageFiles[i]);
        }
    }
   
}

void AlignerApp::load(){
    printf("[load] loading files: \n\t[%s]\n\t[%s]\n", face_file, face_points_file);
    face = imread(face_file);
    if (face.data == NULL){
        printf("error loading image\n");
        exit(-2);
    }
    face_points = loadPoints(face_points_file);

    char *template_points_file = "/Users/ktuite/Desktop/averageman_symmetric.txt";
    template_points = loadPoints(template_points_file);

    char *mask_file = "model/igormask.png";
    mask = imread(mask_file);

    Mat face_with_points = face.clone();

    Mat mask_with_points = imread(mask_file);
    for (int i = 0; i < template_points.size(); i++){
        circle(mask_with_points, template_points[i], 1,  CV_RGB(255, 5, 34), 0, 8, 0);
        circle(face_with_points, face_points[i], 3, CV_RGB(34, 77, 255), 2, 8, 0);
        circle(face_with_points, face_points[i], 3, CV_RGB(34, 77, 255), 2, 8, 0);
    }
    if (visualize){
        imshow ("mask", mask_with_points);
        imshow ("face", face_with_points);
    }
}

void AlignerApp::loadFromList(){
    printf("[loadFromList] loading from file %s\n", list_file);
    FILE *file = fopen ( list_file, "r" );
    if ( file != NULL ) {
        char image_path[256];
        char point_path[256];
        char mask_path[256];
        while( fscanf(file, "%s %s %s\n", image_path, point_path, mask_path) > 0 ) {
            imageFiles.push_back(image_path);
            imagePointFiles.push_back(point_path);
            outputImageFiles.push_back(mask_path);
        }
        fclose (file);
    }
    else {
        perror (list_file);
    }
    printf("[loadFromList] found %d image filenames\n", (int)imageFiles.size());  
}

vector<Point2f> AlignerApp::loadPoints(const char* filename){
    printf("[loadPoints] loading points from file [%s]\n", filename);
    vector<Point2f> points;
    FILE *file = fopen ( filename, "r" );
    if ( file != NULL ) {
        int num_features;
        if (fscanf(file, "%d\n", &num_features) < 1){
            printf("error reading list of feature points\n");
            exit(-1);
        }
        printf("number of points loaded: %d\n", num_features);
        float x, y;
        while( fscanf(file, "%f %f\n", &x, &y) > 0 ) {
            points.push_back(Point2f(x,y));
        }
        fclose (file);
    }
    else {
        perror (filename);
        exit(-1);
    }

    return points;
}

void AlignerApp::dealWithImage(string image, string points, string out){
    face_file = (char*)image.c_str();
    face_points_file = (char*)points.c_str();
    load();
    test1();
    imwrite(out, mask);
}

void AlignerApp::test1(){
    printf("[test1] anchoring based on the eyes and nose\n");

    vector<Point2f> f, c;
    f = face_points;
    c = template_points;

    Mat warp_mat( 2, 3, CV_32FC1 );
    warp_mat = estimateRigidTransform( f, c, false);

    cout << "warp mat" << endl << warp_mat << endl;

    Mat warped_face(500, 500, CV_32FC3);

    Point2f cropFaceSize(225, 250);
    Rect cropFaceROI((500-cropFaceSize.x)/2.0, (500-cropFaceSize.y)/2.0 + 20, cropFaceSize.x, cropFaceSize.y);

    Point2f cropHeadSize(290, 365);
    Rect cropHeadROI((500-cropHeadSize.x)/2.0, (500-cropHeadSize.y)/2.0 + 10, cropHeadSize.x, cropHeadSize.y);

    warpAffine( face, warped_face, warp_mat, warped_face.size() );

    Mat cropFace = warped_face(cropFaceROI);
    Mat cropHead = warped_face(cropHeadROI);

    resize(warped_face, warped_face, Size(), 0.5, 0.5);
    resize(cropHead, cropHead, Size(), 0.5, 0.5);

    if (visualize){
        imshow("warped", warped_face);
        imshow("crop face", cropFace);
        imshow("crop head", cropHead);
        waitKey(0);
    }

    char warped_file[512];
    char crop_face_file[512];
    char crop_head_file[512];

    sprintf(warped_file, "%s_warped.jpg", output_base);
    sprintf(crop_face_file, "%s_cropface.jpg", output_base);
    sprintf(crop_head_file, "%s_crophead.jpg", output_base);

    imwrite(warped_file, warped_face);
    imwrite(crop_head_file, cropHead);
    imwrite(crop_face_file, cropFace);
}

static AlignerApp *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new AlignerApp();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

