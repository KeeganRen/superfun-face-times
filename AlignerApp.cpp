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

using namespace std;
using namespace cv;
#define e3 at<Vec3b>

void error(char *msg) {
    perror(msg);
    exit(1);
}

void PrintUsage() 
{
    printf("\nABOUT THIS CODE AND WHAT IT DOES:\nFor testing out different face alignment methods:\n");
    printf(" - affine anchored on the eyes+mouth\n");
    printf(" - etc...\n\n");
    printf("Usage:  \n" \
            "   \t<face image> <face points>\n" \
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
    
    if (argc == 3){
        face_file = argv[1];
        face_points_file = argv[2];

        load();
        test2();
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
    face_points = loadPoints(face_points_file);

    char *template_points_file = "model/grid-igor-canonical2d.txt";
    template_points = loadPoints(template_points_file);

    char *mask_file = "model/igormask.png";
    mask = imread(mask_file);

    Mat face_with_points = face.clone();

    Mat mask_with_points = imread(mask_file);
    for (int i = 0; i < template_points.size(); i++){
        circle(mask_with_points, template_points[i], 1,  CV_RGB(255, 5, 34), 0, 8, 0);
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
        float x, y;
        while( fscanf(file, "%f %f\n", &x, &y) > 0 ) {
            points.push_back(Point2f(x,y));
        }
        fclose (file);
    }
    else {
        perror (filename);
    }

    return points;
}

void AlignerApp::dealWithImage(string image, string points, string out){
    face_file = (char*)image.c_str();
    face_points_file = (char*)points.c_str();
    load();
    test4();
    imwrite(out, mask);
}

void AlignerApp::test1(){
    printf("[test1] anchoring based on the eyes and nose\n");

    vector<Point2f> f, c;
    if (0){
        f.push_back(middle(face_points[0], face_points[1]));
        f.push_back(middle(face_points[6], face_points[7]));
        f.push_back(middle(face_points[8], face_points[9]));

        c.push_back(middle(template_points[0], template_points[1]));
        c.push_back(middle(template_points[6], template_points[7]));
        c.push_back(middle(template_points[8], template_points[9]));
    }
    else {
        f.push_back(face_points[0]);
        f.push_back(face_points[7]);
        f.push_back(middle(face_points[8], face_points[9]));

        c.push_back(template_points[0]);
        c.push_back(template_points[7]);
        c.push_back(middle(template_points[8], template_points[9]));
    }

    Mat warp_mat( 2, 3, CV_32FC1 );
    warp_mat = getAffineTransform( f, c );

    cout << "warp mat" << endl << warp_mat << endl;

    warpAffine( face, mask, warp_mat, mask.size() );

    if (visualize){
        imshow("warped", mask);
        waitKey(0);
    }
}

void AlignerApp::test2(){
    printf("[test1] anchoring based on the eyes and nose\n");

    vector<Point2f> f, c;

    f.push_back(middle(face_points[0], face_points[1]));
    f.push_back(middle(face_points[6], face_points[7]));
    f.push_back(middle(face_points[8], face_points[9]));
    //f.push_back(middle(face_points[2], face_points[3]));
    //f.push_back(face_points[5]);

    c.push_back(middle(template_points[0], template_points[1]));
    c.push_back(middle(template_points[6], template_points[7]));
    c.push_back(middle(template_points[8], template_points[9]));
    //c.push_back(middle(template_points[2], template_points[3]));
    //c.push_back(template_points[5]);
  
    f[2] = perpendicularPoint(f);
    c[2] = perpendicularPoint(c);

    Mat warp_mat( 2, 3, CV_32FC1 );
    warp_mat = getAffineTransform( f, c );

    cout << "warp mat" << endl << warp_mat << endl;

    Mat warped_face = mask.clone();
    Mat warped_face2 = mask.clone();
    warpAffine( face, warped_face, warp_mat, mask.size() );
    warped_face.copyTo(warped_face2, mask);
    mask = warped_face2;

    if (visualize){
        imshow("warped", warped_face2);
        waitKey(0);
    }
}

void AlignerApp::test3(){
    printf("[test1] anchoring based on the eyes and nose\n");

    vector<Point2f> f, c;

    f.push_back(middle(face_points[0], face_points[1]));
    f.push_back(middle(face_points[6], face_points[7]));
    f.push_back(face_points[5]);

    c.push_back(middle(template_points[0], template_points[1]));
    c.push_back(middle(template_points[6], template_points[7]));
    c.push_back(template_points[5]);
  
    f[2] = perpendicularPoint(f);
    c[2] = perpendicularPoint(c);

    Mat warp_mat( 2, 3, CV_32FC1 );
    warp_mat = getAffineTransform( f, c );

    cout << "warp mat" << endl << warp_mat << endl;

    Mat warped_face = mask.clone();
    Mat warped_face2 = mask.clone();
    warpAffine( face, warped_face, warp_mat, mask.size() );
    warped_face.copyTo(warped_face2, mask);
    mask = warped_face2;

    if (visualize){
        imshow("warped", warped_face2);
        waitKey(0);
    }
}

void AlignerApp::test4(){
    printf("[test1] anchoring based on the eyes and nose\n");

    vector<Point2f> f, c;

    f.push_back(middle(face_points[0], face_points[1]));
    f.push_back(middle(face_points[6], face_points[7]));
    f.push_back(middle(face_points[2], face_points[3]));

    c.push_back(middle(template_points[0], template_points[1]));
    c.push_back(middle(template_points[6], template_points[7]));
    c.push_back(middle(template_points[2], template_points[3]));
  
    f[2] = perpendicularPoint(f);
    c[2] = perpendicularPoint(c);

    Mat warp_mat( 2, 3, CV_32FC1 );
    warp_mat = getAffineTransform( f, c );

    cout << "warp mat" << endl << warp_mat << endl;

    Mat warped_face = mask.clone();
    Mat warped_face2 = mask.clone();
    warpAffine( face, warped_face, warp_mat, mask.size() );
    warped_face.copyTo(warped_face2, mask);
    
    mask = warped_face2;

    if (visualize){
        imshow("warped", warped_face2);
        waitKey(0);
    }
    

}

Point2f AlignerApp::middle(Point2f a, Point2f b){
    Point2f c = Point2f((a.x + b.x)/2.0, (a.y + b.y)/2.0);
    return c;
}

Point2f AlignerApp::perpendicularPoint(vector<Point2f> f){
    // line between eye points ax + by + c = 0
    double a = (f[1].y - f[0].y)/(f[1].x - f[0].x);
    double b = -1;
    double c = -b*f[1].y - a*f[1].x;
    printf("%f * x + %f * y + c = 0\n", a, b, c);

    Point2f eye_mid = middle(f[0], f[1]);

    printf("%f, %f\n", a * f[2].x + b * f[2].y + c, sqrt(a*a + b*b));

    double dist = abs(a * f[2].x + b * f[2].y + c)/sqrt(a*a + b*b);
    printf("disance from nose pt to eye line: %f\n", dist);

    Point2d new_pt = eye_mid - Point2f(a,b)*(1.0/sqrt(a*a + b*b))*dist;
    printf("new point: %f, %f\n", new_pt.x, new_pt.y);
    printf("nose pt: %f %f\n", f[2].x, f[2].y);

    /*
    for (int i = 0; i < 3; i++){
        circle(face, f[i], 3,  CV_RGB(255, 5, 34), 4, 8, 0);
    }

    circle(face, eye_mid, 3,  CV_RGB(0, 200, 34), 4, 8, 0);
    circle(face, new_pt, 3,  CV_RGB(255, 200, 34), 4, 8, 0);
    imshow ("new pts on face", face);
    waitKey(0);
    */

    return new_pt;
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

