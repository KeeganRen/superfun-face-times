/* ScoreCalculatorApp.cpp */

//g++ -I/opt/local/include -L/opt/local/lib -L/usr/lib ScoreCalculatorApp.cpp -lopencv_highgui -lopencv_imgproc -lopencv_core  -o ScoreCalculatorApp && ./ScoreCalculatorApp

#include "ScoreCalculatorApp.h"

#include "FaceLib.h"

#include "getopt.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <opencv/highgui.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <istream>
#include <fstream>
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
            "   \t--face [id of face to score]\n" \
            "   \t--list [file with list of image ids to compare to]\n" \
            "   \t--basePath [path to data directory containing images, face_points, and features]\n" \
            "   \t--metric [hog|fiducials]\n" \
            "   \t--templatePoints [location of landmakr template]\n" \
            "   \t--output [file to save results to]\n" \
            "\n\n" \
            "   \t ./score --face 244 --list scoring_test/notimpressed.txt --basePath ~/Code/FaceServer/data --templatePoints data/averageman_symmetric.txt" \
            "\n");
}

void ScoreCalculatorApp::processOptions(int argc, char **argv){
    while (1){
        static struct option long_options[] = {
            {"help",        0, 0, 'h'},
            {"face",        1, 0, 400},
            {"list",        1, 0, 401},
            {"basePath",    1, 0, 402},
            {"metric",      1, 0, 403},
            {"templatePoints",      1, 0, 404},
            {"output",      1, 0, 405},
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
                faceIdToScore = atoi(optarg);
                break;

            case 401:
                facesToCompareFile = strdup(optarg);
                break;
                
            case 402:
                basePath = strdup(optarg);
                break;

            case 403:
                scoringMetricWord = strdup(optarg);

                if (strcmp(scoringMetricWord, "hog") == 0){
                    metric = HOG_SCORE;
                }
                else {
                    metric = FIDUCIAL_SCORE;
                }

                break;

            case 404:
                templatePointFile = strdup(optarg);
                break;

            case 405:
                outFile = strdup(optarg);
                break;

            default: 
                printf("Unrecognized option %d\n", c);
                break;
        }
    }
}

void ScoreCalculatorApp::init(){
    printf("[init] Running program %s\n", argv[0]);
    
    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }

    faceToScoreFile = NULL;
    facesToCompareFile = NULL;
    basePath = NULL;

    processOptions(argc, argv);

    printf("[init] after process options: \n\t" \
        "face to score file: [%d]\n\t" \
        "faces to compare file: [%s]\n\t" \
        "base path: [%s]\n\t" \
        "metric: [%d]\n\n",
        faceIdToScore, facesToCompareFile, basePath, metric);

    loadAndTransformFacePoints();


    /*
    char *faceAFile = argv[1];
    char *faceBFile = argv[2];
    char *landmarkAFile = argv[3];
    char *landmarkBFile = argv[4];
    char *clusterPath = NULL;
    outPath = argv[5];

    load(faceAFile, faceBFile, landmarkAFile, landmarkBFile, clusterPath);

    FaceLib::computeTransform(A, landmarkA, templatePoints2D, A_xform);
    FaceLib::computeTransform(B, landmarkB, templatePoints2D, B_xform);

    Point2f cropFaceSize(225, 250);
    A_xform.at<double>(0,2) -= (500-cropFaceSize.x)/2.0;
    A_xform.at<double>(1,2) -= (500-cropFaceSize.y)/2.0 + 20;

    B_xform.at<double>(0,2) -= (500-cropFaceSize.x)/2.0;
    B_xform.at<double>(1,2) -= (500-cropFaceSize.y)/2.0 + 20;

    cout << "A_xform" << endl;
    cout << A_xform << endl;
    cout << "B_xform" << endl;
    cout << B_xform << endl;

    A_mask = makeFullFaceMask(A, A_xform);
    B_mask = makeFullFaceMask(B, B_xform);


    matchHistograms();

    swap();
    */
    /*

    Mat A_cropped = makeCroppedFaceMask(A, A_xform);
    Mat B_cropped = makeCroppedFaceMask(B, B_xform);
    imshow("a cropped", A_cropped);
    imshow("b cropped", B_cropped);


    A_cropped.convertTo(A_cropped, CV_8UC3, 1.0*255, 0);
    B_cropped.convertTo(B_cropped, CV_8UC3, 1.0*255, 0);
    A.convertTo(A, CV_8UC3, 1.0*255, 0);

    histogramFunTimes(A_cropped, B_cropped, A, B);
    */

    //

    /*

    Mat A_cropped = makeCroppedFaceMask(A, A_xform);
    gsl_vector *A_gsl = gsl_vector_calloc(num_pixels);
    FaceLib::matToGslVec(A_cropped, A_gsl, d, w, h);
    gsl_vector *A_low_gsl = FaceLib::projectNewFace(num_pixels, num_eigenfaces, A_gsl, m_gsl_meanface, m_gsl_eigenfaces);
    Mat A_low;
    FaceLib::gslVecToMat(A_low_gsl, A_low, d, w, h);
    FaceLib::computeFlowAndStore(A_cropped, A_low, A_vx, A_vy);

    // flow the other way
    Mat B_cropped = makeCroppedFaceMask(B, B_xform);
    gsl_vector *B_gsl = gsl_vector_calloc(num_pixels);
    FaceLib::matToGslVec(B_cropped, B_gsl, d, w, h);
    gsl_vector *B_low_gsl = FaceLib::projectNewFace(num_pixels, num_eigenfaces, B_gsl, m_gsl_meanface, m_gsl_eigenfaces);
    Mat B_low;
    FaceLib::gslVecToMat(B_low_gsl, B_low, d, w, h);
    FaceLib::computeFlowAndStore(B_low, B_cropped, B_vx, B_vy);

    //imshow("A flow", FaceLib::showFlow(A_vx, A_vy));
    //imshow("B flow", FaceLib::showFlow(B_vx, B_vy));

    FaceLib::compositeFlow(A_vx, A_vy, B_vx, B_vy, vx, vy);
    //imshow("composed flow", FaceLib::showFlow(vx, vy));

    animateBlend();

    */
}

void ScoreCalculatorApp::loadAndTransformFacePoints(){
    printf("[loadAndTransformFacePoints] face points from file faceIdToScore [%d] and facesToCompareFile [%s] \n", faceIdToScore, facesToCompareFile);
    
    FaceLib::loadNewFiducialPoints(templatePointFile, templatePoints2D);
    for (int i = 0; i < templatePoints2D.size(); i++){
        templatePoints2D[i].x /= 2.0;
        templatePoints2D[i].y /= 2.0;
    }
    printf("\ttemplate landmarks read in\n");

    FILE *file = fopen ( facesToCompareFile, "r" );
    if ( file != NULL ) {
        int face_id;
        while( fscanf(file, "%d\n", &face_id) > 0 ) {
            if (face_id != faceIdToScore){
                faceIdsToCompare.push_back(face_id);
            }
        }
        fclose (file);
        printf("\tnumber of faces read: %d\n", faceIdsToCompare.size());
    }

    char landmarkFile[200];
    sprintf(landmarkFile, "%s/face_points/%d_2.txt", basePath, faceIdToScore);
    vector<Point2f> user_landmarks;
    FaceLib::loadNewFiducialPoints(landmarkFile, user_landmarks);

    printf("\tlandmark file [%s], [%d] points loaded\n\n", landmarkFile, user_landmarks.size());

    Mat xform;
    FaceLib::computeTransform(user_landmarks, templatePoints2D, xform);

    Mat viz = Mat::zeros(500, 500, CV_8UC3);
    // draw template points 
    for (int i = 0; i < templatePoints2D.size(); i++){
        circle(viz, templatePoints2D[i], 1, CV_RGB(250, 250, 250), 4, 8, 0);
    }

    vector<Point2f> all_landmark_points;
    int num_faces = 0;
    int num_points = user_landmarks.size();

    for (int h = 0; h < faceIdsToCompare.size(); h++){
        int face_id = faceIdsToCompare[h];
        sprintf(landmarkFile, "%s/face_points/%d_2.txt", basePath, face_id);

        vector<Point2f> landmarks;
        FaceLib::loadNewFiducialPoints(landmarkFile, landmarks);
        if (landmarks.size() > 0){
            num_faces++;
            printf("\tlandmark file [%s], [%d] points loaded\n\n", landmarkFile, landmarks.size());

            Mat xform;
            FaceLib::computeTransform(landmarks, templatePoints2D, xform);

            for (int i = 0; i < landmarks.size(); i++){
                Point2f new_xy = applyXform(landmarks[i], xform);
                all_landmark_points.push_back(new_xy);
                circle(viz, new_xy, .9, CV_RGB(50,50,100), 2, 8, 0);
            }
        }
    }

    printf("num faces: %d\n\n", num_faces);

    gsl_landmarks = gsl_matrix_alloc(num_faces, num_points*2);

    for (int i = 0; i < num_faces; i++){
        for (int j = 0; j < num_points; j++){
            Point2f p = all_landmark_points[i*num_points + j];
            gsl_matrix_set(gsl_landmarks, i, j*2 + 0, p.x);
            gsl_matrix_set(gsl_landmarks, i, j*2 + 1, p.y);
        }
    }

    int num_points_in_range = 0;

    FILE *out = fopen(outFile, "w");

    for (int i = 0; i < num_points; i++){
        gsl_vector_view col_x = gsl_matrix_column(gsl_landmarks, i*2 + 0);
        gsl_vector_view col_y = gsl_matrix_column(gsl_landmarks, i*2 + 1);

        double x_mean = gsl_stats_mean(col_x.vector.data, col_x.vector.stride, num_faces);
        double y_mean = gsl_stats_mean(col_y.vector.data, col_y.vector.stride, num_faces);
        double x_var = gsl_stats_variance(col_x.vector.data, col_x.vector.stride, num_faces) * 2.0;
        double y_var = gsl_stats_variance(col_y.vector.data, col_y.vector.stride, num_faces) * 2.0;
        Point2f mean(x_mean, y_mean);

        printf("\tMean: \t\t[%f] [%f]\n", x_mean, y_mean);
        printf("\tVariance: \t[%f] [%f]\n", x_var, y_var);
        double r = sqrt(x_var + y_var);
        circle(viz, mean, r, CV_RGB(150,150,150), 1, 8, 0);

        Point2f user_point = applyXform(user_landmarks[i], xform);

        double x_dist = user_point.x - x_mean;
        double y_dist = user_point.y - y_mean;
        double user_r = sqrt(x_dist*x_dist + y_dist*y_dist);

        printf("\tUSER dist: \t[%f] [%f]\n", x_dist, y_dist);

        bool point = false;
        if (user_r < r){
            printf("point %d is within range!\n", i);
            circle(viz, user_point, 2, CV_RGB(100, 250, 100), 2, 8, 0);
            num_points_in_range ++;
            point = true;
        }
        else {
            circle(viz, user_point, 2, CV_RGB(200, 100, 100), 2, 8, 0);
        }

        fprintf(out, "%f %f %f %f %f %d\n", user_point.x, user_point.y, x_mean, y_mean, r, point);

    }

    fclose(out);

    printf("%d/%d points in range\n", num_points_in_range, num_points);

    //imshow("points", viz);
    //waitKey(0);
}


Point2f ScoreCalculatorApp::applyXform(Point2f pt, Mat &xform){
    float x = pt.x * xform.at<double>(0,0) + pt.y * xform.at<double>(0,1) + xform.at<double>(0,2);
    float y = pt.x * xform.at<double>(1,0) + pt.y * xform.at<double>(1,1) + xform.at<double>(1,2);
    
    return Point2f(x,y);
}

Point2f ScoreCalculatorApp::applyInverseXform(Point2f pt, Mat &xform){
    double det = xform.at<double>(0,0)*xform.at<double>(1,1) - xform.at<double>(0,1)*xform.at<double>(1,0);
    
    float px = pt.x - xform.at<double>(0,2);
    float py = pt.y - xform.at<double>(1,2);

    float x =  px * xform.at<double>(1,1)/det - py*xform.at<double>(0,1)/det;
    float y =  -px * xform.at<double>(1,0)/det + py*xform.at<double>(0,0)/det;
    return Point2f(x,y);
}


static ScoreCalculatorApp *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new ScoreCalculatorApp();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

