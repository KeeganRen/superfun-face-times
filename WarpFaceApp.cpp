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
            "   \t--list [image file and point file on same line, separated by space, one line per image]\n" \
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
            {"templateMesh",    1, 0, 402},
            {"output",          1, 0, 403},
            {"templatePoints",  1, 0, 404},
            {"canonicalPoints", 1, 0, 405},
            {"list",            1, 0, 406},
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

            case 406:
                useList = true;
                sprintf(listFile, "%s", optarg);
                break;

            default: 
                printf("Unrecognized option %d\n", c);
                break;
        }
    }
}

void WarpFaceApp::init(){
    printf("[init] Running program %s\n", argv[0]);
    useList = false;

    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }
    
    processOptions(argc, argv);

    loadTemplateFiles();

    if (useList){
        loadImageList();
    }
    else{
        num_images = 1;
        loadFaceSpecificFiles();
    }


    setupShapeStuff();
    populateTemplateMatrix();
    populateImageMatrix();
    shapeStuff();

    //getColorFromImage();
    //findTransformation();
    
    //makeNewFace();
   
}

void WarpFaceApp::loadImageList(){
    printf("[loadImageList] loading from file %s\n", listFile);
    FILE *file = fopen ( listFile, "r" );
    if ( file != NULL ) {
        char image_path[256];
        char point_path[256];
        while( fscanf(file, "%s %s\n", image_path, point_path) > 0 ) {
            imageFiles.push_back(image_path);
            imagePointFiles.push_back(point_path);
        }
        fclose (file);
    }
    else {
        perror (facePointsFile);
    }
    printf("[loadImageList] found %d image filenames\n", (int)imageFiles.size());
    num_images = (int)imageFiles.size();    
}

void WarpFaceApp::setupShapeStuff(){
    printf(" --- num_images = %d and num_points = %d ---\n", num_images, num_points);

    m_gsl_model = gsl_matrix_calloc(4, num_points);
    m_gsl_images = gsl_matrix_calloc(num_points, num_images);
    m_gsl_s = gsl_matrix_calloc(4, num_points);
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
    num_points = (int)templateMesh.size();


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

void WarpFaceApp::populateTemplateMatrix(){
    printf("[populateTemplateMatrix]");
    for (int i = 0; i < num_points; i++){
        Point3f c = templateColors[i];
        float lum = (0.2126*c.x) + (0.7152*c.y) + (0.0722*c.z);
        Point3f n = templateNormals[i];
        gsl_matrix_set(m_gsl_model, 1, i, n.x);
        gsl_matrix_set(m_gsl_model, 2, i, n.y);
        gsl_matrix_set(m_gsl_model, 3, i, n.z);
    }
}

void WarpFaceApp::populateImageMatrix(){
    printf("[populateImageMatrix]");
    for (int i = 0; i < num_images; i++){
        printf("[populateImageMatrix] image %d\n", i);
        // load the image
        Mat im = imread(imageFiles[i].c_str(), CV_LOAD_IMAGE_COLOR);
        im.convertTo(im, CV_64FC3, 1.0/255, 0);

        // load the points
        const char *filename = imagePointFiles[i].c_str();
        vector<Point2f> facePoints = loadPoints(filename);

        // warp and stuff
        fx = 4000;
        cameraMatrix = Matx33f(fx, 0, 0, 0, fx, 0, 0, 0, 1);
        vector<double> rv(3), tv(3);
        distortion_coefficients = Mat(5,1,CV_64FC1);
        rvec = Mat(3, 1, CV_64FC1);
        tvec = Mat(3, 1,CV_64FC1); 

        vector<Point2f> new_points;
       
        // transform between template and image front-of-face view
        solvePnP(templatePoints, canonicalPoints, cameraMatrix, distortion_coefficients, rvec, tvec, false, CV_ITERATIVE);

        cout << "cameraMatrix: " << cameraMatrix << endl;
        cout << "rvec: " << rvec << endl;
        cout << "tvec: " << tvec << endl;
        cout << "distortion_coefficients: " << distortion_coefficients << endl;

        // project onto image view
        projectPoints(templateMesh, rvec, tvec, cameraMatrix, distortion_coefficients, new_points);

        for (int j = 0; j < new_points.size(); j++){
            double c[3];
            bilinear(c, im, new_points[j].x, new_points[j].y);
            float lum = (0.2126*c[2]) + (0.7152*c[1]) + (0.0722*c[0]);
            gsl_matrix_set(m_gsl_images, j, i, lum);
        }

        for (int j = 0; j < templateMesh.size(); j++){
            circle(im, new_points[j], 4, 1, 0, 8, 0);
        }
        //imshow("face projected back", im);
        //waitKey(0);


        // transform between template and canonical front-of-face view
        solvePnP(templatePoints, canonicalPoints, cameraMatrix, distortion_coefficients, rvec, tvec, false, CV_ITERATIVE);
        projectPoints(templateMesh, rvec, tvec, cameraMatrix, distortion_coefficients, new_points);
        Mat drawable = Mat::zeros(500, 500, CV_32F);
        for (int j = 0; j < templateMesh.size(); j++){
            float color = gsl_matrix_get(m_gsl_images, j, i);
            circle(drawable, new_points[j], 4, color, 0, 8, 0);
        }

        //imshow("drawable", drawable);
        //waitKey(0);

    }

    printf("[populateImageMatrix] DONE!\n");
}

vector<Point2f> WarpFaceApp::loadPoints(const char* filename){
    printf("[loadPoints] loading points from %s\n", filename);
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


void WarpFaceApp::shapeStuff(){
    printf("[shapeStuff]...\n");

    // warp and stuff
    fx = 4000;
    cameraMatrix = Matx33f(fx, 0, 0, 0, fx, 0, 0, 0, 1);
    vector<double> rv(3), tv(3);
    distortion_coefficients = Mat(5,1,CV_64FC1);
    rvec = Mat(3, 1, CV_64FC1);
    tvec = Mat(3, 1,CV_64FC1); 

    vector<Point2f> new_points;
    // transform between template and canonical front-of-face view
    solvePnP(templatePoints, canonicalPoints, cameraMatrix, distortion_coefficients, rvec, tvec, false, CV_ITERATIVE);

    cout << "[shapeStuff] cameraMatrix: " << cameraMatrix << endl;
    cout << "[shapeStuff] rvec: " << rvec << endl;
    cout << "[shapeStuff] tvec: " << tvec << endl;
    cout << "[shapeStuff] distortion_coefficients: " << distortion_coefficients << endl;

    // project onto canonical view
    projectPoints(templateMesh, rvec, tvec, cameraMatrix, distortion_coefficients, new_points);





    // so ive got a fatty matrix... i want to get the mean of it and do SVD and stuff
    gsl_vector *mean = gsl_vector_calloc(num_points);
    for (int i = 0; i < num_images; i++){
        gsl_vector_view col = gsl_matrix_column(m_gsl_images, i);
        gsl_vector_add(mean, &col.vector);
    }
    gsl_vector_scale(mean, 1.0/num_images);
    for (int i = 0; i < num_images; i++){
        gsl_vector_view col = gsl_matrix_column(m_gsl_images, i);
        gsl_vector_sub(&col.vector, mean);
    }

    gsl_vector *S = gsl_vector_calloc(num_images);
    gsl_matrix *V = gsl_matrix_calloc(num_images, num_images);
    gsl_vector *work = gsl_vector_calloc(num_images);

    printf("[shapeStuff] computing SVD!\n");

    int res = gsl_linalg_SV_decomp(m_gsl_images, V, S, work);

    printf("[shapeStuff] SVD computed, result: %d\n", res);

    gsl_matrix *m_gsl_rank4 = gsl_matrix_calloc(4, num_points);


  
    // try visualizing eigenfaces
    for (int i = 0; i < 4; i++){
        gsl_vector_view col = gsl_matrix_column(m_gsl_images, i);
        gsl_vector_view vec_work = gsl_matrix_row(m_gsl_s, i);
        gsl_vector_memcpy(&vec_work.vector, &col.vector);
        gsl_vector_scale(&vec_work.vector, gsl_vector_get(S, i));

        Mat drawable = Mat::zeros(500, 500, CV_32F);
        for (int j = 0; j < new_points.size(); j++){
            float color = gsl_vector_get(&vec_work.vector, j)*.3;
            circle(drawable, new_points[j], 4, color, 0, 8, 0);
        }
        imshow("gmmmmm", drawable);
        waitKey(0);
    }

    solveStuff();

    // outputtt
    for (int i = 0; i < 4; i++){
        gsl_vector_view vec_work = gsl_matrix_row(m_gsl_final_result, i);

        Mat drawable = Mat::zeros(500, 500, CV_32F);
        for (int j = 0; j < new_points.size(); j++){
            float color = gsl_vector_get(&vec_work.vector, j)*.2;
            //printf("%f ", color);
            circle(drawable, new_points[j], 2, color, 0, 8, 0);
        }
        imshow("yeaahhhh", drawable);
        waitKey(0);
    }
}

void WarpFaceApp::solveStuff(){
    printf("[solveStuff] ... \n");
    // copying some photocity matrix solving code
    gsl_matrix* m_gsl_xy = m_gsl_s;
    gsl_matrix* m_gsl_gps = m_gsl_model;
    gsl_matrix* m_gsl_transform = gsl_matrix_alloc(4, 4);

    printf("[solveStuff] ... 1 \n");
  
    // transform = gps * xy' * inv(xy * xy')
    //A = gps * xy'
    //B = xy *xy'
    //C = A * inv(B)
    
    gsl_matrix *A = gsl_matrix_alloc(4,4);
    gsl_matrix *B = gsl_matrix_alloc(4,4);

    printf("[solveStuff] ... 2 \n");
    
    //A = gps * xy'
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, m_gsl_gps, m_gsl_xy, 0.0, A);

    printf("[solveStuff] ... 3 \n");
    
    //B = xy *xy'
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, m_gsl_xy, m_gsl_xy, 0.0, B);

    printf("[solveStuff] ... 4 \n");
    
    // invert B...
    gsl_matrix *inv = gsl_matrix_alloc(4,4);
    gsl_permutation *p = gsl_permutation_alloc (4);
    int s; 
    gsl_linalg_LU_decomp(B, p, &s);
    gsl_linalg_LU_invert(B, p, inv);
    
    // multiply A * inv
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, A, inv, 0.0, m_gsl_transform);
    
    gsl_matrix_free(A);
    gsl_matrix_free(B);
    gsl_matrix_free(inv);
    gsl_permutation_free(p);

    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            printf("%f ", gsl_matrix_get(m_gsl_transform, i, j));
            //gsl_matrix_set(m_gsl_transform, i, j, gsl_matrix_get(m_gsl_transform, i, j)/w);
        }
        printf("\n");
    }

    m_gsl_final_result = gsl_matrix_alloc(4,num_points);    
    //C = transform * xy
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, m_gsl_transform, m_gsl_xy, 0.0, m_gsl_final_result);

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

