/* WarpFaceApp.cpp */

//make -f Makefile.mac warpFace && ./warpFace --list oagTest/images_unwarped_with_points.txt 
//--templateMesh model/grid-igor2.txt --templatePoints model/grid-igor-fiducials2.txt --canonicalPoints model/grid-igor-canonical2d.txt --visualize


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
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

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
            {"visualize",       0, 0, 407},
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

            case 407:
                visualize = true;
                printf("visualize = true\n");
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
    visualize = false;

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

    /* some tests */
    //testIntegrability();
    //testTemplate();
    //testTemplateVsCanonical();


    //findTransformation();


    shapeStuff();

    //getColorFromImage();
    //findTransformation();
    
    //makeNewFace();
   
}

void WarpFaceApp::testIntegrability(){
    printf("[testIntegrability] TESTING... recover depth from template model's normals\n");
    for (int i = 0; i < num_points; i++){
        gsl_matrix_set(m_gsl_final_result, 1, i, templateNormals[i].x);
        gsl_matrix_set(m_gsl_final_result, 2, i, templateNormals[i].y);
        gsl_matrix_set(m_gsl_final_result, 3, i, templateNormals[i].z);
    }
    recoverDepth();
}

void WarpFaceApp::testTemplate(){
    printf("[testTemplate]\n");

    Mat templateImage = Mat::zeros(500, 500, CV_8UC3);
    for (int i = 0; i < templateMesh.size(); i++){
        //Point3d color = templateColors[i];
        Point2f circle_pos = Point2f(templateMesh[i].x, 500-templateMesh[i].y);
        Point3f normal = templateNormals[i];
        Point3d color;
        float f = 100.0;
        color.x = normal.x * f;
        color.y = normal.y * f;
        color.z = normal.z * f;
        circle(templateImage, circle_pos, .7, CV_RGB(color.x, color.y, color.z), 2, 8, 0);
    }
    //Rect cropROI(128, 80, 250, 310);
    //templateImage = templateImage(cropROI);

    for (int i = 0; i < templatePoints.size(); i++){
        Point2f circle_pos = Point2f(templatePoints[i].x, 500-templatePoints[i].y);
        circle(templateImage, circle_pos, 4, CV_RGB(255, 255, 0), 2, 8, 0);
        circle(templateImage, canonicalPoints[i], 4, CV_RGB(255, 0, 255), 2, 8, 0);
    }

    imshow("template mesh warped to canonical points", templateImage);
    imwrite("templateimage.jpg", templateImage);
    cvWaitKey(0);
}

void WarpFaceApp::testTemplateVsCanonical(){
    printf("[testTemplateVsCanonical]\n");

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
    for (int i = 0; i < new_points.size(); i++){
        //Point3d color = templateColors[i];
        Point3f normal = templateNormals[i];
        Point3d color;
        float f = 100.0;
        color.x = normal.x * f;
        color.y = normal.y * f;
        color.z = normal.z * f;
        circle(templateImage, new_points[i], .7, CV_RGB(color.x, color.y, color.z), 2, 8, 0);
    }
    //Rect cropROI(128, 80, 250, 310);
    //templateImage = templateImage(cropROI);
    imshow("template mesh warped to canonical points", templateImage);

    projectPoints(templatePoints, rvec, tvec, cameraMatrix, distortion_coefficients, new_points);
    for (int i = 0; i < new_points.size(); i++){
        circle(templateImage, new_points[i], 3, CV_RGB(255,255,0), 2, 8, 0);
        circle(templateImage, canonicalPoints[i], 3, CV_RGB(0,255,255), 2, 8, 0);
    }

    imshow("template mesh warped to canonical points", templateImage);
    cvWaitKey(0);

    /*
    // transformation between 3D mesh points and image
    solvePnP(templatePoints, facePoints, cameraMatrix, distortion_coefficients, rvec, tvec, false, CV_EPNP);

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
    */
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
    m_gsl_final_result = gsl_matrix_alloc(4,num_points);
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
        //while( fscanf(file, "%f %f %f %f %f %f\n", &x, &y, &z, &nx, &ny, &nz) > 0 ) {
            templateMesh.push_back(Point3f(x,y,z));
            templateNormals.push_back(Point3f(nx,ny,nz));

            //r = 155;
            //g = 155;
            //b = 155;
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
        while( fscanf(file, "%f %f %f\n", &x, &y, &z) > 0 ) {
            printf("canonical points loaded: %f %f %f\n", x, y, z);
            templatePoints.push_back(Point3f(x,y,z));
        }
        fclose (file);
    }
    else {
        perror (templatePointsFile);
    }

    printf("[loadTemplateFiles] huzzah, {{%d}} points loaded into templatePoints\n", (int)templatePoints.size());
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
    solvePnP(templatePoints, facePoints, cameraMatrix, distortion_coefficients, rvec, tvec, false, CV_EPNP);

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
    solvePnP(templatePoints, canonicalPoints, cameraMatrix, distortion_coefficients, rvec, tvec, false, CV_EPNP);

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
    printf("[populateTemplateMatrix]\n");
    for (int i = 0; i < num_points; i++){
        Point3f c = templateColors[i];
        float lum = (0.2126*c.x) + (0.7152*c.y) + (0.0722*c.z);
        lum/=255.0;
        Point3f n = templateNormals[i];
        gsl_matrix_set(m_gsl_model, 0, i, lum);
        gsl_matrix_set(m_gsl_model, 1, i, n.x*lum);
        gsl_matrix_set(m_gsl_model, 2, i, n.y*lum);
        gsl_matrix_set(m_gsl_model, 3, i, n.z*lum);
    }

    if (visualize){
        Mat img = Mat::zeros(500, 2000, CV_32F);
        for (int i = 0; i < num_points; i++){
            Point2f pt = Point2f(templateMesh[i].x, 500 - templateMesh[i].y);
            Point3i color = templateColors[i];
            Point3f n = templateNormals[i];
            double nx = gsl_matrix_get(m_gsl_model, 1, i);
            double ny = gsl_matrix_get(m_gsl_model, 2, i);
            double nz = gsl_matrix_get(m_gsl_model, 3, i);
            circle(img, pt, 1, CV_RGB(color.z/255.0, color.y/255.0, color.x/255.0), 0, 8, 0);
            circle(img, Point2f(pt.x + 500, pt.y), 1, CV_RGB(nx, nx, nx), 0, 8, 0);
            circle(img, Point2f(pt.x + 1000, pt.y), 1, CV_RGB(ny, ny, ny), 0, 8, 0);
            circle(img, Point2f(pt.x + 1500, pt.y), 1, CV_RGB(nz, nz, nz), 0, 8, 0);
        }

        resize(img, img, Size(1000, 250));
        imshow("[albedo] [albedo*nx] [albedo*ny] [albedo*nz]", img);
        //waitKey(0);

        Mat img2 = Mat::zeros(500, 2000, CV_32F);
        for (int i = 0; i < num_points; i++){
            Point2f pt = Point2f(templateMesh[i].x, 500 - templateMesh[i].y);
            double lum = gsl_matrix_get(m_gsl_model, 0, i);
            double nx = gsl_matrix_get(m_gsl_model, 1, i);
            double ny = gsl_matrix_get(m_gsl_model, 2, i);
            double nz = gsl_matrix_get(m_gsl_model, 3, i);
            circle(img2, pt, 1, lum, 0, 8, 0);
            circle(img2, Point2f(pt.x + 500, pt.y), 1, nx/lum, 0, 8, 0);
            circle(img2, Point2f(pt.x + 1000, pt.y), 1, ny/lum, 0, 8, 0);
            circle(img2, Point2f(pt.x + 1500, pt.y), 1, nz/lum, 0, 8, 0);
        }

        resize(img2, img2, Size(1000, 250));
        imshow("[albedo] [nx] [ny] [nz]", img2);
    }   
}

void WarpFaceApp::populateImageMatrix(){
    printf("[populateImageMatrix]");
    for (int i = 0; i < num_images; i++){
        //printf("[populateImageMatrix] image %d\n", i);
        // load the image
        Mat im = imread(imageFiles[i].c_str(), CV_LOAD_IMAGE_COLOR);
        im.convertTo(im, CV_64FC3, 1.0/255, 0);

        // load the points
        const char *filename = imagePointFiles[i].c_str();
        vector<Point2f> facePoints = loadPoints(filename);


        
        printf("[populateImageMatrix] size of facePoints: %d, size of canonicalPoints: %d\n", facePoints.size(), canonicalPoints.size());


        Mat transform = estimateRigidTransform(facePoints, canonicalPoints, true);

        Mat warped = Mat::zeros(500, 500, im.type() );
        warpAffine(im, warped, transform, warped.size() );


        for (int j = 0; j < templateMesh.size(); j++){
            Vec3d c = warped.at<Vec3d>(templateMesh[j].y, templateMesh[j].x);
            float lum = (0.2126*c[2]) + (0.7152*c[1]) + (0.0722*c[0]);
            gsl_matrix_set(m_gsl_images, j, i, lum);
        }


        if (visualize){
            Mat drawable = Mat::zeros(500, 500, CV_32F);
            for (int j = 0; j < templateMesh.size(); j++){
                double c = gsl_matrix_get(m_gsl_images, j, i);
                circle(drawable, Point2f(templateMesh[j].x,500-templateMesh[j].y), 1, CV_RGB(c, c, c), 0, 8, 0);
            }

            //imshow("drawable", drawable);
            //waitKey(0);
        }


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

    if (0){
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
    }

    gsl_vector *S = gsl_vector_calloc(num_images);
    gsl_matrix *V = gsl_matrix_calloc(num_images, num_images);
    gsl_vector *work = gsl_vector_calloc(num_images);

    printf("[shapeStuff] computing SVD!\n");

    int res = gsl_linalg_SV_decomp(m_gsl_images, V, S, work);

    printf("[shapeStuff] SVD computed, result: %d\n", res);

    gsl_matrix *m_gsl_rank4 = gsl_matrix_calloc(4, num_points);


    /*
    // try visualizing the template face
    for (int i = 0; i < 4; i++){
        gsl_vector_view vec_work = gsl_matrix_row(m_gsl_model, i);

        Mat drawable = Mat::zeros(500, 500, CV_32F);
        for (int j = 0; j < new_points.size(); j++){
            float color = gsl_vector_get(&vec_work.vector, j)*.5;
            //printf("%f ", color);
            circle(drawable, new_points[j], 2, color, 0, 8, 0);
        }
        imshow("template face", drawable);
        //waitKey(0);
    }
    */

    for (int i = 0; i < 4; i++){
        gsl_vector_view col = gsl_matrix_column(m_gsl_images, i);
        gsl_vector_view vec_work = gsl_matrix_row(m_gsl_s, i);
        gsl_vector_memcpy(&vec_work.vector, &col.vector);
        gsl_vector_scale(&vec_work.vector, gsl_vector_get(S, i));
    }

    /*
    for (int i = 0; i < num_points; i++){
        double val = gsl_matrix_get(m_gsl_s, 0, i);
        for (int k = 1; k < 4; k++){
            gsl_matrix_set(m_gsl_s, k, i, gsl_matrix_get(m_gsl_s, k, i)/val);
        }
    }
    */


    if (visualize){
        Mat img = Mat::zeros(500, 2000, CV_32F);
        for (int i = 0; i < num_points; i++){
            Point2f pt = Point2f(templateMesh[i].x, 500 - templateMesh[i].y);
            for (int k = 0; k < 4; k++){
                double val = gsl_matrix_get(m_gsl_s, k, i);
                circle(img, Point2f(pt.x + k*500, pt.y), 1, val, 0, 8, 0);
            }
        }

        resize(img, img, Size(1000, 250));
        imshow("[eigenface1] [eigenface2] [eigenface3] [eigenface4]", img);
    }   

  /*
    // try visualizing eigenfaces
    for (int i = 0; i < 4; i++){
        gsl_vector_view col = gsl_matrix_column(m_gsl_images, i);
        gsl_vector_view vec_work = gsl_matrix_row(m_gsl_s, i);
        gsl_vector_memcpy(&vec_work.vector, &col.vector);
        gsl_vector_scale(&vec_work.vector, gsl_vector_get(S, i));

        Mat drawable = Mat::zeros(500, 500, CV_32F);
        for (int j = 0; j < new_points.size(); j++){
            float color = gsl_vector_get(&vec_work.vector, j)*.5;
            circle(drawable, new_points[j], 4, color, 0, 8, 0);
        }
        imshow("gmmmmm", drawable);
        waitKey(0);
    }
    */

    solveStuff();


    if (visualize){
        Mat img0 = Mat::zeros(500, 2000, CV_32F);
        for (int i = 0; i < num_points; i++){
            Point2f pt = Point2f(templateMesh[i].x, 500 - templateMesh[i].y);
            for (int k = 0; k < 4; k++){
                double val = gsl_matrix_get(m_gsl_final_result, k, i);
                circle(img0, Point2f(pt.x + k*500, pt.y), 1, val, 0, 8, 0);
            }
        }

        resize(img0, img0, Size(1000, 250));
        imshow("solved (new images) before albedo divided out", img0);
    } 
    
    for (int i = 0; i < num_points; i++){
        double albedo = gsl_matrix_get(m_gsl_final_result, 0, i);
        for (int k = 1; k < 4; k++){
            gsl_matrix_set(m_gsl_final_result, k, i, gsl_matrix_get(m_gsl_final_result, k, i)/albedo);
        }
    }
    
    
    if (visualize){
        Mat img2 = Mat::zeros(500, 2000, CV_32F);
        for (int i = 0; i < num_points; i++){
            Point2f pt = Point2f(templateMesh[i].x, 500 - templateMesh[i].y);
            for (int k = 0; k < 4; k++){
                double val = gsl_matrix_get(m_gsl_final_result, k, i);
                circle(img2, Point2f(pt.x + k*500, pt.y), 1, val, 0, 8, 0);
            }
        }

        resize(img2, img2, Size(1000, 250));
        imshow("after albedo divided out, should look like [average, nx, ny, nz]", img2);
    }   

    if (visualize){
        Mat img3 = Mat::zeros(500, 2000, CV_32F);
        for (int i = 0; i < num_points; i++){
            Point2f pt = Point2f(templateMesh[i].x, 500 - templateMesh[i].y);
            double nx = gsl_matrix_get(m_gsl_final_result, 1, i);
            double ny = gsl_matrix_get(m_gsl_final_result, 2, i);
            double nz = gsl_matrix_get(m_gsl_final_result, 3, i);

            circle(img3, Point2f(pt.x + 0*500, pt.y), 1, -1*nx/nz, 0, 8, 0);
            circle(img3, Point2f(pt.x + 1*500, pt.y), 1, 1*nx/nz, 0, 8, 0);
            circle(img3, Point2f(pt.x + 2*500, pt.y), 1, -1*ny/nz, 0, 8, 0);
            circle(img3, Point2f(pt.x + 3*500, pt.y), 1, 1*ny/nz, 0, 8, 0);
        }

        resize(img3, img3, Size(1000, 250));
        imshow("zx and zy", img3);
    }   
    
    waitKey(0);


    Mat zx = Mat::zeros(500, 500, CV_32F);
    Mat zy = Mat::zeros(500, 500, CV_32F);
    Mat zx_neg = Mat::zeros(500, 500, CV_32F);
    Mat zy_neg = Mat::zeros(500, 500, CV_32F);
    for (int i = 0; i < num_points; i++){
        Point3f p = templateMesh[i];
        //Point2f p = new_points[i];
        double nx = gsl_matrix_get(m_gsl_final_result, 1, i);
        double ny = gsl_matrix_get(m_gsl_final_result, 2, i);
        double nz = gsl_matrix_get(m_gsl_final_result, 3, i);
        zx.at<float>(p.y, p.x) = -1*nx/nz;
        zy.at<float>(p.y, p.x) = -1*ny/nz;
        zx_neg.at<float>(p.y, p.x) = 1*nx/nz;
        zy_neg.at<float>(p.y, p.x) = 1*ny/nz;
        //cout << "point p: " << p << " zx: " << -1*nx/nz << " zy: " << -1*ny/nz << endl;

    }
    /*
    imshow("zx", zx);
    imshow("zy", zy);
    imshow("zx_neg", zx_neg);
    imshow("zy_neg", zy_neg);
    */

    recoverDepth();

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

       
    //C = transform * xy
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, m_gsl_transform, m_gsl_xy, 0.0, m_gsl_final_result);

}

void WarpFaceApp::recoverDepth(){
    // this function will update the z depth in the templateMesh vector!
    printf("[recoverDepth]\n");

    // build problem 
    int m = templateMesh.size();
    vector<T> coefficients;
    vector<double> b_vec;

    // make a reverse lookup map for finding what index a point (x,y) is... 
    Mat lookup = Mat(500, 500, CV_32S);
    for (int i = 0; i < templateMesh.size(); i++){
        Point3f p = templateMesh[i];
        lookup.at<int>(p.x, p.y) = i;
    }

    // populate coefficients: 
    int coef_count = 0;
    //for each point in templateMesh, get x and y and add some constraints
    for (int i = 0; i < templateMesh.size(); i++){
        Point3f p = templateMesh[i];
        double nx = gsl_matrix_get(m_gsl_final_result, 1, i);
        double ny = gsl_matrix_get(m_gsl_final_result, 2, i);
        double nz = gsl_matrix_get(m_gsl_final_result, 3, i); 


        // todo later: figure out of point is on the boundary or not and do 2 different things
        
        int x1y = lookup.at<int>(p.x+1, p.y);
        int xy = lookup.at<int>(p.x, p.y);
        int xy1 = lookup.at<int>(p.x, p.y+1);

        //printf("x:%3f, \ty: %3f, \tx1y: %d, \txy: %d, \txy1: %d, \tm: %d\n", p.x, p.y, x1y, xy, xy1, m);

        // pixel constraint 1
        b_vec.push_back(-1.0*nx);
        coefficients.push_back(T(coef_count, x1y, nz));
        coefficients.push_back(T(coef_count, xy, -nz));
        coef_count++;

        b_vec.push_back(-1.0*ny);
        coefficients.push_back(T(coef_count, xy1, nz));
        coefficients.push_back(T(coef_count, xy, -nz));
        coef_count++;


    }

    int c = b_vec.size(); // some number of constraints
    printf("coef count: %d, c: %d, m: %d\n", coef_count, c, m);
    SpMat A(c,m); 
    Eigen::VectorXd b(c); 
    printf("done inittig stuff\n");

    A.setFromTriplets(coefficients.begin(), coefficients.end());
    SpMat M(c,c);
    M = A.transpose()*A;

    printf("done setting from triplets stuff\n");
    for (int i = 0; i < b_vec.size(); i++){
        b(i) = b_vec[i];
    }

    Eigen::VectorXd b2 = A.transpose()*b;

    printf("done with all that\n");

    printf("M: %dx%d, b2: %dx1, A: %dx%d, b: %d\n", M.rows(), M.cols(), b2.size(), A.rows(), A.cols(), b.size());

     // Solving:
    Eigen::SimplicialCholesky<SpMat> chol(M); // performs a Cholesky factorization of A
    Eigen::VectorXd x = chol.solve(b2); // use the factorization to solve for the given right hand side

    printf("Solved.\n");

    FILE *f = fopen("meow-ptswithnormals.ply", "w");
    fprintf(f,"ply\n");
    fprintf(f,"format ascii 1.0\n");
    fprintf(f,"comment VCGLIB generated\n");
    fprintf(f,"element vertex %d\n", (int)templateMesh.size());
    fprintf(f,"property float x\n");
    fprintf(f,"property float y\n");
    fprintf(f,"property float z\n");
    fprintf(f,"property float nx\n");
    fprintf(f,"property float ny\n");
    fprintf(f,"property float nz\n");
    fprintf(f,"end_header\n");

    for (int i = 0; i < templateMesh.size(); i++){
        Point3f p = templateMesh[i];
        double nx = gsl_matrix_get(m_gsl_final_result, 1, i);
        double ny = gsl_matrix_get(m_gsl_final_result, 2, i);
        double nz = gsl_matrix_get(m_gsl_final_result, 3, i); 
        fprintf(f, "%f %f %f %f %f %f\n", p.x, p.y, x(i), nx, ny, nz);
    }

    fclose(f);
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

