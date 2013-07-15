/* CrossClusterMorphApp.cpp */

#include "CrossClusterMorphApp.h"
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

#include "FaceLib.h"

#include <Eigen/Sparse>
#include <Eigen/Dense>
using namespace Eigen;
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
            "   \t--face [face image]\n" \
            "   \t--avgface [face image]\n" \
            "   \t--eigenfaces <file> [list of paths to eigenfaces to use]\n" \
            "   \t--visualize [if you want to see images of the progress]\n" \
            "\n");
}

void CrossClusterMorphApp::processOptions(int argc, char **argv){
    while (1){
        static struct option long_options[] = {
            {"help",        0, 0, 'h'},
            {"face",        1, 0, 402},
            {"eigenfaces",  1, 0, 403},
            {"avgface",     1, 0, 404},
            {"visualize",   0, 0, 407},
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
                
            case 402:
                faceFile = optarg;
                break;

            case 403:
                eigenfacesFile = optarg;
                break;

            case 404:
                avgFaceFile = optarg;
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

void CrossClusterMorphApp::init(){
    printf("[init] Running program %s\n", argv[0]);
    visualize = false;

    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }
    
    processOptions(argc, argv);

    loadFace();
    loadEigenfaces();

    projectNewFace();
}

void CrossClusterMorphApp::loadFace(){
    printf("[loadFace] Loading new face %s\n", faceFile);
    faceImage = cvLoadImage(faceFile, CV_LOAD_IMAGE_COLOR);
    faceImage.convertTo(faceImage, CV_64FC3, 1.0/255, 0);

    w = faceImage.rows;
    h = faceImage.cols;
    d = 3;

    avgFace = cvLoadImage(avgFaceFile, CV_LOAD_IMAGE_COLOR);
    avgFace.convertTo(avgFace, CV_64FC3, 1.0/255, 0);

    if (visualize){
        imshow("face", faceImage);
        waitKey(100);
    }
}

void CrossClusterMorphApp::loadEigenfaces(){
    printf("[loadEigenfaces] Loading eigenfaces from list: %s\n", eigenfacesFile);
    vector<string> eigenfacesList;
    FILE *file = fopen ( eigenfacesFile, "r" );
    if ( file != NULL ) {
        char line [ 256 ]; 
        while( fscanf(file, "%s\n", line) > 0 ) {
            printf("read file: %s\n", line);
            eigenfacesList.push_back(string(line));
        }
        fclose (file);
    }
    else {
        perror (eigenfacesFile);
    }

    // try to determine if its grayscale or not
    gsl_vector *test_vec = gsl_vector_calloc(w*h);
    gsl_vector *test_vec3 = gsl_vector_calloc(w*h*3);
    FILE *f = fopen(eigenfacesList[0].c_str(), "rb");
    
    bool alwaysUseThree = true;

    if (alwaysUseThree){
        if (gsl_vector_fread(f, test_vec3)==0){
            printf("[loadEigenfaces] read into THREE- channel vector\n");
            d = 3;
        }
    }
    else {
        if (gsl_vector_fread(f, test_vec)==0){
            printf("[loadEigenfaces] vector read successfully into 1-channel vector\n");
            d = 1;
        }
        else {
            rewind(f);
            if (gsl_vector_fread(f, test_vec3)==0){
                printf("[loadEigenfaces] read into THREE- channel vector\n");
                d = 3;
            }
        }
    }
    printf("DEPTH OF IMAGES: %d\n", d);
    if (d == 1){
        printf("[loadEigenfaces] gotta convert our image to a different number of channels!\n");
        faceImage.convertTo(faceImage, CV_32FC3);
        cvtColor(faceImage,faceImage,CV_RGB2GRAY);
        faceImage.convertTo(faceImage, CV_64F);
    }

    num_pixels = w*h*d;
    num_eigenfaces = eigenfacesList.size();

    // mean face
    m_gsl_meanface = gsl_vector_calloc(num_pixels);
    FaceLib::matToGslVec(avgFace, m_gsl_meanface, d, w, h);
    printf("done supposedly loading mean face into gsl matrix");

    // regular face
    m_gsl_face = gsl_vector_calloc(num_pixels);
    FaceLib::matToGslVec(faceImage, m_gsl_face, d, w, h);
    printf("done supposedly loading regular face into gsl matrix");

    // eigenfaces
    printf("sizes of things. num_pixels: %d, num eigenfaces: %d\n", num_pixels, num_eigenfaces);
    m_gsl_eigenfaces = gsl_matrix_calloc(num_pixels, num_eigenfaces);
    for (int i = 0; i < eigenfacesList.size(); i++){
        gsl_vector_view col = gsl_matrix_column(m_gsl_eigenfaces, i);
        FILE *f = fopen(eigenfacesList[i].c_str(), "rb");
        int success = gsl_vector_fread(f, &col.vector);
        
        if (visualize){
            Mat face;
            FaceLib::gslVecToMat(&(col.vector), face, d, w, h);
            imshow("loaded eigenface", face);
            waitKey(0);
        }
    }
}

void CrossClusterMorphApp::projectNewFace(){
    printf("[projectNewFace] projecting new face onto lower dimensional space!\n");
    gsl_vector *m_gsl_face_minus_mean = gsl_vector_calloc(num_pixels);
    gsl_vector_memcpy(m_gsl_face_minus_mean, m_gsl_face);
    gsl_vector_sub(m_gsl_face_minus_mean, m_gsl_meanface);

    printf("done copying and subtracting vectors\n");

    gsl_vector *m_gsl_eigenvalues = gsl_vector_calloc(num_eigenfaces);
    for (int i = 0; i < 4; i++){
        gsl_vector_view col = gsl_matrix_column(m_gsl_eigenfaces, i);
        double norm = gsl_blas_dnrm2 (&col.vector);
        gsl_vector_scale(&col.vector, 1.0/norm);

        double val;
        gsl_blas_ddot(&col.vector, m_gsl_face_minus_mean, &val);

        printf("dot value: %f, norm of vec: %f\n", val, norm);
        //val /= norm/255.0;

        gsl_vector_set(m_gsl_eigenvalues, i, val);
    }

    gsl_blas_dgemv(CblasNoTrans, 1, m_gsl_eigenfaces, m_gsl_eigenvalues, 0, m_gsl_face_minus_mean);

    gsl_vector_add(m_gsl_face_minus_mean, m_gsl_meanface);
    if (visualize){
        Mat face;
        FaceLib::gslVecToMat(m_gsl_face_minus_mean, face, d, w, h);
        imshow("projected face", face);
        //waitKey(0);
    }

    gsl_vector *m_gsl_projectednewface = m_gsl_face_minus_mean;

    Mat face;
    FaceLib::gslVecToMat(m_gsl_face_minus_mean, face, d, w, h);
    FaceLib::saveAs("proj-into-cluster.jpg", face);


    // magic variables
    double alpha = 0.02;
    double ratio = 0.85;
    int minWidth = 20;
    int nOuterFPIterations = 4;
    int nInnerFPIterations = 1;
    int nSORIterations = 40;
    Mat vx, vy, warp;
    Mat blah;

    CVOpticalFlow::findFlow(vx, vy, warp, face, faceImage, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);

    CVOpticalFlow::warp(blah, faceImage, vx, vy);

    if (visualize){
        imshow("the warped picture", blah);
        imshow("flow", CVOpticalFlow::showFlow(vx, vy));
        waitKey(0);
    }

    

}

static CrossClusterMorphApp *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new CrossClusterMorphApp();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

