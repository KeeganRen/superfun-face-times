/* CollectionFlowApp.cpp */

#include "CollectionFlowApp.h"
#include "getopt.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <opencv/highgui.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <istream>
#include <opencv2/opencv.hpp>
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
            "   \t--input <file> [file containing list of pre-detected/cropped images to process]\n" \
            "   \t--output <path to directory> [dir to put eigenface and low rank faces into]\n" \
            "   \t--visualize [visualize progress]\n" \
            "   \t--gray [convert to grayscale]\n" \
            "\n");
}

void CollectionFlowApp::processOptions(int argc, char **argv){
    while (1){
        static struct option long_options[] = {
            {"help",        0, 0, 'h'},
            {"input",       1, 0, 400},
            {"output",      1, 0, 401},
            {"visualize",   0, 0, 402},
            {"gray",        0, 0, 403},
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
                visualize = true;
                break;

            case 403:
                gray = true;
                break;
                
            default: 
                printf("Unrecognized option %d\n", c);
                break;
        }
    }
}

void CollectionFlowApp::init(){
    printf("[init] Running program %s\n", argv[0]);
    visualize = false;
    gray = false;
    sprintf(outputDir, ".");
    
    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }
    
    processOptions(argc, argv);
    
    loadFacesFromList();
    findImageSizeFromFirstImage();
    openImages();
    if (gray){
        convertImages();
    }

    //putInGslMatrix();
    buildMatrixAndRunPca();
    //outputSomeStuff(); 
}

void CollectionFlowApp::loadFacesFromList(){
    printf("[loadFacesFromList] Loading faces from list: %s\n", inputFile);
    FILE *file = fopen ( inputFile, "r" );
    if ( file != NULL ) {
        char line [ 256 ]; 
        while( fscanf(file, "%s\n", line) > 0 ) {
            printf("\tface file found: %s\n", line);
            faceList.push_back(string(line));
        }
        fclose (file);
    }
    else {
        perror (inputFile);
    }
}

void CollectionFlowApp::findImageSizeFromFirstImage(){
    printf("[findImageSizeFromFirstImage] %d images in list\n", (int)faceList.size());

    const char* firstImage = faceList[0].c_str();
    Mat img = imread(firstImage, CV_LOAD_IMAGE_COLOR);
    w = img.rows;
    h = img.cols;
    d = 3;

    printf("[findImageSizeFromFirstImage] First image %s: (%d x %d) %d channels\n", firstImage, w, h, d);
}

void CollectionFlowApp::openImages(){
    printf("[openImages] %d images in list\n", (int)faceList.size());
    for (int i = 0; i < faceList.size(); i++){
        Mat img = cvLoadImage(faceList[i].c_str(), CV_LOAD_IMAGE_COLOR);
        img.convertTo(img, CV_64FC3, 1.0/255, 0);
        if (img.data != NULL){
            faceImages.push_back(img);
        }
        else {
            printf("Error loading image %s\n", faceList[i].c_str());
        }
    }

}

void CollectionFlowApp::convertImages(){
    printf("[convertImages] %d images in list\n", (int)faceImages.size());
    for (int i = 0; i < faceImages.size(); i++){
        Mat im = faceImages[i];
        Mat gray;
        cvtColor(im, gray, CV_BGR2GRAY);
        faceImages[i] = gray;
    }

    d = 1;
    printf("[convertImages] Now there are %d channels\n", d);

    if (visualize){
        Mat m = faceImages[0];
        imshow("Converted image", m);
    }
}

void CollectionFlowApp::putInGslMatrix(){
    printf("[putInGslMatrix] image size: %dx%dx%d\n", w, h, d);

    Mat im = faceImages[0];
    int num_pixels = w*h*d;
    gsl_vector *m_gsl_vec = gsl_vector_calloc(num_pixels);

    matToGslVec(im, m_gsl_vec);

    /*
    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            if (d == 1){
                int idx = i*h + j;
                double val = im.at<double>(i, j);
                gsl_vector_set(m_gsl_vec, idx, val);
            }
            else if (d == 3){
                int idx = i*h + j;
                Vec3d val = im.at<Vec3d>(i, j);
                for (int k = 0; k < 3; k++){
                    gsl_vector_set(m_gsl_vec, idx*3 + k, val[k]);
                }
            }
        }
    }
    */

    printf("[putInGslMatrix] got that raw data into the gsl matrix\n");

    for (int i = 0; i < 100; i++){
        printf("%f ", i, gsl_vector_get(m_gsl_vec, i));
    }

    printf("[putInGslMatrix] putting it back in\n");

    /*
    if (d == 1){
        Mat m(w, h, CV_64F, m_gsl_vec->data);
        imshow("redone mat", m);
        waitKey(0);
    }
    if (d == 3){
        Mat m(w, h, CV_64FC3, m_gsl_vec->data);
        imshow("redone mat", m);
        waitKey(0);
    }
    */
    imshow("redone mat", im);
    waitKey(0);
}

void CollectionFlowApp::gslVecToMat(gsl_vector *orig, Mat &im){
    gsl_vector* vec = gsl_vector_calloc(orig->size);
    gsl_vector_memcpy(vec, orig);

    if (d == 1){
        Mat m(w, h, CV_64F, vec->data);
        im = m;
    }
    if (d == 3){
        Mat m(w, h, CV_64FC3, vec->data);
        im = m;
    }

    // this will be a mighty fine memory leak some day!
    //gsl_vector_free(vec);
}

void CollectionFlowApp::matToGslVec(Mat &im, gsl_vector *vec){
    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            if (d == 1){
                int idx = i*h + j;
                double val = im.at<double>(i, j);
                gsl_vector_set(vec, idx, val);
            }
            else if (d == 3){
                int idx = i*h + j;
                Vec3d val = im.at<Vec3d>(i, j);
                for (int k = 0; k < 3; k++){
                    gsl_vector_set(vec, idx*3 + k, val[k]);
                }
            }
        }
    }
}

void CollectionFlowApp::saveAs(char* filename, Mat m){
    printf("[saveAs] saving image to file: %s\n", filename);
    Mat rightFormat;
    m.convertTo(rightFormat, CV_8UC3, 1.0*255, 0);
    imwrite(filename, rightFormat);
}

void CollectionFlowApp::buildMatrixAndRunPca(){
    printf("[buildMatrixAndRunPca] Let's do it! Image size: %d x %d\n", w, h);
    int num_pixels = w*h*d;
    int num_images = faceImages.size();

   // the "original" images
    gsl_matrix *m_gsl_mat = gsl_matrix_calloc(num_pixels, num_images);

    // the low rank images
    gsl_matrix *m_gsl_mat_k = gsl_matrix_calloc(num_pixels, num_images);

    for (int i = 0; i < faceImages.size(); i++){
        gsl_vector_view col = gsl_matrix_column(m_gsl_mat, i);
        matToGslVec(faceImages[i], &col.vector);
    }

    printf("[buildMatrixAndRunPca] Matrix populated\n");

    for (int k = 4; k < 20; k++){
        printf("\t[COLLECTION FLOW] RANK %d\n", k);

        gsl_vector *m_gsl_mean = gsl_vector_calloc(num_pixels);

        for (int i = 0; i < faceImages.size(); i++){
            gsl_vector_view col = gsl_matrix_column(m_gsl_mat, i);
            gsl_vector_add(m_gsl_mean, &col.vector);
        }

        gsl_vector_scale(m_gsl_mean, 1.0/num_images);

        printf("[buildMatrixAndRunPca] Mean computed\n");

        Mat m;
        gslVecToMat(m_gsl_mean, m);
        if (visualize){    
            imshow("mean image", m);
        }
        char filename[100];
        sprintf(filename, "%s/mean-rank%d.jpg", outputDir, k);
        saveAs(filename, m);

        // copy m_gsl_mat into m_gsl_mat_k so we can do SVD
        gsl_matrix_memcpy(m_gsl_mat_k, m_gsl_mat);
        
        // subtract mean?
        bool use_mean = true;
        if (use_mean){
            printf("[buildMatrixAndRunPca] Subtracting mean from each image\n");
            for (int i = 0; i < faceImages.size(); i++){
                gsl_vector_view col = gsl_matrix_column(m_gsl_mat_k, i);
                gsl_vector_sub(&col.vector, m_gsl_mean);
            }
            
            gsl_vector_view col = gsl_matrix_column(m_gsl_mat_k, 1);
            Mat face;
            gslVecToMat(&(col.vector), face);
            imshow("face with mean subtracted", face);
        }

        gsl_vector *S = gsl_vector_calloc(num_images);
        gsl_matrix *V = gsl_matrix_calloc(num_images, num_images);
        gsl_vector *work = gsl_vector_calloc(num_images);

        printf("[buildMatrixAndRunPca] computing SVD!\n");

        int res = gsl_linalg_SV_decomp(m_gsl_mat_k, V, S, work);

        printf("[buildMatrixAndRunPca] SVD computed, result: %d\n", res);

        
        gsl_vector *vec_work = gsl_vector_calloc(num_pixels);

        // try visualizing eigenfaces
        for (int i = 0; i <= k; i++){
            gsl_vector_view col = gsl_matrix_column(m_gsl_mat_k, i);
            gsl_vector_memcpy(vec_work, &col.vector);
            gsl_vector_scale(vec_work, gsl_vector_get(S, i));

            Mat eigenface;
            gslVecToMat(vec_work, eigenface);

            if (visualize){
                char eig_title[30];
                sprintf(eig_title, "eigenface #%d", i);
                imshow(eig_title, eigenface);
            }

            char filename[100];
            sprintf(filename, "%s/rank%d-eigen%02d.jpg", outputDir, k, i);
            saveAs(filename, eigenface);
            
        }
        

        gsl_matrix *S_mat = gsl_matrix_calloc(num_images, num_images);
        gsl_matrix_set_zero(S_mat);

        for (int i = 0; i < k; i++){
            printf("\tSVD %d: %f\n", i, gsl_vector_get(S, i));
            gsl_matrix_set(S_mat, i, i, gsl_vector_get(S, i));
        }
        
        gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                           1.0, m_gsl_mat_k, S_mat,
                           0.0, m_gsl_mat_k);

        
        gsl_blas_dgemm (CblasNoTrans, CblasTrans,
                           1.0, m_gsl_mat_k, V,
                           0.0, m_gsl_mat_k);


        if (use_mean){
            printf("[buildMatrixAndRunPca] Adding mean back\n");
            for (int i = 0; i < faceImages.size(); i++){
                gsl_vector_view col = gsl_matrix_column(m_gsl_mat_k, i);
                gsl_vector_add(&col.vector, m_gsl_mean);
            }
        }

        // try visualizing faces in lower space
        for (int i = 0; i < num_images; i++){
            printf("visualizing the low rank images\n");

            Mat m_lowrank;
            gsl_vector_view col_low = gsl_matrix_column(m_gsl_mat_k, i);
            gslVecToMat(&col_low.vector, m_lowrank);

            Mat m_highrank;
            gsl_vector_view col_high = gsl_matrix_column(m_gsl_mat, i);
            gslVecToMat(&col_high.vector, m_highrank);
    
            if (visualize){
                imshow("high rank", m_highrank);
                imshow("low rank", m_lowrank);
            }


            char filename[100];
            sprintf(filename, "%s/rank%d-face%02d-orig.jpg", outputDir, k, i);
            saveAs(filename, m_highrank);
            sprintf(filename, "%s/rank%d-face%02d-low.jpg", outputDir, k, i);
            saveAs(filename, m_lowrank);

            
            printf("[computeFlow] image %d/%d\n", i, num_images);
            // magic variables
            double alpha = 0.03;
            double ratio = 0.85;
            int minWidth = 20;
            int nOuterFPIterations = 4;
            int nInnerFPIterations = 1;
            int nSORIterations = 40;
            
            Mat vx, vy, warp;
            
            CVOpticalFlow::findFlow(vx, vy, warp, m_lowrank, m_highrank, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
            imshow("the warped picture", warp);
            imshow("flow", CVOpticalFlow::showFlow(vx, vy));


            Mat warped;
            CVOpticalFlow::warp(warped, m_highrank, vx, vy);
            
            m_highrank = warped;

            if (visualize){
                imshow("warped back", m_highrank);
                waitKey(0);
            }

            matToGslVec(m_highrank, &col_high.vector);

            sprintf(filename, "%s/rank%d-face%02d-warped.jpg", outputDir, k, i);
            saveAs(filename, m_highrank);
            


            /*
            // TODO: HOW TO COPY 2 PIX INTO ONE?
            Mat pair = Mat::zeros(w*2, h, faceImages[0].type());
            Mat high = pair(Rect(0, 0, w, h));
            Mat low = pair(Rect(w-1, 0, w, h));


            Mat m = pair;

            char filename[100];
            sprintf(filename, "%s/rank%d-face%02d.jpg", outputDir, k, i);
            saveAs(filename, m);

            if (visualize){
                imshow("low rank", m);
                cvWaitKey(0);
            }
            */
            
        }
        
    }
}


static CollectionFlowApp *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new CollectionFlowApp();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}
