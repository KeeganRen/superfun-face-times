/* FaceProcessApp.cpp */

#include "FaceProcessApp.h"
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
            "   \t--inputFile [file containing list of pre-detected/cropped images to process]\n" \
            "\n");
}

void FaceProcessApp::processOptions(int argc, char **argv){
    while (1){
        static struct option long_options[] = {
            {"help",        0, 0, 'h'},
            {"input",       1, 0, 400},
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
                
            default: 
                printf("Unrecognized option %d\n", c);
                break;
        }
    }
}

void FaceProcessApp::init(){
    printf("[init] Running program %s\n", argv[0]);
    
    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }
    
    processOptions(argc, argv);
    
    loadFacesFromList();
    findImageSizeFromFirstImage();
    openImages();
    convertImages();
    buildMatrixAndRunPca();
    //outputSomeStuff(); 
}

void FaceProcessApp::loadFacesFromList(){
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

void FaceProcessApp::findImageSizeFromFirstImage(){
    printf("[findImageSizeFromFirstImage] %d images in list\n", (int)faceList.size());

    const char* firstImage = faceList[0].c_str();
    IplImage* img;
    img = cvLoadImage(firstImage, CV_LOAD_IMAGE_COLOR);
    w = img->width;
    h = img->height;
    cvReleaseImage(&img);

    printf("[findImageSizeFromFirstImage] First image %s: (%d x %d)\n", firstImage, w, h);
}

void FaceProcessApp::openImages(){
    printf("[openImages] %d images in list\n", (int)faceList.size());
    for (int i = 0; i < faceList.size(); i++){
        IplImage* img = cvLoadImage(faceList[i].c_str(), CV_LOAD_IMAGE_COLOR);
        if (img){
            faceImages.push_back(img);
        }
        else {
            printf("Error loading image %s\n", faceList[i].c_str());
        }
    }
}

void FaceProcessApp::convertImages(){
    printf("[convertImages] %d images in list\n", (int)faceImages.size());
    for (int i = 0; i < faceImages.size(); i++){
        IplImage *im_rgb  = faceImages[i];
        IplImage *im_gray = cvCreateImage(cvGetSize(im_rgb),IPL_DEPTH_8U,1);
        cvCvtColor(im_rgb,im_gray,CV_RGB2GRAY);
        faceImages[i] = im_gray;
        cvReleaseImage(&im_rgb);
    }

    if (0){
        Mat m = faceImages[0];
        imshow("gray image", m);
        cvWaitKey(0);
    }
}

void FaceProcessApp::buildMatrixAndRunPca(){
    printf("[buildMatrixAndRunPca] Let's do it! Image size: %d x %d\n", w, h);
    int num_pixels = w*h;
    int num_images = faceImages.size();

    gsl_matrix *m_gsl_mat = gsl_matrix_calloc(num_pixels, num_images);
    for (int i = 0; i < faceImages.size(); i++){
        unsigned char* im = (unsigned char*)(faceImages[i]->imageData);
        for (int j = 0; j < num_pixels; j++){
            gsl_matrix_set(m_gsl_mat, j, i, (double)(im[j]));
        }
    }

    printf("[buildMatrixAndRunPca] Matrix populated\n");

    gsl_vector *m_gsl_mean = gsl_vector_calloc(num_pixels);
    for (int i = 0; i < faceImages.size(); i++){
        gsl_vector_view col = gsl_matrix_column(m_gsl_mat, i);
        gsl_vector_add(m_gsl_mean, &col.vector);
    }
    gsl_vector_scale(m_gsl_mean, 1.0/num_images);

    printf("[buildMatrixAndRunPca] Mean computed\n");

    if (0){
        // try making an image of the mean
        IplImage *im_mean = cvCreateImage(cvSize(w,h),IPL_DEPTH_8U,1);
        for (int j = 0; j < num_pixels; j++){
            (im_mean->imageData)[j] = (unsigned char)gsl_vector_get(m_gsl_mean, j);
        }
        Mat m = im_mean;
        imshow("mean image", m);
        cvWaitKey(0);
    }

    for (int i = 0; i < faceImages.size(); i++){
        gsl_vector_view col = gsl_matrix_column(m_gsl_mat, i);
        gsl_vector_sub(&col.vector, m_gsl_mean);
    }

    printf("[buildMatrixAndRunPca] Mean subtracted from each image\n");

    gsl_vector *S = gsl_vector_calloc(num_images);
    gsl_matrix *V = gsl_matrix_calloc(num_images, num_images);
    gsl_vector *work = gsl_vector_calloc(num_images);

    int res = gsl_linalg_SV_decomp(m_gsl_mat, V, S, work);

    printf("[buildMatrixAndRunPca] SVD computed, result: %d\n", res);

    for (int i = 0; i < 4; i++){
        printf("\tSVD %d: %f\n", i, gsl_vector_get(S, i));
    }

    if (0){
        printf("first 100 values of first eigenface: \n");
        for (int j = 0; j < 100; j++){
            printf("%f ", gsl_matrix_get(m_gsl_mat, j, 0)*gsl_vector_get(S, 0));
        }
        printf("\n");

        // try visualizing eigenfaces
        for (int i = 0; i < 4; i++){
            IplImage *im_mean = cvCreateImage(cvSize(w,h),IPL_DEPTH_8U,1);
            gsl_vector_view col = gsl_matrix_column(m_gsl_mat, i);

            for (int j = 0; j < num_pixels; j++){    
                (im_mean->imageData)[j] = (unsigned char) (gsl_vector_get(&col.vector, j)*gsl_vector_get(S,i) + gsl_vector_get(m_gsl_mean, j));
            }
            Mat m = im_mean;
            imshow("mean image", m);
            printf("showing image %d\n", i);
            cvWaitKey(0);
        }
    }

    for (int i = 0; i < 4; i++){
        IplImage *im_mean = cvCreateImage(cvSize(w,h),IPL_DEPTH_8U,1);
        gsl_vector_view col = gsl_matrix_column(m_gsl_mat, i);

        for (int j = 0; j < num_pixels; j++){    
            (im_mean->imageData)[j] = (unsigned char) (gsl_vector_get(&col.vector, j)*gsl_vector_get(S,i) + gsl_vector_get(m_gsl_mean, j));
        }

        Mat m = im_mean;
        char filename[100];
        sprintf(filename, "eigen%02d.jpg", i);
        printf("[buildMatrixAndRunPca] saving eigenface to file: %s\n", filename);
        imwrite(filename, m);
    }


    gsl_matrix *S_mat = gsl_matrix_calloc(num_images, num_images);
    gsl_matrix_set_zero(S_mat);

    for (int i = 0; i < 4; i++){
        gsl_matrix_set(S_mat, i, i, gsl_vector_get(S, i));
    }
    
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                       1.0, m_gsl_mat, S_mat,
                       0.0, m_gsl_mat);

    
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                       1.0, m_gsl_mat, V,
                       0.0, m_gsl_mat);
    

    // try visualizing faces in lower space
    for (int i = 0; i < num_images; i++){
        IplImage *im_mean = cvCreateImage(cvSize(w,h),IPL_DEPTH_8U,1);
        gsl_vector_view col = gsl_matrix_column(m_gsl_mat, i);

        for (int j = 0; j < num_pixels; j++){    
            (im_mean->imageData)[j] = (unsigned char) (gsl_vector_get(&col.vector, j) + gsl_vector_get(m_gsl_mean, j));
        }

        IplImage* pair = cvCreateImage(cvSize(w*2, h), IPL_DEPTH_8U, 1);
        cvZero(pair);
        cvSetImageROI(pair, cvRect(0, 0, w, h));
        cvCopy(faceImages[i], pair);
        cvSetImageROI(pair, cvRect(w, 0, w, h));
        cvCopy(im_mean, pair);
        cvSetImageROI(pair, cvRect(0, 0, w*2, h));

        Mat m = pair;
        char filename[100];
        sprintf(filename, "lowrankface%02d.jpg", i);
        printf("[buildMatrixAndRunPca] saving low rank face approximation to file: %s\n", filename);
        imwrite(filename, m);
    }

}

static FaceProcessApp *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new FaceProcessApp();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

