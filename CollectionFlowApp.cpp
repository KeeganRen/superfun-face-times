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
    printf("\nABOUT THIS PROGRAM AND WHAT IT DOES: runs collection flow on a collection of images!\n\n");
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
        d = 1;
    }

    buildMatrixAndRunPca();
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

const char* CollectionFlowApp::faceFileName(int i){
    string filename = faceList[i];
    int idx1 = filename.rfind("/");
    //int idx2 = filename.find(".jpg");
    int idx2 = filename.rfind("_"); // pretty specific to this app.. looking for the faceid #
    string sub = filename.substr(idx1+1, idx2-idx1-1);
    return sub.c_str();
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

    double gamma = 1.8;
    double inverse_gamma = 1.0 / gamma;

    Mat lut_matrix(1, 256, CV_8UC1 );
    uchar * ptr = lut_matrix.ptr();
    for( int i = 0; i < 256; i++ )
    ptr[i] = (int)( pow( (double) i / 255.0, inverse_gamma ) * 255.0 );

    Mat mask = imread("jackiechan/mask.png");

    for (int i = 0; i < faceList.size(); i++){
        printf("opening image %d: %s\n", i, faceList[i].c_str());
        Mat img = imread(faceList[i].c_str());
        if (img.data != NULL){
            if (gray){
                cvtColor(img, img, CV_BGR2GRAY);
            }

            Mat result;

            if (0){
                // gamma correction
                LUT( img, lut_matrix, result );
                result.convertTo(result, CV_64FC3, 1.0/255, 0);
            }
            else {
                img.convertTo(result, CV_64FC3, 1.0/255, 0);
            }

            Mat m;
            result.copyTo(m, mask);


            result = m;
            
            faceImages.push_back(result);
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
        if (im.depth() == CV_64F){
            im.convertTo(im, CV_32FC3, 1.0, 0);
        }
        printf("image depth: %d, %d %d\n", im.channels(), im.depth(), CV_64F);
        Mat gray;
        cvtColor(im, gray, CV_BGR2GRAY);
        faceImages[i] = gray;
    }

    d = 1;
    printf("[convertImages] Now there are %d channels\n", d);

    if (visualize){
        Mat m = faceImages[0];
        imshow("Converted image", m);
        waitKey(0);
    }
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

void CollectionFlowApp::gslVecToMatWithBorder(gsl_vector *orig, Mat &im){
    gsl_vector* vec = gsl_vector_calloc(orig->size);
    gsl_vector_memcpy(vec, orig);

    if (d == 1){
        Mat m(w, h, CV_64F, vec->data);
        Mat m_border(w + borderSize*2, h + borderSize*2, CV_64F, -1);
        //m.copyTo(m_border(Rect(borderSize, borderSize, h, w)));
        Mat dst_roi = m_border(Rect(borderSize, borderSize, h, w));
        m.copyTo(dst_roi);
        im = m_border;
    }
    if (d == 3){
        Mat m(w, h, CV_64FC3, vec->data);
        Mat m_border(w + borderSize*2, h + borderSize*2, CV_64FC3);
        
        for (int i = 0; i < m_border.rows; i++){
            for (int j = 0; j < m_border.cols; j++){
                m_border.at<Vec3d>(i, j) = Vec3d(0, 0, 0);
            }
        }

        Mat dst_roi = m_border(Rect(borderSize, borderSize, h, w));
        m.copyTo(dst_roi);
        //m.copyTo(m_border(Rect(borderSize, borderSize, h, w)));
        im = m_border;
    }
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

void CollectionFlowApp::matToGslVecWithBorder(Mat &im, gsl_vector *vec){
    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            if (d == 1){
                int idx = i*h + j;
                double val = im.at<double>(i + borderSize, j + borderSize);
                gsl_vector_set(vec, idx, val);
            }
            else if (d == 3){
                int idx = i*h + j;
                Vec3d val = im.at<Vec3d>(i + borderSize, j + borderSize);
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

void CollectionFlowApp::saveAsCropBorder(char* filename, Mat m){
    printf("[saveAs] saving image to file (but cropping first): %s\n", filename);
    Mat rightFormat;
    m.convertTo(rightFormat, CV_8UC3, 1.0*255, 0);
    Rect cropROI(borderSize, borderSize, h, w);
    rightFormat = rightFormat(cropROI);
    imwrite(filename, rightFormat);
}

void CollectionFlowApp::saveBinaryEigenface(char* filename, gsl_vector *face){
    printf("[saveBinaryEigenface] saving binary file of eigenface to file: %s\n", filename);
    FILE *file = fopen(filename, "wb");
    gsl_vector_fwrite(file, face);
    fclose(file);
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
        // populate matrix of original images
        gsl_vector_view col = gsl_matrix_column(m_gsl_mat, i);
        matToGslVec(faceImages[i], &col.vector);

        // populate this other matrix, too
        gsl_vector_view col_k = gsl_matrix_column(m_gsl_mat_k, i);
        matToGslVec(faceImages[i], &col_k.vector);
    }

    printf("[buildMatrixAndRunPca] Matrix populated\n");

    int ranks[] = {4, 5, 5};
    int len = 3;

    for (int r = 0; r < len; r++){
        int k = ranks[r];
        printf("\t[COLLECTION FLOW] RANK %d (r: %d)\n", k, r);

        gsl_vector *m_gsl_mean = gsl_vector_calloc(num_pixels);

        for (int i = 0; i < faceImages.size(); i++){
            gsl_vector_view col = gsl_matrix_column(m_gsl_mat_k, i);
            gsl_vector_add(m_gsl_mean, &col.vector);
        }

        gsl_vector_scale(m_gsl_mean, 1.0/num_images);

        printf("[buildMatrixAndRunPca] Mean computed\n");

        Mat m;
        gslVecToMat(m_gsl_mean, m);
        if (visualize){    
            imshow("mean image", m);
            waitKey(0);
        }
        char filename[100];
        sprintf(filename, "%s/mean-rank%02d-%d.jpg", outputDir, k, r);
        saveAs(filename, m);

        char generalMeanFilename[100];
        sprintf(generalMeanFilename, "%s/clustermean.jpg", outputDir);
        saveAs(generalMeanFilename, m);

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
            if (visualize)
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
            sprintf(filename, "%s/eigen%02d.jpg", outputDir, i);
            saveAs(filename, eigenface);

            char filenameBin[100];
            sprintf(filenameBin, "%s/eigen%02d.bin", outputDir, i);
            saveBinaryEigenface(filenameBin, vec_work);
            
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

        if (r == len-1){
            printf("dont do flow for last step...\n");
            break;
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

            char filename1[100], filename2[100], filename3[100];
            char flowFile[100], flowImage[100], flowImage2[100];
            const char *faceStr = faceFileName(i);
            printf("face filename: %s\n", faceStr);
            sprintf(filename1, "%s/%s-orig-%d.jpg", outputDir, faceStr, k);
            sprintf(filename2, "%s/%s-low-%d.jpg", outputDir, faceStr, k);
            sprintf(filename3, "%s/%s-warped-%d.jpg", outputDir, faceStr, k);
            sprintf(flowFile,  "%s/%s-flow-%d.bin", outputDir, faceStr, k);
            sprintf(flowImage,  "%s/%s-flow-%d.jpg", outputDir, faceStr, k);
            sprintf(flowImage2,  "%s/%s-flow2-%d.jpg", outputDir, faceStr, k);

            saveAs(filename1, m_highrank);
            saveAs(filename2, m_lowrank);

            
            printf("[computeFlow] image %d/%d\n", i, num_images);
            // magic variables
            double alpha = 0.02; // 0.015 smaller parameter should make it look even more sharp
            double ratio = 0.85; 
            int minWidth = 20;
            int nOuterFPIterations = 7;
            int nInnerFPIterations = 1;
            int nSORIterations = 40; // sometimes use 20
            
            Mat vx, vy, warp;
            Mat vx2, vy2, warp2
            
            CVOpticalFlow::findFlow(vx, vy, warp, m_lowrank, m_highrank, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
            CVOpticalFlow::findFlow(vx2, vy2, warp2, m_highrank, m_lowrank, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
            if (visualize){
                imshow("the warped picture", warp);
                imshow("flow", CVOpticalFlow::showFlow(vx, vy));
                imshow("flow2", CVOpticalFlow::showFlow(vx2, vy2));
            }

            Mat warped;
            if (gray){
                CVOpticalFlow::warpGray(warped, m_highrank, vx, vy);
            }
            else {
                CVOpticalFlow::warp(warped, m_highrank, vx, vy);
            }
            
            m_highrank = warped;

            if (visualize){
                imshow("warped back", m_highrank);
                waitKey(0);
            }

            matToGslVec(m_highrank, &col_low.vector);


            CVOpticalFlow::writeFlow(flowFile, vx, vy);
            saveAs(filename3, m_highrank);
            saveAs(flowImage, CVOpticalFlow::showFlow(vx, vy));
            saveAs(flowImage2, CVOpticalFlow::showFlow(vx2, vy2));
            
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

