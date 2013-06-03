/* IncrementalCFlowApp.cpp */

#include "IncrementalCFlowApp.h"
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
            "   \t--newFace <file>\n" \
            "   \t--oldAvgFace <file> [existing average face]\n" \
            "   \t--numFaces <number>\n" \
            "   \t--eigenfaces <file> [list of paths to eigenfaces to use]\n" \
            "   \t--visualize [visualize progress]\n" \
            "\n");
}

void IncrementalCFlowApp::processOptions(int argc, char **argv){
    while (1){
        static struct option long_options[] = {
            {"help",        0, 0, 'h'},
            {"newFace",     1, 0, 400},
            {"avgFace",     1, 0, 401},
            {"visualize",   0, 0, 402},
            {"eigenfaces",  1, 0, 403},
            {"numFaces",    1, 0, 404},
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
                sprintf(newFaceFile, "%s", optarg);
                break;

            case 401:
                sprintf(avgFaceFile, "%s", optarg);
                break;

            case 402:
                visualize = true;
                break;

            case 403:
                sprintf(eigenfacesFile, "%s", optarg);
                break;
            
            case 404:
                numFaces = atoi(optarg);
                printf("num faces: %d\n", numFaces);
                break;

            default: 
                printf("Unrecognized option %d\n", c);
                break;
        }
    }
}

void IncrementalCFlowApp::init(){
    printf("[init] Running program %s\n", argv[0]);
    visualize = false;
    
    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }
    
    processOptions(argc, argv);
    
    loadFaces();
    findImageSizeFromFirstImage();
    loadEigenfaces();
    
    //printf("face file name: %s\n", faceFileName(newFaceFile));

    projectNewFace();
    warpNewFace();
    makeNewAvg();

    /*
    
    openImages();
    if (gray){
        convertImages();
    }

    buildMatrixAndRunPca();
    */

}

void IncrementalCFlowApp::loadFaces(){
    printf("[loadFaces] Loading new face %s and avg face %s\n", newFaceFile, avgFaceFile);
    newFace = cvLoadImage(newFaceFile, CV_LOAD_IMAGE_COLOR);
    newFace.convertTo(newFace, CV_64FC3, 1.0/255, 0);

    avgFace = cvLoadImage(avgFaceFile, CV_LOAD_IMAGE_COLOR);
    avgFace.convertTo(avgFace, CV_64FC3, 1.0/255, 0);

    if (visualize){
        imshow("new face", newFace);
        imshow("old average face", avgFace);
    }
}

void IncrementalCFlowApp::loadEigenfaces(){
    printf("[loadEigenfaces] Loading eigenfaces from list: %s\n", eigenfacesFile);
    vector<string> eigenfacesList;
    FILE *file = fopen ( eigenfacesFile, "r" );
    if ( file != NULL ) {
        char line [ 256 ]; 
        while( fscanf(file, "%s\n", line) > 0 ) {
            eigenfacesList.push_back(string(line));
        }
        fclose (file);
    }
    else {
        perror (eigenfacesFile);
    }

    num_pixels = w*h*d;
    num_eigenfaces = eigenfacesList.size();

    // mean face
    m_gsl_meanface = gsl_vector_calloc(num_pixels);
    matToGslVec(avgFace, m_gsl_meanface);
    printf("done supposedly loading mean face into gsl matrix");

    // new face
    m_gsl_newface = gsl_vector_calloc(num_pixels);
    matToGslVec(newFace, m_gsl_newface);
    printf("done supposedly loading mean face into gsl matrix");

    // eigenfaces
    printf("sizes of things. num_pixels: %d, num eigenfaces: %d\n", num_pixels, num_eigenfaces);
    m_gsl_eigenfaces = gsl_matrix_calloc(num_pixels, num_eigenfaces);
    for (int i = 0; i < eigenfacesList.size(); i++){
        gsl_vector_view col = gsl_matrix_column(m_gsl_eigenfaces, i);
        FILE *f = fopen(eigenfacesList[i].c_str(), "rb");
        int success = gsl_vector_fread(f, &col.vector);
        
        if (visualize){
            Mat face;
            gslVecToMat(&(col.vector), face);
            imshow("loaded eigenface", face);
        }
    }
}

void IncrementalCFlowApp::projectNewFace(){
    printf("[projectNewFace] projecting new face onto lower dimensional space!\n");
    gsl_vector *m_gsl_face_minus_mean = gsl_vector_calloc(num_pixels);
    gsl_vector_memcpy(m_gsl_face_minus_mean, m_gsl_newface);
    gsl_vector_sub(m_gsl_face_minus_mean, m_gsl_meanface);

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
        gslVecToMat(m_gsl_face_minus_mean, face);
        imshow("projected face", face);
    }

    m_gsl_projectednewface = m_gsl_face_minus_mean;

}

void IncrementalCFlowApp::warpNewFace(){
    printf("[warpNewFace] computing flow on new face\n");
    // magic variables
    double alpha = 0.03;
    double ratio = 0.85;
    int minWidth = 20;
    int nOuterFPIterations = 4;
    int nInnerFPIterations = 1;
    int nSORIterations = 40;
    
    Mat vx, vy, warp;

    Mat m_lowrank, m_highrank;
    gslVecToMat(m_gsl_projectednewface, m_lowrank);
    m_highrank = newFace;
    
    CVOpticalFlow::findFlow(vx, vy, warp, m_lowrank, m_highrank, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);

    CVOpticalFlow::warp(warped, m_highrank, vx, vy);

    if (visualize){
        imshow("the warped picture", warped);
        imshow("flow", CVOpticalFlow::showFlow(vx, vy));
    }
}

void IncrementalCFlowApp::makeNewAvg(){
    gsl_vector *m_gsl_warped = gsl_vector_calloc(num_pixels);
    matToGslVec(warped, m_gsl_warped);

    gsl_vector *m_gsl_meancopy = gsl_vector_calloc(num_pixels);
    gsl_vector_memcpy(m_gsl_meancopy, m_gsl_meanface);

    gsl_vector_scale(m_gsl_meancopy, numFaces);
    gsl_vector_add(m_gsl_meancopy, m_gsl_warped);
    gsl_vector_scale(m_gsl_meancopy, 1.0/(numFaces + 1));

    Mat newAvg;
    gslVecToMat(m_gsl_meancopy, newAvg);
    if (visualize){
        imshow("NEW average", newAvg);
    }
}

const char* IncrementalCFlowApp::faceFileName(char* f){
    string filename = string(f);
    int idx1 = filename.rfind("/");
    //int idx2 = filename.find(".jpg");
    int idx2 = filename.rfind("_"); // pretty specific to this app.. looking for the faceid #
    string sub = filename.substr(idx1+1, idx2-idx1-1);
    return sub.c_str();
}

void IncrementalCFlowApp::findImageSizeFromFirstImage(){
    printf("[findImageSizeFromFirstImage] %s\n", newFaceFile);

    w = newFace.rows;
    h = newFace.cols;
    d = 3;

    printf("[findImageSizeFromFirstImage] First image: (%d x %d) %d channels\n", w, h, d);
}

void IncrementalCFlowApp::gslVecToMat(gsl_vector *orig, Mat &im){
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

void IncrementalCFlowApp::matToGslVec(Mat &im, gsl_vector *vec){
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

void IncrementalCFlowApp::saveAs(char* filename, Mat m){
    printf("[saveAs] saving image to file: %s\n", filename);
    Mat rightFormat;
    m.convertTo(rightFormat, CV_8UC3, 1.0*255, 0);
    imwrite(filename, rightFormat);
}

void IncrementalCFlowApp::saveBinaryEigenface(char* filename, gsl_vector *face){
    printf("[saveBinaryEigenface] saving binary file of eigenface to file: %s\n", filename);
    FILE *file = fopen(filename, "wb");
    gsl_vector_fwrite(file, face);
    fclose(file);
}

void IncrementalCFlowApp::buildMatrixAndRunPca(){
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
        sprintf(filename, "%s/mean-rank%02d.jpg", outputDir, k);
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
            const char *faceStr = faceFileName(newFaceFile);
            printf("face filename: %s\n", faceStr);
            sprintf(filename1, "%s/%s-orig.jpg", outputDir, faceStr);
            sprintf(filename2, "%s/%s-low.jpg", outputDir, faceStr);
            sprintf(filename3, "%s/%s-warped.jpg", outputDir, faceStr);

            //saveAs(filename1, m_highrank);
            saveAs(filename2, m_lowrank);

            
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
            if (visualize){
                imshow("the warped picture", warp);
                imshow("flow", CVOpticalFlow::showFlow(vx, vy));
            }


            Mat warped;
            CVOpticalFlow::warp(warped, m_highrank, vx, vy);
            
            m_highrank = warped;

            if (visualize){
                imshow("warped back", m_highrank);
                waitKey(0);
            }

            matToGslVec(m_highrank, &col_high.vector);


            saveAs(filename3, m_highrank);
            
        }
    }
}


static IncrementalCFlowApp *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new IncrementalCFlowApp();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

