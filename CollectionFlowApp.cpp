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
            "   \t--mask [image to use as mask]\n" \
            "   \t--scale (how to rescale images)\n" \
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
            {"mask",        1, 0, 404},
            {"scale",       1, 0, 405},
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
                inputFile = strdup(optarg);
                break;

            case 401:
                outputDir = strdup(optarg);
                break;

            case 402:
                visualize = true;
                break;

            case 403:
                gray = true;
                break;

            case 404:
                maskFile = strdup(optarg);
                break;

            case 405:
                scale = (float)atof(optarg);
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
    outputDir = ".";
    scale = 1.0;

    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }
    
    processOptions(argc, argv);

    printf("done processing options\n");
    
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
    w = img.rows * scale;
    h = img.cols * scale;
    d = 3;

    printf("[findImageSizeFromFirstImage] First image %s: (%d x %d) %d channels\n", firstImage, w, h, d);
}

Mat CollectionFlowApp::computeImageHistogram(Mat img, Mat img2){
    img.convertTo(img, CV_8UC3, 255, 0);
    img2.convertTo(img2, CV_8UC3, 255, 0);

    // histogram matching stuff
    vector<Mat> img_split;
    split( img, img_split );

    // Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 255 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist, g_hist, r_hist;
    Mat b_hist2, g_hist2, r_hist2;

    // Compute the histograms:
    calcHist( &img_split[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &img_split[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &img_split[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );


    vector<Mat> res_split;
    split( img2, res_split );

    calcHist( &res_split[0], 1, 0, Mat(), b_hist2, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &res_split[1], 1, 0, Mat(), g_hist2, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &res_split[2], 1, 0, Mat(), r_hist2, 1, &histSize, &histRange, uniform, accumulate );

    // todo: cdf 
    Mat cdf_r_hist(b_hist.size(), b_hist.type());
    Mat cdf_g_hist(b_hist.size(), b_hist.type());
    Mat cdf_b_hist(b_hist.size(), b_hist.type());
    Mat cdf_r_hist2(b_hist.size(), b_hist.type());
    Mat cdf_g_hist2(b_hist.size(), b_hist.type());
    Mat cdf_b_hist2(b_hist.size(), b_hist.type());

    for( int i = 0; i < histSize; i++ ){
        if (i == 0){
            cdf_r_hist.at<float>(i) = cvRound(r_hist.at<float>(i));
            cdf_g_hist.at<float>(i) = cvRound(g_hist.at<float>(i));
            cdf_b_hist.at<float>(i) = cvRound(b_hist.at<float>(i));
            cdf_r_hist2.at<float>(i) = cvRound(r_hist2.at<float>(i));
            cdf_g_hist2.at<float>(i) = cvRound(g_hist2.at<float>(i));
            cdf_b_hist2.at<float>(i) = cvRound(b_hist2.at<float>(i));
        }
        else {
            cdf_r_hist.at<float>(i) = cvRound(r_hist.at<float>(i)) + cvRound(cdf_r_hist.at<float>(i-1));
            cdf_g_hist.at<float>(i) = cvRound(g_hist.at<float>(i)) + cvRound(cdf_g_hist.at<float>(i-1));
            cdf_b_hist.at<float>(i) = cvRound(b_hist.at<float>(i)) + cvRound(cdf_b_hist.at<float>(i-1));
            cdf_r_hist2.at<float>(i) = cvRound(r_hist2.at<float>(i)) + cvRound(cdf_r_hist2.at<float>(i-1));
            cdf_g_hist2.at<float>(i) = cvRound(g_hist2.at<float>(i)) + cvRound(cdf_g_hist2.at<float>(i-1));
            cdf_b_hist2.at<float>(i) = cvRound(b_hist2.at<float>(i)) + cvRound(cdf_b_hist2.at<float>(i-1));
        }
    }

    Mat M_r(b_hist.size(), CV_8U);
    Mat M_g(b_hist.size(), CV_8U);
    Mat M_b(b_hist.size(), CV_8U);

    for( int i = 0; i < histSize; i++ ){
        // i is a pixel intensity value
        
        int target_intensity = 255;
        float num_pixels_in_a;
        float num_pixels_in_b;

        // r channel
        num_pixels_in_a = cdf_r_hist2.at<float>(i);

        for (int j = 0; j < histSize; j++){
            num_pixels_in_b = cdf_r_hist.at<float>(j);
            if (num_pixels_in_b > num_pixels_in_a){
                target_intensity = j;
                break;
            }
        }
        M_r.at<char>(i) = target_intensity;

        // g channel
        num_pixels_in_a = cdf_g_hist2.at<float>(i);

        for (int j = 0; j < histSize; j++){
            num_pixels_in_b = cdf_g_hist.at<float>(j);
            if (num_pixels_in_b > num_pixels_in_a){
                target_intensity = j;
                break;
            }
        }
        M_g.at<char>(i) = target_intensity;

        // b channel
        num_pixels_in_a = cdf_b_hist2.at<float>(i);

        for (int j = 0; j < histSize; j++){
            num_pixels_in_b = cdf_b_hist.at<float>(j);
            if (num_pixels_in_b > num_pixels_in_a){
                target_intensity = j;
                break;
            }
        }
        M_b.at<char>(i) = target_intensity;
    }




    // correct one direction
    Mat new_r(img2.size(), CV_8U);
    Mat new_g(img2.size(), CV_8U);
    Mat new_b(img2.size(), CV_8U);

    vector<Mat> new_face;

    LUT( res_split[0], M_b, new_b );
    LUT( res_split[1], M_g, new_g );
    LUT( res_split[2], M_r, new_r );

    //imshow("res_split", res_split[0]);
    //imshow("new b", new_b);
    
    new_face.push_back(new_b);
    new_face.push_back(new_g);
    new_face.push_back(new_r);

    Mat color_corrected;
    merge(new_face, color_corrected);

    /*
    imshow("face to match color", img);
    imshow("uncorrected", img2);
    imshow("color corrected", color_corrected);
    waitKey(0);
    */
    
    color_corrected.convertTo(color_corrected, CV_64FC3, 1.0/255, 0);

    return color_corrected;
}

void CollectionFlowApp::openImages(){
    printf("[openImages] %d images in list\n", (int)faceList.size());

    double gamma = 1.8;
    double inverse_gamma = 1.0 / gamma;

    Mat lut_matrix(1, 256, CV_8UC1 );
    uchar * ptr = lut_matrix.ptr();
    for( int i = 0; i < 256; i++ )
    ptr[i] = (int)( pow( (double) i / 255.0, inverse_gamma ) * 255.0 );

    printf("Mask file: %s\n", maskFile);
    Mat mask = imread(maskFile);
    printf("Mask size: %d x %d\n", mask.rows, mask.cols);


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

            resize(result, result, Size(), scale, scale);

            
            if (i > 0){
                result = computeImageHistogram(faceImages[0], result);
            }
            

            faceImages.push_back(result);
            

            if (visualize){
                imshow("loaded masked image", result);
                waitKey(3);
            }
        }
        else {
            printf("Error loading image %s\n", faceList[i].c_str());
        }
    }

    destroyWindow("loaded masked image");

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
        waitKey(100);
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

    // the eigen images
    gsl_matrix *m_gsl_mat_work = gsl_matrix_calloc(num_pixels, num_images);

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

        Mat vis_img = Mat::zeros(w, h*(k+2), CV_64FC3);
        printf(" vis_img size %d x %d  and  h x w = %d x %d\n", vis_img.rows, vis_img.cols, h, w);

        Mat m;
        gslVecToMat(m_gsl_mean, m);
        if (visualize){    
            Mat dst_roi = vis_img(Rect(0, 0, h, w));
            m.copyTo(dst_roi);
            imshow("mean image and eigenfaces", vis_img);
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
                //imshow(eig_title, eigenface);
                Mat dst_roi = vis_img(Rect(h*(i+1), 0, h, w));
                eigenface.copyTo(dst_roi);
                imshow("mean image and eigenfaces", vis_img);
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
                           0.0, m_gsl_mat_work);

        
        gsl_blas_dgemm (CblasNoTrans, CblasTrans,
                           1.0, m_gsl_mat_work, V,
                           0.0, m_gsl_mat_k);

        if (use_mean){
            printf("[buildMatrixAndRunPca] Adding mean back\n");
            for (int i = 0; i < faceImages.size(); i++){
                gsl_vector_view col = gsl_matrix_column(m_gsl_mat_k, i);
                gsl_vector_view col_o = gsl_matrix_column(m_gsl_mat, i);
                /*printf("\nwhat does the inside of this low rank matrix look like?\n");
                for (int j = 0; j < w*h*d; j++){
                    printf("\t[%f|%f] ", gsl_vector_get(&col.vector, j),  gsl_vector_get(&col_o.vector, j));
                }
                */
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
    
            Mat vis_img = Mat::zeros(w, h*4, CV_64FC3);    

            if (visualize){
                Mat dst_roi = vis_img(Rect(0, 0, h, w));
                m_lowrank.copyTo(dst_roi);
                dst_roi = vis_img(Rect(h, 0, h, w));
                m_highrank.copyTo(dst_roi);
                imshow("high, low, warped, flow", vis_img);
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
            Mat vx2, vy2, warp2;
            
            CVOpticalFlow::findFlow(vx, vy, warp, m_lowrank, m_highrank, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
            //CVOpticalFlow::findFlow(vx2, vy2, warp2, m_highrank, m_lowrank, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);


            Mat warped;
            if (gray){
                CVOpticalFlow::warpGray(warped, m_highrank, vx, vy);
            }
            else {
                CVOpticalFlow::warp(warped, m_highrank, vx, vy);
            }
            
            m_highrank = warped;

            if (visualize){
                Mat dst_roi = vis_img(Rect(h*2, 0, h, w));
                warped.copyTo(dst_roi);
                Mat flow_img = CVOpticalFlow::showNormalizedFlow(vx, vy);
                dst_roi = vis_img(Rect(h*3, 0, h, w));
                flow_img.copyTo(dst_roi);
                imshow("high, low, warped, flow", vis_img);
                waitKey(10);
            }

            matToGslVec(m_highrank, &col_low.vector);

            CVOpticalFlow::writeFlow(flowFile, vx, vy);
            saveAs(filename3, m_highrank);
            saveAs(flowImage, CVOpticalFlow::showFlow(vx, vy));
            //saveAs(flowImage2, CVOpticalFlow::showFlow(vx2, vy2));
            
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

