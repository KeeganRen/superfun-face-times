/* ProjectFace.cpp */

#include "ProjectFace.h"
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
    printf("\nProjects new faces onto existing eigenface basis and warps those new faces\n\n");
    printf("Usage:  \n" \
            "   \t--faceList <file> [list of faces]\n" \
            "   \t--eigenDir <file> [directory containing eigenfaces and mean face]\n" \
            "   \t--extension <extension to add to file>\n" \
            "   \t--visualize [visualize progress]\n" \
            "\n");
}

void ProjectFace::processOptions(int argc, char **argv){
    while (1){
        static struct option long_options[] = {
            {"help",        0, 0, 'h'},
            {"visualize",   0, 0, 402},
            {"eigenDir",  1, 0, 403},
            {"output",      1, 0, 405},
            {"faceList",    1, 0, 406},
            {"extension",    1, 0, 407},
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
                visualize = true;
                break;

            case 403:
                eigenfacesDir = optarg;
                break;
            
            case 405:
                sprintf(outputDir, "%s", optarg);
                break;

            case 406:
                facesFile = optarg;
                break;

            case 407:
                extension = optarg;
                break;

            default: 
                printf("Unrecognized option %d\n", c);
                break;
        }
    }
}

void ProjectFace::init(){
    printf("[init] Running program %s\n", argv[0]);
    visualize = false;
    facesFile = NULL;

    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }
    
    processOptions(argc, argv);
    
    loadFaces();
    findImageSizeFromFirstImage();
    loadEigenfaces();
    
    for (int i = 0; i < faceImages.size(); i++){
        string filename = faceList[i].substr(0, faceList[i].size() - 4);
        printf("face file basename %s\n", filename.c_str());
        double flow_score = projectAndWarpFace(faceImages[i]);

        string warpedFilename = filename + extension + "_projected_flow.jpg";
        string lowRankFilename = filename + extension + "_projected.jpg";
        string flowScoreFilename = filename + extension + "_flow_score.txt";
        string flowFieldFilename = filename + extension + "_flow.jpg";
        string flowBinFilename = filename + extension + "_flow_xy.bin";
        string reverseFlowBinFilename = filename + extension + "_reverse_flow_xy.bin";
        saveAs(warpedFilename.c_str(), warped);
        saveAs(lowRankFilename.c_str(), lowRank);
        saveAs(flowFieldFilename.c_str(), flowField);
        CVOpticalFlow::writeFlow((char*)(flowBinFilename.c_str()), flow_vx, flow_vy);
        CVOpticalFlow::writeFlow((char*)(reverseFlowBinFilename.c_str()), reverse_flow_vx, reverse_flow_vy);

        FILE * pFile;
        pFile = fopen (flowScoreFilename.c_str(),"w");
        if (pFile!=NULL){
            fprintf(pFile, "%f", flow_score);
            fclose (pFile);
        }
    }

    //printf("face file name: %s\n", faceFileName(newFaceFile));

    /*
    if (facesFile == NULL){
        projectNewFace();
        warpNewFace();
        makeNewAvg();
    }
    else {
        makeNewEigenfaces();
    }
    */
    /*
    
    openImages();
    if (gray){
        convertImages();
    }

    buildMatrixAndRunPca();
    */

}

void ProjectFace::loadFaces(){
    printf("[loadFaces] Loading faces from list: %s\n", facesFile);
    FILE *file = fopen ( facesFile, "r" );
    if ( file != NULL ) {
        char line [ 256 ]; 
        while( fscanf(file, "%s\n", line) > 0 ) {
            printf("\tface file found: %s\n", line);
            faceList.push_back(string(line));
        }
        fclose (file);
    }
    else {
        perror (facesFile);
    }

    // open images
    for (int i = 0; i < faceList.size(); i++){
        Mat img = cvLoadImage(faceList[i].c_str(), CV_LOAD_IMAGE_COLOR);
        /*
        if (gray){
            cvtColor(img, img, CV_BGR2GRAY);
        }
        */
        img.convertTo(img, CV_64FC3, 1.0/255, 0);
        if (img.data != NULL){
            faceImages.push_back(img);
        }
        else {
            printf("Error loading image %s\n", faceList[i].c_str());
        }
    }
    if (visualize){
        imshow("first face", faceImages[0]);
        waitKey(100);
    }
}


void ProjectFace::findImageSizeFromFirstImage(){
    printf("[findImageSizeFromFirstImage]\n");

    w = faceImages[0].rows;
    h = faceImages[0].cols;
    d = 3;

    printf("[findImageSizeFromFirstImage] First image: (%d x %d) %d channels\n", w, h, d);
}


void ProjectFace::loadEigenfaces(){
    printf("[loadEigenfaces] Loading eigenfaces from directory: %s\n", eigenfacesDir);

    string basepath = string(eigenfacesDir);

    string avgFaceFile = basepath + "/mean-rank05-2.jpg";
    avgFace = imread(avgFaceFile.c_str(), CV_LOAD_IMAGE_COLOR);
    avgFace.convertTo(avgFace, CV_64FC3, 1.0/255, 0);

    if (visualize){
        imshow("average face", avgFace);
        waitKey(100);
    }

    vector<string> eigenfacesList;
    eigenfacesList.push_back(basepath + "/eigen00.bin");
    eigenfacesList.push_back(basepath + "/eigen01.bin");
    eigenfacesList.push_back(basepath + "/eigen02.bin");
    eigenfacesList.push_back(basepath + "/eigen03.bin");

    for (int i = 0; i < eigenfacesList.size(); i++){
        printf("eigen %d: %s\n", i, eigenfacesList[i].c_str());
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
        avgFace.convertTo(avgFace, CV_32FC3);
        cvtColor(avgFace,avgFace,CV_RGB2GRAY);
        avgFace.convertTo(avgFace, CV_64F);
    }

    num_pixels = w*h*d;
    num_eigenfaces = eigenfacesList.size();

    // mean face
    m_gsl_meanface = gsl_vector_calloc(num_pixels);
    matToGslVec(avgFace, m_gsl_meanface);
    printf("done supposedly loading mean face into gsl matrix");

    // holder for new face (to project and warp)
    m_gsl_newface = gsl_vector_calloc(num_pixels);


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
            waitKey(100);
        }
    }
}

double ProjectFace::projectAndWarpFace(Mat face){
    printf("[projectAndWarpFace]\n");
    newFace = face;
    matToGslVec(newFace, m_gsl_newface);
    projectNewFace();
    double flow_score = warpNewFace();
    return flow_score;
}

void ProjectFace::projectNewFace(){
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
        imshow("original face", newFace);
    }

    m_gsl_projectednewface = m_gsl_face_minus_mean;

}

double ProjectFace::warpNewFace(){
    printf("[warpNewFace] computing flow on new face\n");
    // magic variables
    double alpha = 0.02;
    double ratio = 0.85;
    int minWidth = 20;
    int nOuterFPIterations = 4;
    int nInnerFPIterations = 1;
    int nSORIterations = 40;
    
    Mat vx, vy, warp;

    Mat m_lowrank, m_highrank;
    gslVecToMat(m_gsl_projectednewface, m_lowrank);
    m_highrank = newFace;

    lowRank = m_lowrank;

    /*
    char *newFaceFile  = "newFaceFile.jpg";

    // save low rank picture
    const char *faceStr = faceFileName(newFaceFile);
    char filename[100];
    sprintf(filename, "%s/%s-low.jpg", outputDir, faceStr);
    saveAs(filename, m_lowrank);
    */

    printf("type of image: %d %d %d\n", m_lowrank.type(), m_highrank.type(), avgFace.type());

    Mat gray_low, gray_high;

    m_lowrank.convertTo(gray_low, CV_32FC3);
    cvtColor(gray_low,gray_low,CV_RGB2GRAY);
    gray_low.convertTo(gray_low, CV_64F);

    m_highrank.convertTo(gray_high, CV_32FC3);
    cvtColor(gray_high,gray_high,CV_RGB2GRAY);
    gray_high.convertTo(gray_high, CV_64F);

    CVOpticalFlow::findFlow(vx, vy, warp, gray_low, gray_high, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);

    if (d == 3){
        CVOpticalFlow::warp(warped, m_highrank, vx, vy);
    }
    else {
        CVOpticalFlow::warpGray(warped, m_highrank, vx, vy);
    }

    if (visualize){
        imshow("the warped picture", warped);
        imshow("flow", CVOpticalFlow::showNormalizedFlow(vx, vy));
    }

    flowField = CVOpticalFlow::showNormalizedFlow(vx, vy);
    flow_vx = vx;
    flow_vy = vy;

    CVOpticalFlow::findFlow(reverse_flow_vx, reverse_flow_vy, warp, gray_high, gray_low, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);


    Mat zero_mat =  Mat::zeros(vx.size(), CV_64F);

    double flow_score = norm(vx, zero_mat, NORM_L2) + norm(vy, zero_mat, NORM_L2);
    printf("FLOW SCORE: %f\n", flow_score);

    /*
    // save warped picture
    const char *faceStr2 = faceFileName(newFaceFile);
    char filename2[100];
    sprintf(filename2, "%s/%s-warped.jpg", outputDir, faceStr2);
    saveAs(filename2, warped);
    */

    return flow_score;
}



const char* ProjectFace::faceFileName(char* f){
    string filename = string(f);
    int idx1 = filename.rfind("/");
    int idx2 = filename.find(".jpg");
    //int idx2 = filename.rfind("_"); // pretty specific to this app.. looking for the faceid #
    string sub = filename.substr(idx1+1, idx2-idx1-1);
    return sub.c_str();
}



void ProjectFace::gslVecToMat(gsl_vector *orig, Mat &im){
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

void ProjectFace::matToGslVec(Mat &im, gsl_vector *vec){
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

void ProjectFace::saveAs(const char* filename, Mat m){
    printf("[saveAs] saving image to file: %s\n", filename);
    Mat rightFormat;
    m.convertTo(rightFormat, CV_8UC3, 1.0*255, 0);
    imwrite(filename, rightFormat);
}



static ProjectFace *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new ProjectFace();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

