/* TransferExpression.cpp */

#include "TransferExpression.h"
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
            "   \t--face <file> [single face file]\n" \
            "   \t--faceList <file> [list of faces]\n" \
            "   \t--mask <file> [optional mask to apply to face]\n" \
            "\n" \
            "   \t--sourceDir <file> [directory containing eigenfaces and mean face of SOURCE expression]\n" \
            "   \t--referenceDir <file> [directory containing eigenfaces and mean face of REFERENCE (neutral) expression]\n" \
            "   \t--targetDir <file> [directory containing eigenfaces and mean face of TARGET expression]\n" \
            "\n" \
            "   \t--save <file extension> [root to save image files to]\n" \
            "\n" \
            "   \t--visualize [visualize progress]\n" \
            "\n");
}

void TransferExpression::processOptions(int argc, char **argv){
    while (1){
        static struct option long_options[] = {
            {"help",        0, 0, 'h'},
            {"visualize",   0, 0, 402},
            {"face",        1, 0, 403},
            {"faceList",    1, 0, 405},
            {"sourceDir",   1, 0, 406},
            {"referenceDir",1, 0, 407},
            {"targetDir",   1, 0, 408},
            {"save",        1, 0, 409},
            {"mask",        1, 0, 410},
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
                faceFile = optarg;
                break;
            
            case 405:
                faceListFile = optarg;
                break;

            case 406:
                sourceDir = optarg;
                break;

            case 407:
                referenceDir = optarg;
                break;

            case 408:
                targetDir = optarg;
                break;

            case 409:
                saveExtension = optarg;
                break;

            case 410:
                maskFile = optarg;
                break;

            default: 
                printf("Unrecognized option %d\n", c);
                break;
        }
    }
}

void TransferExpression::init(){
    printf("[init] Running program %s\n", argv[0]);
    visualize = false;
    faceFile = NULL;
    faceListFile = NULL;
    saveExtension = NULL;
    maskFile = NULL;

    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }
    
    processOptions(argc, argv);
    
    load();
    step1();

}

void TransferExpression::load(){
    printf("[load] Loading faces and source, reference, and target expressions\n");
    face = cvLoadImage(faceFile, CV_LOAD_IMAGE_COLOR);
    face.convertTo(face, CV_64FC3, 1.0/255, 0);

    w = face.rows;
    h = face.cols;
    d = 3;
    num_pixels = w*h*d;

    printf("[load] First image: (%d x %d) %d channels\n", w, h, d);

    if (visualize){
        imshow("original face", face);
        waitKey(100);
    }

    if (maskFile){
        mask = imread(maskFile);
        printf("Mask size: %d x %d\n", mask.rows, mask.cols);

        Mat m = Mat(face.rows, face.cols, CV_64FC3);
        m = Scalar(1.0, 1.0, 1.0);    
        face.copyTo(m, mask);
        face = m;
    }

    if (saveExtension){
        char filename[500];
        sprintf(filename, "%s_orig.jpg", saveExtension);
        saveAs(filename, face);
    }

    loadEigenfaces(sourceDir, &m_gsl_source_eigenfaces, &m_gsl_source_meanface);
    loadEigenfaces(referenceDir, &m_gsl_reference_eigenfaces, &m_gsl_reference_meanface);
    loadEigenfaces(targetDir, &m_gsl_target_eigenfaces, &m_gsl_target_meanface);

    loadFlow(sourceDir, flow_source_ref_vx, flow_source_ref_vy, false);
    loadFlow(targetDir, flow_ref_target_vx, flow_ref_target_vy, true);
 
}

void TransferExpression::loadFlow(char* eigenfacesDir, Mat &vx, Mat &vy, bool reverse){
    char flowFilename[1000];
    if (reverse){
        sprintf(flowFilename, "%s/mean-rank05-2reference_reverse_flow_xy.bin", eigenfacesDir);  
    }
    else {
        sprintf(flowFilename, "%s/mean-rank05-2reference_flow_xy.bin", eigenfacesDir);
    }
    printf("[loadFlow] from file [%s]\n", flowFilename);

    CVOpticalFlow::readFlow(flowFilename, vx, vy);

    if (visualize){
        imshow("flow", CVOpticalFlow::showNormalizedFlow(vx, vy));
        waitKey(100);
    }
}

void TransferExpression::loadEigenfaces(char* eigenfacesDir, gsl_matrix **eigenfaces, gsl_vector **meanface){
    printf("[loadEigenfaces] Loading eigenfaces from directory: %s\n", eigenfacesDir);

    string basepath = string(eigenfacesDir);

    string avgFaceFile = basepath + "/mean-rank05-2.jpg";
    Mat avgFace = imread(avgFaceFile.c_str(), CV_LOAD_IMAGE_COLOR);
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
    eigenfacesList.push_back(basepath + "/eigen04.bin");
    eigenfacesList.push_back(basepath + "/eigen05.bin");

    for (int i = 0; i < eigenfacesList.size(); i++){
        printf("eigen %d: %s\n", i, eigenfacesList[i].c_str());
    }


    num_eigenfaces = eigenfacesList.size();

    // mean face
    *meanface = gsl_vector_calloc(num_pixels);
    matToGslVec(avgFace, *meanface);

    // eigenfaces
    printf("sizes of things. num_pixels: %d, num eigenfaces: %d\n", num_pixels, num_eigenfaces);
    *eigenfaces = gsl_matrix_calloc(num_pixels, num_eigenfaces);
    for (int i = 0; i < eigenfacesList.size(); i++){
        gsl_vector_view col = gsl_matrix_column(*eigenfaces, i);
        FILE *f = fopen(eigenfacesList[i].c_str(), "rb");
        int success = gsl_vector_fread(f, &col.vector);
        
        if (visualize){
            Mat face;
            gslVecToMat(&(col.vector), face);
            imshow("loaded eigenface", face);
            waitKey(10);
        }
    }

}

void TransferExpression::step1(){
    printf("[step 1] projecting new face into source and target\n");

    Mat source = projectFace(face, m_gsl_source_eigenfaces, m_gsl_source_meanface);
    Mat neutral = projectFace(face, m_gsl_reference_eigenfaces, m_gsl_reference_meanface);
    Mat target = projectFace(face, m_gsl_target_eigenfaces, m_gsl_target_meanface);

    //test 
    /*
    Mat warpedInputReference = warpFaceToMatch(face, neutral, face);
    imshow("face", face);
    imshow("neutral", neutral);
    imshow("input warped to neutral", warpedInputReference);
    waitKey(0);
    */

    Mat warpedSource = warpFaceToMatch(source, face, source); // save this flow somehow...
    Mat warpedTarget = warpFaceToMatch(target, face, target);

    if (visualize){
        imshow("projected source warped to input", warpedSource);
        imshow("projected target warped to input", warpedTarget);
    }

    if (saveExtension){
        char filename_source[500];
        sprintf(filename_source, "%s_A_source_I.jpg", saveExtension);
        saveAs(filename_source, warpedSource);

        char filename_target[500];
        sprintf(filename_target, "%s_A_target_I.jpg", saveExtension);
        saveAs(filename_target, warpedTarget);
    }

    Mat tex_diff = warpedTarget - warpedSource;
    Mat faceWithTex = face + tex_diff;

    if (visualize){
        imshow("tex diff", tex_diff);
        imshow("textured face", faceWithTex);
    }

    if (saveExtension){
        char filename[500];
        sprintf(filename, "%s_texture.jpg", saveExtension);
        saveAs(filename, faceWithTex);
    }

    Mat vx, vy;
    computeFlow(source, face, vx, vy);


    Mat full_flow_vx = Mat(vx.size(), CV_64F);
    Mat full_flow_vy = Mat(vy.size(), CV_64F);
    CVOpticalFlow::compositeFlow(vx, vy, flow_source_ref_vx, flow_source_ref_vy, full_flow_vx, full_flow_vy);
    CVOpticalFlow::compositeFlow(full_flow_vx, full_flow_vy, flow_ref_target_vx, flow_ref_target_vy, full_flow_vx, full_flow_vy);

    Mat full_flow = CVOpticalFlow::showNormalizedFlow(full_flow_vx, full_flow_vy);

    if (visualize){
        imshow("full flow", full_flow);
    }
    
    if (saveExtension){
        char filename[500];
        sprintf(filename, "%s_fullflow.jpg", saveExtension);
        saveAs(filename, full_flow);
    }

    Mat final;
    CVOpticalFlow::warp(final, faceWithTex, full_flow_vx, full_flow_vy);

    if (visualize){
        imshow("final", final);
        waitKey(0);
    }

    if (saveExtension){
        char filename[500];
        sprintf(filename, "%s_final.jpg", saveExtension);
        saveAs(filename, final);
    }

    
}


Mat TransferExpression::projectFace(Mat &face, gsl_matrix* m_gsl_eigenfaces, gsl_vector* m_gsl_meanface){
    printf("[projectFace]\n");

    gsl_vector *m_gsl_newface = gsl_vector_calloc(num_pixels);
    matToGslVec(face, m_gsl_newface);

    gsl_vector *m_gsl_face_minus_mean = gsl_vector_calloc(num_pixels);
    gsl_vector_memcpy(m_gsl_face_minus_mean, m_gsl_newface);
    gsl_vector_sub(m_gsl_face_minus_mean, m_gsl_meanface);

    printf("num eigenfaces: %d\n", num_eigenfaces);

    gsl_vector *m_gsl_eigenvalues = gsl_vector_calloc(num_eigenfaces);
    for (int i = 0; i < num_eigenfaces; i++){
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

    Mat projectedFace;
    gslVecToMat(m_gsl_face_minus_mean, projectedFace);

   return projectedFace;
}

Mat TransferExpression::warpFaceToMatch(Mat &faceToWarp, Mat &source, Mat &target){
    printf("[warpFaceToMatch] computing flow from source to target and applying warp to faceToWarp\n");
    // magic variables
    double alpha = 0.02;
    double ratio = 0.85;
    int minWidth = 20;
    int nOuterFPIterations = 4;
    int nInnerFPIterations = 1;
    int nSORIterations = 40;

    Mat vx, vy, warp;

    // make gray because that seems to improve things
    if (0){
        Mat gray_source, gray_target;

        source.convertTo(gray_source, CV_32FC3);
        cvtColor(gray_source,gray_source,CV_RGB2GRAY);
        gray_source.convertTo(gray_source, CV_64F);

        target.convertTo(gray_target, CV_32FC3);
        cvtColor(gray_target,gray_target,CV_RGB2GRAY);
        gray_target.convertTo(gray_target, CV_64F);

        CVOpticalFlow::findFlow(vx, vy, warp, gray_source, gray_target, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
    }
    else {
        CVOpticalFlow::findFlow(vx, vy, warp, source, target, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);

    }

    Mat warped;
    CVOpticalFlow::warp(warped, faceToWarp, vx, vy);

    if (visualize){
        imshow("flow field", CVOpticalFlow::showNormalizedFlow(vx, vy));
    }


    return warped;
}

void TransferExpression::computeFlow(Mat &source, Mat &target, Mat &vx, Mat &vy){
    printf("[computeFlow] computing flow from source to target\n");
    // magic variables
    double alpha = 0.02;
    double ratio = 0.85;
    int minWidth = 20;
    int nOuterFPIterations = 4;
    int nInnerFPIterations = 1;
    int nSORIterations = 40;

    Mat warp;

    // make gray because that seems to improve things
    if (0){
        Mat gray_source, gray_target;

        source.convertTo(gray_source, CV_32FC3);
        cvtColor(gray_source,gray_source,CV_RGB2GRAY);
        gray_source.convertTo(gray_source, CV_64F);

        target.convertTo(gray_target, CV_32FC3);
        cvtColor(gray_target,gray_target,CV_RGB2GRAY);
        gray_target.convertTo(gray_target, CV_64F);

        CVOpticalFlow::findFlow(vx, vy, warp, gray_source, gray_target, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
    }
    else {
        CVOpticalFlow::findFlow(vx, vy, warp, source, target, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);

    }
}

void TransferExpression::gslVecToMat(gsl_vector *orig, Mat &im){
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

void TransferExpression::matToGslVec(Mat &im, gsl_vector *vec){
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

void TransferExpression::saveAs(const char* filename, Mat m){
    printf("[saveAs] saving image to file: %s\n", filename);
    Mat rightFormat;
    m.convertTo(rightFormat, CV_8UC3, 1.0*255, 0);
    imwrite(filename, rightFormat);
}

static TransferExpression *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new TransferExpression();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

