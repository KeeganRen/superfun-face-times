/* AgingApp.cpp */

#include "AgingApp.h"

#include "FaceLib.h"

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


using namespace std;
using namespace cv;

void error(char *msg) {
    perror(msg);
    exit(1);
}

void PrintUsage() 
{
    printf("Usage:  \n" \
            "   \tproject one face into a bunch of other clusters (e.g different ages)\n" \
            "   \t<image path> <face_id> <reference_cluster_id> <cluster_1> <cluster_2> ... <cluster_n>\n" \
            "\n");
}

void AgingApp::init(){
    printf("[init] Running program %s\n", argv[0]);
    visualize = false;

    if (argc < 4 ){
        PrintUsage();
        exit(0);
    }
    
    data_path = argv[1];
    face_id = atoi(argv[2]);
    ref_cluster_id = atoi(argv[3]);

    clusters = NULL;
    int num_clusters = argc - 4;
    printf(" number of other clusters: %d\n", num_clusters);

    if (num_clusters > 0){
        clusters = (int*)malloc(num_clusters*sizeof(int));
        for (int i = 0; i < num_clusters; i++){
            clusters[i] = atoi(argv[i+4]);
            printf("%d ", clusters[i]);
        }
    }

    printf("  [data path: %s]\n", data_path);
    printf("  [loading face %d into reference cluster %d]\n", face_id, ref_cluster_id);

    // load the face
    char faceFile[256];
    sprintf(faceFile, "%simages/%d_mask2.jpg", data_path, face_id);
    printf("faceFile: %s\n", faceFile);
    face = cvLoadImage(faceFile, CV_LOAD_IMAGE_COLOR);
    face.convertTo(face, CV_64FC3, 1.0/255, 0);
    w = face.rows;
    h = face.cols;
    d = 3;
    num_pixels = w*h*d;

    if (visualize){
        imshow("loaded face", face);
        waitKey(500);
    }

    loadCluster(ref_cluster_id, ref_mean_face, &ref_mean, &ref_eigenfaces);
    Mat projected_face = projectFace(ref_mean, ref_eigenfaces);
    saveRelitFace(projected_face, ref_cluster_id);

    // warp face to global reference
    Mat reshaped_face = FaceLib::computeFlowAndWarp(face, projected_face);

    for (int i = 0; i < num_clusters; i++){
        int cluster_id = clusters[i];
        printf("--- cluster id: %d\n", cluster_id);
        loadCluster(cluster_id, cluster_mean_face, &cluster_mean, &cluster_eigenfaces);
        Mat cluster_projected_face = projectFace(cluster_mean, cluster_eigenfaces);
        saveRelitFace(cluster_projected_face, cluster_id); 

        Mat reshaped_cluster_avg = FaceLib::computeFlowAndWarp(cluster_projected_face, projected_face);
        char file[256];
        sprintf(file, "%swork/aging/face%d_cluster%d_refcluster%d.jpg", data_path, face_id, cluster_id, ref_cluster_id);
        FaceLib::saveAs(file, reshaped_cluster_avg);

        Mat diff_im = reshaped_cluster_avg - projected_face;
        
        char file1[256];
        sprintf(file1, "%swork/aging/face%d_cluster%d_refcluster%d_texdiff.jpg", data_path, face_id, cluster_id, ref_cluster_id);
        FaceLib::saveAs(file1, diff_im);

        diff_im = reshaped_face + diff_im;
        //imshow("difference image (texture)", diff_im);

        char file2[256];
        sprintf(file2, "%swork/aging/face%d_cluster%d_refcluster%d_tex.jpg", data_path, face_id, cluster_id, ref_cluster_id);
        FaceLib::saveAs(file2, diff_im);

        Mat unflowed = FaceLib::computeFlowAndApplyFlow(projected_face, cluster_projected_face, diff_im);
        //imshow("unflowed", unflowed);

        char file3[256];
        sprintf(file3, "%swork/aging/face%d_cluster%d_refcluster%d_texshape.jpg", data_path, face_id, cluster_id, ref_cluster_id);
        FaceLib::saveAs(file3, unflowed);
        //waitKey(0);
    }

}

void AgingApp::loadCluster(int c_id, Mat &mean_mat, gsl_vector **mean, gsl_matrix **eigenfaces){
    printf("[loadCluster] loading cluster %d\n", c_id);

    char file[256];
    sprintf(file, "%swork/%d/cflow/mean-rank05-1.jpg", data_path, c_id);
    mean_mat = cvLoadImage(file, CV_LOAD_IMAGE_COLOR);
    mean_mat.convertTo(mean_mat, CV_64FC3, 1.0/255, 0);
    *mean = gsl_vector_calloc(num_pixels);
    FaceLib::matToGslVec(mean_mat, *mean, d, w, h);
    if (0 && visualize){
        imshow("mean face", mean_mat);
        waitKey(500);
    }

    vector<string> eigenfacesList;
    for (int i = 0; i < 4; i++){
        char file[256];
        sprintf(file, "%swork/%d/cflow/eigen%02d.bin", data_path, c_id, i);
        printf("eigenface file: %s\n", file);
        eigenfacesList.push_back(string(file));
    }

    num_eigenfaces = eigenfacesList.size();

    *eigenfaces = gsl_matrix_calloc(num_pixels, num_eigenfaces);
    for (int i = 0; i < eigenfacesList.size(); i++){
        gsl_vector_view col = gsl_matrix_column(*eigenfaces, i);
        FILE *f = fopen(eigenfacesList[i].c_str(), "rb");
        int success = gsl_vector_fread(f, &col.vector);
        
        if (0 && visualize){
            Mat face;
            FaceLib::gslVecToMat(&(col.vector), face, d, w, h);
            imshow("loaded eigenface", face);
            waitKey(200);
        }
    }
}

Mat AgingApp::projectFace(gsl_vector *m_gsl_meanface, gsl_matrix *m_gsl_eigenfaces){
    printf("[projectFace]\n");

    gsl_vector *m_gsl_face = gsl_vector_calloc(num_pixels);
    FaceLib::matToGslVec(face, m_gsl_face, d, w, h);

    printf("done copying face mat to gsl\n");

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

    Mat projected_face;
    FaceLib::gslVecToMat(m_gsl_face_minus_mean, projected_face, d, w, h);

    if (visualize){
        imshow("projected face", projected_face);
        waitKey(1000);
    }

    return projected_face;
}

void AgingApp::saveRelitFace(Mat &projected_face, int ref_cluster_id){
    char file[256];
    sprintf(file, "%swork/aging/face%d_cluster%d_relit.jpg", data_path, face_id, ref_cluster_id);
    printf("file to save: %s\n", file);
    FaceLib::saveAs(file, projected_face);
}

static AgingApp *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new AgingApp();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

