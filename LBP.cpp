#include <string>
#include <stdio.h>
#include <vector>
#include <list>
#include <climits>
#include <opencv/highgui.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>

extern "C" {
    #include <vl/generic.h>
    #include <vl/lbp.h>
}

using namespace cv;

float* matToFloatVec(const Mat& im);
float chiSquared(float *f1, float *f2, int num_features);

int main (int argc, char** argv){

    if (argc != 3) {
        printf("Usage: %s <queryFace> <face> \n"
            "  Returns: chi-squared distance between lbp features of faces\n"
            "\n", 
            argv[0]);
            return -1;
    }

    char* face_file_1 = argv[1];
    char* face_file_2 = argv[2];


    Mat face1 = imread(face_file_1);
    Mat face2 = imread(face_file_2);

    cvtColor(face1,face1,CV_RGB2GRAY);
    cvtColor(face2,face2,CV_RGB2GRAY);

    float *face1_floats = matToFloatVec(face1);
    float *face2_floats = matToFloatVec(face2);


    VlLbp *extractor = vl_lbp_new(VlLbpUniform, false);
    int dim = vl_lbp_get_dimension(extractor);

    //printf("lbp extractor dimensions: %d\n", dim);

    int cell_size = 50;
    int w = face1.cols;
    int h = face1.rows;

    int num_rows = floor(h/cell_size);
    int num_cols = floor(w/cell_size);
    int num_features = num_rows*num_cols*dim;
    //printf("%d features in this lbp-h vector\n", num_rows*num_cols*dim);


    float face1_features[num_rows*num_cols*dim];
    float face2_features[num_rows*num_cols*dim];

    vl_lbp_process(extractor, face1_features, face1_floats, w, h, cell_size);
    vl_lbp_process(extractor, face2_features, face2_floats, w, h, cell_size);

    //printf("%f %f %f\n", face1_features[0], face1_features[1], face1_features[2]);

    float x = chiSquared(face1_features, face2_features, num_features);
    printf("%f\n", x);

    return 0;
}

float* matToFloatVec(const Mat& im) {
    int w = im.cols;
    int h = im.rows;
    //printf("converting [%dx%d] image from Mat to float*\n", w, h);

    float *dst = (float*)malloc(w*h*sizeof(float));
    //float dst[w*h];

    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            int idx = i*h + j;
            uchar val = im.at<uchar>(i, j);
            dst[idx] = float(val);
        }
    }

    return dst;
}

float chiSquared(float *f1, float *f2, int num_features){

    if (f1[0] == -1 || f2[0] == -1){
        //fprintf(stderr, "[compareFeatures] one of the pair of features does not exist\n");
        return FLT_MAX;
    }

    float l1 = 0;
    for (int i = 0; i < num_features; i++){
        float nominator = (f1[i] - f2[i]) * (f1[i] - f2[i]);
        float denominator = (f1[i] + f2[i]);
        if ((f1[i] + f2[i]) != 0){
            float dist = nominator / denominator;
            l1 += dist;
        }
    }
    return l1;
}


