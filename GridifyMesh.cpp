#include <string>
#include <stdio.h>
#include <vector>
#include <list>
#include <climits>
#include <opencv/highgui.h>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;


int main (int argc, char** argv){
    int num_faces = 0;
    int num_features = 0;

    if (argc < 3) {
        printf("Usage: %s <3d normal colored points as text> <mask image>\n"
            "  Returns: points that have been manipulated into corresponding with unmasked face pixels\n"
            "\n", 
            argv[0]);
            return -1;
    }


    int max_pts = 20000;
    float pts[max_pts][3];
    int colors[max_pts][3];
    float normals[max_pts][3];

    printf("loading mesh points from [%s]\n", argv[1]);
    FILE* file = fopen ( argv[1], "r" );
    int num_points = 0;
    if ( file != NULL ) {
        float x, y, z, nx, ny, nz;
        int r, g, b;
        float a;
        while( fscanf(file, "%f %f %f %f %f %f %d %d %d %d \n", &x, &y, &z, &nx, &ny, &nz, &r, &g, &b, &a) > 0 ) {
            pts[num_points][0] = x - 137.5;
            pts[num_points][1] = y - 145.0;
            pts[num_points][2] = z;

            normals[num_points][0] = nx;
            normals[num_points][1] = ny;
            normals[num_points][2] = nz;

            colors[num_points][0] = r;
            colors[num_points][1] = g;
            colors[num_points][2] = b;

            num_points++;
        }
        fclose (file);
    }
    else {
        perror (argv[1]);
    }
    
    printf("number of points read: %d\n", num_points);

    Mat maskImage = imread(argv[2]);
    if (maskImage.data == NULL){
        printf("Error loading image %s\n", argv[2]);
        exit(1);
    }
    cvtColor(maskImage, maskImage, CV_BGR2GRAY);

    /*for (int i = 0; i < num_points; i++){
        circle(maskImage, Point2f(pts[i][0], pts[i][1]), 1, 1, 0, 8, 0);
    }*/

    int count = 0;
    
    for (int i = 0; i < maskImage.cols; i++){
        for (int j = 0; j < maskImage.rows; j++){
            uchar g = maskImage.at<uchar>(j, i);
            if (g == 255){
                float dist = 10000000.0;
                int ii, jj;
                float zz;
                int magic_index = 0;
                for (int k = 0; k < num_points; k++){
                    float new_d = pow((i - pts[k][0]),2) + pow((j - pts[k][1]),2);
                    if (new_d < dist){
                        ii = pts[k][0];
                        jj = pts[k][1];
                        zz = pts[k][2];
                        magic_index = k;
                        dist = new_d;
                    }
                }
                printf("%d %d %f %f %f %f %d %d %d 1.0\n", i, j, zz, normals[magic_index][0], normals[magic_index][1], normals[magic_index][2], colors[magic_index][0], colors[magic_index][1], colors[magic_index][2]);
                count ++;
            }
            
        }
    }
    
    printf("num: %d\n", count);

    //imshow("mask", maskImage);
    //waitKey(0);


    fflush(stdout);
    fflush(stderr);

    return 0;
}

int countFeatures(std::string feature_file){
    FILE *file = fopen(feature_file.c_str(), "r");
    if (!file){
        //fprintf(stderr, "[countFeatures] error reading feature file: %s\n", feature_file.c_str());
        return -1;
    }

    int c = 0;
    char buf[1024];
    //while (fscanf(file, "%s\n", buf) != EOF){
    while (fgets(buf, 1024, file) != NULL){
        c++;
    }

    fclose(file);
    return c;
}

void readFeatures(int *features, std::string feature_file){
    const char *filename= feature_file.c_str();
    FILE *file = fopen(filename, "r");
    if (!file){
        //fprintf(stderr, "[readFeatures] error reading feature file: %s\n", feature_file.c_str());
        features[0] = -1;
        return;
    }

    int x, y, s, r;
    char buf[1024];
    int c = 0;
    while (1){
        if (fscanf(file, "%d %d %d %d", &x, &y, &s, &r) == EOF){
            break;
        }
        
        for (int i = 0; i < 128; i++){
            int x = fscanf(file, " %d", &(features[c*128 + i]));
        }
        c++;
    }

    fclose(file);
}

int compareFeatures(int *f1, int *f2, int num_features){

    if (f1[0] == -1 || f2[0] == -1){
        //fprintf(stderr, "[compareFeatures] one of the pair of features does not exist\n");
        return INT_MAX;
    }

    int l1 = 0;
    for (int i = 0; i < num_features*128; i++){
        int dist = f1[i] - f2[i];
        if (dist < 0)
            dist *= -1;
        l1 += dist;
    }
    return l1;
}
