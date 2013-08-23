/* AverageFaces.cpp */

//g++ -I/opt/local/include -L/opt/local/lib -L/usr/lib starter.cpp -lopencv_highgui -lopencv_imgproc -lopencv_core  -o AverageFaces && ./AverageFaces

#include "AverageFaces.h"

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
#define e3 at<Vec3b>

void error(char *msg) {
    perror(msg);
    exit(1);
}

void PrintUsage() 
{
    printf("Usage:  \n" \
            "   \t[./program] list_of_images.txt output_image.jpg\n" \
            "\n");
}

void AverageFaces::init(){
    printf("[init] Running program %s\n", argv[0]);
    
    if (argc < 3 ){
        PrintUsage();
        exit(0);
    }

    char* file_list = argv[1];
    char* output_image = argv[2];

    printf("file list: [%s], output image: [%s]\n", file_list, output_image);

    loadAndMakeAverage(file_list, output_image);
}

void AverageFaces::loadAndMakeAverage(char *file_list, char *output_image){
    printf("[loadAndMakeAverage] loading things\n");
    //a = imread(a_file);
    //a.convertTo(a, CV_64FC3, 1.0/255, 0);
    
    vector<string> imageFiles;

    FILE *file = fopen ( file_list, "r" );
    if ( file != NULL ) {
        char image_path[256];
        while( fscanf(file, "%s\n", image_path) > 0 ) {
            imageFiles.push_back(image_path);
        }
        fclose (file);
    }
    else {
        perror (file_list);
    }

    Mat final_blend = imread(imageFiles[0].c_str(), CV_LOAD_IMAGE_COLOR);
    final_blend.convertTo(final_blend, CV_64FC3, 1.0/255, 0);

    double alpha;
    double beta;

    for (int i = 1; i < imageFiles.size(); i++){
        //printf("\t loading image %d [%s]\n", i, imageFiles[i].c_str());
        // load the image
        Mat im = imread(imageFiles[i].c_str(), CV_LOAD_IMAGE_COLOR);
        im.convertTo(im, CV_64FC3, 1.0/255, 0);
        

        alpha = 1.0/(i+1);
        beta = (1 - alpha);
        
        addWeighted(im, alpha, final_blend, beta, 0.0, final_blend);

        //imshow("loaded image", im);
        //imshow("blended image", final_blend);
        //waitKey(80);
        im.release();
    }

    final_blend.convertTo(final_blend, CV_8UC3, 1.0*255, 0);
    imwrite(output_image, final_blend);
}


static AverageFaces *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new AverageFaces();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

