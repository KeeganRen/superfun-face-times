/* AverageFaces.cpp */

//g++ -I/opt/local/include -L/opt/local/lib -L/usr/lib starter.cpp -lopencv_highgui -lopencv_imgproc -lopencv_core  -o AverageFaces && ./AverageFaces

#include "AverageFaces.h"

#include "getopt.h"
#include <string.h>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <opencv/highgui.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>  // Video write


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
    char* output_vid_file = NULL;
    if (argc >= 4){
        output_vid_file = argv[3];
    }

    printf("file list: [%s], output image: [%s]\n", file_list, output_image);

    loadAndMakeAverage(file_list, output_image, output_vid_file);
}

void AverageFaces::loadAndMakeAverage(char *file_list, char *output_image, char* output_vid_file){
    printf("[loadAndMakeAverage] loading things\n");
    //a = imread(a_file);
    //a.convertTo(a, CV_64FC3, 1.0/255, 0);
    
    vector<string> imageFiles;

    FILE *file = fopen ( file_list, "r" );
    if ( file != NULL) {
        char image_path[256];
        int r = fscanf(file, "%s\n", image_path); 
        while( r != EOF && r > 0 ) {
            ifstream ifile(image_path);
            if (ifile) {
                imageFiles.push_back(image_path);
            }
            else {
                printf("file not found: %s\n", image_path);
            }
            r = fscanf(file, "%s\n", image_path); 
            
        }
        fclose (file);
    }
    else {
        printf("couldnt open the file");
        perror (file_list);
        exit(-1);
    }

    if (imageFiles.size() == 0){
        exit(-1);
    }


    Mat final_blend = imread(imageFiles[0].c_str(), CV_LOAD_IMAGE_COLOR);
    final_blend.convertTo(final_blend, CV_64FC3, 1.0/255, 0);

    Mat convert_holder;
    VideoWriter outputVideo;

    if (output_vid_file){
        convert_holder = imread(imageFiles[0].c_str(), CV_LOAD_IMAGE_COLOR);
        outputVideo.open(output_vid_file, CV_FOURCC('U','2','6','3'), 25, final_blend.size(), true);
    }

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
        
        if (output_vid_file){
            final_blend.convertTo(convert_holder, CV_8UC3, 1.0*255, 0);
            outputVideo << convert_holder;
            outputVideo << convert_holder;
        }
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

