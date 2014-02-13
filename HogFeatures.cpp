/* HogFeatures.cpp */

//g++ -I/opt/local/include -L/opt/local/lib -L/usr/lib HogFeatures.cpp -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_objdetect -o HogFeatures && ./HogFeatures

#include "HogFeatures.h"

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
            "   \t--image <image file> \n" \
            "   \t  [OR] \n" \
            "   \t--list <file with list of files> \n\n" \
            "   \t  [AND] \n" \
            "   \t--output <file to save hog features to... one image per line> \n" \
            "\n");
}

void HogFeatures::processOptions(int argc, char **argv){
    while (1){
        static struct option long_options[] = {
            {"help",        0, 0, 'h'},
            {"image",       1, 0, 400},
            {"output",      1, 0, 401},
            {"list",        1, 0, 402},
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
                imageFile = optarg;
                break;

            case 401:
                outputFile = optarg;
                break;

            case 402:
                imageListFile = optarg;
                break;

            default: 
                printf("Unrecognized option %d\n", c);
                break;
        }
    }
}


void HogFeatures::init(){
    printf("[init] Running program %s\n", argv[0]);
    imageFile = NULL;
    imageListFile = NULL;
    outputFile = NULL;
    
    if (argc < 3 ){
        PrintUsage();
        exit(0);
    }

    processOptions(argc, argv);


    if (outputFile){
        printf("output file: %s\n", outputFile);
    }
    else {
        printf("need a file to save features to...\n");
        exit(0);
    }

    if (imageFile){
        printf("image file: %s\n", imageFile);
        images.push_back(imageFile);
    }
    else if (imageListFile){
        printf("image list file: %s\n", imageListFile);
        FILE *file = fopen ( imageListFile, "r" );
        if ( file != NULL ) {
            char image_path[256];
            while( fscanf(file, "%s\n", image_path) > 0 ) {
                images.push_back(image_path);
            }
        }
    }

    printf("number of images to extract hog features on: %d\n", int(images.size()));

    FILE *hog_file = fopen ( outputFile, "w" );
    HOGDescriptor d;
    d.winSize = Size(32, 32);
    d.blockSize = Size(32, 32);
    d.cellSize = Size(16, 16);

    vector<float> ders;
    vector<Point>locs;


    for (int i = 0; i < images.size(); i++){
        Mat img = imread(images[i].c_str());
        if (img.data != NULL){

            cvtColor(img, img, CV_BGR2GRAY);
        
            d.compute(img,ders,Size(32,32),Size(0,0),locs);
        
            /*
            cout << "HOG descriptor size is " << d.getDescriptorSize() << endl;
            cout << "img dimensions: " << img.cols << " width x " << img.rows << "height" << endl;
            cout << "Found " << ders.size() << " descriptor values" << endl;
            cout << "Nr of locations specified : " << locs.size() << endl;

            cout << "locs size: " << locs.size() << endl;;
            cout << "ders size: " << ders.size() << endl;;
            */

            for(int i=0;i<ders.size();i++) {
                fprintf(hog_file, "%f ", ders.at(i));
            }

            fprintf(hog_file, "\n");
        }

    }

    fclose (hog_file);
}


static HogFeatures *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new HogFeatures();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

