/* StarterApp.cpp */

//g++ -I/opt/local/include -L/opt/local/lib -L/usr/lib starter.cpp -lopencv_highgui -lopencv_imgproc -lopencv_core  -o starterApp && ./starterApp

#include "starter.h"

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
            "   \tstarter program\n" \
            "\n");
}

void StarterApp::init(){
    printf("[init] Running program %s\n", argv[0]);
    
    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }

    load();
}

void StarterApp::load(){
    printf("[load] loading things\n");
    //a = imread(a_file);
    //a.convertTo(a, CV_64FC3, 1.0/255, 0);
}


static StarterApp *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new StarterApp();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

