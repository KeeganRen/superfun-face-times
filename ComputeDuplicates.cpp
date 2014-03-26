/* ComputeDuplicates.cpp */

//g++ -I/opt/local/include -L/opt/local/lib -L/usr/lib ComputeDuplicates.cpp -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_objdetect -o ComputeDuplicates && ./ComputeDuplicates

#include "ComputeDuplicates.h"

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
            "   \t--list <list of cropped (same size) images> \n" \
            "   \t--output <file to results to> \n" \
            "\n");
}

void ComputeDuplicates::processOptions(int argc, char **argv){
    while (1){
        static struct option long_options[] = {
            {"help",        0, 0, 'h'},
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


void ComputeDuplicates::init(){
    printf("[init] Running program %s\n", argv[0]);
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
        printf("need a file to save results to...\n");
        exit(0);
    }

    if (imageListFile){
        printf("image list file: %s\n", imageListFile);
        FILE *file = fopen ( imageListFile, "r" );
        if ( file != NULL ) {
            char image_path[256];
            while( fscanf(file, "%s\n", image_path) > 0 ) {
                imageNames.push_back(image_path);
            }
        }
    }

    printf("number of images: %d\n", int(imageNames.size()));

    loadAllImages();
    matchBuildList();
    saveMatchList();
    
}

void ComputeDuplicates::loadAllImages(){
    for (int i = 0; i < imageNames.size(); i++){
        Mat img = imread(imageNames[i].c_str());
        if (img.data != NULL){
            images.push_back(img);
        }
    }

    printf("number of images loaded: %d\n", int(images.size()));
}

void ComputeDuplicates::match(){
    int dupes = 0;

    for (int i = 0; i < images.size(); i++){
        for (int j = i + 1; j < images.size(); j++){
            Mat A = images[i];
            Mat B = images[j];
            double n = norm(A,B,NORM_L1);
            if (n < 2000000){
                dupes += 1;
                //printf("%s %s\t %f\n", imageNames[i].c_str(), imageNames[j].c_str(), n);
            }   
        }
    }

    printf("%d/%d duplicates (%f percent)", dupes, images.size(), float(dupes)/images.size());
}

void ComputeDuplicates::matchBuildList(){
    int dupes = 0;
    int unique = 0;
    double threshold = 2000000.0;

    for (int i = 0; i < images.size(); i++){
        bool in_list = false;

        // compare against stuff in list already
        for( map<int, vector<int> >::iterator ii=matches.begin(); ii!=matches.end(); ++ii){
            int j = (*ii).first;

            Mat A = images[i];
            Mat B = images[j];

            double n = norm(A,B,NORM_L1);
            if (n < threshold){
                matches[j].push_back(i);
                dupes += 1;
                in_list = true;
                break;
            }

        }

        if (!in_list){
            bool no_matches_found = true;

            for (int j = i + 1; j < images.size(); j++){
                Mat A = images[i];
                Mat B = images[j];
                double n = norm(A,B,NORM_L1);
                if (n < threshold){
                    matches[i].push_back(i);
                    matches[i].push_back(j);
                    dupes += 1;
                    no_matches_found = false;
                    break;
                }   
            }

            if (no_matches_found){
                unique += 1;
            }
        }
    }

    printf("%d/%d duplicates (%f percent)\n", dupes, images.size(), float(dupes)/images.size());
    printf("%d/%d unique (%f percent)\n", unique, images.size(), float(unique)/images.size());
}


bool sortMatches(vector<int> a, vector<int> b){
    int s1 = a.size();
    int s2 = b.size();

    return s1 > s2;
}


void ComputeDuplicates::saveMatchList(){
    vector<vector<int> > match_vector;
    for( map<int, vector<int> >::iterator ii=matches.begin(); ii!=matches.end(); ++ii){
        match_vector.push_back((*ii).second);
    }

    sort(match_vector.begin(),match_vector.end(),&sortMatches);

    FILE* f = fopen(outputFile, "w");
    for(int i = 0; i < match_vector.size(); i++){
        vector<int> list = match_vector[i];
        fprintf(f, "%d ", list.size());

        for (int j = 0; j < list.size(); j++){
            fprintf(f, "%s ", imageNames[list[j]].c_str());
        }

        fprintf(f, "\n");
    }
    /*
    for( map<int, vector<int> >::iterator ii=matches.begin(); ii!=matches.end(); ++ii){
        int i = (*ii).first;
        vector<int> list = (*ii).second;
        fprintf(f, "%d ", list.size());
        fprintf(f, "%s ", imageNames[i].c_str());

        for (int j = 0; j < list.size(); j++){
            fprintf(f, "%s ", imageNames[list[j]].c_str());
        }

        fprintf(f, "\n");
    }
    */
}

static ComputeDuplicates *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new ComputeDuplicates();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

