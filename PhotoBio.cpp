/* PhotoBioApp.cpp */

//g++ -I/opt/local/include -L/opt/local/lib -L/usr/lib starter.cpp -lopencv_highgui -lopencv_imgproc -lopencv_core  -o PhotoBioApp && ./PhotoBioApp

#include "PhotoBio.h"

#include "getopt.h"
#include <string.h>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <time.h>
#include <opencv/highgui.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <istream>
#include <opencv2/opencv.hpp>


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
            "   \t./photoBio <file w/ [image_id] [feature path] [category] on each line> <file to write sequence to>\n" \
            "\n");
}

void PhotoBioApp::init(){
    printf("[init] Running program %s\n", argv[0]);
    
    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }

    load(argv[1]);
    loadFeaturesAndDoStuff(argv[2]);

}

void PhotoBioApp::load(char *listFile){
    printf("[load] loading stuff from file %s\n", listFile);
    FILE *file = fopen ( listFile, "r" );
    if ( file != NULL ) {
        int image_id;
        char feature_path[256];
        int cat_id;
        while( fscanf(file, "%d %s %d\n", &image_id, feature_path, &cat_id) > 0 ) {
            ifstream ifile(feature_path);
            if (ifile) {
                imageIds.push_back(image_id);
                featureFiles.push_back(feature_path);
                imageCategories.push_back(cat_id);
            }
            else {
                printf("\t oh noes! file [%s] not found\n", feature_path);
            }
        }
        fclose (file);
    }
}

void PhotoBioApp::loadFeaturesAndDoStuff(char* outfile){
    num_faces = featureFiles.size();
    fprintf(stderr, "Number of faces: %d\n", num_faces);

    num_features = countFeatures(featureFiles[0]);
    if (num_features <= 0){
        return;
    }
    fprintf(stderr, "Number of features per face: %d\n", num_features);

    int m[num_faces][num_features*128];

    for (int i = 0; i < num_faces; i++){
        //printf("\tFeature file: %s\n", featureFiles[i].c_str());
        readFeatures(m[i], featureFiles[i]);
    }

    /* Create a new array of points */
    pts = annAllocPts(num_faces, num_features*128);

    for (int i = 0; i < num_faces; i++) {
        for (int j = 0; j < num_features*128; j++){
            pts[i][j] = m[i][j];
        }
    }

    tree = new ANNkd_tree(pts, num_faces, num_features*128);


    srand (time(NULL));
    int starting_id = rand() % num_faces;
    findNextPic(starting_id);

    FILE *file = fopen ( outfile, "w" );
    fprintf(file, "convert -delay 20 -loop 0 ");
    for (int i = 0; i < seq.size(); i++){ 
        //fprintf(file, "%d\n", seq[i]);
        fprintf(file, "/Users/ktuite/Code/FaceServer/data/images/%d_warped.jpg ", imageIds[seq[i]]);
    }
    fprintf(file, "$1");
    fclose(file);

}

void PhotoBioApp::findNextPic(int idx){
    if (idx == -1){
        return;
    }

    seq.push_back(idx);

    int next_idx = -1;

    int k = num_faces;
    ANNpoint queryPt;
    ANNidxArray nnIdx = new ANNidx[k];
    ANNdistArray dists = new ANNdist[k];

    queryPt = pts[idx];
    tree->annkSearch(queryPt, k, nnIdx, dists, 0.0);
    for (int i = 0; i < k; i++){
        if ( find(seq.begin(), seq.end(), nnIdx[i])==seq.end() ){
            cout << "not in list" << endl;
            next_idx = nnIdx[i];
            break;
        }
        //cout << i << " " << nnIdx[i] << " " << imageIds[nnIdx[i]] << " " << dists[i] << endl;
    }

    cout << "sequence: ";
    for (int i = 0; i < seq.size(); i++){
        cout << seq[i] << " ";
    }
    cout << endl << endl;

    findNextPic(next_idx);
}

int PhotoBioApp::countFeatures(string feature_file){
    FILE *file = fopen(feature_file.c_str(), "r");
    if (!file){
        fprintf(stderr, "[countFeatures] error reading feature file: %s\n", feature_file.c_str());
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

void PhotoBioApp::readFeatures(int *features, string feature_file){
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
            fscanf(file, " %d", &(features[c*128 + i]));
        }
        c++;
    }

    fclose(file);
}


static PhotoBioApp *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new PhotoBioApp();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

