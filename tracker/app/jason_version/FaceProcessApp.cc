/* FaceProcessApp.cpp */

#include "FaceProcessApp.h"
#include "getopt.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <opencv/highgui.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"

/*
#include "forest.hpp"
#include "multi_part_sample.hpp"
#include "head_pose_sample.hpp"
#include "face_utils.hpp"
#include <istream>
#include <opencv2/opencv.hpp>
#include <boost/progress.hpp>
#include "face_forest.hpp"
#include "feature_channel_factory.hpp"
#include "timing.hpp"
*/ 

void PrintUsage() 
{
    printf("Usage:  ...\n");
}

void FaceProcessApp::processOptions(int argc, char **argv){
    while (1){
        static struct option long_options[] = {
            {"help",        0, 0, 'h'},
			{"image",		1, 0, 'i'},
			{"points",		1, 0, 'p'},
			{"pointImage",	1, 0, 400},
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
				
			case 'i':
				imageFile = strdup(optarg);
				break;
				
			case 'p':
				pointFile = strdup(optarg);
				break;
				
			case 400:
				pointImageFile = strdup(optarg);
				break;
                
			default: 
                printf("Unrecognized option %d\n", c);
                break;
        }
    }
}

void FaceProcessApp::init(){
    printf("[init] Running program %s\n", argv[0]);
	
	if (argc < 2 ){
        PrintUsage();
        exit(0);
    }
	
	ftFile  =	"model/face2.tracker";
	triFile = "model/face.tri";
	conFile = "model/face.con";
	canonicalFace = "bin/canonical-k-2.txt";
	
    processOptions(argc, argv);
	
	loadCanonicalFace();
	
	startFaceForestTracker();
	startFaceTracker();
	
	processFace();

}

void FaceProcessApp::startFaceTracker(){
	printf("[startFaceTracker]\n");
	model = new FACETRACKER::Tracker(ftFile);
	printf("[startFaceTracker] tracker model loaded\n");
    cv::Mat tri=FACETRACKER::IO::LoadTri(triFile);
	printf("[startFaceTracker] triangles loaded\n");
    cv::Mat con=FACETRACKER::IO::LoadCon(conFile);
	printf("[startFaceTracker] connections loaded\n");
}

void FaceProcessApp::startFaceForestTracker(){
	/*
	printf("[startFaceForestTracker]\n");
	
    // Get data files for the face forest stuff 
    string ffd_config_file = "../tracker_eth/data/config_ffd.txt";
    string headpose_config_file = "../tracker_eth/data/config_headpose.txt";
    string face_cascade = "../tracker_eth/data/haarcascade_frontalface_alt.xml";
	
    
    // parse config file for face forest stuff
    ForestParam mp_param;
    assert(loadConfigFile(ffd_config_file, mp_param));
    printf("[startFaceForestTracker] config file loaded\n");
	
    FaceForestOptions option;
    option.face_detection_option.path_face_cascade = face_cascade;
	
    ForestParam head_param;
    assert(loadConfigFile(headpose_config_file, head_param));
    option.head_pose_forest_param = head_param;
    option.mp_forest_param = mp_param;
	
    // ready to run?!
    ff = new FaceForest(option); // this is the line that takes 4 seconds
	
	printf("[startFaceForestTracker] finished loading and starting up face forest\n");
	 */
}

bool FaceProcessApp::processFace(){
	printf("Processing face in file: %s\n", imageFile);
	
	cv::Mat gray,im; 
	IplImage* I;
	
	I = cvLoadImage(imageFile, CV_LOAD_IMAGE_COLOR); 
	
	if(!I){
		printf("Error loading image\n");
		return false;
	}
	
	frame = I;
	im = frame;
	
	cv::cvtColor(im,gray,CV_BGR2GRAY);
	
	//set other tracking parameters
	std::vector<int> wSize1(1); wSize1[0] = 7;
	std::vector<int> wSize2(3); wSize2[0] = 11; wSize2[1] = 9; wSize2[2] = 7;
	int nIter = 5; 
	double clamp=3,fTol=0.01; 
	int fpd = -1; 
	bool fcheck = false;
	bool success = false;
	
	fflush(stdout);
	
	//track this image
    std::vector<int> wSize; 
	wSize = wSize2;
	if (model->Track(gray, wSize, fpd, nIter, clamp, fTol, fcheck) == 0){
		int idx = model->_clm.GetViewIdx();
		write(pointFile, model->_shape, model->_clm._visi[idx]);
		align(model->_shape, model->_clm._visi[idx]);
		success = true;
	}
	
	model->FrameReset();
	
	return success;
}

void FaceProcessApp::loadCanonicalFace(){
	
	FILE *f1 = fopen(canonicalFace, "r");
    double x,y;
    while (fscanf(f1, "%lf %lf", &x, &y) != EOF){
		canonicalPoints.push_back(cv::Point2f(x,y));
    }
	
	printf("Size of canonical points vector: %d\n", int(canonicalPoints.size()));
}

void FaceProcessApp::align(cv::Mat &shape,cv::Mat &visi){
	std::vector<cv::Point2f> facePoints;
	
	
	double x,y;
	int i, n;
	n = canonicalPoints.size();
	
	for(i = 0; i < n; i++){    
		if(visi.at<int>(i,0) == 0)continue;
		x = shape.at<double>(i,0);
		y = shape.at<double>(i+n,0);
		facePoints.push_back(cv::Point2f(x,y));
	}
	
	printf("Size of facePoints vector: %d\n", int(facePoints.size()));

	
	cv::Mat canonicalMat = cv::Mat(canonicalPoints);
	cv::Mat faceMat = cv::Mat(facePoints);

	cv::Mat transform = cv::estimateRigidTransform(facePoints,canonicalPoints, false);
	
	cout << "transform = "<< endl << " "  << transform << endl << endl;
	
	
	// do the warping
	// frame is the source
//	imshow("orig image", frame);
	
	cv::Mat warped = cv::Mat::zeros(500, 500, frame.type() );
	cv::warpAffine(frame, warped, transform, warped.size() );
	
	imshow("warped", warped);
	
	cv::Rect myROI(130, 145, 225, 250);
	
	cv::rectangle(warped, cvPoint(myROI.x, myROI.y), cvPoint(myROI.x+ myROI.width, myROI.y + myROI.height), CV_RGB(255, 0, 0), 1, 8, 0);
	
	imwrite(pointImageFile, warped);
	

	
}

void FaceProcessApp::write(char* outFile, cv::Mat &shape,cv::Mat &visi) {
	if (std::strcmp(outFile, "none") == 0) return;
	
	int i,n = shape.rows/2; cv::Point p1,p2; cv::Scalar c;
	double x,y;
	std::ofstream myfile;
	myfile.open (outFile);
	
	//draw points
	for(i = 0; i < n; i++){    
		if(visi.at<int>(i,0) == 0)continue;
		x = shape.at<double>(i,0);
		y = shape.at<double>(i+n,0);
		myfile << x << " " << y << std::endl;
	}
	
	myfile.close();
	
	return;
}

static FaceProcessApp *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new FaceProcessApp();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

