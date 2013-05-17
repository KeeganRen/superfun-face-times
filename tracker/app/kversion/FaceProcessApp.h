#include <stdlib.h>
#include <vector>
#include <string>

#include "face_forest.hpp"

using namespace std;

class FaceProcessApp
{
public:
    void processOptions(int argc, char **argv);
    void init();
	void startFaceTracker();
	void startFaceForestTracker();
	void loadCanonicalFace();
	bool processFace();
	void write(char* outFile, cv::Mat &shape,cv::Mat &visi);
	void align(cv::Mat &shape,cv::Mat &visi);
	
    int argc;
    char **argv;
	
	char *ftFile;
	char *triFile;
	char *conFile;
	
	char *canonicalFace;
	
	char *imageFile;
	char *pointFile;
	char *pointImageFile;
	
	std::vector<cv::Point2f> canonicalPoints;
	cv::Mat frame;

	FaceForest *ff;
};
