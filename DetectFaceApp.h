#include <stdlib.h>
#include <string>
#include <FaceTracker/Tracker.h>

using namespace std;
using namespace cv;

class DetectFaceApp
{
public:
    void processOptions(int argc, char **argv);
    void init();
    void loadCanonicalFace();
    void startFaceTracker();
    void loadFacesFromList();
    void detectFace(Mat &faceImage, int i);
    void alignFace(Mat &frame, Mat &shape, Mat &visi);
    void visualizeFrame(string title, Mat &frame);

    int argc;
    char **argv;
    
    char *ftFile;
    char *triFile;
    char *conFile;
    char *canonicalFace;
    char *maskFile;
    Mat maskImage;

    bool align;
    bool mask;
    bool visualize;
    bool small;

    FACETRACKER::Tracker *model;
    std::vector<cv::Point2f> canonicalPoints;

    char inputFile[1024];
    char outputDir[1024];

    vector<string> faceList;
    vector<Mat> faceImages;
};
