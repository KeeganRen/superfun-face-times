#include <stdlib.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class WarpFaceApp
{
public:
    void processOptions(int argc, char **argv);
    void init();
    void loadFaceImage();
    void loadFacePoints();
    void loadFaceMesh();
    void populateModelPoints();
    void findTransformation();
    void makeNewFace();

    int argc;
    char **argv;
    
    char faceImageFile[1024];
    char facePointsFile[1024];
    char faceMeshFile[1024];
    char outFaceFile[1024];

    IplImage* faceImage;
    vector<Point2f> facePoints;
    vector<Point3f> modelPoints;
    vector<string> pointLabels;
    vector<Point3f> faceMesh;

    Matx33f cameraMatrix;
    Mat distortion_coefficients;
    Mat rvec;
    Mat tvec;
};
