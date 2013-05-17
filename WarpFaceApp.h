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
    void loadTemplateFiles();
    void loadFaceSpecificFiles();
    void getColorFromImage();
    void findTransformation();
    void makeNewFace();

    void clip(int &a, int lo, int hi);
    void bilinear(double *out, Mat im, float c, float r);

    int argc;
    char **argv;
    
    char faceImageFile[1024];
    char facePointsFile[1024];
    char templateMeshFile[1024];
    char templatePointsFile[1024];
    char canonicalPointsFile[1024];
    char outFaceFile[1024];

    vector<string> pointLabels;

    Mat faceImage;
    vector<Point2f> facePoints;
    

    vector<Point2f> canonicalPoints;
    
    vector<Point3f> templateMesh;
    vector<Point3f> templateNormals;
    vector<Point3f> templateColors;

    vector<Point3f> templatePoints;

    float fx;
    Matx33f cameraMatrix;
    Mat distortion_coefficients;
    Mat rvec;
    Mat tvec;
};
