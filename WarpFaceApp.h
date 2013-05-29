#include <stdlib.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <gsl/gsl_linalg.h>


using namespace std;
using namespace cv;

class WarpFaceApp
{
public:
    void processOptions(int argc, char **argv);
    void init();
    void loadImageList();
    void loadTemplateFiles();
    void loadFaceSpecificFiles();
    void getColorFromImage();
    

    void setupShapeStuff();
    void populateTemplateMatrix();
    void populateImageMatrix();
    void shapeStuff();
    void solveStuff();

    void recoverDepth();

    void clip(int &a, int lo, int hi);
    void bilinear(double *out, Mat im, float c, float r);

    vector<Point2f> loadPoints(const char* filename);

    void testIntegrability();
    void testTemplate();
    void testTemplateVsCanonical();

    int argc;
    char **argv;
    
    char faceImageFile[1024];
    char facePointsFile[1024];
    char templateMeshFile[1024];
    char templatePointsFile[1024];
    char canonicalPointsFile[1024];
    char outFaceFile[1024];
    char listFile[1024];
    bool useList;

    bool visualize;

    int num_images;
    int num_points;

    vector<string> imageFiles;
    vector<string> imagePointFiles;

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

    gsl_matrix *m_gsl_model;
    gsl_matrix *m_gsl_images;
    gsl_matrix *m_gsl_s;
    gsl_matrix *m_gsl_final_result;
};
