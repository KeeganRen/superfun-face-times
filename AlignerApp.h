#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class AlignerApp
{
public:
    void init();
    void load();
    void loadFromList();
    vector<Point2f> loadPoints(const char* filename);
    void dealWithImage(string image, string points, string out);
    void test1();
    void test2();
    void test3();
    void test4();
    Point2f perpendicularPoint(vector<Point2f> f);

    Point2f middle(Point2f a, Point2f b);

    int argc;
    char **argv;
    
    char *face_file;
    char *face_points_file;
    char *list_file;

    Mat face;
    Mat mask;

    vector<Point2f> face_points;
    vector<Point2f> template_points;

    vector<string> imageFiles;
    vector<string> imagePointFiles;
    vector<string> outputImageFiles;

    bool visualize;
};
