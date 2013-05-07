#include <stdlib.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;

class FaceProcessApp
{
public:
    void processOptions(int argc, char **argv);
    void init();
    void loadFacesFromList();
    void findImageSizeFromFirstImage();
    void openImages();
	void convertImages();
	void buildMatrixAndRunPca();

    int argc;
    char **argv;
	
	char inputFile[1024];
	vector<string> faceList;
	int w, h;
	vector<IplImage*> faceImages;
};
