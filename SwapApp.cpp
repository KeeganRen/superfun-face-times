/* SwapApp.cpp */

//g++ -I/opt/local/include -L/opt/local/lib -L/usr/lib SwapApp.cpp -lopencv_highgui -lopencv_imgproc -lopencv_core  -o SwapApp && ./SwapApp

#include "SwapApp.h"

#include "FaceLib.h"

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
            "   \t<full face A> <full face B> <A landmarks> <B landmarks> <cluster dir> <output dir>\n" \
            "\n");
}

void SwapApp::init(){
    printf("[init] Running program %s\n", argv[0]);
    
    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }

    char *faceAFile = argv[1];
    char *faceBFile = argv[2];
    char *landmarkAFile = argv[3];
    char *landmarkBFile = argv[4];
    char *clusterPath = argv[5];
    outPath = argv[6];

    load(faceAFile, faceBFile, landmarkAFile, landmarkBFile, clusterPath);

    FaceLib::computeTransform(A, landmarkA, templatePoints2D, A_xform);
    FaceLib::computeTransform(B, landmarkB, templatePoints2D, B_xform);

    cout << "A_xform" << endl;
    cout << A_xform << endl;

    A_mask = makeFullFaceMask(A, A_xform);
    B_mask = makeFullFaceMask(B, B_xform);

    swap();

    /*

    Mat A_cropped = makeCroppedFaceMask(A, A_xform);
    gsl_vector *A_gsl = gsl_vector_calloc(num_pixels);
    FaceLib::matToGslVec(A_cropped, A_gsl, d, w, h);
    gsl_vector *A_low_gsl = FaceLib::projectNewFace(num_pixels, num_eigenfaces, A_gsl, m_gsl_meanface, m_gsl_eigenfaces);
    Mat A_low;
    FaceLib::gslVecToMat(A_low_gsl, A_low, d, w, h);
    FaceLib::computeFlowAndStore(A_cropped, A_low, A_vx, A_vy);

    // flow the other way
    Mat B_cropped = makeCroppedFaceMask(B, B_xform);
    gsl_vector *B_gsl = gsl_vector_calloc(num_pixels);
    FaceLib::matToGslVec(B_cropped, B_gsl, d, w, h);
    gsl_vector *B_low_gsl = FaceLib::projectNewFace(num_pixels, num_eigenfaces, B_gsl, m_gsl_meanface, m_gsl_eigenfaces);
    Mat B_low;
    FaceLib::gslVecToMat(B_low_gsl, B_low, d, w, h);
    FaceLib::computeFlowAndStore(B_low, B_cropped, B_vx, B_vy);

    //imshow("A flow", FaceLib::showFlow(A_vx, A_vy));
    //imshow("B flow", FaceLib::showFlow(B_vx, B_vy));

    FaceLib::compositeFlow(A_vx, A_vy, B_vx, B_vy, vx, vy);
    //imshow("composed flow", FaceLib::showFlow(vx, vy));

    animateBlend();

    */
}

void SwapApp::load(char *faceAFile, char *faceBFile, char *landmarkAFile, char *landmarkBFile, char* clusterPath){
    printf("[load] loading:\n");
    printf("\t%s\n", faceAFile);
    printf("\t%s\n", faceBFile);
    printf("\t%s\n", landmarkAFile);
    printf("\t%s\n", landmarkBFile);

    A = imread(faceAFile);
    A.convertTo(A, CV_64FC3, 1.0/255, 0);

    B = imread(faceBFile);
    B.convertTo(B, CV_64FC3, 1.0/255, 0);

    FaceLib::loadFiducialPoints(landmarkAFile, landmarkA);
    FaceLib::loadFiducialPoints(landmarkBFile, landmarkB);

    string templateFaceFile = "data/grid-igor-canonical2d.txt";
    FaceLib::loadFiducialPoints(templateFaceFile, templatePoints2D);

    string maskImagefile = "data/igormask.png";
    maskImage = imread(maskImagefile);

    char meanfaceFile[256];
    sprintf(meanfaceFile, "%s/cflow/clustermean.jpg", clusterPath);
    printf("\treading mean face from: %s\n", meanfaceFile);
    Mat meanfaceMat = imread(meanfaceFile);
    meanfaceMat.convertTo(meanfaceMat, CV_64FC3, 1.0/255, 0);

    w = meanfaceMat.rows;
    h = meanfaceMat.cols;
    d = 3;
    num_pixels = w*d*h;

    m_gsl_meanface = gsl_vector_calloc(num_pixels);
    FaceLib::matToGslVec(meanfaceMat, m_gsl_meanface,  d, w, h);

    // eigenfaces
    num_eigenfaces = 4;
    printf("sizes of things. num_pixels: %d, num eigenfaces: %d\n", num_pixels, num_eigenfaces);
    m_gsl_eigenfaces = gsl_matrix_calloc(num_pixels, num_eigenfaces);
    for (int i = 0; i < num_eigenfaces; i++){
        char eigenfaceFile[256];
        sprintf(eigenfaceFile, "%s/cflow/eigen%02d.bin", clusterPath, i);
        printf("\treading eigenface from file: %s\n", eigenfaceFile);

        gsl_vector_view col = gsl_matrix_column(m_gsl_eigenfaces, i);
        FILE *f = fopen(eigenfaceFile, "rb");
        int success = gsl_vector_fread(f, &col.vector);        
    }

    // TODO: loadEigenfaces()
    // project each masked face into eigenfaces
    // compute flow in the right directions
    // composite flow
    // use flow
}

Mat SwapApp::makeFullFaceMask(Mat &frame, Mat &transform){

    Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC3 );

    Mat inverse;
    invertAffineTransform(transform, inverse);
    warpAffine(maskImage, mask, inverse, mask.size() );

    return mask;
}


Mat SwapApp::makeCroppedFaceMask(Mat &frame, Mat &transform){

    Mat warped = Mat::zeros(192, 139, frame.type() );
    Mat mask = Mat::zeros(192, 139, frame.type() );

    warpAffine(frame, warped, transform, warped.size() );
    warped.copyTo(mask, maskImage);

    return mask;
}

void SwapApp::swap(){
    printf("[swapping]\n");

    Mat inverseAXform;
    invertAffineTransform(A_xform, inverseAXform);


    Mat aaa = Mat::eye( 3, 3, CV_32FC1 );
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 3; j++){
            aaa.at<float>(i, j) = inverseAXform.at<double>(i, j);
        }
    }

    cout << "invsers" << inverseAXform << endl;
    cout << "aaa" << aaa << endl;

    Mat bbb = Mat::eye( 3, 3, CV_32FC1 );
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 3; j++){
            bbb.at<float>(i, j) = B_xform.at<double>(i, j);
        }
    }


    Mat BtoAXform =  Mat( 3, 3, CV_32FC1 );
    BtoAXform = aaa*bbb;

    Mat xxx =  Mat( 2, 3, CV_32FC1 );
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 3; j++){
            xxx.at<float>(i, j) = BtoAXform.at<float>(i, j);
        }
    }
 
    Mat transformed = Mat::zeros(A.rows, A.cols, A.type() );
    warpAffine(B, transformed, xxx, transformed.size());
    BBlendedToA = Mat::zeros(A.rows, A.cols, A.type() );
    transformed.copyTo(BBlendedToA, A_mask);
    BBlendedToA = FaceLib::montage(transformed, A, A_mask);
    


    Mat inverseXXX;
    invertAffineTransform(xxx, inverseXXX);
    Mat transformedB = Mat::zeros(B.rows, B.cols, B.type() );
    warpAffine(A, transformedB, inverseXXX, transformedB.size());
    ABlendedToB = Mat::zeros(B.rows, B.cols, B.type() );
    transformedB.copyTo(ABlendedToB, B_mask);
    ABlendedToB = FaceLib::montage(transformedB, B, B_mask);


    /*
    imshow("b blended to A", BBlendedToA);
    imshow("a blended to b", ABlendedToB);
    waitKey(0);
    */

    char filename[256];
    sprintf(filename, "%s/swap_a.jpg", outPath);
    FaceLib::saveAs(filename, BBlendedToA);

    char filename2[256];
    sprintf(filename2, "%s/swap_b.jpg", outPath);
    FaceLib::saveAs(filename2, ABlendedToB);

}

void SwapApp::animateBlend(){
    printf("[animateBlend] %d, %d\n", A_mask.rows, A_mask.cols);

    Mat img = A.clone();

    for (float alpha = 0.0; alpha <= 1.1; alpha += 0.1){
        for (int x=0; x<A_mask.cols; x++) {
            for (int y=0; y<A_mask.rows; y++) {
                uchar c = A_mask.at<Vec3b>(y,x)[0];
                //printf("%d \n", (int)c);
                if ((int)c > 0){
                    Point2f xy = Point2f(x,y);
                    Point2f xform_xy = applyXform(xy, A_xform);
                    Point2f ixform_xy = applyInverseXform(xform_xy, A_xform);

                    double flow_x = CVOpticalFlow::bilinearFlow(vx, xform_xy.x, xform_xy.y);
                    double flow_y = CVOpticalFlow::bilinearFlow(vy, xform_xy.x, xform_xy.y);

                    
                    Point2f new_xy_mask_A = Point2f(xform_xy.x + flow_x*(alpha), xform_xy.y + flow_y*(alpha));
                    Point2f new_xy_A = applyInverseXform(new_xy_mask_A, A_xform);
                    Vec3d color_A = A.at<Vec3d>(new_xy_A.y, new_xy_A.x);
                    
                    Point2f new_xy_mask_B = Point2f(xform_xy.x + flow_x*(alpha-1), xform_xy.y + flow_y*(alpha-1));
                    Point2f new_xy_B = applyInverseXform(new_xy_mask_B, A_xform);
                    Vec3d color_B = BBlendedToA.at<Vec3d>(new_xy_B.y, new_xy_B.x);

                    
                    Vec3d color = (1-alpha)*color_A + alpha*color_B;

                    circle(img, xy, 1, CV_RGB(color[2], color[1], color[1]), 0, 8, 0);

                }
            }
        }
        cout << "ALPHA " << alpha << endl;
        //imshow("transform", img);
        char filename[256];
        sprintf(filename, "%s/swaparoo%.02f.jpg", outPath, alpha);
        FaceLib::saveAs(filename, img);
        //waitKey(0);
    }

    
}

Point2f SwapApp::applyXform(Point2f pt, Mat &xform){
    float x = pt.x * xform.at<double>(0,0) + pt.y * xform.at<double>(0,1) + xform.at<double>(0,2);
    float y = pt.x * xform.at<double>(1,0) + pt.y * xform.at<double>(1,1) + xform.at<double>(1,2);
    
    return Point2f(x,y);
}

Point2f SwapApp::applyInverseXform(Point2f pt, Mat &xform){
    double det = xform.at<double>(0,0)*xform.at<double>(1,1) - xform.at<double>(0,1)*xform.at<double>(1,0);
    
    float px = pt.x - xform.at<double>(0,2);
    float py = pt.y - xform.at<double>(1,2);

    float x =  px * xform.at<double>(1,1)/det - py*xform.at<double>(0,1)/det;
    float y =  -px * xform.at<double>(1,0)/det + py*xform.at<double>(0,0)/det;
    return Point2f(x,y);
}

static SwapApp *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new SwapApp();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

