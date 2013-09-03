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


    matchHistograms();

    swap();

    /*

    Mat A_cropped = makeCroppedFaceMask(A, A_xform);
    Mat B_cropped = makeCroppedFaceMask(B, B_xform);
    imshow("a cropped", A_cropped);
    imshow("b cropped", B_cropped);


    A_cropped.convertTo(A_cropped, CV_8UC3, 1.0*255, 0);
    B_cropped.convertTo(B_cropped, CV_8UC3, 1.0*255, 0);
    A.convertTo(A, CV_8UC3, 1.0*255, 0);

    histogramFunTimes(A_cropped, B_cropped, A, B);
    */

    //

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

    //string maskImagefile = "data/igormask.png";
    string maskImagefile = "data/igormask-smaller.png";
    maskImage = imread(maskImagefile);
    cout << endl << "  mask type: " << maskImage.type() << endl;

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
    warpAffine(B_color_corrected, transformed, xxx, transformed.size());
    BBlendedToA = Mat::zeros(A.rows, A.cols, A.type() );
    transformed.copyTo(BBlendedToA, A_mask);
    BBlendedToA = FaceLib::montage(transformed, A, A_mask);
    


    Mat inverseXXX;
    invertAffineTransform(xxx, inverseXXX);
    Mat transformedB = Mat::zeros(B.rows, B.cols, B.type() );
    warpAffine(A, transformedB, inverseXXX, transformedB.size());
    //warpAffine(A_color_corrected, transformedB, inverseXXX, transformedB.size());
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


void SwapApp::histogramFunTimes(Mat &A_cropped, Mat &B_cropped, Mat &A, Mat &B){
    printf("histogram fun times!");

    /// Separate the image in 3 places ( B, G and R )
    vector<Mat> bgr_planes;
    split( A_cropped, bgr_planes );

    vector<Mat> A_split;
    split( A, A_split );

    vector<Mat> bgr_planes2;
    split( B_cropped, bgr_planes2 );

    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist, g_hist, r_hist;
    Mat b_hist2, g_hist2, r_hist2;

    /// Compute the histograms:
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

    calcHist( &bgr_planes2[0], 1, 0, Mat(), b_hist2, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes2[1], 1, 0, Mat(), g_hist2, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes2[2], 1, 0, Mat(), r_hist2, 1, &histSize, &histRange, uniform, accumulate );

    // todo: cdf 
    Mat cdf_r_hist(b_hist.size(), b_hist.type());
    Mat cdf_g_hist(b_hist.size(), b_hist.type());
    Mat cdf_b_hist(b_hist.size(), b_hist.type());
    Mat cdf_r_hist2(b_hist.size(), b_hist.type());
    Mat cdf_g_hist2(b_hist.size(), b_hist.type());
    Mat cdf_b_hist2(b_hist.size(), b_hist.type());

    for( int i = 0; i < histSize; i++ ){
        if (i == 0){
            cdf_r_hist.at<float>(i) = cvRound(r_hist.at<float>(i));
            cdf_g_hist.at<float>(i) = cvRound(g_hist.at<float>(i));
            cdf_b_hist.at<float>(i) = cvRound(b_hist.at<float>(i));
            cdf_r_hist2.at<float>(i) = cvRound(r_hist2.at<float>(i));
            cdf_g_hist2.at<float>(i) = cvRound(g_hist2.at<float>(i));
            cdf_b_hist2.at<float>(i) = cvRound(b_hist2.at<float>(i));
        }
        else {
            cdf_r_hist.at<float>(i) = cvRound(r_hist.at<float>(i)) + cvRound(cdf_r_hist.at<float>(i-1));
            cdf_g_hist.at<float>(i) = cvRound(g_hist.at<float>(i)) + cvRound(cdf_g_hist.at<float>(i-1));
            cdf_b_hist.at<float>(i) = cvRound(b_hist.at<float>(i)) + cvRound(cdf_b_hist.at<float>(i-1));
            cdf_r_hist2.at<float>(i) = cvRound(r_hist2.at<float>(i)) + cvRound(cdf_r_hist2.at<float>(i-1));
            cdf_g_hist2.at<float>(i) = cvRound(g_hist2.at<float>(i)) + cvRound(cdf_g_hist2.at<float>(i-1));
            cdf_b_hist2.at<float>(i) = cvRound(b_hist2.at<float>(i)) + cvRound(cdf_b_hist2.at<float>(i-1));
        }
    }

    Mat M_r(b_hist.size(), CV_8U);
    Mat M_g(b_hist.size(), CV_8U);
    Mat M_b(b_hist.size(), CV_8U);

    for( int i = 0; i < histSize; i++ ){
        // i is a pixel intensity value
        
        int target_intensity = 0;
        float num_pixels_in_a;
        float num_pixels_in_b;

        // r channel
        num_pixels_in_a = cdf_r_hist.at<float>(i);

        for (int j = 0; j < histSize; j++){
            num_pixels_in_b = cdf_r_hist2.at<float>(j);
            if (num_pixels_in_b >= num_pixels_in_a){
                target_intensity = j;
                break;
            }
        }

        M_r.at<char>(i) = target_intensity;

        // g channel
        num_pixels_in_a = cdf_g_hist.at<float>(i);

        for (int j = 0; j < histSize; j++){
            num_pixels_in_b = cdf_g_hist2.at<float>(j);
            if (num_pixels_in_b >= num_pixels_in_a){
                target_intensity = j;
                break;
            }
        }

        M_g.at<char>(i) = target_intensity;

        // b channel
        num_pixels_in_a = cdf_b_hist.at<float>(i);

        for (int j = 0; j < histSize; j++){
            num_pixels_in_b = cdf_b_hist2.at<float>(j);
            if (num_pixels_in_b >= num_pixels_in_a){
                target_intensity = j;
                break;
            }
        }

        M_b.at<char>(i) = target_intensity;

    }


    Mat new_r(A.size(), CV_8U);
    Mat new_g(A.size(), CV_8U);
    Mat new_b(A.size(), CV_8U);

    vector<Mat> new_face;


    LUT( A_split[0], M_b, new_b );
    LUT( A_split[1], M_g, new_g );
    LUT( A_split[2], M_r, new_r );
    
    new_face.push_back(new_b);
    new_face.push_back(new_g);
    new_face.push_back(new_r);

    Mat result;
    merge(new_face, result);

    imshow("B cropped ", B_cropped);
    imshow("A cropped ", A_cropped);
    
    imshow("color corrected", result);
    

    waitKey(0);

  /*
  printf("before calc prob denisty\n\n\n");

 cvtColor( A_cropped, A_cropped, CV_BGR2GRAY );

  /// Apply Histogram Equalization
  equalizeHist( A_cropped, A_cropped );
  imshow("histogram equalized on A", A_cropped);
  waitKey(0);

  Mat result;
  LUT( bgr_planes[0], b_hist, result );
  imshow("lut result", result);
  waitKey(0);

  printf("after calc prob density\n");
  */

  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
         Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
         Scalar( 255, 0, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
         Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
         Scalar( 0, 255, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
         Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
         Scalar( 0, 0, 255), 2, 8, 0  );
  }

  /// Display
  namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
  imshow("calcHist Demo", histImage );


}


void SwapApp::matchHistograms(){
    printf("matching histograms");

    Mat A_cropped = makeCroppedFaceMask(A, A_xform);
    Mat B_cropped = makeCroppedFaceMask(B, B_xform);
    A_cropped.convertTo(A_cropped, CV_8UC3, 1.0*255, 0);
    B_cropped.convertTo(B_cropped, CV_8UC3, 1.0*255, 0);

    vector<Mat> bgr_planes;
    split( A_cropped, bgr_planes );

    vector<Mat> bgr_planes2;
    split( B_cropped, bgr_planes2 );

    // full images
    Mat A2(A.size(), A.type());
    Mat B2(B.size(), B.type());

    A.convertTo(A2, CV_8UC3, 1.0*255, 0);
    B.convertTo(B2, CV_8UC3, 1.0*255, 0);

    vector<Mat> A_split;
    split( A2, A_split );

    vector<Mat> B_split;
    split( B2, B_split );


    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist, g_hist, r_hist;
    Mat b_hist2, g_hist2, r_hist2;

    /// Compute the histograms:
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

    calcHist( &bgr_planes2[0], 1, 0, Mat(), b_hist2, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes2[1], 1, 0, Mat(), g_hist2, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes2[2], 1, 0, Mat(), r_hist2, 1, &histSize, &histRange, uniform, accumulate );

    // todo: cdf 
    Mat cdf_r_hist(b_hist.size(), b_hist.type());
    Mat cdf_g_hist(b_hist.size(), b_hist.type());
    Mat cdf_b_hist(b_hist.size(), b_hist.type());
    Mat cdf_r_hist2(b_hist.size(), b_hist.type());
    Mat cdf_g_hist2(b_hist.size(), b_hist.type());
    Mat cdf_b_hist2(b_hist.size(), b_hist.type());

    for( int i = 0; i < histSize; i++ ){
        if (i == 0){
            cdf_r_hist.at<float>(i) = cvRound(r_hist.at<float>(i));
            cdf_g_hist.at<float>(i) = cvRound(g_hist.at<float>(i));
            cdf_b_hist.at<float>(i) = cvRound(b_hist.at<float>(i));
            cdf_r_hist2.at<float>(i) = cvRound(r_hist2.at<float>(i));
            cdf_g_hist2.at<float>(i) = cvRound(g_hist2.at<float>(i));
            cdf_b_hist2.at<float>(i) = cvRound(b_hist2.at<float>(i));
        }
        else {
            cdf_r_hist.at<float>(i) = cvRound(r_hist.at<float>(i)) + cvRound(cdf_r_hist.at<float>(i-1));
            cdf_g_hist.at<float>(i) = cvRound(g_hist.at<float>(i)) + cvRound(cdf_g_hist.at<float>(i-1));
            cdf_b_hist.at<float>(i) = cvRound(b_hist.at<float>(i)) + cvRound(cdf_b_hist.at<float>(i-1));
            cdf_r_hist2.at<float>(i) = cvRound(r_hist2.at<float>(i)) + cvRound(cdf_r_hist2.at<float>(i-1));
            cdf_g_hist2.at<float>(i) = cvRound(g_hist2.at<float>(i)) + cvRound(cdf_g_hist2.at<float>(i-1));
            cdf_b_hist2.at<float>(i) = cvRound(b_hist2.at<float>(i)) + cvRound(cdf_b_hist2.at<float>(i-1));
        }
    }

    Mat M_r(b_hist.size(), CV_8U);
    Mat M_g(b_hist.size(), CV_8U);
    Mat M_b(b_hist.size(), CV_8U);

    for( int i = 0; i < histSize; i++ ){
        // i is a pixel intensity value
        
        int target_intensity = 255;
        float num_pixels_in_a;
        float num_pixels_in_b;

        // r channel
        num_pixels_in_a = cdf_r_hist.at<float>(i);

        for (int j = 0; j < histSize; j++){
            num_pixels_in_b = cdf_r_hist2.at<float>(j);
            if (num_pixels_in_b > num_pixels_in_a){
                target_intensity = j;
                break;
            }
        }
        M_r.at<char>(i) = target_intensity;

        // g channel
        num_pixels_in_a = cdf_g_hist.at<float>(i);

        for (int j = 0; j < histSize; j++){
            num_pixels_in_b = cdf_g_hist2.at<float>(j);
            if (num_pixels_in_b > num_pixels_in_a){
                target_intensity = j;
                break;
            }
        }
        M_g.at<char>(i) = target_intensity;

        // b channel
        num_pixels_in_a = cdf_b_hist.at<float>(i);

        for (int j = 0; j < histSize; j++){
            num_pixels_in_b = cdf_b_hist2.at<float>(j);
            if (num_pixels_in_b > num_pixels_in_a){
                target_intensity = j;
                break;
            }
        }
        M_b.at<char>(i) = target_intensity;

    }

    // now the other way!
    Mat M_r2(b_hist.size(), CV_8U);
    Mat M_g2(b_hist.size(), CV_8U);
    Mat M_b2(b_hist.size(), CV_8U);

    for( int i = 0; i < histSize; i++ ){
        // i is a pixel intensity value
        
        int target_intensity = 255;
        float num_pixels_in_a;
        float num_pixels_in_b;

        // r channel
        num_pixels_in_a = cdf_r_hist2.at<float>(i);

        for (int j = 0; j < histSize; j++){
            num_pixels_in_b = cdf_r_hist.at<float>(j);
            if (num_pixels_in_b > num_pixels_in_a){
                target_intensity = j;
                break;
            }
        }
        M_r2.at<char>(i) = target_intensity;

        // g channel
        num_pixels_in_a = cdf_g_hist2.at<float>(i);

        for (int j = 0; j < histSize; j++){
            num_pixels_in_b = cdf_g_hist.at<float>(j);
            if (num_pixels_in_b > num_pixels_in_a){
                target_intensity = j;
                break;
            }
        }
        M_g2.at<char>(i) = target_intensity;

        // b channel
        num_pixels_in_a = cdf_b_hist2.at<float>(i);

        for (int j = 0; j < histSize; j++){
            num_pixels_in_b = cdf_b_hist.at<float>(j);
            if (num_pixels_in_b > num_pixels_in_a){
                target_intensity = j;
                break;
            }
        }
        M_b2.at<char>(i) = target_intensity;

    }


    // correct one direction
    Mat new_r(A.size(), CV_8U);
    Mat new_g(A.size(), CV_8U);
    Mat new_b(A.size(), CV_8U);

    vector<Mat> new_face;

    LUT( A_split[0], M_b, new_b );
    LUT( A_split[1], M_g, new_g );
    LUT( A_split[2], M_r, new_r );
    
    new_face.push_back(new_b);
    new_face.push_back(new_g);
    new_face.push_back(new_r);

    merge(new_face, A_color_corrected);

    //imshow("A color corrected", A_color_corrected);
    


    // correct other direction
    Mat new_r2(B.size(), CV_8U);
    Mat new_g2(B.size(), CV_8U);
    Mat new_b2(B.size(), CV_8U);

    vector<Mat> new_face2;

    LUT( B_split[0], M_b2, new_b2 );
    LUT( B_split[1], M_g2, new_g2 );
    LUT( B_split[2], M_r2, new_r2 );
    
    new_face2.push_back(new_b2);
    new_face2.push_back(new_g2);
    new_face2.push_back(new_r2);

    merge(new_face2, B_color_corrected);

    //imshow("B color corrected", B_color_corrected);


    A_color_corrected.convertTo(A_color_corrected, CV_64FC3, 1.0/255, 0);
    B_color_corrected.convertTo(B_color_corrected, CV_64FC3, 1.0/255, 0);
    //waitKey(0);
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

