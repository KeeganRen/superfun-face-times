#include "CVOpticalFlow.h"

#include "FaceLib.h"

void FaceLib::gslVecToMat(gsl_vector *orig, Mat &im, int d, int w, int h){
    gsl_vector* vec = gsl_vector_calloc(orig->size);
    gsl_vector_memcpy(vec, orig);

    if (d == 1){
        Mat m(w, h, CV_64F, vec->data);
        im = m;
    }
    if (d == 3){
        Mat m(w, h, CV_64FC3, vec->data);
        im = m;
    }

    // this will be a mighty fine memory leak some day!
    //gsl_vector_free(vec);
}

void FaceLib::matToGslVec(Mat &im, gsl_vector *vec, int d, int w, int h){
    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            if (d == 1){
                int idx = i*h + j;
                double val = im.at<double>(i, j);
                gsl_vector_set(vec, idx, val);
            }
            else if (d == 3){
                int idx = i*h + j;
                Vec3d val = im.at<Vec3d>(i, j);
                for (int k = 0; k < 3; k++){
                    gsl_vector_set(vec, idx*3 + k, val[k]);
                }
            }
        }
    }
}

void FaceLib::gslVecToMatWithBorder(gsl_vector *orig, Mat &im, int d, int w, int h){
    gsl_vector* vec = gsl_vector_calloc(orig->size);
    gsl_vector_memcpy(vec, orig);

    if (d == 1){
        Mat m(w, h, CV_64F, vec->data);
        Mat m_border(w + borderSize*2, h + borderSize*2, CV_64F, -1);
        //m.copyTo(m_border(Rect(borderSize, borderSize, h, w)));
        Mat dst_roi = m_border(Rect(borderSize, borderSize, h, w));
        m.copyTo(dst_roi);
        im = m_border;
    }
    if (d == 3){
        Mat m(w, h, CV_64FC3, vec->data);
        Mat m_border(w + borderSize*2, h + borderSize*2, CV_64FC3);
        
        for (int i = 0; i < m_border.rows; i++){
            for (int j = 0; j < m_border.cols; j++){
                m_border.at<Vec3d>(i, j) = Vec3d(0, 0, 0);
            }
        }

        //m.copyTo(m_border(Rect(borderSize, borderSize, h, w)));
        Mat dst_roi = m_border(Rect(borderSize, borderSize, h, w));
        m.copyTo(dst_roi);
        im = m_border;
    }
}

void FaceLib::matToGslVecWithBorder(Mat &im, gsl_vector *vec, int d, int w, int h){
    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            if (d == 1){
                int idx = i*h + j;
                double val = im.at<double>(i + borderSize, j + borderSize);
                gsl_vector_set(vec, idx, val);
            }
            else if (d == 3){
                int idx = i*h + j;
                Vec3d val = im.at<Vec3d>(i + borderSize, j + borderSize);
                for (int k = 0; k < 3; k++){
                    gsl_vector_set(vec, idx*3 + k, val[k]);
                }
            }
        }
    }
}

Mat FaceLib::addBorder(Mat im){
    int w = im.cols;
    int h = im.rows;

    Mat m_border(h + borderSize*2, w + borderSize*2, CV_64FC3);
    Mat m(im);

    for (int i = 0; i < m_border.rows; i++){
        for (int j = 0; j < m_border.cols; j++){
            m_border.at<Vec3d>(i, j) = Vec3d(0, 0, 0);
        }
    }

    //m.copyTo(m_border(Rect(borderSize, borderSize, w, h)));
    Mat dst_roi = m_border(Rect(borderSize, borderSize, w, h));
    m.copyTo(dst_roi);

    return m_border;
}

Mat FaceLib::removeBorder(Mat im){
    int w = im.cols - borderSize*2;
    int h = im.rows - borderSize*2;
    printf("removing border w: %d h: %d\n", w, h);

    Mat rightFormat;
    Rect cropROI(borderSize, borderSize, w, h);
    rightFormat = im(cropROI);
    if (rightFormat.type() == CV_8UC3){
        rightFormat.convertTo(rightFormat, CV_64FC3, 1.0/255, 0);
    }
    return rightFormat;
}

Mat FaceLib::removeFlowBorder(Mat x){
    int w = x.cols - borderSize*2;
    int h = x.rows - borderSize*2;
    printf("removing border from FLOW field w: %d h: %d\n", w, h);

    Mat cropped;
    Rect cropROI(borderSize, borderSize, w, h);
    cropped = x(cropROI);
    return cropped;
}

void FaceLib::saveAs(char* filename, Mat m){
    printf("[saveAs] saving image to file: %s\n", filename);
    if (m.type() == CV_8UC3){
        imwrite(filename, m);
    }
    else {
        Mat rightFormat;
        m.convertTo(rightFormat, CV_8UC3, 1.0*255, 0);
        imwrite(filename, rightFormat);
    }
}

Mat FaceLib::showFlow(Mat &vx, Mat &vy){
    return CVOpticalFlow::showFlow(vx, vy);
}

void FaceLib::compositeFlow(Mat &vx_ab, Mat &vy_ab, Mat &vx_cd, Mat &vy_cd, Mat &out_x, Mat &out_y){
    out_x = Mat(vx_ab.size(), CV_64F);
    out_y = Mat(vy_ab.size(), CV_64F);
    CVOpticalFlow::compositeFlow(vx_ab, vy_ab, vx_cd, vy_cd, out_x, out_y);
}

Mat FaceLib::computeFlowAndWarp(Mat &face2, Mat &face1){
    printf("[computeFlowAndWarp]\n");
    Mat vx, vy, warp;
            
    Mat face1_border = addBorder(face1);
    Mat face2_border = addBorder(face2);

    CVOpticalFlow::findFlow(vx, vy, warp, face1_border, face2_border, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
    Mat warp2 = removeBorder(warp);
    
    return warp2;
}

Mat FaceLib::computeFlowAndApplyFlow(Mat &face2, Mat &face1, Mat &faceToWarp){
    Mat vx, vy, warp;
            
    Mat face1_border = addBorder(face1);
    Mat face2_border = addBorder(face2);
    Mat faceToWarp_border = addBorder(faceToWarp);

    CVOpticalFlow::findFlow(vx, vy, warp, face1_border, face2_border, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);

    Mat output;
    CVOpticalFlow::warp(output, faceToWarp_border, vx, vy);
    output = removeBorder(output);
    return output;
}

void FaceLib::computeFlowAndStore(Mat &face2, Mat &face1, Mat &vx, Mat &vy){
    printf("[computeFlowAndStore]\n");
    Mat warp;
            
    Mat face1_border = addBorder(face1);
    Mat face2_border = addBorder(face2);

    CVOpticalFlow::findFlow(vx, vy, warp, face1_border, face2_border, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
    
    vx = removeFlowBorder(vx);
    vy = removeFlowBorder(vy);
}

void FaceLib::loadFiducialPoints(string file, std::vector<cv::Point2f> &point_vector){    
    point_vector.clear();

    FILE *f1 = fopen(file.c_str(), "r");
    
    if (!f1){
        printf("couldn't open fiducial point file [%s]\n", file.c_str());
        exit(0);
    }
    
    double x,y;
    while (fscanf(f1, "%lf %lf", &x, &y) != EOF){
        point_vector.push_back(Point2f(x,y));
    }
    
    printf("Size of points vector loaded from [%s]: %d\n", file.c_str(), int(point_vector.size()));
}

void FaceLib::computeTransform(Mat &frame, vector<Point2f> detectedPoints, vector<Point2f> templatePoints, Mat &xform){
    printf("[computeTransform] detected pts %d template pts %d\n", detectedPoints.size(), templatePoints.size());

    vector<Point2f> f, c;

    f.push_back(middle(detectedPoints[0], detectedPoints[1]));
    f.push_back(middle(detectedPoints[6], detectedPoints[7]));
    f.push_back(middle(detectedPoints[2], detectedPoints[3]));

    c.push_back(middle(templatePoints[0], templatePoints[1]));
    c.push_back(middle(templatePoints[6], templatePoints[7]));
    c.push_back(middle(templatePoints[2], templatePoints[3]));
  
    f[2] = perpendicularPoint(f);
    c[2] = perpendicularPoint(c);

    xform = Mat( 2, 3, CV_32FC1 );
    xform = getAffineTransform( f, c );

    /*
    Mat warped = Mat::zeros(192, 139, frame.type() );
    Mat masked = Mat::zeros(192, 139, frame.type() );
    warpAffine(frame, warped, transform, warped.size() );
    warped.copyTo(masked, maskImage);

    imshow("masked", masked);
    waitKey(0);
    */
}

Point2f FaceLib::middle(Point2f a, Point2f b){
    Point2f c = Point2f((a.x + b.x)/2.0, (a.y + b.y)/2.0);
    return c;
}

Point2f FaceLib::perpendicularPoint(vector<Point2f> f){
    // line between eye points ax + by + c = 0
    double a = (f[1].y - f[0].y)/(f[1].x - f[0].x);
    double b = -1;
    double c = -b*f[1].y - a*f[1].x;
    double dist = abs(a * f[2].x + b * f[2].y + c)/sqrt(a*a + b*b);

    Point2f eye_mid = middle(f[0], f[1]);
    Point2d new_pt = eye_mid - Point2f(a,b)*(1.0/sqrt(a*a + b*b))*dist;

    return new_pt;
}

#define MAX_ITERATION 300

inline double clampValue (double val) {
  if (val > 1) return 1;
  else if (val < 0) return 0;
  else return (double)val;
}

Mat FaceLib::montage (Mat &srcImage, Mat &destImage, Mat &maskImage) {
  int w = srcImage.cols;
  int h = srcImage.rows;
  
  if (w != maskImage.cols || h != maskImage.rows) {
    cerr << "mask size doesn't match src size" << endl;
    return destImage;
  }

  vector<Point2i> destPoints;
  vector<double> constants;
  
  int size=0;
  for (int y=0; y<h; y++) {
    for (int x=0; x<w; x++) {
            
      uchar c = maskImage.at<Vec3b>(y,x)[0];
      
      if (c > 0) {
        //printf("input (%d,%d)\n", x, y);
        destPoints.push_back(Point2i(x,y));
            
        int constant[3] = {0};
        double sum1[3]={0};
        double sum2[3]={0};
        // right
        if (x < srcImage.cols-1) {
          for (int i=0; i<3; i++) {
            double val1 = destImage.at<Vec3d>(y,x+1)[i] - destImage.at<Vec3d>(y,x)[i];
            double val2 = srcImage.at<Vec3d>(y,x+1)[i] - srcImage.at<Vec3d>(y,x)[i];
            sum1[i]+=val1;
            sum2[i]+=val2;
          }
        }
        // left
        if (x > 0) {
          for (int i=0; i<3; i++) {
            double val1 = destImage.at<Vec3d>(y,x-1)[i] - destImage.at<Vec3d>(y,x)[i];
            double val2 = srcImage.at<Vec3d>(y,x-1)[i] - srcImage.at<Vec3d>(y,x)[i];
            sum1[i]+=val1;
            sum2[i]+=val2;
          }
        }
        // top
        if (y > 0) {
          for (int i=0; i<3; i++) {
            double val1 = destImage.at<Vec3d>(y-1,x)[i] - destImage.at<Vec3d>(y,x)[i];
            double val2 = srcImage.at<Vec3d>(y-1,x)[i] - srcImage.at<Vec3d>(y,x)[i];
            sum1[i]+=val1;
            sum2[i]+=val2;
          }
        }
        // bottom
        if (y < srcImage.rows-1) {
          for (int i=0; i<3; i++) {
            double val1 = destImage.at<Vec3d>(y+1,x)[i] - destImage.at<Vec3d>(y,x)[i];
            double val2 = srcImage.at<Vec3d>(y+1,x)[i] - srcImage.at<Vec3d>(y,x)[i];
            sum1[i]+=val1;
            sum2[i]+=val2;
          }
        }
        for (int i=0; i<3; i++) {
          constants.push_back(sum2[i]);
        }
      }
      size++;
    } 
  }
  
  printf("destPoints size=%d\n", (int)destPoints.size());
  printf("constants size=%d\n", (int)constants.size());

  Mat final(destImage.rows, destImage.cols, CV_64FC3);
  
  for (int x=0; x<destImage.cols; x++) 
    for (int y=0; y<destImage.rows; y++) 
        final.at<Vec3d>(y,x) = destImage.at<Vec3d>(y,x);
    

  // ヤコビ法で連立一次方程式を解く
  // "I solve a system of linear equations in the Jacobi method"
  for (int loop=0; loop<MAX_ITERATION; loop++) {
    int n = destPoints.size();
    for (int p=0; p<n; p++) {
      //int destIndex = destPoints[p];
      int y = destPoints[p].y;
      int x = destPoints[p].x;
      //printf("check (%d,%d)\n", x, y);
      double values[3] = {0};
      // right
      int count = 0;
      if (x < destImage.cols-1) {
        count++;
        for (int i=0; i<3; i++) {
          values[i] += final.at<Vec3d>(y, x+1)[i];
        }
      }
      // left
      if (x > 0) {
        count++;
        for (int i=0; i<3; i++) {
          values[i] += final.at<Vec3d>(y, x-1)[i];
        }
      }
      // top
      if (y > 0) {
        count++;
        for (int i=0; i<3; i++) {
          values[i] += final.at<Vec3d>(y-1, x)[i];
        }
      }
      // bottom
      if (y < destImage.rows-1) {
        count++;
        for (int i=0; i<3; i++) {
          values[i] += final.at<Vec3d>(y+1, x)[i];
        }
      }

      // 更新
      for (int j=0; j<3; j++) {
        double newval = (values[j] - constants[p*3+j]) / count;
        double oldval = final.at<Vec3d>(y,x)[j];
        final.at<Vec3d>(y,x)[j] = newval;
      }
    }
  }
  
  Mat result = destImage.clone();
  //Mat(destImage.rows, destImage.cols, destImage.type());

    int n = destPoints.size();
    for (int p=0; p<n; p++) {
      int y = destPoints[p].y;
      int x = destPoints[p].x;
      for (int i=0; i<3; i++) {
        result.at<Vec3d>(y,x)[i] = clampValue(final.at<Vec3d>(y,x)[i]);
      }
    }
  

  return result;
}

gsl_vector* FaceLib::projectNewFace(int num_pixels, int num_eigenfaces, gsl_vector *new_face, gsl_vector *meanface, gsl_matrix *eigenfaces){
    printf("[projectNewFace] projecting new face onto lower dimensional space! num_pixels: %d\n", num_pixels);
    gsl_vector *face_minus_mean = gsl_vector_calloc(num_pixels);
    gsl_vector_memcpy(face_minus_mean, new_face);
    gsl_vector_sub(face_minus_mean, meanface);

    gsl_vector *eigenvalues = gsl_vector_calloc(num_eigenfaces);
    for (int i = 0; i < 4; i++){
        gsl_vector_view col = gsl_matrix_column(eigenfaces, i);
        double norm = gsl_blas_dnrm2 (&col.vector);
        gsl_vector_scale(&col.vector, 1.0/norm);

        double val;
        gsl_blas_ddot(&col.vector, face_minus_mean, &val);

        printf("dot value: %f, norm of vec: %f\n", val, norm);
        //val /= norm/255.0;

        gsl_vector_set(eigenvalues, i, val);
    }

    gsl_blas_dgemv(CblasNoTrans, 1, eigenfaces, eigenvalues, 0, face_minus_mean);

    gsl_vector_add(face_minus_mean, meanface);

    return face_minus_mean;
}
