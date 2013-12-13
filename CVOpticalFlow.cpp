/* This code is from Ce Liu's Optical Flow code: http://people.csail.mit.edu/celiu/OpticalFlow/ */
/* This particular file is from Supasorn Aek Suwajanakorn with some additions by Kathleen Tuite. */

#include "CVOpticalFlow.h"

void CVOpticalFlow::writeFlow(char* filename, Mat &vx, Mat &vy){
    int rows = vx.rows;
    int cols = vx.cols;

    ofstream f;
    f.open(filename, ios::out | ios::binary);
    
    f.write(reinterpret_cast<char*>(&rows), sizeof(int));
    f.write(reinterpret_cast<char*>(&cols), sizeof(int));

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            double x = vx.at<double>(i, j);
            double y = vy.at<double>(i, j);

            f.write(reinterpret_cast<char*>(&x), sizeof(double));
            f.write(reinterpret_cast<char*>(&y), sizeof(double));
        }
    }
    
    f.close();
}

void CVOpticalFlow::readFlow(char* filename, Mat &vx, Mat &vy){
    ifstream f;
    f.open(filename, ios::in | ios::binary);
    int rows, cols;
    f.read(reinterpret_cast<char*>(&rows), sizeof(int));
    f.read(reinterpret_cast<char*>(&cols), sizeof(int));

    vx = Mat(rows, cols, CV_64F);
    vy = Mat(rows, cols, CV_64F);

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            double x, y;

            f.read(reinterpret_cast<char*>(&x), sizeof(double));
            f.read(reinterpret_cast<char*>(&y), sizeof(double));
            vx.at<double>(i, j) = x;
            vy.at<double>(i, j) = y;
        }
    }
    
    f.close();
}

Mat CVOpticalFlow::showFlow(Mat &vxi, Mat &vyi) {
  Mat vx = vxi.clone();
  Mat vy = vyi.clone();
  int thresh = 1e9;
  Mat out = Mat(vx.size(), CV_64FC3);
  double maxrad = -1;
  for (int i=0; i<out.rows; i++) {
    for (int j=0; j<out.cols; j++) {
      if (fabs(vx.at<double>(i, j)) > thresh) 
        vx.at<double>(i, j) = 0;
      if (fabs(vy.at<double>(i, j)) > thresh) 
        vy.at<double>(i, j) = 0;
      double rad = vx.at<double>(i, j) * vx.at<double>(i, j) + vy.at<double>(i, j) * vy.at<double>(i, j);
      maxrad = max(maxrad, rad);
    }
  }
  maxrad = sqrt(maxrad);
  for (int i=0; i<out.rows; i++) {
    for (int j=0; j<out.cols; j++) {
      vx.at<double>(i, j) /= out.cols;
      vy.at<double>(i, j) /= out.rows;
      //vx.at<double>(i, j) /= maxrad;
      //vy.at<double>(i, j) /= maxrad;
      out.at<Vec3d>(i, j)[0] = vx.at<double>(i, j) * 0.5 + 0.5;
      out.at<Vec3d>(i, j)[1] = vy.at<double>(i, j) * 0.5 + 0.5;
      out.at<Vec3d>(i, j)[2] = 0;
    }
  }
  return out;
}
inline void CVOpticalFlow::clip(int &a, int lo, int hi) {
  a = (a < lo) ? lo : (a>=hi ? hi-1: a);
}

void CVOpticalFlow::bilinear(double *out, Mat &im, double r, double c, int channels) {
  int r0 = r, r1 = r+1;
  int c0 = c, c1 = c+1;
  clip(r0, 0, im.rows);
  clip(r1, 0, im.rows);
  clip(c0, 0, im.cols);
  clip(c1, 0, im.cols);

  double tr = r - r0;
  double tc = c - c0;
  for (int i=0; i<channels; i++) {
    double ptr00 = im.at<Vec3d>(r0, c0)[i];
    double ptr01 = im.at<Vec3d>(r0, c1)[i];
    double ptr10 = im.at<Vec3d>(r1, c0)[i];
    double ptr11 = im.at<Vec3d>(r1, c1)[i];
    out[i] = ((1-tr) * (tc * ptr01 + (1-tc) * ptr00) + tr * (tc * ptr11 + (1-tc) * ptr10));
  }
}

void CVOpticalFlow::bilinearGray(double *out, Mat &im, double r, double c, int channels) {
  int r0 = r, r1 = r+1;
  int c0 = c, c1 = c+1;
  clip(r0, 0, im.rows);
  clip(r1, 0, im.rows);
  clip(c0, 0, im.cols);
  clip(c1, 0, im.cols);

  double tr = r - r0;
  double tc = c - c0;
  double ptr00 = im.at<double>(r0, c0);
  double ptr01 = im.at<double>(r0, c1);
  double ptr10 = im.at<double>(r1, c0);
  double ptr11 = im.at<double>(r1, c1);
  out[0] = ((1-tr) * (tc * ptr01 + (1-tc) * ptr00) + tr * (tc * ptr11 + (1-tc) * ptr10));
  
}

void CVOpticalFlow::warp(Mat &out, Mat &im, Mat &vx, Mat &vy) {
  if(im.type()!=CV_64FC3) {
    printf("Error: unsupported typed. Required CV_64FC3");
    return ;
  }
  out = Mat(im.size(), CV_64FC3);
  for (int i=0; i<out.rows; i++) {
    for (int j=0; j<out.cols; j++) {
      bilinear(&out.at<Vec3d>(i, j)[0], im, i+vy.at<double>(i, j), j+vx.at<double>(i, j), 3);
    }
  }
}

void CVOpticalFlow::warpGray(Mat &out, Mat &im, Mat &vx, Mat &vy) {
  printf("warp gray!\n");
  if(im.type()!=CV_64F) {
    printf("Error: unsupported typed. Required CV_64F");
    return ;
  }
  out = Mat(im.size(), CV_64F);
  for (int i=0; i<out.rows; i++) {
    for (int j=0; j<out.cols; j++) {
      bilinearGray(&out.at<double>(i, j), im, i+vy.at<double>(i, j), j+vx.at<double>(i, j), 1);
    }
  }
}

void CVOpticalFlow::warpInterpolation(Mat &out, Mat &im1, Mat &im2, Mat &vx, Mat &vy, float dt) {
  if(im1.type()!=CV_64FC3) {
    printf("Error: unsupported typed. Required CV_64FC3");
    return ;
  }
  printf("warp interpolation, dt: %f\n", dt);

  out = Mat(im1.size(), CV_64FC3);
  for (int i=0; i<out.rows; i++) {
    for (int j=0; j<out.cols; j++) {
      double a[3], b[3];
      bilinear(a, im2, i+(dt)*vy.at<double>(i, j), j+(dt)*vx.at<double>(i, j), 3);
      bilinear(b, im1, i+(dt-1)*vy.at<double>(i, j), j+(dt-1)*vx.at<double>(i, j), 3);
      for (int c = 0; c < 3; c++){
        out.at<Vec3d>(i, j)[c] = (1-dt)*a[c] + (dt)*b[c];
      }
    }
  }
}

void CVOpticalFlow::warpInterpolationGray(Mat &out, Mat &im1, Mat &im2, Mat &vx, Mat &vy, float dt) {
  if(im1.type()!=CV_64F) {
    printf("Error: unsupported typed. Required CV_64F");
    return ;
  }
  printf("warp interpolation, dt: %f\n", dt);

  out = Mat(im1.size(), CV_64F);
  for (int i=0; i<out.rows; i++) {
    for (int j=0; j<out.cols; j++) {
      double a, b;
      bilinearGray(&a, im2, i+(dt)*vy.at<double>(i, j), j+(dt)*vx.at<double>(i, j), 1);
      bilinearGray(&b, im1, i+(dt-1)*vy.at<double>(i, j), j+(dt-1)*vx.at<double>(i, j), 1);
 
        out.at<Vec3d>(i, j) = (1-dt)*a + (dt)*b;
      
    }
  }
}

void CVOpticalFlow::findFlow(Mat &vx, Mat &vy, Mat &warp, Mat &im1, Mat &im2, double alpha, double ratio, int minWidth, int nOuterFPIterations, int nInnerFPIterations, int nSORIterations) {
  DImage iim1, iim2;
  DImage ivx, ivy, iwarp;
  iim1.fromMat(im1);
  iim2.fromMat(im2);
  OpticalFlow::Coarse2FineFlow(ivx, ivy, iwarp, iim1, iim2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
  vx = Mat(im1.size(), CV_64F);
  vy = Mat(im2.size(), CV_64F);
  memcpy(vx.data, ivx.pData, sizeof(double) * vx.rows * vx.cols);
  memcpy(vy.data, ivy.pData, sizeof(double) * vy.rows * vy.cols);
  iwarp.toMat(warp);
}


void CVOpticalFlow::compositeFlow(Mat &ax, Mat &ay, Mat &bx, Mat &by, Mat &outx, Mat &outy) {
  for (int i=0; i<ax.rows; i++) {
    for (int j=0; j<ax.cols; j++) {
      double flow_ax = ax.at<double>(i, j);
      double flow_ay = ay.at<double>(i, j);

      double new_flow_x = flow_ax + bilinearFlow(bx, i + flow_ax, j + flow_ay);
      double new_flow_y = flow_ay + bilinearFlow(by, i + flow_ax, j + flow_ay);

      outx.at<double>(i, j) = new_flow_x;
      outy.at<double>(i, j) = new_flow_y;

    }
  }  
}

double CVOpticalFlow::bilinearFlow(Mat &im, double r, double c) {
  int r0 = r, r1 = r+1;
  int c0 = c, c1 = c+1;
  clip(r0, 0, im.rows);
  clip(r1, 0, im.rows);
  clip(c0, 0, im.cols);
  clip(c1, 0, im.cols);


  double tr = r - r0;
  double tc = c - c0;
  double ptr00 = im.at<double>(r0, c0);
  double ptr01 = im.at<double>(r0, c1);
  double ptr10 = im.at<double>(r1, c0);
  double ptr11 = im.at<double>(r1, c1);
  double out = ((1-tr) * (tc * ptr01 + (1-tc) * ptr00) + tr * (tc * ptr11 + (1-tc) * ptr10));
  
  return out;
}
