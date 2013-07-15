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
        m.copyTo(m_border(Rect(borderSize, borderSize, h, w)));
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

        m.copyTo(m_border(Rect(borderSize, borderSize, h, w)));
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

    m.copyTo(m_border(Rect(borderSize, borderSize, w, h)));

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

Mat FaceLib::computeFlowAndWarp(Mat &face2, Mat &face1){
    Mat vx, vy, warp;
            
    Mat face1_border = addBorder(face1);
    Mat face2_border = addBorder(face2);

    CVOpticalFlow::findFlow(vx, vy, warp, face1_border, face2_border, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
    Mat warp2 = removeBorder(warp);
    if (1){
        imshow("the warped picture", warp);
        //imshow("the warped picture wo border", warp2);
        //imshow("flow", CVOpticalFlow::showFlow(vx, vy));
        //waitKey(0);
    }

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

