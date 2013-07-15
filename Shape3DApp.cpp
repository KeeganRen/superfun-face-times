/* Shape3DApp.cpp */

#include "Shape3DApp.h"
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
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

#include <Eigen/Sparse>
#include <Eigen/Dense>
using namespace Eigen;
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

using namespace cv;

void error(char *msg) {
    perror(msg);
    exit(1);
}

void PrintUsage() 
{
    printf("Usage:  \n" \
            "   \t--input <file> [file with one image per line]\n" \
            "   \t--templateMesh <file> [template 3D point cloud]\n" \
            "   \t--output <file> [json 3d object to save warped face to]\n" \
            "   \t--ply <file> [ply file to save warped face to]\n" \
            "   \t--visualize [if you want to see images of the progress]\n" \
            "\n");
}

void Shape3DApp::processOptions(int argc, char **argv){
    while (1){
        static struct option long_options[] = {
            {"help",        0, 0, 'h'},
            {"templateMesh",    1, 0, 402},
            {"output",          1, 0, 403},
            {"input",           1, 0, 406},
            {"visualize",       0, 0, 407},
            {"ply",             1, 0, 408},
            {0,0,0,0} 
        };

        int option_index;;
        int c = getopt_long(argc, argv, "f:do:a:i:x",
                long_options, &option_index);

        if (c == -1)
            break;

        switch (c) {
            case 'h':
                PrintUsage();
                exit(0);
                break;
                
            case 402:
                sprintf(templateMeshFile, "%s", optarg);
                break;

            case 403:
                sprintf(outFaceFile, "%s", optarg);
                break;

            case 406:
                sprintf(listFile, "%s", optarg);
                break;

            case 407:
                visualize = true;
                printf("visualize = true\n");
                break;

            case 408:
                outFacePly = optarg;
                break;

            default: 
                printf("Unrecognized option %d\n", c);
                break;
        }
    }
}

void Shape3DApp::init(){
    printf("[init] Running program %s\n", argv[0]);
    useList = false;
    visualize = false;
    outFacePly = NULL;

    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }
    
    processOptions(argc, argv);

    loadTemplateFiles();
    loadImageList();


    setupShapeStuff();
    populateTemplateMatrix();


    /* some tests */
    //testIntegrability();
    //testTemplate();
    
    populateImageMatrix();
    shapeStuff();

}

void Shape3DApp::testIntegrability(){
    printf("[testIntegrability] TESTING... recover depth from template model's normals\n");
    for (int i = 0; i < num_points; i++){
        gsl_matrix_set(m_gsl_final_result, 1, i, templateNormals[i].x);
        gsl_matrix_set(m_gsl_final_result, 2, i, templateNormals[i].y);
        gsl_matrix_set(m_gsl_final_result, 3, i, templateNormals[i].z);
    }
    recoverDepth();
}

void Shape3DApp::testTemplate(){
    printf("[testTemplate]\n");

    Mat templateImage = Mat::zeros(192, 139, CV_8UC3);
    for (int i = 0; i < templateMesh.size(); i++){
        //Point3d color = templateColors[i];
        Point2f circle_pos = Point2f(templateMesh[i].x,templateMesh[i].y);
        Point3f normal = templateNormals[i];
        Point3d color;
        float f = 100.0;
        color.x = normal.x * f;
        color.y = normal.y * f;
        color.z = normal.z * f;
        circle(templateImage, circle_pos, .7, CV_RGB(color.x, color.y, color.z), 1, 8, 0);
    }
    //Rect cropROI(128, 80, 250, 310);
    //templateImage = templateImage(cropROI);

    for (int i = 0; i < templatePoints.size(); i++){
        Point2f circle_pos = Point2f(templatePoints[i].x, templatePoints[i].y);
        circle(templateImage, circle_pos, 4, CV_RGB(255, 255, 0), 2, 8, 0);
        circle(templateImage, canonicalPoints[i], 4, CV_RGB(255, 0, 255), 2, 8, 0);
    }

    imshow("template mesh warped to canonical points", templateImage);
    imwrite("templateimage.jpg", templateImage);
    cvWaitKey(0);
}

void Shape3DApp::loadImageList(){
    printf("[loadImageList] loading from file %s\n", listFile);
    FILE *file = fopen ( listFile, "r" );
    if ( file != NULL ) {
        char image_path[256];
        while( fscanf(file, "%s\n", image_path) > 0 ) {
            imageFiles.push_back(image_path);
        }
        fclose (file);
    }
    else {
        perror (facePointsFile);
    }
    printf("[loadImageList] found %d image filenames\n", (int)imageFiles.size());
    num_images = (int)imageFiles.size();

    Mat m = imread(imageFiles[0]);
    w = m.cols;
    h = m.rows;
    printf("Image size: %d by %d\n", w, h);
}

void Shape3DApp::setupShapeStuff(){
    printf(" --- num_images = %d and num_points = %d ---\n", num_images, num_points);

    m_gsl_model = gsl_matrix_calloc(4, num_points);
    m_gsl_images = gsl_matrix_calloc(num_points, num_images);
    m_gsl_images_orig = gsl_matrix_calloc(num_points, num_images);
    m_gsl_s = gsl_matrix_calloc(4, num_points);
    m_gsl_final_result = gsl_matrix_alloc(4,num_points);
}


void Shape3DApp::loadTemplateFiles(){
    FILE *file;
    printf("[loadTemplateFiles] loading template face mesh from file %s\n", templateMeshFile);
    file = fopen ( templateMeshFile, "r" );
    if ( file != NULL ) {
        float x, y, z, nx, ny, nz;
        int r, g, b, a;
        while( fscanf(file, "%f %f %f %f %f %f %d %d %d %d \n", &x, &y, &z, &nx, &ny, &nz, &r, &g, &b, &a) > 0 ) {
        //while( fscanf(file, "%f %f %f %f %f %f\n", &x, &y, &z, &nx, &ny, &nz) > 0 ) {
            templateMesh.push_back(Point3f(x,y,z));
            templateNormals.push_back(Point3f(nx,ny,nz));

            //r = 155;
            //g = 155;
            //b = 155;
            templateColors.push_back(Point3f(r,g,b));
        }
        fclose (file);
    }
    else {
        perror (templateMeshFile);
    }
    printf("[loadTemplateFiles] yay, {{%d}} points loaded into templateMesh\n", (int)templateMesh.size());
    num_points = (int)templateMesh.size();
}

void Shape3DApp::clip(int &a, int lo, int hi) {
    a = (a < lo) ? lo : (a>=hi ? hi-1: a);
}

void Shape3DApp::bilinear(double *out, Mat im, float c, float r){
    int r0 = r, r1 = r+1;
    int c0 = c, c1 = c+1;
    clip(r0, 0, im.rows);
    clip(r1, 0, im.rows);
    clip(c0, 0, im.cols);
    clip(c1, 0, im.cols);

    double tr = r - r0;
    double tc = c - c0;
    for (int i=0; i<3; i++) {
        double ptr00 = im.at<Vec3d>(r0, c0)[i];
        double ptr01 = im.at<Vec3d>(r0, c1)[i];
        double ptr10 = im.at<Vec3d>(r1, c0)[i];
        double ptr11 = im.at<Vec3d>(r1, c1)[i];
        out[i] = ((1-tr) * (tc * ptr01 + (1-tc) * ptr00) + tr * (tc * ptr11 + (1-tc) * ptr10));

    }
}

void Shape3DApp::populateTemplateMatrix(){
    printf("[populateTemplateMatrix]\n");
    for (int i = 0; i < num_points; i++){
        Point3f c = templateColors[i];
        float lum = (0.2126*c.x) + (0.7152*c.y) + (0.0722*c.z);
        lum/=255.0;
        Point3f n = templateNormals[i];
        gsl_matrix_set(m_gsl_model, 0, i, lum);
        gsl_matrix_set(m_gsl_model, 1, i, n.x*lum);
        gsl_matrix_set(m_gsl_model, 2, i, n.y*lum);
        gsl_matrix_set(m_gsl_model, 3, i, n.z*lum);
    }

    if (visualize){
        Mat img = Mat::zeros(h, w*4, CV_32F);
        for (int i = 0; i < num_points; i++){
            Point2f pt = Point2f(templateMesh[i].x,  templateMesh[i].y);
            Point3i color = templateColors[i];
            Point3f n = templateNormals[i];
            double nx = gsl_matrix_get(m_gsl_model, 1, i);
            double ny = gsl_matrix_get(m_gsl_model, 2, i);
            double nz = gsl_matrix_get(m_gsl_model, 3, i);
            circle(img, pt, 1, CV_RGB(color.z/255.0, color.y/255.0, color.x/255.0), 0, 8, 0);
            circle(img, Point2f(pt.x + 1*w, pt.y), 1, CV_RGB(nx, nx, nx), 0, 8, 0);
            circle(img, Point2f(pt.x + 2*w, pt.y), 1, CV_RGB(ny, ny, ny), 0, 8, 0);
            circle(img, Point2f(pt.x + 3*w, pt.y), 1, CV_RGB(nz, nz, nz), 0, 8, 0);
        }

        //resize(img, img, Size(1000, 250));
        imshow("[albedo] [albedo*nx] [albedo*ny] [albedo*nz]", img);
        //waitKey(0);

        Mat img2 = Mat::zeros(h, w*4, CV_32F);
        for (int i = 0; i < num_points; i++){
            Point2f pt = Point2f(templateMesh[i].x, templateMesh[i].y);
            double lum = gsl_matrix_get(m_gsl_model, 0, i);
            double nx = gsl_matrix_get(m_gsl_model, 1, i);
            double ny = gsl_matrix_get(m_gsl_model, 2, i);
            double nz = gsl_matrix_get(m_gsl_model, 3, i);
            circle(img2, pt, 1, lum, 0, 8, 0);
            circle(img2, Point2f(pt.x + 1*w, pt.y), 1, nx/lum, 0, 8, 0);
            circle(img2, Point2f(pt.x + 2*w, pt.y), 1, ny/lum, 0, 8, 0);
            circle(img2, Point2f(pt.x + 3*w, pt.y), 1, nz/lum, 0, 8, 0);
        }

        //resize(img2, img2, Size(1000, 250));
        imshow("[albedo] [nx] [ny] [nz]", img2);
    }   
}

void Shape3DApp::populateImageMatrix(){
    printf("[populateImageMatrix]");
    for (int i = 0; i < num_images; i++){
        printf("[populateImageMatrix] image %d (%s)\n", i, imageFiles[i].c_str());
        // load the image
        Mat im = imread(imageFiles[i].c_str(), CV_LOAD_IMAGE_COLOR);
        im.convertTo(im, CV_64FC3, 1.0/255, 0);

        for (int j = 0; j < templateMesh.size(); j++){
            Vec3d c = im.at<Vec3d>(templateMesh[j].y, templateMesh[j].x);
            float lum = (0.2126*c[2]) + (0.7152*c[1]) + (0.0722*c[0]);
            gsl_matrix_set(m_gsl_images, j, i, lum);
        }


        if (visualize){
            Mat drawable = Mat::zeros(h, w, CV_32F);
            for (int j = 0; j < templateMesh.size(); j++){
                double c = gsl_matrix_get(m_gsl_images, j, i);
                circle(drawable, Point2f(templateMesh[j].x,templateMesh[j].y), 1, CV_RGB(c, c, c), 0, 8, 0);
            }

            imshow("loaded face", drawable);
            waitKey(100);
        }
    }   

    gsl_matrix_memcpy(m_gsl_images_orig, m_gsl_images);
    printf("[populateImageMatrix] DONE!\n");
}

void Shape3DApp::shapeStuff(){
    printf("[shapeStuff]...\n");

    if (0){
        // so ive got a fatty matrix... i want to get the mean of it and do SVD and stuff
        gsl_vector *mean = gsl_vector_calloc(num_points);
        for (int i = 0; i < num_images; i++){
            gsl_vector_view col = gsl_matrix_column(m_gsl_images, i);
            gsl_vector_add(mean, &col.vector);
        }
        gsl_vector_scale(mean, 1.0/num_images);
        for (int i = 0; i < num_images; i++){
            gsl_vector_view col = gsl_matrix_column(m_gsl_images, i);
            gsl_vector_sub(&col.vector, mean);
        }
    }

    gsl_vector *S = gsl_vector_calloc(num_images);
    gsl_matrix *V = gsl_matrix_calloc(num_images, num_images);
    gsl_vector *work = gsl_vector_calloc(num_images);

    printf("[shapeStuff] computing SVD!\n");

    int res = gsl_linalg_SV_decomp(m_gsl_images, V, S, work);

    printf("[shapeStuff] SVD computed, result: %d\n", res);

    gsl_matrix *m_gsl_rank4 = gsl_matrix_calloc(4, num_points);


    for (int i = 0; i < 4; i++){
        gsl_vector_view col = gsl_matrix_column(m_gsl_images, i);
        gsl_vector_view vec_work = gsl_matrix_row(m_gsl_s, i);
        gsl_vector_memcpy(&vec_work.vector, &col.vector);
        gsl_vector_scale(&vec_work.vector, gsl_vector_get(S, i));
    }


    if (visualize){
        Mat img = Mat::zeros(h, w*4, CV_32F);
        for (int i = 0; i < num_points; i++){
            Point2f pt = Point2f(templateMesh[i].x, templateMesh[i].y);
            for (int k = 0; k < 4; k++){
                double val = gsl_matrix_get(m_gsl_s, k, i);
                circle(img, Point2f(pt.x + k*w, pt.y), 1, val, 0, 8, 0);
            }
        }

        //resize(img, img, Size(1000, 250));
        imshow("[eigenface1] [eigenface2] [eigenface3] [eigenface4]", img);
    }   

    solveStuff();


    if (visualize){
        Mat img0 = Mat::zeros(h, w*4, CV_32F);
        for (int i = 0; i < num_points; i++){
            Point2f pt = Point2f(templateMesh[i].x, templateMesh[i].y);
            for (int k = 0; k < 4; k++){
                double val = gsl_matrix_get(m_gsl_final_result, k, i);
                circle(img0, Point2f(pt.x + k*w, pt.y), 1, val, 0, 8, 0);
            }
        }

        //resize(img0, img0, Size(1000, 250));
        imshow("solved (new images) before albedo divided out", img0);
    } 
    


    for (int i = 0; i < num_points; i++){
        double albedo = gsl_matrix_get(m_gsl_final_result, 0, i);
        for (int k = 1; k < 4; k++){
            gsl_matrix_set(m_gsl_final_result, k, i, gsl_matrix_get(m_gsl_final_result, k, i)/albedo);
        }
    }
    

    

    if (visualize){
        Mat img2 = Mat::zeros(h, w*4, CV_32F);
        for (int i = 0; i < num_points; i++){
            Point2f pt = Point2f(templateMesh[i].x, templateMesh[i].y);
            for (int k = 0; k < 4; k++){
                double val = gsl_matrix_get(m_gsl_final_result, k, i);
                circle(img2, Point2f(pt.x + k*w, pt.y), 1, val, 0, 8, 0);
            }
        }

        //resize(img2, img2, Size(1000, 250));
        imshow("after albedo divided out, should look like [average, nx, ny, nz]", img2);
    }   

    if (visualize){
        Mat img3 = Mat::zeros(h, w*4, CV_32F);
        for (int i = 0; i < num_points; i++){
            Point2f pt = Point2f(templateMesh[i].x, templateMesh[i].y);
            double nx = gsl_matrix_get(m_gsl_final_result, 1, i);
            double ny = gsl_matrix_get(m_gsl_final_result, 2, i);
            double nz = gsl_matrix_get(m_gsl_final_result, 3, i);

            circle(img3, Point2f(pt.x + 0*w, pt.y), 1, -1*nx/nz, 0, 8, 0);
            circle(img3, Point2f(pt.x + 1*w, pt.y), 1, 1*nx/nz, 0, 8, 0);
            circle(img3, Point2f(pt.x + 2*w, pt.y), 1, -1*ny/nz, 0, 8, 0);
            circle(img3, Point2f(pt.x + 3*w, pt.y), 1, 1*ny/nz, 0, 8, 0);
        }

        //resize(img3, img3, Size(1000, 250));
        imshow("zx and zy", img3);
    }   
    
    computeLightDistribution();


    recoverDepth();

    if (visualize){
        waitKey(0);
    }

}

void Shape3DApp::solveStuff(){
    printf("[solveStuff] ... \n");
    // copying some photocity matrix solving code
    gsl_matrix* m_gsl_xy = m_gsl_s;
    gsl_matrix* m_gsl_gps = m_gsl_model;
    gsl_matrix* m_gsl_transform = gsl_matrix_alloc(4, 4);

    printf("[solveStuff] ... 1 \n");
  
    // transform = gps * xy' * inv(xy * xy')
    //A = gps * xy'
    //B = xy *xy'
    //C = A * inv(B)
    
    gsl_matrix *A = gsl_matrix_alloc(4,4);
    gsl_matrix *B = gsl_matrix_alloc(4,4);

    printf("[solveStuff] ... 2 \n");
    
    //A = gps * xy'
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, m_gsl_gps, m_gsl_xy, 0.0, A);

    printf("[solveStuff] ... 3 \n");
    
    //B = xy *xy'
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, m_gsl_xy, m_gsl_xy, 0.0, B);

    printf("[solveStuff] ... 4 \n");
    
    // invert B...
    gsl_matrix *inv = gsl_matrix_alloc(4,4);
    gsl_permutation *p = gsl_permutation_alloc (4);
    int s; 
    gsl_linalg_LU_decomp(B, p, &s);
    gsl_linalg_LU_invert(B, p, inv);
    
    // multiply A * inv
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, A, inv, 0.0, m_gsl_transform);
    
    gsl_matrix_free(A);
    gsl_matrix_free(B);
    gsl_matrix_free(inv);
    gsl_permutation_free(p);

    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            printf("%f ", gsl_matrix_get(m_gsl_transform, i, j));
            //gsl_matrix_set(m_gsl_transform, i, j, gsl_matrix_get(m_gsl_transform, i, j)/w);
        }
        printf("\n");
    }

       
    //C = transform * xy
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, m_gsl_transform, m_gsl_xy, 0.0, m_gsl_final_result);

}

void Shape3DApp::computeLightDistribution(){
    printf("[computeLightDistribution]\n");
    MatrixXd imgs(num_images,num_points);
    MatrixXd weights(num_images,4);
    MatrixXd basis(4,num_points);

    for (int i = 0; i < num_images; i++){
        for (int j = 0; j < num_points; j++){
            imgs(i, j) = gsl_matrix_get(m_gsl_images_orig, j, i);
        }
    }

    printf("m_gsl_images populated\n");

    for (int i = 0; i < 4; i++){
        for (int j = 0; j < num_points; j++){
            basis(i, j) = gsl_matrix_get(m_gsl_final_result, i, j);
        }
    }

    printf("matrices populated\n");


    MatrixXd bbt(4,4);
    bbt = basis * basis.transpose();
    weights = imgs * basis.transpose() * bbt.inverse();

    printf("matrix math happened\n");

    for (int i = 0; i < num_images; i++){
        float a = weights(i, 0);
        float x = weights(i, 1);
        float y = weights(i, 2);
        float z = weights(i, 3);
        float dist = sqrt(x*x + y*y + z*z);
        //printf("image %d (%s): \t%f \t%f \t%f \t%f \t%f\n", i, imageFiles[i].c_str(), a, x/dist, y/dist, z/dist, dist);
        //printf("%f \t%f \t%f\n", x/dist, y/dist, z/dist);

        float incline = acos(z/dist)-3.14159/2.0;
        float azimuth = atan(y/x);
        printf("incline: %f \tazimuth: %f\n", incline, azimuth);


        Mat img = Mat::zeros(h, w*2, CV_32F);
        for (int j = 0; j < num_points; j++){
            Point2f pt = Point2f(templateMesh[j].x, templateMesh[j].y);
            double al = gsl_matrix_get(m_gsl_final_result, 0, j);
            double nx = gsl_matrix_get(m_gsl_final_result, 1, j);
            double ny = gsl_matrix_get(m_gsl_final_result, 2, j);
            double nz = gsl_matrix_get(m_gsl_final_result, 3, j);

            double c = gsl_matrix_get(m_gsl_images_orig, j, i);
            double c2 = a*al + x*nx + y*ny + z*nz;
            //printf("(%f, %f)    ", c, c2);

            circle(img, Point2f(pt.x + 0*w, pt.y), 1, c, 0, 8, 0);
            circle(img, Point2f(pt.x + 1*w, pt.y), 1, c2, 0, 8, 0);
        }

        //resize(img3, img3, Size(1000, 250));
        imshow("relit face", img);
        waitKey(0);
    }

}

void Shape3DApp::recoverDepth(){
    // this function will update the z depth in the templateMesh vector!
    printf("[recoverDepth]\n");

    // build problem 
    int m = templateMesh.size();
    vector<T> coefficients;
    vector<double> b_vec;

    // make a reverse lookup map for finding what index a point (x,y) is... 
    Mat lookup = Mat(300, 300, CV_32S, Scalar(0));
    for (int i = 0; i < templateMesh.size(); i++){
        Point3f p = templateMesh[i];
        lookup.at<int>(p.x, p.y) = i;
    }

    // populate coefficients: 
    int coef_count = 0;
    //for each point in templateMesh, get x and y and add some constraints
    for (int i = 0; i < templateMesh.size(); i++){
        Point3f p = templateMesh[i];
        double nx = gsl_matrix_get(m_gsl_final_result, 1, i);
        double ny = gsl_matrix_get(m_gsl_final_result, 2, i);
        double nz = gsl_matrix_get(m_gsl_final_result, 3, i); 


        // todo later: figure out of point is on the boundary or not and do 2 different things
        
        int x1y = lookup.at<int>(p.x+1, p.y);
        int xy = lookup.at<int>(p.x, p.y);
        int xy1 = lookup.at<int>(p.x, p.y+1);


        if (xy1 > -1 && xy > -1 && x1y > -1){
            // pixel constraint 1
            b_vec.push_back(-1.0*nx);
            coefficients.push_back(T(coef_count, x1y, nz));
            coefficients.push_back(T(coef_count, xy, -nz));
            coef_count++;

            b_vec.push_back(-1.0*ny);
            coefficients.push_back(T(coef_count, xy1, nz));
            coefficients.push_back(T(coef_count, xy, -nz));
            coef_count++;
        }
        /* 
         * boundary conditions smishons
        else {
            b_vec.push_back(0);
            coefficients.push_back(T(coef_count, xy1, -nx));
            coefficients.push_back(T(coef_count, xy, nx));
            coefficients.push_back(T(coef_count, x1y, ny));
            coefficients.push_back(T(coef_count, xy, -ny));
            coef_count++;
        }
        */


    }

    int c = b_vec.size(); // some number of constraints
    printf("coef count: %d, c: %d, m: %d\n", coef_count, c, m);
    SpMat A(c,m); 
    Eigen::VectorXd b(c); 
    printf("done initing stuff\n");

    A.setFromTriplets(coefficients.begin(), coefficients.end());
    printf("done setting from triplets\n");
    SpMat M(c,c);
    M = A.transpose()*A;

    printf("done setting from triplets stuff\n");
    for (int i = 0; i < b_vec.size(); i++){
        b(i) = b_vec[i];
    }

    Eigen::VectorXd b2 = A.transpose()*b;

    printf("done with all that\n");

    printf("M: %dx%d, b2: %dx1, A: %dx%d, b: %d\n", M.rows(), M.cols(), b2.size(), A.rows(), A.cols(), b.size());

     // Solving:
    Eigen::SimplicialCholesky<SpMat> chol(M); // performs a Cholesky factorization of A
    Eigen::VectorXd x = chol.solve(b2); // use the factorization to solve for the given right hand side

    double min_depth = x.minCoeff();
    for (int i = 0; i < x.size(); i++){
        x(i) = x(i) - min_depth;
    }


    //x = Scaling(2.0)*x;
    printf("not scaling\n");

    printf("Solved.\n");

    

    if (outFacePly){
        printf("printing ply file\n");
        FILE *f = fopen(outFacePly, "w");

    
        fprintf(f,"ply\n");
        fprintf(f,"format ascii 1.0\n");
        fprintf(f,"comment VCGLIB generated\n");
        fprintf(f,"element vertex %d\n", (int)templateMesh.size());
        fprintf(f,"property float x\n");
        fprintf(f,"property float y\n");
        fprintf(f,"property float z\n");
        fprintf(f,"property float nx\n");
        fprintf(f,"property float ny\n");
        fprintf(f,"property float nz\n");
        fprintf(f,"end_header\n");

        for (int i = 0; i < templateMesh.size(); i++){
            Point3f p = templateMesh[i];
            double nx = gsl_matrix_get(m_gsl_final_result, 1, i);
            double ny = gsl_matrix_get(m_gsl_final_result, 2, i);
            double nz = gsl_matrix_get(m_gsl_final_result, 3, i); 
            fprintf(f, "%f %f %f %f %f %f\n", p.x, p.y, x(i), nx, ny, nz);
        }

        fclose(f);
    }
    

    else {
        FILE *f = fopen(outFaceFile, "w");

        fprintf(f, "{\n");
        fprintf(f, "\"metadata\": { \"formatVersion\" : 3 },\n");
        fprintf(f, "\"scale\" : 5,\n");
        fprintf(f, "\"materials\": [],\n");
        fprintf(f, "\"vertices\": [");
        
        //print vertices!
        for (int i = 0; i < templateMesh.size(); i++){
            Point3f p = templateMesh[i];
            fprintf(f, "%f,%f,%f", p.x-65, 100-p.y, x(i));
            if (i < templateMesh.size()-1){
                fprintf(f,",");
            }
        }

        fprintf(f, "],\n");
        fprintf(f, "\"morphTargets\": [],\n");
        fprintf(f, "\"normals\": [],\n");
        fprintf(f, "\"colors\": [],\n");
        fprintf(f, "\"uvs\": [[]],\n");
        fprintf(f, "\"faces\": [");
        
        bool printedSomethingYet = false;
        // print faces!
        for (int i = 0; i < templateMesh.size(); i++){
            Point3f p = templateMesh[i];
            int x = p.x;
            int y = p.y;

            int x1y = lookup.at<int>(p.x+1, p.y);
            int x1y1 = lookup.at<int>(p.x+1, p.y+1);
            int xy1 = lookup.at<int>(p.x, p.y+1);

            if (x1y != 0 && x1y1 != 0 && xy1 != 0){
                // make a quad
                if (printedSomethingYet){
                    fprintf(f,",");
                }
                else {
                    printedSomethingYet = true;
                }
                fprintf(f,"1,%d,%d,%d,%d", i, xy1, x1y1, x1y);
            }
        }

        fprintf(f, "],\n");
        fprintf(f, "\"edges\" : []\n");
        fprintf(f, "}\n");

        fclose(f);
    }

}

static Shape3DApp *the_app = NULL;

/******************
 **     Main!!    **
 *******************/

int main(int argc, char **argv){

   
    the_app = new Shape3DApp();
    the_app->argc = argc;
    the_app->argv = argv;

    the_app->init();

    return 0;
}

