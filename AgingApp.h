#include <stdlib.h>
#include <string>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

class AgingApp
{
public:
    void init();
    void loadCluster(int c_id, Mat &mean_mat, gsl_vector **mean, gsl_matrix **eigenfaces);
    Mat projectFace(gsl_vector *mean, gsl_matrix *eigenfaces);
    void saveRelitFace(Mat &projected_face, int ref_cluster_id);

    int argc;
    char **argv;
    
    bool visualize;

    char *data_path;
    int face_id;
    int ref_cluster_id;
    int *clusters;

    int w, h, d, num_pixels;
    int num_eigenfaces;

    Mat face;
    Mat ref_mean_face;
    Mat cluster_mean_face;
    gsl_vector *ref_mean;
    gsl_matrix *ref_eigenfaces;
    gsl_vector *cluster_mean;
    gsl_matrix *cluster_eigenfaces;

};
