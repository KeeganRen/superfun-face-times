#include <string>
#include <stdio.h>
#include <vector>
#include <list>
#include <climits>

int countFeatures(std::string feature_file);
void readFeatures(int *features, std::string feature_file);
int compareFeatures(int *f1, int *f2, int num_features);

class Distance {
public:
    Distance(int id, int dist) :
    m_id(id), m_dist(dist)
    {}
    
    int m_id; /* face id */
    int m_dist; /* distance or score */
};

bool compare_distances(Distance d1, Distance d2){
    return (d1.m_dist < d2.m_dist);
}


int main (int argc, char** argv){
    int num_faces = 0;
    int num_features = 0;

    if (argc < 3) {
        printf("Usage: %s <queryFaceFeatures> <faceFeatures0> <faceFeatures<1> ... <faceFeaturesN>\n"
            "  Returns:\n"
            "\n"
            "  <face id> <distance> (best)\n"
            "  <face id> <distance> (second best)\n"
            "  ... and so on\n"
            "\n", 
            argv[0]);
            return -1;
    }

    std::vector<std::string> feature_files;
    for (int i = 1; i < argc; i++){ 
        feature_files.push_back(std::string(argv[i]));
    }

    num_faces = feature_files.size();
    //fprintf(stderr, "Number of faces: %d\n", num_faces);

    num_features = countFeatures(feature_files[0]);
    if (num_features <= 0){
        return -1;
    }
    //fprintf(stderr, "Number of features per face: %d\n", num_features);

    int all_features[num_faces][num_features*128]; 
    for (int i = 0; i < num_faces; i++){
        //printf("\tFeature file: %s\n", feature_files[i].c_str());
        readFeatures(all_features[i], feature_files[i]);
    }


    std::list<Distance> scores;
    for (int i = 1; i < num_faces; i++){
        int score = compareFeatures(all_features[0], all_features[i], num_features);
        scores.push_back(Distance(i, score));
    }

    scores.sort(compare_distances);
    std::list<Distance>::iterator it;
    for (it = scores.begin(); it != scores.end(); ++it){
        if (it->m_dist != INT_MAX){
            printf("%d %d\n", it->m_id, it->m_dist);
        }
    }


    fflush(stdout);
    fflush(stderr);

    return 0;
}

int countFeatures(std::string feature_file){
    FILE *file = fopen(feature_file.c_str(), "r");
    if (!file){
        //fprintf(stderr, "[countFeatures] error reading feature file: %s\n", feature_file.c_str());
        return -1;
    }

    int c = 0;
    char buf[1024];
    //while (fscanf(file, "%s\n", buf) != EOF){
    while (fgets(buf, 1024, file) != NULL){
        c++;
    }

    fclose(file);
    return c;
}

void readFeatures(int *features, std::string feature_file){
    const char *filename= feature_file.c_str();
    FILE *file = fopen(filename, "r");
    if (!file){
        //fprintf(stderr, "[readFeatures] error reading feature file: %s\n", feature_file.c_str());
        features[0] = -1;
        return;
    }

    int x, y, s, r;
    char buf[1024];
    int c = 0;
    while (1){
        if (fscanf(file, "%d %d %d %d", &x, &y, &s, &r) == EOF){
            break;
        }
        
        for (int i = 0; i < 128; i++){
            int x = fscanf(file, " %d", &(features[c*128 + i]));
        }
        c++;
    }

    fclose(file);
}

int compareFeatures(int *f1, int *f2, int num_features){

    if (f1[0] == -1 || f2[0] == -1){
        //fprintf(stderr, "[compareFeatures] one of the pair of features does not exist\n");
        return INT_MAX;
    }

    int l1 = 0;
    for (int i = 0; i < num_features*128; i++){
        int dist = f1[i] - f2[i];
        if (dist < 0)
            dist *= -1;
        l1 += dist;
    }
    return l1;
}
