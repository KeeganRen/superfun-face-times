#include <FaceTracker/Tracker.h>
#include <opencv/highgui.h>
#include <iostream>
#include <fstream>

int main(int argc, const char** argv){
    /* necessary files: 
    
            face2.tracker
            face.con
            face.tri
            
            output image
            output points file
            
            canonical face to align to
    */
        
    /* init face tracker */
    FACETRACKER::Tracker model(ftFile);
    cv::Mat tri=FACETRACKER::IO::LoadTri(triFile);
    cv::Mat con=FACETRACKER::IO::LoadCon(conFile);
    
    /* flow:
            
            load image
            make grayscale or something
            track face (find points)
            draw things
            save things / write files
            
            reset in case we're going to do it again on another image!
    */
    
    /* extra flow:
            
            load canonical face
            compute alignment (estimateRigidTransform)
            warp image (probably with warpAffine)
            crop image
            save more files
    */
}

