Hello, friends! I am writing code to do things with faces. Let's face it... aw crap, I can't think of anything to say next. 

## Things you should know: ##
I code on a mac (snow leopard, whatever, my cat is out of date). But I use basic things like OpenCV and GSL (GNU Scientific Library) and my code also works on linux. 

## Libraries you'll need ##
* OpenCV (I installed this with MacPorts, but any package manager should work. Installing it by hand didn't work for me.)
* GSL (For doing matrix stuff. http://www.gnu.org/software/gsl/ Super easy to install from source.) 
* Boost (Uh, I don't actually know what this does except one of the face trackers uses it. http://www.boost.org/ I remember it being relatively easy to install from source.)


###warpFace
This program is still called warpFace because it takes a picture of a face and a 3D template model of a face, and using the 3D template model, warps the *pose* of the face in the image to a frontal pose. 

Now, that part of the code is still there, but the overall purpose of this program is to produce a 3D reconstruction of a face from a bunch of images. So, the input looks like this:
* --list: a text file with each line containing the path to a face photo and a text file of the detected fiducial points in that face photo
* --templateMesh: a text file of [x y z nx ny nz r g b] points of a template 3D face
* --templatePoints: a text file of [x,y,z] 3D points in the template 3D face corresponding to the detected fiducial points in the images
* --canonicalPoints: a text file of [x y] 2D points of where in an image the canonical fiducial points should be

The program will run as follows:
* Align all face images to be matched with the 3D face template
* Run SVD 
* Do some other magic to find the new face normals at each 3D point and then recover the new surface from those normals

I run it like this: 
     
    ./warpFace --list oagTest/images_with_points.txt --templateMesh model/igor.txt --templatePoints model/igor-canonical.txt --canonicalPoints model/canonical_faceforest.txt


###collectionFlow
http://grail.cs.washington.edu/cflow/

    ./collectionFlow --input oagTest/imagesCropped.txt --output cfout/


###morphApp
Takes 4 images (a,b,c,d) and computes a morph between the first and the last by going through the middle two. Essentially, if you have the output of Collection Flow, this lets you composite two flow fields (a->b, c->d, assuming the flow from b->c is the identity) and get a final flow/morph from a->d.

    ./morphApp /Users/ktuite/Desktop/rank4-face49-orig.jpg /Users/ktuite/Desktop/rank19-face49-low.jpg /Users/ktuite/Desktop/rank19-face11-low.jpg /Users/ktuite/Desktop/rank4-face11-orig.jpg


###flowFace
Computes optical flow between 2 images and does a little interpolation morph between them.

    ./flowFace --image1 ~/Desktop/rank4-face49-orig.jpg --image2 ~/Desktop/rank19-face49-low.jpg

###detectFace
Uses the face detection from http://www.ict.csiro.au/staff/jason.saragih/

    ./detectFaces --input kath/warped.txt --mask --small --output kath/