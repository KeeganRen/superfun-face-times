Hello, friends! I am writing code to do things with faces. Let's face it... aw crap, I can't think of anything to say next. 

## Things you should know: ##
I code on a mac (snow leopard, whatever, my cat is out of date). But I use basic things like OpenCV and GSL (GNU Scientific Library) and my code also works on linux. 

## There are a couple different programs here right now: ##
### eigenface
It takes a list of images (faces, presumably, that are already detected and cropped and aligned) and does some eigenface stuff. It saves pictures of the eigenfaces and low rank (rank 4) faces to whatever directory you specify with --output. Sometimes I get worried this code is messed up, but it may be because I don't have enough data or it's too crazy or something. If you spot something wrong, please tell me!
    ./eigenface --input imageList.txt --output test/ 

###warpFace
This is in progress. In that it makes weird pictures right now but I guess it's working how it's supposed to. 
It takes an image. It also takes a list of the 10 fiducial points detected by this guy's code: http://www.dantone.me/projects-2/facial-feature-detection/ It also takes a "face mesh" which is basically a face-shaped point cloud where the points are conveniently enough in a grid formation so we can use them to warp and color a new image.
    ./warpFace --faceImage 108.jpg --facePoints 108.txt --faceMesh smaller-face.txt --output warp108.jpg
