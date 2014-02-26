
CC = g++
CFLAGS=-c -Wall -O3 -g

OPENCV_PREFIX = /usr/local/Cellar/opencv/2.4.6.1/

LIBDIR = -L/usr/X11R6/lib -I$(OPENCV_PREFIX)/include -L/usr/lib  
INCDIR = -I/opt/local/include -L$(OPENCV_PREFIX)/lib -I/Users/ktuite/Library -I/usr/include -I/usr/include/opencv -I/homes/grail/ktuite/library -I/usr/local/Cellar/eigen/3.2.0/include/eigen3

LIBS = -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_calib3d -lgsl -lopencv_video -lblas

ANN_LIB = -L/homes/grail/ktuite/library/ann_1.1.2/lib -L/Users/ktuite/Library/ann_1.1.2/lib
ANN_INC = -I/homes/grail/ktuite/library/ann_1.1.2/include -I/Users/ktuite/Library/ann_1.1.2/include

COLLECTIONFLOWFACE=collectionFlow
COLLECTIONFLOWSOURCES=CollectionFlowApp.cpp

INCCFLOW=incrementalCFlow
INCCFLOWSOURCES=IncrementalCFlowApp.cpp

WARPFACE=warpFace
WARPSOURCES=WarpFaceApp.cpp

SHAPEFACE=shape3D
SHAPESOURCES=Shape3DApp.cpp

XMORPH=xCluster
XMORPHSOURCES=CrossClusterMorphApp.cpp

AGING=agingDemo
AGINGSOURCES=AgingApp.cpp


FLOWFACE=flowFace
FLOWSOURCES=GaussianPyramid.cpp OpticalFlow.cpp CVOpticalFlow.cpp Stochastic.cpp
FLOWOBJECTS=$(FLOWSOURCES:.cpp=.o)
FLOWFLAGS=-D_LINUX_MAC -D_OPENCV -I$(OPENCV_PREFIX)/include/opencv 

FACELIBSOURCES=FaceLib.cpp
FACELIBOBJECTS=$(FACELIBSOURCES:.cpp=.o)

MORPHFACE=morphApp

ALIGNFACE=alignFace
ALIGNFACESOURCES=AlignerApp.cpp

SWAPFACE=swapFace
SWAPFACESOURCES=SwapApp.cpp

SCOREFACE=score
SCOREFACERESOURCES=ScoreCalculatorApp.cpp

AVGFACE=averageFaces
AVGFACESOURCES=AverageFaces.cpp

PHOTOBIO=photoBio
PHOTOBIOSOURCES=PhotoBio.cpp

HOGFEATURES=hogFeatures
HOGFEATURESSOURCES=HogFeatures.cpp

NEARESTFACE=nearestFace
NEARESTFACESOURCES=NearestFace.cpp

all: $(COLLECTIONFLOWFACE) $(WARPFACE) $(FLOWFACE) $(MORPHFACE) $(SHAPEFACE) $(INCCFLOW) $(SWAPFACE) $(AVGFACE) $(PHOTOBIO)

clean: 
	rm -f *.o $(COLLECTIONFLOWFACE) $(WARPFACE) $(FLOWFACE) $(MORPHFACE)
	    

$(NEARESTFACE): $(NEARESTFACESOURCES) 
	$(CC) $(NEARESTFACESOURCES) -o $@ 

gridifyMesh: GridifyMesh.cpp 
	$(CC) GridifyMesh.cpp $(INCDIR) $(LIBDIR) -o $@ -lopencv_highgui -lopencv_imgproc -lopencv_core
    
$(COLLECTIONFLOWFACE): $(COLLECTIONFLOWSOURCES)  $(FLOWOBJECTS) 
	$(CC) $(FLOWFLAGS) $(INCDIR) $(LIBDIR) $(COLLECTIONFLOWSOURCES) $(FLOWOBJECTS)  -o $@ $(LIBS)

$(INCCFLOW): $(INCCFLOWSOURCES)  $(FLOWOBJECTS) 
	$(CC) $(FLOWFLAGS) $(INCDIR) $(LIBDIR) $(INCCFLOWSOURCES) $(FLOWOBJECTS)  -o $@ $(LIBS)

$(WARPFACE): $(WARPSOURCES) 
	$(CC) $(INCDIR) $(LIBDIR) $(WARPSOURCES) -o $@ $(LIBS)

$(SHAPEFACE): $(SHAPESOURCES) 
	$(CC) $(INCDIR) $(LIBDIR) $(SHAPESOURCES) -o $@ $(LIBS)

$(AGING): $(AGINGSOURCES) $(FACELIBOBJECTS) $(FLOWOBJECTS)
	$(CC) $(FLOWFLAGS) $(INCDIR) $(LIBDIR) $(AGINGSOURCES) $(FLOWOBJECTS) $(FACELIBOBJECTS)  -o $@ $(LIBS)

$(XMORPH): $(XMORPHSOURCES) $(FACELIBOBJECTS) $(FLOWOBJECTS)
	$(CC) $(FLOWFLAGS) $(INCDIR) $(LIBDIR) $(XMORPHSOURCES) $(FACELIBOBJECTS) $(FLOWOBJECTS) -o $@ $(LIBS)

$(FLOWFACE):  FlowFaceApp.cpp $(FLOWOBJECTS) 
	$(CC) $(FLOWFLAGS) $(INCDIR) $(LIBDIR) FlowFaceApp.cpp $(FLOWOBJECTS) -o $@ $(LIBS)

$(MORPHFACE):  MorphApp.cpp $(FLOWOBJECTS) 
	$(CC) $(FLOWFLAGS) $(INCDIR) $(LIBDIR) MorphApp.cpp $(FLOWOBJECTS) -o $@ $(LIBS)

$(ALIGNFACE):  $(ALIGNFACESOURCES)  
	$(CC) $(INCDIR) $(LIBDIR) $(ALIGNFACESOURCES) -o $@ $(LIBS)

$(SWAPFACE): $(SWAPFACESOURCES) $(FACELIBOBJECTS) $(FLOWOBJECTS) 
	$(CC) $(FLOWFLAGS) $(INCDIR) $(LIBDIR) $(SWAPFACESOURCES) $(FLOWOBJECTS) $(FACELIBOBJECTS)  -o $@ $(LIBS)

$(SCOREFACE): $(SCOREFACERESOURCES) $(FACELIBOBJECTS) $(FLOWOBJECTS) 
	$(CC) $(FLOWFLAGS) $(INCDIR) $(LIBDIR) $(SCOREFACERESOURCES) $(FLOWOBJECTS) $(FACELIBOBJECTS)  -o $@ $(LIBS)

$(AVGFACE):  $(AVGFACESOURCES)  
	$(CC) $(INCDIR) $(LIBDIR) $(AVGFACESOURCES) -o $@ $(LIBS)

$(HOGFEATURES):  $(HOGFEATURESSOURCES)  
	$(CC) $(INCDIR) $(LIBDIR) $(HOGFEATURESSOURCES) -o $@ $(LIBS) -lopencv_objdetect

$(PHOTOBIO):  $(PHOTOBIOSOURCES)  
	$(CC) $(INCDIR) $(LIBDIR) $(ANN_LIB) $(ANN_INC) $(PHOTOBIOSOURCES) -o $@ $(LIBS) -lANN

.cpp.o:
	$(CC) $(FLOWFLAGS) $(INCDIR) $< -o $@ -c
