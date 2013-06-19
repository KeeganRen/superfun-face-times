
CC = g++
CFLAGS=-c -Wall -O3 -g

OPENCV_PREFIX = /opt/local

LIBDIR = -L/usr/X11R6/lib -I$(OPENCV_PREFIX)/include -L/usr/lib 
INCDIR = -I/opt/local/include -L$(OPENCV_PREFIX)/lib -I/Users/ktuite/Library -I/usr/include -I/usr/include/opencv -I/homes/grail/ktuite/library

LIBS = -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_calib3d -lgsl -lopencv_video 

COLLECTIONFLOWFACE=collectionFlow
COLLECTIONFLOWSOURCES=CollectionFlowApp.cpp

INCCFLOW=incrementalCFlow
INCCFLOWSOURCES=IncrementalCFlowApp.cpp

WARPFACE=warpFace
WARPSOURCES=WarpFaceApp.cpp

SHAPEFACE=shape3D
SHAPESOURCES=Shape3DApp.cpp

FLOWFACE=flowFace
FLOWSOURCES=GaussianPyramid.cpp OpticalFlow.cpp CVOpticalFlow.cpp Stochastic.cpp
FLOWOBJECTS=$(FLOWSOURCES:.cpp=.o)
FLOWFLAGS=-D_LINUX_MAC -D_OPENCV -I/opt/local/include/opencv 

MORPHFACE=morphApp

ALIGNFACE=alignFace
ALIGNFACESOURCES=AlignerApp.cpp

all: $(COLLECTIONFLOWFACE) $(WARPFACE) $(FLOWFACE) $(MORPHFACE) $(SHAPEFACE) $(INCCFLOW)

clean: 
	rm -f *.o $(COLLECTIONFLOWFACE) $(WARPFACE) $(FLOWFACE) $(MORPHFACE)
	    
$(COLLECTIONFLOWFACE): $(COLLECTIONFLOWSOURCES)  $(FLOWOBJECTS) 
	$(CC) $(FLOWFLAGS) $(INCDIR) $(LIBDIR) $(COLLECTIONFLOWSOURCES) $(FLOWOBJECTS)  -o $@ $(LIBS)

$(INCCFLOW): $(INCCFLOWSOURCES)  $(FLOWOBJECTS) 
	$(CC) $(FLOWFLAGS) $(INCDIR) $(LIBDIR) $(INCCFLOWSOURCES) $(FLOWOBJECTS)  -o $@ $(LIBS)

$(WARPFACE): $(WARPSOURCES) 
	$(CC) $(INCDIR) $(LIBDIR) $(WARPSOURCES) -o $@ $(LIBS)

$(SHAPEFACE): $(SHAPESOURCES) 
	$(CC) $(INCDIR) $(LIBDIR) $(SHAPESOURCES) -o $@ $(LIBS)

$(FLOWFACE):  FlowFaceApp.cpp $(FLOWOBJECTS) 
	$(CC) $(FLOWFLAGS) $(INCDIR) $(LIBDIR) FlowFaceApp.cpp $(FLOWOBJECTS) -o $@ $(LIBS)

$(MORPHFACE):  MorphApp.cpp $(FLOWOBJECTS) 
	$(CC) $(FLOWFLAGS) $(INCDIR) $(LIBDIR) MorphApp.cpp $(FLOWOBJECTS) -o $@ $(LIBS)

$(ALIGNFACE):  $(ALIGNFACESOURCES)  
	$(CC) $(INCDIR) $(LIBDIR) $(ALIGNFACESOURCES) -o $@ $(LIBS)

.cpp.o:
	$(CC) $(FLOWFLAGS) $(INCDIR) $< -o $@ -c
