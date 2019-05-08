
# Path
DARKNET_PATH ?= ../darknet
MOT_DEEPSORT_PATH ?= ../tracker
# OPENCV_PATH ?= ../../../opencv-4.0.1


# Common definitions
GCC ?= g++ -g
NVCC ?= $(CUDA_PATH)/bin/nvcc

CCFLAGS := -fPIC -std=c++11 -lstdc++fs -Wall -O0 -Wunused-function -Wunused-variable
NVCCFLAGS := --compiler-options "-fPIC -Wall -O0 -std=c++11 -mavx -mavx2" -std=c++11 -lstdc++fs


# Common includes and paths
INCLUDES += -I$(DARKNET_PATH)/include
INCLUDES += -I$(MOT_DEEPSORT_PATH)/include


# Link applications against stub libraries provided in the SDKs.
LDFLAGS += -L$(MOT_DEEPSORT_PATH) -ldeepsort
LDFLAGS += -Wl,-rpath="/usr/local/lib"
LDFLAGS += -lopencv_objdetect -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lopencv_videoio
LDFLAGS += -L$(DARKNET_PATH) -ldarknet
LDFLAGS += -Wl,-rpath="$(DARKNET_PATH)"
LDFLAGS += -Wl,-rpath="$(MOT_DEEPSORT_PATH)"


#
EXE_NAME ?= my_demo

all: $(EXE_NAME)

%.o: %.cpp
	$(GCC) $(CCFLAGS) $(INCLUDES) -c $<

%.o: $(CLIENT_SERVER_PROTO)/%.cpp
	$(GCC) $(CCFLAGS) $(INCLUDES) -c $<

%.o: $(CLIENT_SERVER_PROTO)/%.cc
	$(GCC) $(CCFLAGS) $(INCLUDES) -c $<

$(EXE_NAME): my_demo.o
	$(GCC) $(CCFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf *.o $(EXE_NAME)
