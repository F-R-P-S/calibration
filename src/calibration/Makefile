CXX = g++
CXXFLAGS = -std=c++17 $(shell pkg-config --cflags opencv4) -I/usr/include/eigen3
LDFLAGS = $(shell pkg-config --libs opencv4)

SRC = cameraCalibration.cpp
TARGET = eyeInHand

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
