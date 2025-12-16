CXX = g++
NVCC = nvcc

DEBUG ?= 0

ifeq ($(DEBUG),1)
  CXXFLAGS = -g -O0 -DDEBUG -Wall -march=native -funroll-loops -Wno-unused-result
  NVFLAGS  = -g -G -O0 -DDEBUG --ptxas-options=-v
else
  CXXFLAGS = -O2 -Wall -march=native -funroll-loops -Wno-unused-result
  NVFLAGS  = -O2 --ptxas-options=-v
endif

# Add include path here
INCLUDES = -Iinclude

CXXFLAGS += $(INCLUDES)
NVFLAGS  += $(INCLUDES)

LDLIBS = -lm

SRC_CPP  = src/word2vec.cpp src/vocab.cpp
SRC_CUDA = src/skipgram.cu

OBJ = build/word2vec.o build/vocab.o build/skipgram.o

TARGET = word2vec

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(NVFLAGS) $(OBJ) -o $@ $(LDLIBS)

build/%.o: src/%.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/%.o: src/%.cu
	@mkdir -p build
	$(NVCC) $(NVFLAGS) -c $< -o $@
clean:
	rm -rf build $(TARGET)

.PHONY: all clean
