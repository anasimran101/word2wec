CXX = g++
CXXFLAGS = -g -O0 -DDEBUG -march=native -Wall -funroll-loops \
           -Wno-unused-result -pthread -Iinclude

LDFLAGS = -lm

SRC = src/word2vec.cpp src/vocab.cpp
OBJ = build/word2vec.o build/vocab.o

TARGET = word2vec

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

build/%.o: src/%.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build $(TARGET)
