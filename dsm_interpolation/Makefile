CXX = g++
CXXFLAGS = -Wall -fPIC -O2 -fopenmp
LDFLAGS = -fopenmp
TARGET = dsm_interpolation.so
SRC = dsm_interpolation.cpp
OBJ = dsm_interpolation.o

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) -shared $(LDFLAGS) $(OBJ) -o $(TARGET)

$(OBJ): $(SRC)
	$(CXX) $(CXXFLAGS) -c $(SRC)

clean:
	rm -f $(OBJ) $(TARGET)
