CXX = g++
CXXFLAGS = -Wall -fPIC
TARGET = dsm_interpolation.so
SRC = dsm_interpolation.cpp
OBJ = dsm_interpolation.o

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) -shared $(OBJ) -o $(TARGET)

$(OBJ): $(SRC)
	$(CXX) $(CXXFLAGS) -c $(SRC)

clean:
	rm -f $(OBJ) $(TARGET)