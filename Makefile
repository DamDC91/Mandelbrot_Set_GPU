CXX = nvcc
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
INC_DIR = include

EXEC_NAME = app
CXXFLAGS = -std=c++11 -I $(INC_DIR)
LIBS = -lyaml-cpp -lsfml-graphics -lsfml-window -lsfml-system

SRC_FILES = $(wildcard $(SRC_DIR)/*.cu)

OBJ_FILES = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC_FILES))

all : $(EXEC_NAME)
 

$(EXEC_NAME) : $(OBJ_FILES)
	$(CXX) -g -o $(BIN_DIR)/$(EXEC_NAME) $(OBJ_FILES) $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.*
	$(CXX) $(CXXFLAGS) -o $@ -c $< 


.PHONY: clean

clean :
	rm $(BIN_DIR)/$(EXEC_NAME) $(OBJ_FILES)
