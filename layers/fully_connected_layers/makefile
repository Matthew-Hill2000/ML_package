# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3 -pthread -fopenmp

# Output executable
TARGET = FC_test

# Project root directory (adjust if needed)
ROOT_DIR = ../..

# Source files needed
SRCS = FC_test.cpp \
       $(ROOT_DIR)/tensor/tensor_view.cpp \
       $(ROOT_DIR)/tensor/tensor_view_accessors.cpp \
       $(ROOT_DIR)/tensor/tensor_view_overloads.cpp \
       $(ROOT_DIR)/tensor/tensor_view_maths.cpp \
       $(ROOT_DIR)/tensor/tensor_storage.cpp \
       $(ROOT_DIR)/layers/fully_connected_layers/fully_connected_layer.cpp \
       $(ROOT_DIR)/layers/fully_connected_layers/fully_connected_layer_accessors.cpp

# Generate object file paths
OBJS = $(SRCS:.cpp=.o)

# Include paths
INCLUDES = -I$(ROOT_DIR)

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ -fopenmp

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)

# Run the test
run: all
	./$(TARGET)

# Print OpenMP info (useful for debugging)
omp-info:
	@echo "Checking OpenMP configuration..."
	@echo "#include <omp.h>" > omp_check.cpp
	@echo "#include <stdio.h>" >> omp_check.cpp
	@echo "int main() { printf(\"Max threads: %d\\n\", omp_get_max_threads()); return 0; }" >> omp_check.cpp
	@$(CXX) -fopenmp omp_check.cpp -o omp_check
	@./omp_check
	@rm omp_check.cpp omp_check

.PHONY: all clean run omp-info