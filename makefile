# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3 -march=native -pthread -fopenmp

# Output executable
TARGET = main

# Directory structure
SRC_DIR = .
OBJ_DIR = obj

# Find all CPP files (excluding test files)
SRCS := $(shell find $(SRC_DIR) -name "*.cpp" -not -name "*_test.cpp")

# Generate object file paths
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

# Include paths
INCLUDES := -I.

# Default target
all: directories $(TARGET)

# Debug information
print-info:
	@echo "Source files:"
	@for src in $(SRCS); do echo "  $$src"; done
	@echo
	@echo "Object files:"
	@for obj in $(OBJS); do echo "  $$obj"; done

# Create necessary directories for object files
directories:
	@mkdir -p $(OBJ_DIR)
	@for dir in $(sort $(dir $(OBJS))); do \
		mkdir -p $$dir; \
	done

# Link the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ -fopenmp

# Compile source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

# Run the executable
run: all
	./$(TARGET)

.PHONY: all clean directories run print-info