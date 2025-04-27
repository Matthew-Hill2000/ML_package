#ifndef DATASET_H
#define DATASET_H

#include "../tensor/tensor_view.h"
#include <string>
#include <vector>
#include <random>
#include <algorithm>

// The Dataset class manages collections of images and their labels
// Provides functionality for loading, batching, and accessing training data
class Dataset
{
private:
    Tensor data;                // Stores images in shape (image, channels, width, height)
    Tensor labels;              // Stores labels for each image
    int numImages;              // Number of images in the dataset
    int channels;               // Number of channels per image
    int width;                  // Width of each image
    int height;                 // Height of each image
    int numLabels;              // Number of unique label classes
    std::vector<int> indices;// For shuffling data order
    int batch_size = 1;         // Default batch size

public:
    // Default constructor - creates an empty dataset
    Dataset() : numImages(0), channels(0), width(0), height(0), numLabels(0) {}
    
    // Constructor with specific dimensions
    Dataset(int numImages, int channels, int width, int height, int numLabels);

    // Add a single image with its label
    void addImage(const Tensor &image, double image_label);

    // Add multiple images with their labels
    void addImages(const std::vector<Tensor> &images, std::vector<double> image_labels);

    // Get a specific image as a tensor
    Tensor getImage(int index) const;

    // Get label as one-hot encoded tensor
    Tensor getLabel(int index) const;

    // Set the number of samples per batch
    void setBatchSize(int new_batch_size);

    // Get the current batch size
    int getBatchSize() const;

    // Get the total number of batches
    int getNumBatches() const;

    // Randomly reorder the dataset
    void shuffle();

    // Get a batch of inputs as a tensor
    TensorView getBatchInputs(int batch_index) const;

    // Get a batch of labels as a tensor
    TensorView getBatchLabels(int batch_index) const;

    // Get the entire dataset tensor
    const Tensor &getData() const { return data; }

    // Get dataset dimensions
    int getNumImages() const { return numImages; }
    int getChannels() const { return channels; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }

    // Load images from CSV files
    bool loadFromCSVFiles(const std::vector<std::string> &filePaths, int numImages, int numLabels, int res);

    // Get a summary of the dataset
    std::string summary() const;

    // Reset the dataset to empty state
    void clear();
};

#endif // DATASET_H