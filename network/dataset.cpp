#include <iostream>
#include <algorithm>
#include <random>
#include <iterator>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "dataset.h"
#include <unordered_set>
#include <omp.h>

// Loads dataset from CSV files with specified limits
bool Dataset::loadFromCSVFiles(const std::vector<std::string> &filePaths, int numImages, int numLabels, int res)
{
    // Clear any existing data
    clear();
    this -> numLabels = numLabels; // Set the number of labels

    // Check if file paths are empty
    if (filePaths.empty())
    {
        std::cerr << "No file paths provided" << std::endl;
        return false;
    }

    // Check if numImages is valid
    if (numImages < 0)
    {
        std::cerr << "Invalid number of images specified" << std::endl;
        return false;
    }
    {
        // Reserve vectors with estimated capacity
        std::vector<double> data_values;
        data_values.reserve(numImages * res * res); // Reserve space for specified number of images

        std::vector<int> label_values;
        label_values.reserve(numImages);

        int totalProcessedCount = 0; // Track total processed images across all files

        for (const std::string &filename : filePaths)
        {
            std::ifstream file(filename);

            if (!file.is_open())
            {
                std::cerr << "Could not open file: " << filename << std::endl;
                return false;
            }

            std::string line;
            int fileProcessedCount = 0;

            // Process each line (each image)
            while (std::getline(file, line) && (numImages <= 0 || totalProcessedCount < numImages))
            {
                std::stringstream ss(line);
                std::string cell;

                // Get the digit label (first column)
                std::getline(ss, cell, ',');
                int label = std::stoi(cell);

                // Add label to our dataset
                label_values.push_back(label);

                // Read pixel values
                while (std::getline(ss, cell, ','))
                {
                    double value = std::stod(cell);

                    // Track min/max values during first few images
                    static double min_value = 255.0;
                    static double max_value = 0.0;
                    if (totalProcessedCount < 10)
                    {
                        min_value = std::min(min_value, value);
                        max_value = std::max(max_value, value);
                    }

                    // Normalize values to [0,1] range
                    if (max_value > 1.0) // If data is in [0,255] range
                    {
                        value /= 255.0;  // Scale DOWN to [0,1]
                    }

                    // Add the normalized pixel value to our image data
                    data_values.push_back(value);
                }

                fileProcessedCount++;
                totalProcessedCount++;

                // Print progress
                if (totalProcessedCount % 1000 == 0)
                {
                    std::cout << "Processed " << totalProcessedCount << " samples" << std::endl;
                }

                // Break if we've processed the requested number of images
                if (numImages > 0 && totalProcessedCount >= numImages)
                {
                    std::cout << "Reached requested limit of " << numImages << " images" << std::endl;
                    break;
                }
            }

            std::cout << "Processed " << fileProcessedCount << " samples from " << filename << std::endl;

            // If we've processed all requested images, or if this is the last file
            if (numImages <= 0 || totalProcessedCount >= numImages || &filename == &filePaths.back())
            {
                // Set the actual number of images processed
                this->numImages = totalProcessedCount;
                channels = 1; // Assuming grayscale images
                width = 28;   // Image width
                height = 28;  // Image height

                // Create tensors with the actual number of processed images
                std::vector<int> dimensions = {totalProcessedCount, 1, 28, 28};

                // Create a new tensor with the appropriate dimensions
                this->data = Tensor(dimensions);

                // Create strides for proper indexing
                std::vector<int> strides = {28 * 28, 28 * 28, 28, 1};

                // Fill the tensor with our data
                for (int i = 0; i < totalProcessedCount * 28 * 28; i++)
                {
                    this->data.set_value_direct(i, data_values[i]);
                }

                // Find unique labels
                std::unordered_set<int> label_set(label_values.begin(), label_values.end());

                // Create a tensor for the labels
                std::vector<int> label_dims = {totalProcessedCount, 1};
                this->labels = Tensor(label_dims);

                // Fill the labels tensor
                for (int i = 0; i < totalProcessedCount; i++)
                {
                    this->labels.set_value({i, 0}, label_values[i]);
                }

                break; // Stop processing files if we've reached the limit
            }
        }

        indices.resize(numImages);
        for (int i{0}; i < numImages; i++)
        {
            indices[i] = i;
        }

        std::cout << "Successfully loaded a total of " << this->numImages << " samples" << std::endl;
        return true;
    }
}

// Constructor to initialize a dataset with the specified dimensions
Dataset::Dataset(int numImages, int channels, int width, int height, int numLabels)
    : numImages(numImages), channels(channels), width(width), height(height), numLabels(numLabels)
{
    // Initialize empty data tensor with appropriate dimensions
    std::vector<int> dims = {numImages, channels, height, width};
    data = Tensor(dims);

    // Initialize empty labels tensor
    std::vector<int> labelDims = {numImages, 1};
    labels = Tensor(labelDims);

    // Initialize indices for shuffling
    indices.resize(numImages);
    for (int i{0}; i < numImages; i++)
    {
        indices[i] = i;
    }
}

// Adds a single image and its label to the dataset
void Dataset::addImage(const Tensor &image, double image_label)
{
    // Check if image has correct dimensions
    auto imageDims = image.get_dimensions();
    if (imageDims.size() != 3 || imageDims[0] != channels ||
        imageDims[1] != height || imageDims[2] != width)
    {
        throw std::invalid_argument("Image dimensions don't match expected format");
    }

    // If this is the first image, set up the dataset
    if (numImages == 0)
    {
        channels = imageDims[0];
        height = imageDims[1];
        width = imageDims[2];

        // Initialize data with room for one image
        std::vector<int> dims = {1, channels, height, width};
        data = Tensor(dims);

        // Copy image data
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    data[{0, c, h, w}] = image[{c, h, w}];
                }
            }
        }

        numImages = 1;
    }
    else
    {
        // Create a new tensor with room for one more image
        std::vector<int> newDims = {numImages + 1, channels, height, width};
        Tensor newData(newDims);

        // Copy existing data
        for (int i = 0; i < numImages; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        newData[{i, c, h, w}] = data[{i, c, h, w}];
                    }
                }
            }
        }

        // Add the new image
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    newData[{numImages, c, h, w}] = image[{c, h, w}];
                }
            }
        }

        // Update the dataset
        data = newData;
        numImages++;

        // Extend labels tensor if it exists
        if (labels.get_n_values() > 0)
        {
            std::vector<int> newLabelDims = {numImages, 1};
            Tensor newLabels(newLabelDims);

            // Copy existing labels
            for (int i = 0; i < numImages - 2; i++)
            {
                newLabels[{i, 0}] = labels[{i, 0}];
            }

            // Add the new label
            newLabels[{numImages - 1, 0}] = image_label;

            labels = newLabels;
        }
    }
}

// Adds multiple images and their labels to the dataset
void Dataset::addImages(const std::vector<Tensor> &images, std::vector<double> image_labels)
{
    if (images.size() != image_labels.size())
    {
        throw std::invalid_argument("Number of images and labels must match");
    }

    // Add each image and its corresponding label
    for (int i = 0; i < static_cast<int>(images.size()); i++)
    {
        addImage(images[i], image_labels[i]);
    }
}

// Returns a view of a specific image from the dataset
Tensor Dataset::getImage(int index) const
{
    if (index < 0 || index >= numImages)
    {
        throw std::out_of_range("Image index out of bounds");
    }

    // Create a view into the data tensor for the specific image
    return data[index];
}

// Returns a one-hot encoded label for a specific image
Tensor Dataset::getLabel(int index) const
{
    if (index < 0 || index >= numImages)
    {
        throw std::out_of_range("Label index out of bounds");
    }

    std::vector<int> one_hot_size = {1, numLabels};
    Tensor labelTensor(one_hot_size);
    int label_value = labels[{index, 0}]; // Get the label value

    labelTensor[{0, label_value}] = 1.0; // Set the label to 1 for the specified index

    // Return the label for the specified image
    return labelTensor;
}

// Sets the number of samples to include in each batch
void Dataset::setBatchSize(int new_batch_size)
{
    if (new_batch_size == 0)
    {
        throw std::invalid_argument("Batch size must be greater than 0");
    }
    batch_size = new_batch_size;
}

// Returns the current batch size
int Dataset::getBatchSize() const
{
    return batch_size;
}

// Returns the total number of batches based on dataset size and batch size
int Dataset::getNumBatches() const
{
    return (numImages + batch_size - 1) / batch_size; // Ceiling division
}

// Randomly reorders the indices to shuffle the dataset
void Dataset::shuffle()
{
    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(indices.begin(), indices.end(), g);
}

// Returns a tensor containing a batch of input images
TensorView Dataset::getBatchInputs(int batch_index) const
{
    if (batch_index >= getNumBatches())
    {
        throw std::out_of_range("Batch index out of bounds");
    }

    int start = batch_index * batch_size;
    int end = std::min(start + batch_size, numImages);

    // Create a new tensor for the batch
    std::vector<int> batch_dims = {end - start, channels, height, width};
    Tensor batch(batch_dims);

    // Fill the batch tensor with data from the dataset
    for (int i =start ; i < end; i++)
    {
        int index = indices[i]; // Get the shuffled index
        
        // Use explicit element-wise copy
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    batch[{i - start, c, h, w}] = data[{index, c, h, w}];
                }
            }
        }
    }

    return batch;
}

// Returns a tensor containing a batch of one-hot encoded labels
TensorView Dataset::getBatchLabels(int batch_index) const
{
    if (batch_index >= getNumBatches())
    {
        throw std::out_of_range("Batch index out of bounds");
    }

    int start = batch_index * batch_size;
    int end = std::min(start + batch_size, numImages);

    // Create a new tensor for the labels in the batch
    std::vector<int> label_dims = {end - start, numLabels};
    Tensor batch_labels(label_dims);

    // Fill the batch labels tensor with data from the dataset
    for (int i = start; i < end; i++)
    {
        int index = indices[i];               // Get the shuffled index
        int label_value = labels[{index, 0}]; // Get the label value

        batch_labels[{i - start, label_value}] = 1.0;
    }

    return batch_labels;
}

// Resets the dataset to an empty state
void Dataset::clear()
{
    // Reset all data members
    std::vector<int> emptyDims = {0, 0, 0, 0};
    data = Tensor(emptyDims);
    labels = Tensor({0, 0});
    numImages = 0;
    channels = 0;
    width = 0;
    height = 0;
}

// Creates a text summary of the dataset
std::string Dataset::summary() const
{
    std::stringstream ss;
    ss << "Dataset Summary:" << std::endl;
    ss << "----------------" << std::endl;
    ss << "Number of images: " << numImages << std::endl;
    ss << "Image dimensions: " << channels << " x " << height << " x " << width << std::endl;
    ss << "Labels dimensions: " << labels.get_dimensions()[0] << std::endl;

    return ss.str();
}