#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "softmax.h"

// Define network architecture parameters
#define INPUT_SIZE 784         // 28x28 pixel input
#define HIDDEN_LAYERS 2        // Number of hidden layers
#define HIDDEN_SIZE 256        // Nodes in each hidden layer
#define OUTPUT_SIZE 10         // Number of output classes (digits 0-9)
#define LEARNING_RATE 0.01     // Learning rate for weight updates

// Activation function: ReLU
double relu(double x) {
    return x > 0 ? x : 0;
}

// Derivative of ReLU function
double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

// Structure to represent the neural network
typedef struct {
    int num_layers;        // Total number of layers in the network
    int *layers;           // Array containing the size of each layer
    double **neurons;      // 2D array for neuron activations
    double **net_inputs;   // 2D array for storing net inputs to neurons
    double **biases;       // 2D array for biases of each neuron
    double ***weights;     // 3D array for weights between layers
    double **errors;       // 2D array for storing error terms
} NeuralNetwork;

// Function prototypes
NeuralNetwork *create_network(int input_size, int hidden_layers, int hidden_size, int output_size);
void free_network(NeuralNetwork *nn);
void forward_propagation(NeuralNetwork *nn, double *input);
void back_propagation(NeuralNetwork *nn, double *target);
void update_weights(NeuralNetwork *nn);
void train(NeuralNetwork *nn, double **inputs, double **targets, int samples, int epochs);
void read_data(const char *images_file, const char *labels_file, double **inputs, double **targets, int samples);
void test_network(NeuralNetwork *nn, double **inputs, double **targets, int samples);

int main() {
    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Create the neural network
    NeuralNetwork *nn = create_network(INPUT_SIZE, HIDDEN_LAYERS, HIDDEN_SIZE, OUTPUT_SIZE);

    int train_samples = 60000;     // Number of training samples
    int test_samples = 10000;      // Number of test samples
    int epochs = 4;               // Number of training epochs

    // Allocate memory for training inputs and targets
    double **train_inputs = malloc(train_samples * sizeof(double *));
    double **train_targets = malloc(train_samples * sizeof(double *));
    for (int i = 0; i < train_samples; i++) {
        train_inputs[i] = malloc(INPUT_SIZE * sizeof(double));
        train_targets[i] = malloc(OUTPUT_SIZE * sizeof(double));
    }

    // Read training data from files
    read_data("dataset_work_dir/training_images.txt", "dataset_work_dir/training_labels.txt", train_inputs, train_targets, train_samples);

    // Train the network
    train(nn, train_inputs, train_targets, train_samples, epochs);

    // Free training data
    for (int i = 0; i < train_samples; i++) {
        free(train_inputs[i]);
        free(train_targets[i]);
    }
    free(train_inputs);
    free(train_targets);

    // Allocate memory for test inputs and targets
    double **test_inputs = malloc(test_samples * sizeof(double *));
    double **test_targets = malloc(test_samples * sizeof(double *));
    for (int i = 0; i < test_samples; i++) {
        test_inputs[i] = malloc(INPUT_SIZE * sizeof(double));
        test_targets[i] = malloc(OUTPUT_SIZE * sizeof(double));
    }

    // Read test data from files
    read_data("dataset_work_dir/test_images.txt", "dataset_work_dir/test_labels.txt", test_inputs, test_targets, test_samples);

    // Test the network
    test_network(nn, test_inputs, test_targets, test_samples);

    // Free test data
    for (int i = 0; i < test_samples; i++) {
        free(test_inputs[i]);
        free(test_targets[i]);
    }
    free(test_inputs);
    free(test_targets);

    // Free allocated memory for the network
    free_network(nn);

    return 0;
}

// Function to create and initialize the neural network
NeuralNetwork *create_network(int input_size, int hidden_layers, int hidden_size, int output_size) {
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    nn->num_layers = hidden_layers + 2; // Total layers: input + hidden + output
    nn->layers = malloc(nn->num_layers * sizeof(int));

    // Define the size of each layer
    nn->layers[0] = input_size; // Input layer
    for (int i = 1; i <= hidden_layers; i++) {
        nn->layers[i] = hidden_size; // Hidden layers
    }
    nn->layers[nn->num_layers - 1] = output_size; // Output layer

    // Allocate memory for neurons, biases, net inputs, errors
    nn->neurons = malloc(nn->num_layers * sizeof(double *));
    nn->net_inputs = malloc(nn->num_layers * sizeof(double *));
    nn->biases = malloc(nn->num_layers * sizeof(double *));
    nn->errors = malloc(nn->num_layers * sizeof(double *));
    nn->weights = malloc((nn->num_layers - 1) * sizeof(double **));

    // Initialize neurons, biases, and weights
    for (int i = 0; i < nn->num_layers; i++) {
        nn->neurons[i] = calloc(nn->layers[i], sizeof(double));
        nn->net_inputs[i] = calloc(nn->layers[i], sizeof(double));
        nn->biases[i] = calloc(nn->layers[i], sizeof(double));
        nn->errors[i] = calloc(nn->layers[i], sizeof(double));

        // Initialize weights and biases for layers beyond input
        if (i > 0) {
            nn->weights[i - 1] = malloc(nn->layers[i] * sizeof(double *));
            for (int j = 0; j < nn->layers[i]; j++) {
                nn->weights[i - 1][j] = malloc(nn->layers[i - 1] * sizeof(double));
                // Initialize weights and biases with small random values
                for (int k = 0; k < nn->layers[i - 1]; k++) {
                    nn->weights[i - 1][j][k] = ((double)rand() / RAND_MAX) - 0.5;
                }
                nn->biases[i][j] = ((double)rand() / RAND_MAX) - 0.5;
            }
        }
    }

    return nn;
}

// Function to free allocated memory for the neural network
void free_network(NeuralNetwork *nn) {
    for (int i = 0; i < nn->num_layers; i++) {
        free(nn->neurons[i]);
        free(nn->net_inputs[i]);
        free(nn->biases[i]);
        free(nn->errors[i]);

        // Free weights for layers beyond input
        if (i > 0) {
            for (int j = 0; j < nn->layers[i]; j++) {
                free(nn->weights[i - 1][j]);
            }
            free(nn->weights[i - 1]);
        }
    }
    free(nn->neurons);
    free(nn->net_inputs);
    free(nn->biases);
    free(nn->errors);
    free(nn->weights);
    free(nn->layers);
    free(nn);
}

// Function for forward propagation through the network
void forward_propagation(NeuralNetwork *nn, double *input) {
    // Set input layer neurons
    memcpy(nn->neurons[0], input, nn->layers[0] * sizeof(double));

    // Propagate through hidden layers using ReLU activation
    for (int i = 1; i < nn->num_layers - 1; i++) {
        for (int j = 0; j < nn->layers[i]; j++) {
            double sum = nn->biases[i][j];
            // Calculate weighted sum of inputs
            for (int k = 0; k < nn->layers[i - 1]; k++) {
                sum += nn->weights[i - 1][j][k] * nn->neurons[i - 1][k];
            }
            nn->net_inputs[i][j] = sum;         // Store net input before activation
            nn->neurons[i][j] = relu(sum);      // Apply ReLU activation function
        }
    }

    // Process output layer with softmax activation
    int output_layer = nn->num_layers - 1;
    for (int j = 0; j < nn->layers[output_layer]; j++) {
        double sum = nn->biases[output_layer][j];
        // Calculate weighted sum of inputs
        for (int k = 0; k < nn->layers[output_layer - 1]; k++) {
            sum += nn->weights[output_layer - 1][j][k] * nn->neurons[output_layer - 1][k];
        }
        nn->net_inputs[output_layer][j] = sum;  // Store net input before activation
    }
    // Apply softmax activation function to output layer
    softmax(nn->net_inputs[output_layer], nn->neurons[output_layer], nn->layers[output_layer]);
}

// Function for backpropagation to compute error terms
void back_propagation(NeuralNetwork *nn, double *target) {
    int output_layer = nn->num_layers - 1;

    // Compute error at output layer (Softmax with Cross-Entropy Loss)
    for (int i = 0; i < nn->layers[output_layer]; i++) {
        double output = nn->neurons[output_layer][i];
        nn->errors[output_layer][i] = output - target[i];  // Derivative simplifies to output - target
    }

    // Backpropagate errors to hidden layers
    for (int i = nn->num_layers - 2; i > 0; i--) {
        for (int j = 0; j < nn->layers[i]; j++) {
            double error = 0.0;
            // Sum weighted errors from the next layer
            for (int k = 0; k < nn->layers[i + 1]; k++) {
                error += nn->errors[i + 1][k] * nn->weights[i][k][j];
            }
            double derivative = relu_derivative(nn->net_inputs[i][j]);
            nn->errors[i][j] = error * derivative;
        }
    }
}

// Function to update weights and biases based on error terms
void update_weights(NeuralNetwork *nn) {
    for (int i = 1; i < nn->num_layers; i++) {
        // Update biases and weights for each neuron
        for (int j = 0; j < nn->layers[i]; j++) {
            nn->biases[i][j] -= LEARNING_RATE * nn->errors[i][j];
            for (int k = 0; k < nn->layers[i - 1]; k++) {
                nn->weights[i - 1][j][k] -= LEARNING_RATE * nn->errors[i][j] * nn->neurons[i - 1][k];
            }
        }
    }
}

// Function to train the neural network with given inputs and targets
void train(NeuralNetwork *nn, double **inputs, double **targets, int samples, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        // Iterate over each training sample
        for (int i = 0; i < samples; i++) {
            forward_propagation(nn, inputs[i]);   // Forward pass
            back_propagation(nn, targets[i]);     // Backward pass
            update_weights(nn);                   // Update weights and biases

            // Calculate loss (Cross-Entropy Loss)
            int output_layer = nn->num_layers - 1;
            for (int j = 0; j < nn->layers[output_layer]; j++) {
                double output = nn->neurons[output_layer][j];
                double target = targets[i][j];
                if (target > 0) {
                    total_loss += -target * log(output + 1e-15); // Add small epsilon to prevent log(0)
                }
            }
        }
        // Display loss for the current epoch
        printf("Epoch %d, Loss: %f\n", epoch + 1, total_loss / samples);
    }
}

// Function to read input data and labels from text files
void read_data(const char *images_file, const char *labels_file, double **inputs, double **targets, int samples) {
    FILE *f_images = fopen(images_file, "r");
    FILE *f_labels = fopen(labels_file, "r");

    if (!f_images || !f_labels) {
        fprintf(stderr, "Error opening data files.\n");
        exit(1);
    }

    char line[16384]; // Buffer to hold a line of input

    for (int i = 0; i < samples; i++) {
        // Read image data
        if (fgets(line, sizeof(line), f_images) == NULL) {
            fprintf(stderr, "Error reading image data at line %d\n", i + 1);
            exit(1);
        }
        // Parse image data
        char *token = strtok(line, " \t\n");
        for (int j = 0; j < INPUT_SIZE; j++) {
            if (token == NULL) {
                fprintf(stderr, "Not enough data in image line %d\n", i + 1);
                exit(1);
            }
            double pixel_value = atof(token);
            inputs[i][j] = pixel_value / 255.0; // Normalize pixel value to [0,1]
            token = strtok(NULL, " \t\n");
        }

        // Read label data
        if (fgets(line, sizeof(line), f_labels) == NULL) {
            fprintf(stderr, "Error reading label data at line %d\n", i + 1);
            exit(1);
        }
        // Parse label data
        int label = atoi(line);
        if (label < 0 || label >= OUTPUT_SIZE) {
            fprintf(stderr, "Invalid label %d at line %d\n", label, i + 1);
            exit(1);
        }
        // Initialize target vector (one-hot encoding)
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            targets[i][k] = 0.0;
        }
        targets[i][label] = 1.0;
    }

    fclose(f_images);
    fclose(f_labels);
}

// Function to test the neural network on test data
void test_network(NeuralNetwork *nn, double **inputs, double **targets, int samples) {
    int correct = 0;
    for (int i = 0; i < samples; i++) {
        forward_propagation(nn, inputs[i]);

        // Get the predicted label (index with highest probability)
        int predicted_label = 0;
        double max_prob = nn->neurons[nn->num_layers - 1][0];
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (nn->neurons[nn->num_layers - 1][j] > max_prob) {
                max_prob = nn->neurons[nn->num_layers - 1][j];
                predicted_label = j;
            }
        }

        // Get the true label
        int true_label = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (targets[i][j] == 1.0) {
                true_label = j;
                break;
            }
        }

        if (predicted_label == true_label) {
            correct++;
        }
    }

    double accuracy = ((double)correct / samples) * 100.0;
    printf("Test Accuracy: %.2f%% (%d/%d correct)\n", accuracy, correct, samples);
}
