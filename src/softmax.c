#include <math.h>
#include "softmax.h"

void softmax(double* input, double* output, int length) {
    double max = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        output[i] = exp(input[i] - max);
        sum += output[i];
    }

    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}