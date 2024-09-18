#include <stdio.h>
#include "softmax.h"  

int main() {
    double input[] = {1.0, 2.0, 3.0};
    int length = sizeof(input) / sizeof(input[0]);
    double output[length];

    // Call the softmax function
    softmax(input, output, length);

    printf("Softmax result:\n");
    for (int i = 0; i < length; i++) {
        printf("%f\n", output[i]);
    }

    return 0;
}