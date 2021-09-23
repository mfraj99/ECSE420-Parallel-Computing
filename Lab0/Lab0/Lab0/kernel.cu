
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

// rectify pixel values of a png image
// input - pointer to array of pixels of input image
// output - pointer to array of pixels for output image
// n - limit of the numebr of threads
__global__ void rectification(unsigned char* input, unsigned char* output, int n)
{
    int i = threadIdx.x;
    if (i < n) {
        if ((int)input[i] < 127) {
            output[i] = (unsigned char)127;
        }
        else {
            output[i] = input[i];
        }
    }
}

void process(char* input_filename, char* output_filename)
{
    printf("process lmao");

    unsigned error;
    unsigned char* image, * new_image;
    unsigned width, height;

    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    new_image = (unsigned char*)malloc(width * height * 4 * sizeof(unsigned char));

    // process image
    unsigned char value;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

            value = image[4 * width * i + 4 * j];

            new_image[4 * width * i + 4 * j + 0] = value; // R
            new_image[4 * width * i + 4 * j + 1] = value; // G
            new_image[4 * width * i + 4 * j + 2] = value; // B
            new_image[4 * width * i + 4 * j + 3] = image[4 * width * i + 4 * j + 3]; // A
        }
    }

    lodepng_encode32_file(output_filename, new_image, width, height);

    free(image);
    free(new_image);
}

int main(int argc, char** argv)
{

    printf("test lmao");

    process("Test_1.png", "output.png");

    if (strcmp(argv[0], "rectify") == 0) {
        char* png_input = argv[1];
        char* png_output = argv[2];
        int thread_number = atoi(argv[3]);

        unsigned error;
        unsigned char* image = 0;
        unsigned width, height;

        //process(png_input, png_output);

        /*cudaMallocManaged((void**)&image, width * height * 4 * sizeof(unsigned char));
        cudaMallocManaged((void**)&new_image, width * height * 4 * sizeof(unsigned char));

        error = lodepng_decode32_file(&image, &width, &height, png_input);
        if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
        new_image = (unsigned char*)malloc(width * height * 4 * sizeof(unsigned char));

        rectification << < 1, thread_number >> > (image, new_image, thread_number);

        cudaDeviceSynchronize();

        lodepng_encode32_file(png_output, new_image, width, height);

        cudaFree(image);
        cudaFree(new_image);

        free(image);
        free(new_image);*/

        return 0;
    }

    return 0;
}
