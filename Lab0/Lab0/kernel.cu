
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

// rectify pixel values of a png image
// input - pointer to array of pixels of input image
// output - pointer to array of pixels for output image
// n - limit of the numebr of threads
__global__ void rectification(unsigned char* input, unsigned char* output, int n, int width, int height)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        for (int j = 0; j < width; j++) {
            int png_index = 4 * width * index + 4 * j;

            if ((int)input[png_index] < 127) {
                output[png_index] = (unsigned char)127;
            }
            else {
                output[png_index] = input[png_index];
            }

            if ((int)input[png_index + 1] < 127) {
                output[png_index + 1] = (unsigned char)127;
            }
            else {
                output[png_index + 1] = input[png_index + 1];
            }

            if ((int)input[png_index + 2] < 127) {
                output[png_index + 2] = (unsigned char)127;
            }
            else {
                output[png_index + 2] = input[png_index + 2];
            }

            output[png_index + 3] = input[png_index + 3];
        }
    }
        
}

int main(int argc, char** argv)
{

    char* png_input = "Test_1.png";
    char* png_output = "output.png";
    int thread_number = 32;

    unsigned error;
    unsigned char* image;
    unsigned char* image_cuda;
    unsigned char* new_image_cuda;
    unsigned width, height;

    error = lodepng_decode32_file(&image, &width, &height, png_input);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

    cudaMallocManaged(&image_cuda, width * height * 4 * sizeof(unsigned char));
    cudaMallocManaged(&new_image_cuda, width * height * 4 * sizeof(unsigned char));

    error = lodepng_decode32_file(&image_cuda, &width, &height, png_input);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

    //memcpy(image_cuda, image, width * height * 4 * sizeof(unsigned char));

    for (int i = 0; i < 100; i++) {
        printf("lel:%u%u\n", (unsigned)image[i], (unsigned)image_cuda[i]);
    }

    int threads_per_block = height/thread_number;

    rectification <<< (height+threads_per_block-1)/threads_per_block, threads_per_block >>> (image_cuda, new_image_cuda, height, width, height);

    cudaDeviceSynchronize();
    
    /*for (int i = 0; i < 100; i++) {
        printf("lol:%u%u\n", (unsigned)image[i], (unsigned)new_image_cuda[i]);
    }*/

    //testing sequential version of rectification algorithm

    //unsigned char zero = (unsigned char)127;

    //for (int i = 0; i < height; i++) {
    //    for (int j = 0; j < width; j++) {	
    //    
    //        int index = 4 * i * width + 4 * j;
    //        printf("%d", (int)image[index]);
    //        printf("%u", (unsigned)image[index]);

    //        if ((int)image[index] < 127) {
    //            new_image[index] = zero;
    //        }
    //        else {
    //            new_image[index] = image[index];
    //        }

    //        if ((int)image[index + 1] < 127) {
    //            new_image[index + 1] = zero;
    //        }
    //        else {
    //            new_image[index + 1] = image[index + 1];
    //        }

    //        if ((int)image[index + 2] < 127) {
    //            new_image[index + 2] = zero;
    //        }
    //        else {
    //            new_image[index + 2] = image[index + 2];
    //        }

    //        new_image[index + 3] = image[index + 3];
    //    }
    //}

    //cudaMemcpy(final_image, new_image, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    lodepng_encode32_file(png_output, new_image_cuda, width, height);

    cudaFree(image_cuda);
    cudaFree(new_image_cuda);

    /*free(image);
    free(new_image);*/

   /* if (strcmp(argv[0], "rectify") == 0) {
        char* png_input = argv[1];
        char* png_output = argv[2];
        int thread_number = atoi(argv[3]);

        

        return 0;
    }*/

    return 0;
}
