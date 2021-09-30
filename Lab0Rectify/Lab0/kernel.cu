
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lodepng.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>

// rectify pixel values of a png image
// input - pointer to array of pixels of input image
// output - pointer to array of pixels for output image
// n - limit of the numebr of threads
__global__ void rectification(unsigned char* input, unsigned char* output, int height, int width, int n)
{
    //int i = threadIdx.x + blockIdx.x * blockDim.x;
    //int j = threadIdx.y + blockIdx.x * blockDim.y;
     
    //int png_index = 4*width*j + 4*i;
    //int png_index = 4 * width * threadIdx.x + 4 * blockIdx.x * blockDim.x;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("%u %u", i, j);
    if (index < n) {
        int png_index = 4 * index;
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

int main(int argc, char** argv)
{

    char* png_input = "Test_1.png";
    char* png_output = "output.png";
    int thread_number = 32;

    unsigned error;
    
    unsigned char* image_host;
    unsigned char* image_cuda;
    unsigned char* new_image_cuda;
    unsigned width, height;
    unsigned* width_cuda;
    unsigned* height_cuda;

    

    error = lodepng_decode32_file(&image_host, &width, &height, png_input);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

    cudaMalloc((void**)&image_cuda, width * height * 4 * sizeof(unsigned char));
    cudaMalloc((void**)&new_image_cuda, width * height * 4 * sizeof(unsigned char));

    cudaMemcpy(image_cuda, image_host, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //memcpy(image_cuda, image, width * height * 4 * sizeof(unsigned char));
    int  number_of_blocks = (width * height / thread_number);
    dim3 grid(number_of_blocks, 1, 1);
    dim3 block(thread_number, 1, 1);

    struct GpuTimer timer;
    timer.Start();

    rectification <<<grid, block>>> (image_cuda, new_image_cuda, height, width, height*width);
    timer.Stop();
    printf("timer: %f", timer.Elapsed());
    cudaDeviceSynchronize();

    /*for (int i = 0; i < 100; i++) {
        printf("lel:%u%u\n", (unsigned)image[i], (unsigned)image_cuda[i]);
    }*/

   

   // rectification <<< (height+threads_per_block-1)/threads_per_block, threads_per_block >>> (image_cuda, new_image_cuda, height, width, height);

    
    
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
    //memcpy(image, new_image_cuda, width * height * 4 * sizeof(unsigned char));
    /*for (int i = 0; i < 100; i++) {
        printf("lol:%u%u\n", (unsigned)image_host[i], (unsigned)final_image[i]);
    }*/
    unsigned char* final_image = new unsigned char[width * height * 4]();
    cudaMemcpy(final_image, new_image_cuda, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    lodepng_encode32_file(png_output, final_image, width, height);
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
