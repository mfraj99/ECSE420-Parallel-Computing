// Written by Michael Frajman and Shi Tong Li - ECSE420 Fall2021 Group 8
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
__global__ void rectification(unsigned char* input, unsigned char* output, int n)
{
    //calculate the index of the current thread 
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        //the png_index is 4*index since every pixel contains 4 unsigned char values
        int png_index = 4 * index;
        //rectifying R value
        if ((int)input[png_index] < 127) {
            output[png_index] = (unsigned char)127;
        }
        else {
            output[png_index] = input[png_index];
        }
        //rectifying G value
        if ((int)input[png_index + 1] < 127) {
            output[png_index + 1] = (unsigned char)127;
        }
        else {
            output[png_index + 1] = input[png_index + 1];
        }
        //rectifying B value
        if ((int)input[png_index + 2] < 127) {
            output[png_index + 2] = (unsigned char)127;
        }
        else {
            output[png_index + 2] = input[png_index + 2];
        }
        //A value stays the same
        output[png_index + 3] = input[png_index + 3];
    }
      
}

int main(int argc, char** argv)
{

    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //HARD CODED VARIABLES FOR INPUT OUTPUT
    char* png_input = "Test_3.png";
    char* png_output = "output.png";
    int thread_number = 128;
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////

    //declaring variables used with lodepng and cuda
    unsigned error;
    unsigned char* image_host; //stores png info on host
    unsigned char* image_cuda; //copy of png info on gpu
    unsigned char* new_image_cuda; //output of png on gpu
    unsigned width, height; //width and height of png image
  

    
    //using lodepng decode to store the values of pixels of the image inside image_host
    error = lodepng_decode32_file(&image_host, &width, &height, png_input);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

    //allocating gpu memory
    cudaMalloc((void**)&image_cuda, width * height * 4 * sizeof(unsigned char));
    cudaMalloc((void**)&new_image_cuda, width * height * 4 * sizeof(unsigned char));

    //copying png information onto the gpu
    cudaMemcpy(image_cuda, image_host, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //calculating the number of blocks needed, the grid, and block size
    int  number_of_blocks = (width * height / thread_number);
    dim3 grid(number_of_blocks, 1, 1);
    dim3 block(thread_number, 1, 1);

    //timer for test
    struct GpuTimer timer;
    timer.Start();

    //calling the rectification function on the gpu
    rectification <<<grid, block>>> (image_cuda, new_image_cuda, height*width);
    timer.Stop();
    printf("timer: %f", timer.Elapsed());
    cudaDeviceSynchronize();

    //declaring variable to store the gpu output onto the host machine
    unsigned char* final_image = new unsigned char[width * height * 4]();
    //copying the gpu output onto the host machine
    cudaMemcpy(final_image, new_image_cuda, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    //encode the resulting gpu output
    lodepng_encode32_file(png_output, final_image, width, height);
    cudaFree(image_cuda);
    cudaFree(new_image_cuda);
   
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //testing sequential version of rectification algorithm
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //unsigned char zero = (unsigned char)127;
    //for (int i = 0; i < height; i++) {
    //    for (int j = 0; j < width; j++) {	
    //    
    //        int index = 4 * i * width + 4 * j;
    //        printf("%d", (int)image[index]);
    //        printf("%u", (unsigned)image[index]);
    //
    //        if ((int)image[index] < 127) {
    //            new_image[index] = zero;
    //        }
    //        else {
    //            new_image[index] = image[index];
    //        }
    //
    //        if ((int)image[index + 1] < 127) {
    //            new_image[index + 1] = zero;
    //        }
    //        else {
    //            new_image[index + 1] = image[index + 1];
    //        }
    //
    //        if ((int)image[index + 2] < 127) {
    //            new_image[index + 2] = zero;
    //        }
    //        else {
    //            new_image[index + 2] = image[index + 2];
    //        }
    //
    //        new_image[index + 3] = image[index + 3];
    //    }
    //}

    return 0;
}
