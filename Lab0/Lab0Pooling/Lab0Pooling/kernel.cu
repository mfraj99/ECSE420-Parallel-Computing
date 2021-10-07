//Written by Shi Tong Li and Michael Frajman - ECSE420 Fall2021 Group8
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lodepng.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>

// pools pixel values of a png image
// input - pointer to array of pixels of input image
// output - pointer to array of pixels for output image
// width - width of input image
// n - limit of the numebr of threads
__global__ void pooling(unsigned char* input, unsigned char* output, int width, int n)
{
    //calculate the current index and row position of the current thread
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int row = index / width;

    //if the index is even and is on an even row of the image, do pooling operation
    if (index < n && row%2==0 && index%2==0) {
        //get a 2x2 grid
        int png_index_1 = 4 * index; //first pixel of the first row
        int png_index_2 = 4 * (index)+4; //second pixel of the first row
        int png_index_3 = 4 * (index) + 4*width; //first pixel of the second row
        int png_index_4 = 4 * (index) + 4*(width+1); //second pixel of the second row
        //compare R and get the biggest value
        int r = input[png_index_1];
        if (r < (int)input[png_index_2]) {
            r = input[png_index_2];
        }
        if (r < (int)input[png_index_3]) {
            r = input[png_index_3];
        }
        if (r < (int)input[png_index_4]) {
            r = input[png_index_4];
        }
        //compare G and get the biggest value
        int g = input[png_index_1+1];
        if (g < (int)input[png_index_2+1]) {
            g = input[png_index_2+1];
        }
        if (g < (int)input[png_index_3+1]) {
            g = input[png_index_3+1];
        }
        if (g < (int)input[png_index_4+1]) {
            g = input[png_index_4+1];
        }
        //compare B and get the biggest value
        int b = input[png_index_1+2];
        if (b < (int)input[png_index_2+2]) {
            b = input[png_index_2+2];
        }
        if (b < (int)input[png_index_3+2]) {
            b = input[png_index_3+2];
        }
        if (b < (int)input[png_index_4+2]) {
            b = input[png_index_4+2];
        }
        //compare A and get the biggest value
        int a = input[png_index_1 + 3];
        if (a < (int)input[png_index_2 + 3]) {
            a = input[png_index_2 + 3];
        }
        if (a < (int)input[png_index_3 + 3]) {
            a = input[png_index_3 + 3];
        }
        if (a < (int)input[png_index_4 + 3]) {
            a= input[png_index_4 + 3];
        }
        //calculate the new position of the resulting pixel on the rebuilt output image
        int new_image_index = (index % width) / 2 + (width / 2) * (row / 2);
        //store the RGBA values of the resulting pixel to the corresponding position in the output array
        output[4 * new_image_index] = (unsigned char)r;
        output[4 * new_image_index + 1] = (unsigned char)g;
        output[4 * new_image_index + 2] = (unsigned char)b;
        output[4 * new_image_index + 3] = (unsigned char)a;
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
    int thread_number = 256;
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
    cudaMalloc((void**)&new_image_cuda, width * height * sizeof(unsigned char));

    //copying png information onto the gpu
    cudaMemcpy(image_cuda, image_host, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //calculating the number of blocks needed, the grid, and block size
    int  number_of_blocks = (width * height / thread_number);
    dim3 grid(number_of_blocks, 1, 1);
    dim3 block(thread_number, 1, 1);

    //timer for test
    struct GpuTimer timer;
    timer.Start();

    //calling gpu function for pooling
    pooling <<<grid, block>>> (image_cuda, new_image_cuda, width, height*width);
    timer.Stop();
    printf("timer: %f", timer.Elapsed());
    cudaDeviceSynchronize();

    //declaring a variable to store the ouput of the gpu pooling function
    unsigned char* final_image = new unsigned char[width * height]();
    //copy the ouput of gpu onto the variable on the host machine
    cudaMemcpy(final_image, new_image_cuda, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    //encode the output array into a png image a quarter of the size of the first
    lodepng_encode32_file(png_output, final_image, width/2, height/2);
    cudaFree(image_cuda);
    cudaFree(new_image_cuda);

    return 0;
}
