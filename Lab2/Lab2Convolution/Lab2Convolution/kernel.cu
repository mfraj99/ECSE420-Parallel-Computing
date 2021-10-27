//Written by Shi Tong Li and Michael Frajman - ECSE420 Fall2021 Group8
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lodepng.h"
#include "gputimer.h"
#include "wm.h"
#include <stdio.h>
#include <stdlib.h>

// pools pixel values of a png image
// input - pointer to array of pixels of input image
// output - pointer to array of pixels for output image
// width - width of input image
// n - limit of the numebr of threads
__global__ void convolution(unsigned char* input, unsigned char* output, int width, int height, int n, float w_cuda[3][3])
{
    //calculate the current index and row position of the current thread
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int row = index / width +1;

    //if the index is even and is on an even row of the image, do pooling operation
    if (index<n && index % width != 0 && index % width != width - 1 && row != 1 && index < width * (height - 1)) {
        //get a 3x3 grid
        int png_index_11 = 4 * index - 4* width-4; //first pixel of the first row
        int png_index_12 = 4 * index - 4 * width;
        int png_index_13 = 4 * index - 4 * width +4;
        int png_index_21 = 4 * index-4;//first  pixel of the second row
        int png_index_22 = 4 * index;
        int png_index_23 = 4 * index + 4;
        int png_index_31 = 4 * (index)+4 * width-4; 
        int png_index_32 = 4 * (index)+4 * width;
        int png_index_33 = 4 * (index)+4 * width+4; //first pixel of the second row
        
        int r = w_cuda[0][0] * (int)input[png_index_11] + w_cuda[0][1] * (int)input[png_index_12] + w_cuda[0][2] * (int)input[png_index_13] + w_cuda[1][0] * (int)input[png_index_21] + w_cuda[1][1] * (int)input[png_index_22] + w_cuda[1][2] * (int)input[png_index_23] + w_cuda[2][0] * (int)input[png_index_31] + (int)w_cuda[2][1] * (int)input[png_index_32] + w_cuda[2][2] * (int)input[png_index_33];
        int g = w_cuda[0][0] * (int)input[png_index_11+1] + w_cuda[0][1] * (int)input[png_index_12+1] + w_cuda[0][2] * (int)input[png_index_13+1] + w_cuda[1][0] * (int)input[png_index_21+1] + w_cuda[1][1] * (int)input[png_index_22+1] + w_cuda[1][2] * (int)input[png_index_23+1] + w_cuda[2][0] * (int)input[png_index_31+1] + w_cuda[2][1] * (int)input[png_index_32+1] + w_cuda[2][2] * (int)input[png_index_33+1];
        int b = w_cuda[0][0] * (int)input[png_index_11+2] + w_cuda[0][1] * (int)input[png_index_12+2] + w_cuda[0][2] * (int)input[png_index_13+2] + w_cuda[1][0] * (int)input[png_index_21+2] + w_cuda[1][1] * (int)input[png_index_22+2] + w_cuda[1][2] * (int)input[png_index_23+2] + w_cuda[2][0] * (int)input[png_index_31+2] + w_cuda[2][1] * (int)input[png_index_32+2] + w_cuda[2][2] * (int)input[png_index_33+2];
        int a = w_cuda[0][0] * (int)input[png_index_11+3] + w_cuda[0][1] * (int)input[png_index_12+3] + w_cuda[0][2] * (int)input[png_index_13+3] + w_cuda[1][0] * (int)input[png_index_21+3] + w_cuda[1][1] * (int)input[png_index_22+3] + w_cuda[1][2] * (int)input[png_index_23+3] + w_cuda[2][0] * (int)input[png_index_31+3] + w_cuda[2][1] * (int)input[png_index_32+3] + w_cuda[2][2] * (int)input[png_index_33+3];
        if (r > 255) {
            r = 255;
        }
        if (r < 0) {
            r = 0;
        }
        if (g > 255) {
            g = 255;
        }
        if (g < 0) {
            g = 0;
        }
        if (b > 255) {
            b = 255;
        }
        if (b < 0) {
            b = 0;
        }
        if (a > 255) {
            a = 255;
        }
        if (a < 0) {
            a = 0;
        }
        //calculate the new position of the resulting pixel on the rebuilt output image
        int new_image_index = index - width - 2 * (row - 2) - 1;
        //store the RGBA values of the resulting pixel to the corresponding position in the output array
        output[4 * new_image_index] = (unsigned char)r;
        output[4 * new_image_index + 1] = (unsigned char)g;
        output[4 * new_image_index + 2] = (unsigned char)b;
        output[4 * new_image_index + 3] = input[4*index+3];
    }

}

int main(int argc, char** argv)
{
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //HARD CODED VARIABLES FOR INPUT OUTPUT
    char* png_input = "Test_1.png";
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
    cudaMalloc((void**)&new_image_cuda, width * height* 4 * sizeof(unsigned char));

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
    convolution<<< grid, block >>> (image_cuda, new_image_cuda, width, height, height * width, w);
    timer.Stop();
    printf("timer: %f", timer.Elapsed());
    cudaDeviceSynchronize();

    
    //declaring a variable to store the ouput of the gpu pooling function
    unsigned char* final_image = new unsigned char[4*(width-2) * (height-2)]();
    //copy the ouput of gpu onto the variable on the host machine
    cudaMemcpy(final_image, new_image_cuda, 4*(width - 2) * (height - 2) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    //encode the output array into a png image a quarter of the size of the first
    

    //for (int index = 0; index < (int)height * (int)width; index++) {
    //    int row = index / (int)width + 1;

    //    //if the index is even and is on an even row of the image, do pooling operation
    //    if (index % width !=0 && index % width != width-1 && row != 1 && index < width*(height-1)) {
    //        //get a 3x3 grid
    //        int png_index_11 = 4 * index - 4 * width - 4; //first pixel of the first row
    //        int png_index_12 = 4 * index - 4 * width;
    //        int png_index_13 = 4 * index - 4 * width + 4;
    //        int png_index_21 = 4 * index - 4;//first  pixel of the second row
    //        int png_index_22 = 4 * index;
    //        int png_index_23 = 4 * index + 4;
    //        int png_index_31 = 4 * (index)+4 * width - 4;
    //        int png_index_32 = 4 * (index)+4 * width;
    //        int png_index_33 = 4 * (index)+4 * width + 4; //first pixel of the second row

    //        int r = (int)(w[0][0] * (float)image_host[png_index_11] + w[0][1] * (float)image_host[png_index_12] + w[0][2] * (float)image_host[png_index_13] + w[1][0] * (float)image_host[png_index_21] + w[1][1] * (float)image_host[png_index_22] + w[1][2] * (float)image_host[png_index_23] + w[2][0] * (float)image_host[png_index_31] + w[2][1] * (float)image_host[png_index_32] + w[2][2] * (float)image_host[png_index_33]);
    //        int g = (int)(w[0][0] * (float)image_host[png_index_11 + 1] + w[0][1] * (float)image_host[png_index_12 + 1] + w[0][2] * (float)image_host[png_index_13 + 1] + w[1][0] * (float)image_host[png_index_21 + 1] + w[1][1] * (float)image_host[png_index_22 + 1] + w[1][2] * (float)image_host[png_index_23 + 1] + w[2][0] * (float)image_host[png_index_31 + 1] + w[2][1] * (float)image_host[png_index_32 + 1] + w[2][2] * (float)image_host[png_index_33 + 1]);
    //        int b = (int)(w[0][0] * (float)image_host[png_index_11 + 2] + w[0][1] * (float)image_host[png_index_12 + 2] + w[0][2] * (float)image_host[png_index_13 + 2] + w[1][0] * (float)image_host[png_index_21 + 2] + w[1][1] * (float)image_host[png_index_22 + 2] + w[1][2] * (float)image_host[png_index_23 + 2] + w[2][0] * (float)image_host[png_index_31 + 2] + w[2][1] * (float)image_host[png_index_32 + 2] + w[2][2] * (float)image_host[png_index_33 + 2]);
    //        int a = (int)(w[0][0] * (float)image_host[png_index_11 + 3] + w[0][1] * (float)image_host[png_index_12 + 3] + w[0][2] * (float)image_host[png_index_13 + 3] + w[1][0] * (float)image_host[png_index_21 + 3] + w[1][1] * (float)image_host[png_index_22 + 3] + w[1][2] * (float)image_host[png_index_23 + 3] + w[2][0] * (float)image_host[png_index_31 + 3] + w[2][1] * (float)image_host[png_index_32 + 3] + w[2][2] * (float)image_host[png_index_33 + 3]);
    //        if (r > 255) {
    //            r = 255;
    //        }
    //        if (r < 0) {
    //            r = 0;
    //        }
    //        if (g > 255) {
    //            g = 255;
    //        }
    //        if (g < 0) {
    //            g = 0;
    //        }
    //        if (b > 255) {
    //            b = 255;
    //        }
    //        if (b < 0) {
    //            b = 0;
    //        }
    //        if (a > 255) {
    //            a = 255;
    //        }
    //        if (a < 0) {
    //            a = 0;
    //        }
    //       
    //        //calculate the new position of the resulting pixel on the rebuilt output image
    //        int new_image_index = index - width - 2*(row-2)-1;
    //        //store the RGBA values of the resulting pixel to the corresponding position in the output array
    //        final_image[4 * new_image_index] = (unsigned char)r;
    //        final_image[4 * new_image_index + 1] = (unsigned char)g;
    //        final_image[4 * new_image_index + 2] = (unsigned char)b;
    //        final_image[4 * new_image_index + 3] = image_host[4*index+3];
    //    }
    //}







    lodepng_encode32_file(png_output, final_image, width-2, height-2);
    cudaFree(image_cuda);
    cudaFree(new_image_cuda);

    return 0;
}
