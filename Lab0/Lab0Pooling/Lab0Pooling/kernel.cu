
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lodepng.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>

// rectify pixel values of a png image
__global__ void pooling(unsigned char* input, unsigned char* output, int width, int height, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int row = index / width;
    int column = index / height;

    if (index < n && row%2==0 && column%2==0) {
        int png_index_1 = 4 * index;
        int png_index_2 = 4 * (index)+4;
        int png_index_3 = 4 * (index) + 4*width;
        int png_index_4 = 4 * (index) + 4*(width+1);
        //compare R
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
        //compare G
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
        //compare B
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
        //compare A
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
        //int new_image_index_row = (index % (width/2))/2;
        //int new_image_index = (index%width)/2 + height*(column/4);
        int new_image_index = (index % width) / 2 + (width / 2) * (row / 2);

        output[4 * new_image_index] = (unsigned char)r;
        output[4 * new_image_index + 1] = (unsigned char)g;
        output[4 * new_image_index + 2] = (unsigned char)b;
        output[4 * new_image_index + 3] = (unsigned char)a;
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
    cudaMalloc((void**)&new_image_cuda, width * height * sizeof(unsigned char));

    cudaMemcpy(image_cuda, image_host, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //memcpy(image_cuda, image, width * height * 4 * sizeof(unsigned char));
    int  number_of_blocks = (width * height / thread_number);
    dim3 grid(number_of_blocks, 1, 1);
    dim3 block(thread_number, 1, 1);

    struct GpuTimer timer;
    timer.Start();

    pooling <<<grid, block>>> (image_cuda, new_image_cuda, width, height, height*width);
    timer.Stop();
    printf("timer: %f", timer.Elapsed());
    cudaDeviceSynchronize();

    unsigned char* final_image = new unsigned char[width * height]();
    cudaMemcpy(final_image, new_image_cuda, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    lodepng_encode32_file(png_output, final_image, width/2, height/2);
    cudaFree(image_cuda);
    cudaFree(new_image_cuda);

    return 0;
}
