//Written by Shi Tong Li and Michael Frajman - ECSE420 Fall2021 Group8
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>



__global__ void update_interior(double* element_grid, double* element_grid_u1, double* element_grid_u2, int N, int n, int elements_per_thread)
{
    //index is multiplied by elements_per_thread to offset
    int index = (threadIdx.x + blockIdx.x * blockDim.x)* elements_per_thread;
    int row = index / N + 1;
    //repeat for every element per thread 
    for (int i = 0; i < elements_per_thread; i++) {
        if (index < n && index % N != 0 && index % N != N - 1 && row != 1 && index < N * (N - 1)) {
            element_grid[index] = (0.5 * (element_grid_u1[index - N] + element_grid_u1[index + N] + element_grid_u1[index - 1] + element_grid_u1[index + 1] - 4 * element_grid_u1[index]) + 2 * element_grid_u1[index] - (1 - 0.0002) * (element_grid_u2[index])) / (1 + 0.0002);
        }
        index++;
    }
    


}

__global__ void update_boundary(double* element_grid, int N, int n, int elements_per_thread) {
    //index is multiplied by elements_per_thread to offset
    int index = (threadIdx.x + blockIdx.x * blockDim.x)* elements_per_thread;

    int row = index / N + 1;
    //repeat for every element per thread 

    for (int i = 0; i < elements_per_thread; i++) {
        if (index < n && row == 1 && index % N != 0 && index % N != N - 1) {
            element_grid[index] = 0.75 * element_grid[index + N];
        }
        if (index < n && index >= N * (N - 1) && index % N != 0 && index % N != N - 1) {
            element_grid[index] = 0.75 * element_grid[index - N];
        }
        if (index < n && index % N == 0 && row != 1 && index < N * (N - 1)) {
            element_grid[index] = 0.75 * element_grid[index + 1];
        }
        if (index < n && index % N == N - 1 && row != 1 && index < N * (N - 1)) {
            element_grid[index] = 0.75 * element_grid[index - 1];
        }
        index++;
    }
}


//no need to do 16 elements per thread since there are only 4 items to update
__global__ void update_corners(double* element_grid, int N, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index == 0) {
        element_grid[0] = 0.75 * element_grid[N];
    }
    if (index == 1) {
        element_grid[(N - 1) * N] = 0.75 * element_grid[(N - 1) * N - N];

    }
    if (index == 2) {
        element_grid[N - 1] = 0.75 * element_grid[N - 1 - 1];

    }
    if (index == 3) {
        element_grid[N * N - 1] = 0.75 * element_grid[N * N - 1 - 1];

    }
}

int main(int argc, char** argv)
{
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //HARD CODED VARIABLES FOR INPUT OUTPUT
    int T = 16; //number of iterations
    const int N = 512;
    int thread_number = 1024;
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    double *element_grid = new double[N*N]();
    double *element_grid_u1 = new double[N * N]();
    double *element_grid_u2 = new double[N * N]();
    

    double* cuda_grid;
    double* cuda_grid_u1;
    double* cuda_grid_u2;

    //allocating gpu memory unified
    cudaMallocManaged((void**)&element_grid, N*N * sizeof(double));
    cudaMallocManaged((void**)&element_grid_u1, N * N * sizeof(double));
    cudaMallocManaged((void**)&element_grid_u2, N * N * sizeof(double));
    //initiate drum hit
    cudaMemset(element_grid, 0, N * N * sizeof(double));
    cudaMemset(element_grid_u1, 0, N * N * sizeof(double));
    cudaMemset(element_grid_u2, 0, N * N * sizeof(double));
    element_grid_u1[(N * N / 2) + N / 2] = 1;

    cudaMalloc((void**)&cuda_grid, N * N * sizeof(double));
    cudaMalloc((void**)&cuda_grid_u1, N * N * sizeof(double));
    cudaMalloc((void**)&cuda_grid_u2, N * N * sizeof(double));


    //copy to device
    cudaMemcpy(cuda_grid, element_grid, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_grid_u1, element_grid_u1, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_grid_u2, element_grid_u2, N * N * sizeof(double), cudaMemcpyHostToDevice);

    //loop for number of iterations
    //calculating the number of blocks needed, the grid, and block size
    int  number_of_blocks = (int)sqrt(N*N/thread_number);
    int elements_per_thread = N * N / (thread_number * number_of_blocks);
    dim3 grid(number_of_blocks, 1, 1);
    dim3 block(thread_number, 1, 1);

    struct GpuTimer timer;
    timer.Start();
    for (int i = 0; i < T; i++) {
        //do interior element first
        
        update_interior <<< grid, block >>> (cuda_grid, cuda_grid_u1, cuda_grid_u2, N, N * N, elements_per_thread);
        cudaDeviceSynchronize();
        
        update_boundary <<< grid, block >>> (cuda_grid, N, N * N, elements_per_thread);
        cudaDeviceSynchronize();

        update_corners <<< grid, 1 >>> (cuda_grid, N, 4);
        cudaDeviceSynchronize();
        //print required statement to terminal
        cudaMemcpy(element_grid, cuda_grid, N * N * sizeof(double), cudaMemcpyDeviceToHost);

        printf("\n (%d, %d): %f", N / 2, N / 2, element_grid[(N * N / 2) + N / 2]);
        //u1 becomes u1, and current values become u1
        cudaMemcpy(cuda_grid_u2, cuda_grid_u1, N * N * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(cuda_grid_u1, cuda_grid, N * N * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    timer.Stop();
    printf("\ntimer: %f", timer.Elapsed());

    return 0;
}
