//Written by Shi Tong Li and Michael Frajman ECSE420 Fal2021

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>

#define AND     0
#define OR      1
#define NAND    2
#define NOR     3
#define XOR     4
#define XNOR    5



__global__ void logic_gates(int* input1_array, int* input2_array, int* gate_array, int* output, int n)
{

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        int bit;
        int input1 = input1_array[index];
        int input2 = input2_array[index];
        int gate = gate_array[index];
        switch (gate) {
        case AND:
            bit = input1 & input2;
            break;
        case OR:
            bit = input1 | input2;
            break;
        case NAND:
            bit = !(input1 & input2);
            break;
        case NOR:
            bit = !(input1 | input2);
            break;
        case XOR:
            if ((input1 == 0 && input2 == 1) || (input1 == 1 && input2 == 0)) {
                bit = 1;
            }
            else {
                bit = 0;
            }
            break;
        case XNOR:
            if ((input1 == 0 && input2 == 0) || (input1 == 1 && input2 == 1)) {
                bit = 1;
            }
            else {
                bit = 0;
            }
            break;
        }
        output[index] = bit;
    }

}

int main()
{
    FILE* input, * output;
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //HARD CODED VARIABLES FOR INPUT OUTPUT
    input = fopen("input_10000.txt", "r");
    output = fopen("output_10000.txt", "w");
    int thread_number = 1024;
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////


    //count how many lines are there in the program, taken from https://stackoverflow.com/questions/12733105/c-function-that-counts-lines-in-file
    int number_of_lines = 0;
    while (!feof(input)) {
        char ch = fgetc(input);
        if (ch == '\n') {
            number_of_lines++;
        }
    }
    rewind(input);
    char line[10];

    //declaring variables used 
    unsigned error;
    int* input1_unified;
    int* input2_unified;
    int* gate_unified;
    int* output_unified;


    



    //allocating gpu memory
    cudaMallocManaged((void**)&input1_unified, number_of_lines * sizeof(int));
    cudaMallocManaged((void**)&input2_unified, number_of_lines * sizeof(int));
    cudaMallocManaged((void**)&gate_unified, number_of_lines * sizeof(int));
    cudaMallocManaged((void**)&output_unified, number_of_lines * sizeof(int));

    //calculating the number of blocks needed, the grid, and block size
    int  number_of_blocks = (number_of_lines / thread_number) + 1;
    dim3 grid(number_of_blocks, 1, 1);
    dim3 block(thread_number, 1, 1);

    for (int i = 0; i < number_of_lines; i++) {
        fgets(line, sizeof(line), input);
        input1_unified[i] = (int)(line[0] - '0');
        input2_unified[i] = (int)(line[2] - '0');
        gate_unified[i] = (int)(line[4] - '0');

    }

    //timer for test
    struct GpuTimer timer;
    timer.Start();

    //calling the rectification function on the gpu
    logic_gates <<< grid, block >>> (input1_unified, input2_unified, gate_unified, output_unified, number_of_lines);
    timer.Stop();
    printf("timer: %f", timer.Elapsed());
    cudaDeviceSynchronize();

    
    for (int i = 0; i < number_of_lines; i++) {
        fprintf(output, "%d\n", output_unified[i]);
    }
    fclose(input);
    fclose(output);
    cudaFree(input1_unified);
    cudaFree(input2_unified);
    cudaFree(gate_unified);
    cudaFree(output_unified);

    return 0;
}