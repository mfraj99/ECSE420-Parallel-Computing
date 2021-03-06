//Written by Shi Tong Li and Michael Frajman ECSE420 Fall2021

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>

//definitions of the logic gates
#define AND     0
#define OR      1
#define NAND    2
#define NOR     3
#define XOR     4
#define XNOR    5


// do logic gate operation on two input arrays
// input1_array - pointer to array of binary values of input1
// input2_array - pointer to array of binary values of input2
// gate_array - pointer to array of values of gate
// output - pointer to array of binary values of the resulting output
// n - limit of the numebr of threads
__global__ void logic_gates(int* input1_array, int* input2_array, int* gate_array, int* output, int n)
{
    //calculate index of the current thread
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        //get the input1, input2, and gate value for the corresponding index
        int bit;
        int input1 = input1_array[index];
        int input2 = input2_array[index];
        int gate = gate_array[index];
        //do operation depending on the value of the gate
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
        //write resulting bit to the output array with the correct index
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
    input = fopen("input_100000.txt", "r");
    output = fopen("output_100000.txt", "w");
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
    //make input point to the beginning of file
    rewind(input);
    char line[10];

    //declaring variables used 
    unsigned error;
    int* input1_unified;
    int* input2_unified;
    int* gate_unified;
    int* output_unified;

    struct GpuTimer timer2;
    timer2.Start();

    //allocating gpu memory unified
    cudaMallocManaged((void**)&input1_unified, number_of_lines * sizeof(int));
    cudaMallocManaged((void**)&input2_unified, number_of_lines * sizeof(int));
    cudaMallocManaged((void**)&gate_unified, number_of_lines * sizeof(int));
    cudaMallocManaged((void**)&output_unified, number_of_lines * sizeof(int));

    timer2.Stop();
    printf("Memorytimer: %f\n", timer2.Elapsed());

    //calculating the number of blocks needed, the grid, and block size
    int  number_of_blocks = (number_of_lines / thread_number) + 1;
    dim3 grid(number_of_blocks, 1, 1);
    dim3 block(thread_number, 1, 1);

    //parse the input1, input2, and gate values and store them in arrays
    for (int i = 0; i < number_of_lines; i++) {
        fgets(line, sizeof(line), input);
        input1_unified[i] = (int)(line[0] - '0');
        input2_unified[i] = (int)(line[2] - '0');
        gate_unified[i] = (int)(line[4] - '0');

    }

    //timer for test
    struct GpuTimer timer;
    timer.Start();

    //calling the logic gate function on the gpu
    logic_gates <<< grid, block >>> (input1_unified, input2_unified, gate_unified, output_unified, number_of_lines);
    timer.Stop();
    printf("GPUtimer: %f", timer.Elapsed());
    cudaDeviceSynchronize();

    //write back the output of the gpu function (stored in an array) to the output file
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