//Written by Shi Tong Li and Michael Frajman ECSE420 Fal2021

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>

//definition of the logic gates
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
__global__ void logic_gates(int* input1_array, int *input2_array, int *gate_array, int *output, int n)
{
    //calculate the index of the current thread
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
    input = fopen("input_1000000.txt", "r");
    output = fopen("output_1000000.txt", "w");
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
    int* input1_host;
    input1_host = new int[number_of_lines];
    int* input1_cuda;
    int* input2_host = new int[number_of_lines];
    int* input2_cuda;
    int* gate_host = new int[number_of_lines];
    int* gate_cuda;
    int* output_cuda;

    //parse the input1, input2, and gate values and store them in arrays
    for (int i = 0; i < number_of_lines; i++) {
        fgets(line, sizeof(line), input);
        input1_host[i] = (int)(line[0]-'0');
        input2_host[i] = (int)(line[2] - '0');
        gate_host[i] = (int)(line[4] - '0');
        
    }
    
    struct GpuTimer timer2;
    timer2.Start();
    
    //allocating gpu memory
    cudaMalloc((void**)&input1_cuda, number_of_lines*sizeof(int));
    cudaMalloc((void**)&input2_cuda, number_of_lines * sizeof(int));
    cudaMalloc((void**)&gate_cuda, number_of_lines * sizeof(int));
    cudaMalloc((void**)&output_cuda, number_of_lines * sizeof(int));
    //copying input information onto the gpu
    cudaMemcpy(input1_cuda, input1_host, number_of_lines * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(input2_cuda, input2_host, number_of_lines * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gate_cuda, gate_host, number_of_lines * sizeof(int), cudaMemcpyHostToDevice);
    
    timer2.Stop();
    printf("Memorytimer: %f\n", timer2.Elapsed());
    
    //calculating the number of blocks needed, the grid, and block size
    int  number_of_blocks = (number_of_lines / thread_number)+1;
    dim3 grid(number_of_blocks, 1, 1);
    dim3 block(thread_number, 1, 1);

    //timer for test
    struct GpuTimer timer1;
    timer1.Start();

    //calling the logic gate function on the gpu
    logic_gates <<<grid, block >>> (input1_cuda, input2_cuda, gate_cuda, output_cuda, number_of_lines);
    timer1.Stop();
    printf("GPUtimer: %f", timer1.Elapsed());
    cudaDeviceSynchronize();

    //declaring variable to store the gpu output onto the host machine
    int *output_host = new int[number_of_lines]();
    //copying the gpu output onto the host machine
    cudaMemcpy(output_host, output_cuda,number_of_lines* sizeof(int), cudaMemcpyDeviceToHost);
    //write back the output of the gpu function (stored in an array) to the output file
    for (int i = 0; i < number_of_lines; i++) {
        fprintf(output, "%d\n", output_host[i]);
    }
    fclose(input);
    fclose(output);
    cudaFree(input1_cuda);
    cudaFree(input2_cuda);
    cudaFree(gate_cuda);
    cudaFree(output_cuda);

    return 0;
}