//Written by Michael Frajman and Shi Tong Li ECSE420 Fall2021


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

    //input points to beginning
    rewind(input);
    char line[10];

    struct GpuTimer timer;
    timer.Start();
    
    //for each line, get the inputs and the gate
    for (int i = 0; i < number_of_lines; i++) {
        fgets(line, sizeof(line), input);
        int input1 = (int)(line[0]-'0');
        int input2 = (int)(line[2]-'0');
        int gate = (int)(line[4]-'0');

        int bit;
        //do the corresponding logic operation dictated by the gate
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
        //write result bit to output file
        fprintf(output, "%d\n", bit);
    }
    timer.Stop();
    printf("timer: %f", timer.Elapsed());

    //close both files
    fclose(input);
    fclose(output);
    return 0;
}
