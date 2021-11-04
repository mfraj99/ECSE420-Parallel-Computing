//Written by Shi Tong Li and Michael Frajman - ECSE420 Fall2021 Group8
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


int main(int argc, char** argv)
{
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //HARD CODED VARIABLES FOR INPUT OUTPUT
    int T = 4; //number of iterations
    const int N = 4;
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    double element_grid[N * N] = { 0.0 };
    double element_grid_u1[N * N] = { 0.0 };
    double element_grid_u2[N * N] = { 0.0 };
    //initiate drum hit
    element_grid_u1[(N * N / 2) + N/2] = 1;

    //loop for number of iterations
    for (int i = 0; i < T; i++) {
        //do interior element first
        for (int index = 0; index < N * N; index++) {
            int row = index / N + 1;

            if (index % N != 0 && index % N != N - 1 && row != 1 && index < N * (N - 1)) {
                element_grid[index] = (0.5*(element_grid_u1[index-N]+element_grid_u1[index+N]+element_grid_u1[index-1]+element_grid_u1[index+1]-4*element_grid_u1[index])+2*element_grid_u1[index]-(1-0.0002)*(element_grid_u2[index]))/(1+0.0002);
            }
            
        }
        //do boundary without corners
        for (int index = 0; index < N * N; index++) {
            int row = index / N + 1;
            if (row == 1 && index % N != 0 && index % N != N - 1) {
                element_grid[index] = 0.75 * element_grid[index + N];
            }
            if (index >= N * (N - 1) && index % N != 0 && index % N != N - 1) {
                element_grid[index] = 0.75 * element_grid[index - N];
            }
            if (index % N == 0 && row != 1 && index < N * (N - 1)) {
                element_grid[index] = 0.75 * element_grid[index + 1];
            }
            if (index % N == N - 1 && row != 1 && index < N * (N - 1)) {
                element_grid[index] = 0.75 * element_grid[index - 1];
            }
        }
        //do corners
        element_grid[0] = 0.75 * element_grid[N];
        element_grid[(N - 1) * N] = 0.75 * element_grid[(N - 1) * N - N];
        element_grid[N - 1] = 0.75 * element_grid[N - 1 - 1];
        element_grid[N * N - 1] = 0.75 * element_grid[N * N - 1 - 1];
        //print required statement to terminal
        printf("\n (%d, %d): %f", N / 2, N / 2, element_grid[(N * N / 2) + N / 2]);
        //u1 becomes u1, and current values become u1
        memcpy(element_grid_u2, element_grid_u1, sizeof(element_grid_u2));
        memcpy(element_grid_u1, element_grid, sizeof(element_grid_u1));
    }
    return 0;
}
