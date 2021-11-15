
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"


#include <stdio.h>
#include <stdlib.h>

//definition of the logic gates
#define AND     0
#define OR      1
#define NAND    2
#define NOR     3
#define XOR     4
#define XNOR    5

int read_input_one_two_four(int** input1, char* filepath) {
	FILE* fp = fopen(filepath, "r");
	if (fp == NULL) {
		fprintf(stderr, "Couldn't open file for reading\n");
		exit(1);
	}

	int counter = 0;
	int len;
	int length = fscanf(fp, "%d", &len);
	*input1 = (int*)malloc(len * sizeof(int));

	int temp1;

	while (fscanf(fp, "%d", &temp1) == 1) {
		(*input1)[counter] = temp1;

		counter++;
	}

	fclose(fp);
	return len;




}
int read_input_three(int** input1, int** input2, int** input3, int** input4, char* filepath) {
	FILE* fp = fopen(filepath, "r");
	if (fp == NULL) {
		fprintf(stderr, "Couldn't open file for reading\n");
		exit(1);
	}

	int counter = 0;
	int len;
	int length = fscanf(fp, "%d", &len);
	*input1 = (int*)malloc(len * sizeof(int));
	*input2 = (int*)malloc(len * sizeof(int));
	*input3 = (int*)malloc(len * sizeof(int));
	*input4 = (int*)malloc(len * sizeof(int));



	int temp1;
	int temp2;
	int temp3;
	int temp4;
	while (fscanf(fp, "%d,%d,%d,%d", &temp1, &temp2, &temp3, &temp4) == 4) {
		(*input1)[counter] = temp1;
		(*input2)[counter] = temp2;
		(*input3)[counter] = temp3;
		(*input4)[counter] = temp4;
		counter++;
	}

	fclose(fp);
	return len;

}

__device__ int gate_solver(int gate, int input1, int input2)
{
	int bit;
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
	return bit;
}

__global__ void global_queue(int* currLevelNodes_h, int *nodePtrs_h, int *nodeNeighbors_h, int *nodeVisited_h, int *nodeOutput_h, int *nodeGate_h, int *nodeInput_h, int *nextLevelNodes_h, int *numNextLevelNodes_h, int numCurrLevelNodes)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while(index<numCurrLevelNodes){
		int node = currLevelNodes_h[index];
		for (int j = nodePtrs_h[node]; j < nodePtrs_h[node + 1]; j++) {
			int neighbor = nodeNeighbors_h[j];
			if (!nodeVisited_h[neighbor]) {
				nodeVisited_h[neighbor] = 1;
				nodeOutput_h[neighbor] = gate_solver(nodeGate_h[neighbor], nodeOutput_h[node], nodeInput_h[neighbor]);
				atomicExch(&nextLevelNodes_h[atomicAdd(numNextLevelNodes_h, 1)], neighbor);
			}
		}
		index += stride;

	}

}



int main() {
	FILE* nextlevelnodes_file, * nodeoutput_file;
	nextlevelnodes_file = fopen("nextLevelNodes.txt", "w");
	nodeoutput_file = fopen("nodeOutput.txt", "w");
	// Variables
	int numNodePtrs;
	int numNodes;
	int* nodePtrs_h;
	int* nodeNeighbors_h;
	int* nodeVisited_h;
	int numTotalNeighbors_h;
	int* currLevelNodes_h;
	int numCurrLevelNodes;
	int* numNextLevelNodes_h;
	int* nodeGate_h;
	int* nodeInput_h;
	int* nodeOutput_h;



	//output
	int* nextLevelNodes_h = (int*)malloc(sizeof(int) * 100000);




	numNodePtrs = read_input_one_two_four(&nodePtrs_h, "input1.raw");

	numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, "input2.raw");

	numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h, "input3.raw");

	numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, "input4.raw");

	//cuda variables
	int* cuda_nodePtrs_h;
	int* cuda_nodeNeighbors_h;
	int* cuda_nodeVisited_h;
	int* cuda_currLevelNodes_h;
	int* cuda_nodeGate_h;
	int* cuda_nodeInput_h;
	int* cuda_nodeOutput_h;
	int* cuda_nextLevelNodes_h;
	int* cuda_numNextLevelNodes_h;

	//explicit memory allocation
	cudaMalloc((void**)&cuda_nodePtrs_h, numNodePtrs*sizeof(int));
	cudaMalloc((void**)&cuda_nodeNeighbors_h, numTotalNeighbors_h*sizeof(int));
	cudaMalloc((void**)&cuda_nodeVisited_h, numNodes*sizeof(int));
	cudaMalloc((void**)&cuda_currLevelNodes_h, numCurrLevelNodes*sizeof(int));
	cudaMalloc((void**)&cuda_nodeGate_h, numNodes*sizeof(int));
	cudaMalloc((void**)&cuda_nodeInput_h, numNodes*sizeof(int));
	cudaMalloc((void**)&cuda_nodeOutput_h, numNodes*sizeof(int));
	cudaMalloc((void**)&cuda_nextLevelNodes_h, sizeof(int) * numTotalNeighbors_h);
	cudaMalloc((void**)&cuda_numNextLevelNodes_h, sizeof(int));
	/*cudaMallocManaged(&numNextLevelNodes_h, sizeof(int));
	int temp = 0;
	numNextLevelNodes_h = &temp;*/

	//copy to device
	cudaMemcpy(cuda_nodePtrs_h, nodePtrs_h, numNodePtrs*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_nodeNeighbors_h, nodeNeighbors_h, numTotalNeighbors_h * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_nodeVisited_h, nodeVisited_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_currLevelNodes_h, currLevelNodes_h, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_nodeGate_h, nodeGate_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_nodeInput_h, nodeInput_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_nodeOutput_h, nodeOutput_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemset((void**)&cuda_numNextLevelNodes_h, 0, sizeof(int));

	//calculating the number of blocks needed, the grid, and block size
	int blocksize = 32;
	int  number_of_blocks = 10;


	global_queue <<<number_of_blocks, blocksize >>> (cuda_currLevelNodes_h, cuda_nodePtrs_h, cuda_nodeNeighbors_h, cuda_nodeVisited_h, cuda_nodeOutput_h, cuda_nodeGate_h, cuda_nodeInput_h, cuda_nextLevelNodes_h, cuda_numNextLevelNodes_h, numCurrLevelNodes);
	
	cudaMemcpy(nextLevelNodes_h, cuda_nextLevelNodes_h, sizeof(int)*numTotalNeighbors_h, cudaMemcpyDeviceToHost);
	cudaMemcpy(nodeOutput_h, cuda_nodeOutput_h, sizeof(int)*numNodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(numNextLevelNodes_h, cuda_numNextLevelNodes_h, sizeof(int), cudaMemcpyDeviceToHost);

	fprintf(nextlevelnodes_file, "%d\n", *numNextLevelNodes_h);
	fprintf(nodeoutput_file, "%d\n", numNodes);
	printf("\n %d", numNextLevelNodes_h);
	
	for (int l = 0; l < *numNextLevelNodes_h; l++) {
		fprintf(nextlevelnodes_file, "%d\n", nextLevelNodes_h[l]);
	}
	for (int m = 0; m < numNodes; m++) {
		fprintf(nodeoutput_file, "%d\n", nodeOutput_h[m]);
	}
	fclose(nextlevelnodes_file);
	fclose(nodeoutput_file);
	return 0;
}
