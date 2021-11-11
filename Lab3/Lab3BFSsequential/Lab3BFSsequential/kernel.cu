
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "read_input.h"

#include <stdio.h>
#include <stdlib.h>


//definition of the logic gates
#define AND     0
#define OR      1
#define NAND    2
#define NOR     3
#define XOR     4
#define XNOR    5

int gate_solver(int gate, int input1, int input2)
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

int main() {
	FILE *nextlevelnodes_file, * nodeoutput_file;
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
	int numNextLevelNodes_h=0;
	int* nodeGate_h;
	int* nodeInput_h;
	int* nodeOutput_h;



	//output
	int* nextLevelNodes_h = (int*) malloc(sizeof(int) * 100000);


	numNodePtrs = read_input_one_two_four(&nodePtrs_h, "input1.raw");

	numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, "input2.raw");

	numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h, "input3.raw");

	numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, "input4.raw");

	for (int i = 0; i < numCurrLevelNodes; i++) {
		int node = currLevelNodes_h[i];
		for (int j = nodePtrs_h[node]; j < nodePtrs_h[node+1]; j++) {
			int neighbor = nodeNeighbors_h[j];
			if (!nodeVisited_h[neighbor]) {
				nodeVisited_h[neighbor] = 1;
				nodeOutput_h[neighbor] = gate_solver(nodeGate_h[neighbor], nodeOutput_h[node], nodeInput_h[neighbor]);
				nextLevelNodes_h[numNextLevelNodes_h] = neighbor;
				++(numNextLevelNodes_h);
			}
		}

	}
	fprintf(nextlevelnodes_file, "%d\n", numNextLevelNodes_h);
	fprintf(nodeoutput_file, "%d\n", numNodes);
	for (int l = 0; l < numNextLevelNodes_h; l++) {
		fprintf(nextlevelnodes_file, "%d\n", nextLevelNodes_h[l]);
	}
	for (int m = 0; m < numNodes; m++) {
		fprintf(nodeoutput_file, "%d\n", nodeOutput_h[m]);
	}
	fclose(nextlevelnodes_file);
	fclose(nodeoutput_file);
	return 0;
}
