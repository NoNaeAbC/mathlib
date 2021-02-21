//
// Created by af on 25.01.21.
//
#include <iostream>

#define AML_CUDA

#include "amathlib.h"


__global__ void kernel(double height, double width, int accuracy, int *results) {
	int i = blockIdx.x * blockDim.x + threadIdx.x; //width
	int j = blockIdx.y * blockDim.y + threadIdx.y; //height
	results[i + j] = 10000000;
	return;
	/*
	if (!(i >= width || j >= height)) {

		CU_Complex64 c = {((double) j) / ((double) height / 2.0f) - 1.5f,
						  ((double) i) / ((double) width / 2.0f) - 1.0f};
		CU_Complex64 z = c;
		for (int x = 0; x < accuracy; ++x) {
			z = z * z + c;
			if (z.abs_gt(2)) {
				results[i + j * (int) width] = accuracy;
				break;
			}
		}
		results[i] = accuracy;
	} else {
		results[i] = 10;
	}
	 */
}


int main() {

	// Run kernel
	int width = 250;
	int height = 80;
	int accuracy = 1000;

	int *deviceResults;
	cudaMalloc(&deviceResults, width * height * sizeof(int));

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
				   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

	kernel<<< numBlocks, threadsPerBlock>>>(width, height, accuracy, deviceResults);
	std::cout << " : " << width << " : " << c << std::endl;

	int *results = (int *) malloc(width * height * sizeof(int));

	cudaMemcpy(results, deviceResults, width * height * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(deviceResults);
	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			if (results[y + x * width] >= accuracy) {
				std::cout << "#";
			} else if (results[y + x * width] == 10) {
				std::cout << ".";
			} else {
				std::cout << "_";
			}
		}
		std::cout << "\n";
	}

	free(results);

}
