//
// Created by af on 25.01.21.
//
#include <iostream>

#define AML_CUDA

#include "amathlib.h"

__global__ void kernel(double width, double height, int accuracy, int *results) {
	int i = blockIdx.x * blockDim.x + threadIdx.x; //width
	if (i >= (int) width) {
		return;
	}
	int j = blockIdx.y * blockDim.y + threadIdx.y; //height
	if (j >= (int) height) {
		return;
	}
	results[i + j * (int) width] = accuracy;
	results[i + j * (int) width] = 0;
	if (!(i >= width || j >= height)) {

		CU_Complex64 c = {CU_AML::mapLinear((double) j, 0.0, (double) height, -1.5, 0.5),
						  CU_AML::mapLinear((double) i, 0.0, (double) width, -1.0, 1.0)};
		CU_Complex64 z = c;
		for (int x = 0; x < accuracy; ++x) {
			z = z * z + c;
			if (z.abs_gt(2)) {
				results[i + j * (int) width] = accuracy;
				break;
			}
		}
	}
}


int main() {
	const int width = 250;
	const int height = 80;
	const int accuracy = 100000;

	int *deviceResults;
	cudaMalloc(&deviceResults, width * height * sizeof(int));

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
				   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

	kernel<<< numBlocks, threadsPerBlock>>>(width, height, accuracy, deviceResults);
	std::cout << " : " << width << " : " << std::endl;

	int *results = (int *) malloc(width * height * sizeof(int));

	cudaMemcpy(results, deviceResults, width * height * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(deviceResults);
	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			if (results[y + x * (int) width] >= accuracy) {
				std::cout << " ";
			} else if (results[y + x * (int) width] == 10) UNLIKELY {
				std::cout << ".";
			} else LIKELY {
				std::cout << "#";
			}
		}
		std::cout << "\n";
	}

	free(results);

}
