//
// Created by af on 08.03.21.
//
#include <opencl-c.h>

int main() {
	int width = 250;
	int height = 80;
	int accuracy = 100000;

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
			} else if (results[y + x * (int) width] == 10)
				UNLIKELY{
						std::cout << ".";
				}
			else
				LIKELY{
						std::cout << "#";
				}
		}
		std::cout << "\n";
	}

	free(results);
}

