
#include <iostream>
#include <chrono>

#define AML_CUDA

#include "../amathlib.h"

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


uint64_t getTime() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::system_clock::now().time_since_epoch()).count();
}

int width = 250;          // example values
int height = 80;          // example values
int accuracy = 100000;    // example values way to high, just to get useful timing
int main() {
	std::cout << "width recommended 250" << std::endl;
	std::string input;
	std::getline(std::cin, input);
	if (!input.empty()) {
		std::istringstream stream(input);
		stream >> width;
	}
	std::cout << "height recommended 80" << std::endl;
	std::getline(std::cin, input);
	if (!input.empty()) {
		std::istringstream stream(input);
		stream >> height;
	}
	std::cout << "accuracy recommended 100000" << std::endl;
	std::getline(std::cin, input);
	if (!input.empty()) {
		std::istringstream stream(input);
		stream >> accuracy;
	}

	uint32_t begin;
	uint32_t end;

	begin = getTime();


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

	end = getTime();
	std::cout << "Time 0 : " << ((double) (end - begin)) / 1000.0f << std::endl;
}
