//
// Created by af on 26.01.21.
//


#define AML_USE_ARRAY_STRICT
#define AML_USE_STD_COMPLEX

#include <stdint.h>
#include "../amathlib.h"
#include <iostream>
#include <chrono>

/*
 * This sample shows how to implement the Mandelbrot set in C++. It is implemented in two ways a fast and a slow one.
 * It is recommended to know what the Mandelbrot set is, not only because it makes it easy to follow, but because not knowing what it is makes you suck.
 * Refer to https://en.wikipedia.org/wiki/Mandelbrot_set when needed.
 *
 * To compile on x86-64 choose clang or gcc and run
 * 		clang++ -O3 -ffast-math -march=native mandelbrot.cpp -o mandelbrot
 *	OR
 *		g++ -O3 -ffast-math -march=native mandelbrot.cpp -o mandelbrot
 *
 * "-O3" and "-ffast-math" are optimisation flags, you can try to run the code with just "-O0" or without -march=native
 *
 * On non x86-64 platforms remove -march=native
 */

uint64_t getTime() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::system_clock::now().time_since_epoch()).count();
}

__attribute__((noinline)) void printImage(int iterations) {
	if (0 == iterations) {
		std::cout << "#";
	} else {
		std::cout << " ";
	}
}

double stdMandelbrot(const int height, const int width, int accuracy) {

	uint64_t begin;
	uint64_t end;
	const double d_width = width;
	const double d_height = height;

	begin = getTime();
	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			std::complex<double> c(((double) x) / (d_height / 2.0) - 1.5,
								   ((double) y) / (d_width / 2.0) - 1.0);
			std::complex<double> z = c;
			int iterations = accuracy;
			while ((iterations) && (norm(z) < (4.0))) {
				iterations--;
				z = z * z + c;
			}
			printImage(iterations);
		}
		std::cout << "\n";
	}

	end = getTime();
	double t0 = ((double) (end - begin)) / 1000.0f;

	std::cout << std::endl;
	return t0;
}


double amlMandelbrot(const int height, const int width, const int accuracy) {
	uint64_t begin;
	uint64_t end;
	const double d_width = width;
	const double d_height = height;
	begin = getTime();
	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			int iterations = accuracy;
			const Complex64 c = {AML::mapLinear((double) x, 0.0, d_height, -1.5, 0.5),
								 AML::mapLinear((double) y, 0.0, d_width, -1.0, 1.0)};
			Complex64 z = c;
			while ((iterations > 0) && (z.abs_lt(2))) {
				iterations--;
				z.square()->add(c);
				//z.multiply(z)->add(c);
				//z = z * z + c;
			}
			printImage(iterations);
		}
		std::cout << "\n";
	}
	end = getTime();
	double t1 = ((double) (end - begin)) / 1000.0f;

	std::cout << std::endl;
	return t1;
}


double simdMandelbrot(const int height, const int width, const int accuracy) {
	uint64_t begin;
	uint64_t end;
	const double d_width = width;
	const double d_height = height;

	begin = getTime();
	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width / IDEAL_COMPLEX_64_SIZE; y++) {
			IDEAL_COMPLEX_64_TYPE C;
			IDEAL_COMPLEX_64_TYPE Z;
			for (Complex64Ptr c_ptr : C) {
				*c_ptr = {AML::mapLinear((double) x, 0.0, d_height, -1.5, 0.5),
						  AML::mapLinear((double) y * IDEAL_COMPLEX_64_SIZE + (int) c_ptr, 0.0, d_width, -1.0,
										 1.0)};
			}
			Z = C;
			IDEAL_COMPLEX_64_VECTOR_TYPE result(accuracy);
			int i = 0;
			IDEAL_COMPLEX_64_MASK_TYPE finished;
			IDEAL_COMPLEX_64_MASK_TYPE nowFinished;
			IDEAL_COMPLEX_64_MASK_TYPE alreadyFinished;
			for (; i < accuracy; ++i) {
				Z = Z * Z + C;
				finished = Z.abs_gt(2);
				bool anyFinished = finished.anyTrue();
				if (anyFinished) {
					alreadyFinished = finished;
					result.set(i, finished);
					break;
				}
			}
			if (!(finished.allTrue())) {
				for (; i < accuracy; ++i) {
					Z.square(!finished)->add(C, !finished);
					//complex64_Z = (complex64_Z * complex64_Z) + complex64_C; Identical result but slower on my machine
					finished = Z.abs_gt(2);
					nowFinished = finished && !alreadyFinished;
					if (nowFinished.anyTrue()) {
						alreadyFinished = finished;
						result.set(i, nowFinished);
					}
					if (finished.allTrue()) {
						break;
					}
				}
			}
			for (int index = 0; index < IDEAL_COMPLEX_64_SIZE; index++) {
				if (result[index] >= accuracy) {
					std::cout << "#";
				} else {
					std::cout << " ";
				}
			}
		}
		std::cout << "\n";
	}
	end = getTime();


	return ((double) (end - begin)) / 1000.0f;

}

int width = 250;          // example values
int height = 80;          // example values
int accuracy = 1000000;    // example values way to high, just to get useful timing
int main(int argc, const char **argv) {
	if (argc <= 1) {
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
	}


	double t0 = 0.0;//stdMandelbrot(height, width, accuracy);


	//

	double t1 = 0.0;//amlMandelbrot(height, width, accuracy);


	double t2 = simdMandelbrot(height, width, accuracy);


	std::cout << "Time 0 : " << t0 << std::endl;
	std::cout << "Time 1 : " << t1 << std::endl;
	std::cout << "Time 2 : " << t2 << std::endl;
}
