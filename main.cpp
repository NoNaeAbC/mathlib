
#define USE_FMA
#define USE_CONCEPTS
#define AML_USE_GMP

#define AML_USE_STD_COMPLEX

#include "amathlib.h"
#include <iostream>
#include <algorithm>


int main() {

#if defined(__NVCC__)
	std::cout << "USE CUDA" << std::endl;
#endif
#if defined(__EMSCRIPTEN__)
	std::cout << "WEB" << std::endl;
#endif

#if defined(DEBUG)
	std::cout << "debug" << std::endl;
#endif

#if defined(USE_AVX)
	std::cout << "USE_AVX" << std::endl;
#endif

#if defined(USE_CONCEPTS)
	std::cout << "C++ 20" << std::endl;
#endif

	std::cout << "size : " << sizeof(VectorDouble4D) << " align : " << alignof(VectorDouble4D) << std::endl;
	std::cout << "size double : " << sizeof(double) << " size float : " << sizeof(float) << std::endl;
	std::cout << "size : " << sizeof(Complex32) << " align : " << alignof(Complex32) << std::endl;
	std::cout << "size : " << sizeof(Array2Complex32) << " align : " << alignof(Array2Complex32) << std::endl;
	std::cout << "size : " << sizeof(Array4Complex32) << " align : " << alignof(Array4Complex32) << std::endl;
	std::cout << "size : " << sizeof(Array8Complex32) << " align : " << alignof(Array8Complex32) << std::endl;
	std::cout << "size : " << sizeof(AmlNumber) << " align : " << alignof(AmlNumber) << std::endl;

	{
		AmlNumber a = 1.0;
		AmlNumber b = 1.0;

		AmlNumber c = a + ((AmlNumber) 1.0f) / b;
		while (AML::precisionSufficient(a, c)) {
			b *= 2.0;
			c = a + ((AmlNumber) 1.0f) / b;
		}
		std::cout << (double) b << std::endl;
	}

	{
		float a = 1.0;
		float b = 1.0;

		float c = a + ((float) 1.0f) / b;
		while (AML::precisionSufficient(a, c)) {
			b *= 1.5;
			c = a + ((float) 1.0f) / b;
		}
		std::cout << (double) b << std::endl;
	}
	{
		long double a = 1.0;
		long double b = 1.0;

		long double c = a + ((long double) 1.0f) / b;
		while (AML::precisionSufficient(a, c)) {
			b *= 2.0;
			c = a + ((long double) 1.0f) / b;
		}
		std::cout << (double) b << std::endl;
	}
	AML_DEFAULT_PRECISION 50;
	{
		AML_DEFAULT_PRECISION 2000;
		std::cout << AML::defaultPrecision << std::endl;

		AmlNumber a = 1.0;
		AmlNumber b = 1.0;

		AmlNumber c = a + ((AmlNumber) 1.0f) / b;
		while (AML::precisionSufficient(a, c)) {
			b *= 2.0;
			c = a + ((AmlNumber) 1.0f) / b;
		}
		std::cout << (double) b << std::endl;
	}
	std::cout << AML::defaultPrecision << std::endl;

	{
		float a = 0.0;
		float b = 1.0;

		float c = a + ((float) 1.0f) / b;
		while (AML::precisionSufficient(a, c, [](float x) { return x / 10000000000000000000000000000000000000.0f; })) {
			b *= 1.5;
			c = a + ((float) 1.0f) / b;
		}
		std::cout << (double) b << std::endl;
	}

	return 0;

}
