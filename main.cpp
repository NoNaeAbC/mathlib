
#define USE_FMA
#define USE_CONCEPTS
#define AML_USE_GMP

#define AML_USE_STD_COMPLEX

#include "amathlib.h"
#include <iostream>
#include <algorithm>

struct A {
	double a[0];
	double b[0];
};

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
	if (std::is_constant_evaluated()) {
		std::cout << "CONST" << std::endl;
	}

	std::cout << "size : " << sizeof(Vector4D_64) << " align : " << alignof(Vector4D_64) << std::endl;
	std::cout << "size double : " << sizeof(double) << " size float : " << sizeof(float) << std::endl;
	std::cout << "size : " << sizeof(Complex32) << " align : " << alignof(Complex32) << std::endl;
	std::cout << "size : " << sizeof(Array2Complex32) << " align : " << alignof(Array2Complex32) << std::endl;
	std::cout << "size : " << sizeof(Array4Complex32) << " align : " << alignof(Array4Complex32) << std::endl;
	std::cout << "size : " << sizeof(Array8Complex32) << " align : " << alignof(Array8Complex32) << std::endl;
	std::cout << "size : " << sizeof(AmlNumber) << " align : " << alignof(AmlNumber) << std::endl;
	std::cout << "size : " << sizeof(Array3_64) << " align : " << alignof(Array3_64) << std::endl;
	std::cout << "size : " << sizeof(ArrayN_64<5>) << " align : " << alignof(ArrayN_64<5>) << " array N" << std::endl;

	ArrayN_64<2> a;
	ArrayN_64<3> a1;
	ArrayN_64<4> a2;
	ArrayN_64<5> a3;
	ArrayN_64<9> a4;

	return 0;

}
