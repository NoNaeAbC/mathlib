
#define USE_FMA
#define USE_CONCEPTS
#define AML_USE_GMP

#define AML_USE_STD_COMPLEX

#include <iostream>
#include "amathlib.h"
#include <algorithm>

int main() {
	std::cout << sizeof(Matrix4x4_64) << std::endl;
	std::cout << sizeof(Matrix4x4_32) << std::endl;
	Matrix4x4_64 a(2.0);
	Matrix4x4_32 b(2.0);
	b = b * b;
	std::cout << b[0][0] << std::endl;
	std::cout << b[1][1] << std::endl;
	std::cout << b[2][2] << std::endl;
	std::cout << b[3][3] << std::endl;
	return 0;

}
