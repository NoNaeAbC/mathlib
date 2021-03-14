
#define USE_FMA
#define USE_CONCEPTS
#define AML_USE_GMP

#define AML_USE_STD_COMPLEX

#include "amathlib.h"
#include <iostream>
#include <algorithm>

int main() {
	std::cout << sizeof(Matrix4x4_64) << std::endl;
	std::cout << sizeof(Matrix4x4_32) << std::endl;
	return 0;

}
