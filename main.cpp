
#define USE_FMA
#define USE_CONCEPTS
#define AML_USE_GMP

#define AML_USE_STD_COMPLEX

#include <iostream>
#include "amathlib.h"
#include "amlavg.h"
#include <algorithm>

int main() {

	auto a = 2 + 2_i;

	std::cout << sqrt(a) << std::endl;

	auto b = sqrt(a);

	std::cout << a << std::endl;

}
