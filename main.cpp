
#define USE_FMA
#define USE_CONCEPTS
#define AML_USE_GMP

#define AML_USE_STD_COMPLEX

#include <iostream>
#include "amathlib.h"
#include "amlavg.h"
#include <algorithm>

int main() {

	auto a = 2 + 2_if;

	std::cout << *a.sqrt() << std::endl;
	std::cout << a << std::endl;

}
