//
// Created by af on 28.02.21.
//

#include <iostream>

#define AML_USE_GMP

#include "amathlib.h"

#define TYPE_X AmlNumber<10000>


inline const char *getTruthValue(bool a) {
	return a ? "true" : "false";
}

int main() {
	AmlNumber a1 = 20l;
	AmlNumber a2 = 30l;
	AmlNumber a3 = AML::arithmeticMean(a1, a2);
	std::cout << (double) (a3) << std::endl;

}