//
// Created by af on 02.03.21.
//

#define AML_USE_STD_COMPLEX

#define AML_USE_GMP

#include <stdint.h>
#include "../amathlib.h"
#include <iostream>
#include <chrono>

uint64_t getTime() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::system_clock::now().time_since_epoch()).count();
}

AmlNumber width = 250l;          // example values
AmlNumber height = 80l;          // example values
AmlNumber accuracy = 1000l;    // example values way to high, just to get useful timing
int main() {
	uint64_t begin;
	uint64_t end;

	begin = getTime();

	for (AmlNumber x = 0l; x < height; x += 1l) {
		for (AmlNumber y = 0l; y < width; y += 1l) {
			ComplexN c((x) / (height / (AmlNumber) 2.0f) - (AmlNumber) 1.5f,
					   (y) / (width / (AmlNumber) 2.0f) - (AmlNumber) 1.0f);
			ComplexN z = c;
			int result = (double) accuracy;
			for (int i = 0; i < (double) accuracy; ++i) {
				z = z * z + c;
				if (z.abs_gt(2l)) {
					result = i;
					break;
				}
			}
			if (result >= (double) accuracy) {
				std::cout << "#";
			} else {
				std::cout << " ";
			}
		}
		std::cout << "\n";
	}
	end = getTime();
	double t1 = ((double) (end - begin)) / 1000.0f;
	std::cout << "Time 1 : " << ((double) (end - begin)) / 1000.0f << std::endl;
}