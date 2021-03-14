//
// Created by af on 14.03.21.
//

#include "../amathlib.h"
#include <iostream>
#include <chrono>

void doNotOptimize(void *m) {
	asm volatile("" : : "g"(m): "memory");
}

inline uint64_t getTime() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::system_clock::now().time_since_epoch()).count();
}

int main() {
	Matrix4x4_32 a(2.0);
	Matrix4x4_32 b(1.0);
	const int amount = 100000;
	long begin = getTime();
	doNotOptimize(&a);
	doNotOptimize(&b);
	for (int i = 0; i < amount; i++) {
		Matrix4x4_32 c = b * a;
		doNotOptimize(&c);
	}
	long end = getTime();
	std::cout << (double) (end - begin) / amount << std::endl;
	//Matrix4x4_64 c = a;
}
