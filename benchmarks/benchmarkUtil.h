//
// Created by af on 14.03.21.
//

#ifndef MATHLIB_BENCHMARKUTIL_H
#define MATHLIB_BENCHMARKUTIL_H

#include <benchmark/benchmark.h>


void doNotOptimize(void *m) {
	asm volatile("" : : "g"(m): "memory");
}


#endif //MATHLIB_BENCHMARKUTIL_H
