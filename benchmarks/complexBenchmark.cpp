//
// Created by af on 14.03.21.
//


#include "benchmarkUtil.h"

#define AML_USE_STD_COMPLEX

#include "../amathlib.h"

static void BM_Mul(benchmark::State &state) {
	Complex64 a = 2 + 3_i;
	Complex64 b = 4 + 2_i;
	for (auto _ : state) {
		doNotOptimize(&a);
		doNotOptimize(&b);
		Complex64 c = a * a + b;
		doNotOptimize(&c);
	}

}

BENCHMARK(BM_Mul);


static void BM_MulSTD(benchmark::State &state) {
	std::complex<double> a(2, 3);
	std::complex<double> b(4, 2);
	for (auto _ : state) {
		doNotOptimize(&a);
		doNotOptimize(&b);
		std::complex<double> c = a * a + b;
		doNotOptimize(&c);
	}
}

BENCHMARK(BM_MulSTD);

BENCHMARK_MAIN();
