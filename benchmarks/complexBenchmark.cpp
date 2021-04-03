//
// Created by af on 14.03.21.
//


#include "benchmarkUtil.h"

#define AML_USE_STD_COMPLEX

#include "../amathlib.h"

#define var auto
#define let const auto

static void BM_Mul(benchmark::State &state) {
	var a = 2 + 3_if;
	var b = 4 + 2_if;
	doNotOptimize(&a);
	doNotOptimize(&b);
	for (auto _ : state) {
		var c = a * a + b;
		var e = a * a + b;
		doNotOptimize(&c);
		doNotOptimize(&e);
	}

}

BENCHMARK(BM_Mul);


static void BM_MulSq(benchmark::State &state) {
	var a = 2 + 3_if;
	var b = 4 + 2_if;
	doNotOptimize(&a);
	doNotOptimize(&b);
	var d = a;
	for (auto _ : state) {
		var c = square(d) + b;
		var e = square(d) + b;
		doNotOptimize(&c);
		doNotOptimize(&e);
	}
}

BENCHMARK(BM_MulSq);

static void BM_MulSTD(benchmark::State &state) {
	std::complex<float> a(2, 3);
	std::complex<float> b(4, 2);
	doNotOptimize(&a);
	doNotOptimize(&b);
	for (auto _ : state) {
		std::complex<float> c = a * a + b;
		std::complex<float> d = b * b + a;
		doNotOptimize(&c);
		doNotOptimize(&d);
	}
}

BENCHMARK(BM_MulSTD);

BENCHMARK_MAIN();
