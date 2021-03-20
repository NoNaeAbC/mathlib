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

constexpr inline Complex64 square(const Complex64 a) {
	return Complex64(a.c.c[0] * a.c.c[0] - a.c.c[1] * a.c.c[1], a.c.c[0] * a.c.c[1] + a.c.c[0] * a.c.c[1]);
}

static void BM_MulSq(benchmark::State &state) {
	Complex64 a = 2 + 3_i;
	Complex64 b = 4 + 2_i;
	doNotOptimize(&a);
	doNotOptimize(&b);
	for (auto _ : state) {
		Complex64 c = square(a) + b;
		doNotOptimize(&c);
	}
}

BENCHMARK(BM_MulSq);

constexpr inline Complex64 fsa(const Complex64 a, const Complex64 b) {
	return Complex64(a.c.c[0] * a.c.c[0] - a.c.c[1] * a.c.c[1] + b.c.c[0],
					 a.c.c[0] * a.c.c[1] + a.c.c[0] * a.c.c[1] + b.c.c[1]);
}

static void BM_MulFMA(benchmark::State &state) {
	Complex64 a = 2 + 3_i;
	Complex64 b = 4 + 2_i;
	doNotOptimize(&a);
	doNotOptimize(&b);
	for (auto _ : state) {
		Complex64 c = fsa(a, b);
		doNotOptimize(&c);
	}
}

BENCHMARK(BM_MulFMA);

static void BM_MulFMA_Inline(benchmark::State &state) {
	Complex64 a2 = 2 + 3_i;
	Complex64 b2 = 4 + 2_i;
	doNotOptimize(&a2);
	doNotOptimize(&b2);
	//const Complex64 a = a2;
	//const Complex64 b = b2;
	for (auto _ : state) {
		const Complex64 a = a2;
		const Complex64 b = b2;
		Complex64 c(a.c.c[0] * a.c.c[0] - a.c.c[1] * a.c.c[1] + b.c.c[0],
					a.c.c[0] * a.c.c[1] + a.c.c[0] * a.c.c[1] + b.c.c[1]);
		doNotOptimize(&c);
	}
}

BENCHMARK(BM_MulFMA_Inline);


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
