//
// Created by af on 14.03.21.
//

#include "benchmarkUtil.h"

#include "../amathlib.h"
#include <glm/ext/matrix_double4x4.hpp>
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/ext/matrix_transform.hpp>

static void BM_Mul(benchmark::State &state) {
	Matrix4x4_64 a(2.0);
	Matrix4x4_64 b(1.0);
	for (auto _ : state) {
		doNotOptimize(&a);
		doNotOptimize(&b);
		Matrix4x4_64 c = b * a;
		//Matrix4x4_64 c = a;
		doNotOptimize(&c);
	}

}
// Register the function as a benchmark
BENCHMARK(BM_Mul);

static void BM_GLM(benchmark::State &state) {
	glm::dmat4 a(2.0);
	glm::dmat4 b(1.0);
	for (auto _ : state) {
		doNotOptimize(&a);
		doNotOptimize(&b);
		glm::dmat4 c = b * a;
		doNotOptimize(&c);
	}

}
// Register the function as a benchmark
BENCHMARK(BM_GLM);


// Define another benchmark
static void BM_Mul32(benchmark::State &state) {
	Matrix4x4_32 a(2.0);
	Matrix4x4_32 b(2.0);
	for (auto _ : state) {
		doNotOptimize(&a);
		doNotOptimize(&b);
		Matrix4x4_32 c = a * b;
		doNotOptimize(&c);
	}
}

BENCHMARK(BM_Mul32);

static void BM_GLM32(benchmark::State &state) {
	glm::mat4 a(2.0);
	glm::mat4 b(2.0);
	for (auto _ : state) {
		doNotOptimize(&a);
		doNotOptimize(&b);
		glm::mat4 c = a * b;
		doNotOptimize(&c);
	}

}
// Register the function as a benchmark
BENCHMARK(BM_GLM32);

static void BM_MulN(benchmark::State &state) {
	for (auto _ : state) {
		Matrix4x4_32 a(2.0);
		Matrix4x4_32 b(2.0);
		doNotOptimize(&a);
		doNotOptimize(&b);
	}
}

BENCHMARK(BM_MulN);

static void BM_GLMN(benchmark::State &state) {
	for (auto _ : state) {
		glm::mat4 a(2.0);
		glm::mat4 b(2.0);
		doNotOptimize(&a);
		doNotOptimize(&b);
	}

}
// Register the function as a benchmark
BENCHMARK(BM_GLMN);


static void BM_MulA(benchmark::State &state) {
	Matrix4x4_32 a(2.0);
	for (auto _ : state) {
		doNotOptimize(&a);
		Matrix4x4_32 c = a;
		doNotOptimize(&c);

	}
}

BENCHMARK(BM_MulA);

static void BM_GLMA(benchmark::State &state) {
	glm::mat4 a(2.0);
	for (auto _ : state) {
		doNotOptimize(&a);
		glm::mat4 c = a;
		doNotOptimize(&c);

	}

}
// Register the function as a benchmark
BENCHMARK(BM_GLMA);




BENCHMARK_MAIN();