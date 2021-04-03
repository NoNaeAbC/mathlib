//
// Created by af on 14.03.21.
//


#include "benchmarkUtil.h"

#include "../amathlib.h"
#include <glm/ext/matrix_double4x4.hpp>
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/ext/matrix_transform.hpp>

static void BM_MatrixMultiplicationBoth(benchmark::State &state) {
	Matrix4x4_64 a(2.0);
	Matrix4x4_64 b(1.0);
	doNotOptimize(&a);
	doNotOptimize(&b);
	for (auto _ : state) {
		Matrix4x4_64 c = a * b;
		doNotOptimize(&c);
	}

}

BENCHMARK(BM_MatrixMultiplicationBoth);

static void BM_MatrixMultiplicationSecond(benchmark::State &state) {
	Matrix4x4_64 a(2.0);
	Matrix4x4_64 b(1.0);
	doNotOptimize(&a);
	for (auto _ : state) {
		Matrix4x4_64 c = a * b;
		doNotOptimize(&c);
	}

}

BENCHMARK(BM_MatrixMultiplicationSecond);

static void BM_MatrixMultiplicationFirst(benchmark::State &state) {
	Matrix4x4_64 a(2.0);
	Matrix4x4_64 b(1.0);
	doNotOptimize(&b);
	for (auto _ : state) {
		Matrix4x4_64 c = a * b;
		doNotOptimize(&c);
	}
}

BENCHMARK(BM_MatrixMultiplicationFirst);

static void BM_MatrixMultiplicationNone(benchmark::State &state) {
	const Matrix4x4_64 a(2.0);
	const Matrix4x4_64 b(1.0);
	for (auto _ : state) {
		Matrix4x4_64 c = a * b;
		doNotOptimize(&c);
	}
}

BENCHMARK(BM_MatrixMultiplicationNone);

static void BM_GLM_MatrixMultiplicationBoth(benchmark::State &state) {
	glm::dmat4 a(2.0);
	glm::dmat4 b(1.0);
	doNotOptimize(&a);
	doNotOptimize(&b);
	for (auto _ : state) {
		glm::dmat4 c = a * b;
		doNotOptimize(&c);
	}

}

BENCHMARK(BM_GLM_MatrixMultiplicationBoth);

static void BM_GLM_MatrixMultiplicationSecond(benchmark::State &state) {
	glm::dmat4 a(2.0);
	glm::dmat4 b(1.0);
	doNotOptimize(&a);
	for (auto _ : state) {
		glm::dmat4 c = a * b;
		doNotOptimize(&c);
	}

}

BENCHMARK(BM_GLM_MatrixMultiplicationSecond);

static void BM_GLM_MatrixMultiplicationFirst(benchmark::State &state) {
	glm::dmat4 a(2.0);
	glm::dmat4 b(1.0);
	doNotOptimize(&b);
	for (auto _ : state) {
		glm::dmat4 c = a * b;
		doNotOptimize(&c);
	}

}

BENCHMARK(BM_GLM_MatrixMultiplicationFirst);


static void BM_GLM_MatrixMultiplicationNone(benchmark::State &state) {
	glm::dmat4 a(2.0);
	glm::dmat4 b(1.0);
	for (auto _ : state) {
		glm::dmat4 c = a * b;
		doNotOptimize(&c);
	}

}

BENCHMARK(BM_GLM_MatrixMultiplicationNone);


BENCHMARK_MAIN();
