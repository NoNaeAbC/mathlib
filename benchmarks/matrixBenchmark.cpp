//
// Created by af on 14.03.21.
//

#include "benchmarkUtil.h"

#include "../amathlib.h"
#include <glm/ext/matrix_double4x4.hpp>
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/ext/matrix_transform.hpp>

static void BM_Mul(benchmark::State &state) {
	Matrix4x4_32 a(2.0);
	Matrix4x4_32 b(1.0);
	for (auto _ : state) {
		doNotOptimize(&a);
		doNotOptimize(&b);
		Matrix4x4_32 c = b * a;
		//Matrix4x4_64 c = a;
		doNotOptimize(&c);
	}

}

BENCHMARK(BM_Mul);

static void BM_Mul1(benchmark::State &state) {
	Matrix4x4_32 a(2.0);
	Matrix4x4_32 b(1.0);
	for (auto _ : state) {
		doNotOptimize(&a);
		Matrix4x4_32 c = b * a;
		//Matrix4x4_64 c = a;
		doNotOptimize(&c);
	}

}

BENCHMARK(BM_Mul1);

static void BM_Mul2(benchmark::State &state) {
	Matrix4x4_32 a(2.0);
	Matrix4x4_32 b(1.0);
	for (auto _ : state) {
		Matrix4x4_32 c = b * a;
		//Matrix4x4_64 c = a;
		doNotOptimize(&c);
	}

}

BENCHMARK(BM_Mul2);

static void BM_GLM(benchmark::State &state) {
	glm::mat4 a(2.0);
	glm::mat4 b(1.0);
	for (auto _ : state) {
		doNotOptimize(&a);
		doNotOptimize(&b);
		glm::mat4 c = b * a;
		doNotOptimize(&c);
	}

}

BENCHMARK(BM_GLM);

static void BM_GLM1(benchmark::State &state) {
	glm::mat4 a(2.0);
	glm::mat4 b(1.0);
	for (auto _ : state) {
		doNotOptimize(&a);
		glm::mat4 c = b * a;
		doNotOptimize(&c);
	}

}

BENCHMARK(BM_GLM1);


static void BM_GLM2(benchmark::State &state) {
	glm::mat4 a(2.0);
	glm::mat4 b(1.0);
	for (auto _ : state) {
		glm::mat4 c = b * a;
		doNotOptimize(&c);
	}

}

BENCHMARK(BM_GLM2);


BENCHMARK_MAIN();