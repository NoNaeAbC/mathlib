//
// Created by af on 16.01.21.
//

#if  !defined(MATH_LIB_A_MATH_LIB_H) || defined(USE_CUDA)
#define MATH_LIB_A_MATH_LIB_H

#include <iterator>

#if defined(AML_USE_STD_COMPLEX)

#include <complex>

#endif

#if defined(USE_OPENCL)

#define AML_NO_STRING

#endif

#if !defined(AML_NO_STRING)

#include <string>
#include <sstream>

#endif

#include <stdint.h>

#if !defined(USE_OPENCL)

#include <math.h>

#endif

#define AML_SAFE_MATH

#define AML_LN10 2.3025850929940456840179914546843642076011014886287729760333279009675726096773524802359972050895982983419677840422862486334095254

#if defined(AML_CUDA) && !defined(USE_CUDA)
#define USE_CUDA

#include "amathlib.h"
#undef USE_CUDA
#endif

#if defined(AML_PREFIX)
#undef AML_PREFIX
#undef AML_FUNCTION
#endif

#if defined(USE_CUDA)
#define AML_PREFIX(name) CU_ ## name
#define AML_FUNCTION __device__ __forceinline__
#else
#define AML_PREFIX(name) name
#define AML_FUNCTION inline

#endif

#define DEBUG_TO_INDEX(row, column) ((column - 1) * 4 + (row-1))

//#define USE_AVX512
//#define USE_AVX
//#define USE_SSE

//#define USE_SSE
//#define USE_SSE2
//#define USE_SSE3
//#define USE_SSE41
//#define USE_SSE42

//#define USE_AVX
//#define USE_AVX2
//#define USE_FMA


//#define USE_AVX512
//#define USE_AVX512F
//#define USE_KNC

#if __has_cpp_attribute(unlikely) && __has_cpp_attribute(likely)
#define UNLIKELY [[unlikely]]
#define LIKELY [[likely]]

#else
#define UNLIKELY
#define LIKELY

#endif

#if defined(__cpp_concepts)

#define USE_CONCEPTS

#endif


#ifndef DEBUG

#if defined(__x86_64__) && !defined(__global__)
#ifdef __SSE__
#define USE_SSE
#endif


#ifdef __SSE2__
#define USE_SSE2
#endif


#ifdef __SSE2__
#define USE_SSE2
#endif


#ifdef __SSE4_1__
#define USE_SSE41
#endif


#ifdef __SSE4_2__
#define USE_SSE42
#endif

#ifdef __AVX__
#define USE_AVX
#endif


#ifdef __AVX2__
#define USE_AVX2
#endif


#ifdef __FMA__
#define USE_FMA
#endif

#ifdef __AVX512F__
#define USE_AVX512F
#endif


#endif // INTEL

#endif //NDEBUG

//#if defined(DEBUG)
#if defined(__x86_64__) && !defined(__global__)
#if defined(USE_AVX512F)
#define USE_AVX512
#endif
#if defined(USE_AVX512)
#define USE_AVX512F
#define USE_FMA
#endif
#if defined(USE_FMA)
#define USE_AVX2
#endif
#if defined(USE_AVX2)
#define USE_AVX
#endif
#if defined(USE_AVX)
#define USE_SSE42
#endif
#if defined(USE_SSE42)
#define USE_SSE41
#endif
#if defined(USE_SSE41)
#define USE_SSE3
#endif
#if defined(USE_SSE3)
#define USE_SSE2
#endif
#if defined(USE_SSE2)
#define USE_SSE1
#endif
#if defined(USE_SSE1)
#define USE_SSE
#endif

//#endif // DEBUG

#if defined(USE_AVX512) && !defined(__global__)

#include <immintrin.h>

#endif


#if defined(USE_AVX) && !defined(__global__)

#include <immintrin.h>

#endif
#if defined(USE_SSE) && !defined(__global__)

#include <emmintrin.h>

#endif //USE_SSE
#endif //INTEL

#if defined(__EMSCRIPTEN__)
#include <wasm_simd128.h>
#define USE_WASM_SIMD
#endif

#if defined(__ARM_NEON)

#include <arm_neon.h>

#define USE_NEON


#endif


union AML_PREFIX(doublevec4) {
#ifdef USE_AVX
	__m256d avx;
#endif
#ifdef USE_SSE
	__m128d sse[2];
#endif
#ifdef USE_NEON
	float64x2_t neon[2];
#endif
	double c[4];
};

union AML_PREFIX(doublevec8) {
#ifdef USE_AVX512
	__m512d avx512;
#endif
#ifdef USE_AVX
	__m256d avx[2];
#endif
#ifdef USE_SSE
	__m128d sse[4];
#endif
#ifdef USE_NEON
	float64x2_t neon[4];
#endif
	double c[8];
};

union AML_PREFIX(doublemat4x4) {
#ifdef USE_AVX512
	__m512d avx512[2];
#endif
#ifdef USE_AVX
	__m256d avx[4];
#endif
#ifdef USE_SSE
	__m128d sse[8];
#endif
#ifdef USE_NEON
	float64x2_t neon[8];
#endif
	double c[16];
};

union AML_PREFIX(doublevec2) {
#ifdef USE_SSE
	__m128d sse;
#endif
#ifdef USE_NEON
	float64x2_t neon;
#endif
	double c[2];
};

union AML_PREFIX(doublevec1) {
	double c;
};


union AML_PREFIX(floatvec4) {
#ifdef USE_SSE
	__m128 sse;
#endif
#ifdef USE_NEON
	float32x4_t neon;
#endif
	float c[4];
};

union AML_PREFIX(floatvec8) {
#ifdef USE_AVX
	__m256 avx;
#endif
#ifdef USE_SSE
	__m128 sse[2];
#endif
#ifdef USE_NEON
	float32x4_t neon[2];
#endif
	float c[8];
};


union AML_PREFIX(floatvec2) {
#ifdef USE_NEON
	float32x2_t neon;
#endif
	float c[2];
};

union AML_PREFIX(floatvec1) {
	float c;
};


union AML_PREFIX(u8vec1) { // limited use
	uint8_t c;
};

union AML_PREFIX(u8vec2) {
	uint8_t c[2];
};


union AML_PREFIX(u8vec3) {
	uint8_t c[3];
};


union AML_PREFIX(u8vec4) {
	uint8_t c[4];
};


union AML_PREFIX(u8vec8) {
	uint8_t c[8];
};


union AML_PREFIX(u8vec16) {
#if defined(USE_SSE)
	__m128i sse;
#endif
	uint8_t c[16];
};


union AML_PREFIX(u8vec32) {
#if defined(USE_AVX)
	__m256i avx;
#endif
#if defined(USE_SSE)
	__m128i sse[4];
#endif
	uint8_t c[32];
};


union AML_PREFIX(u8vec64) {

#if defined(USE_AVX512)
	__m512i avx512;
#endif
#if defined(USE_AVX)
	__m256i avx[2];
#endif
#if defined(USE_SSE)
	__m128i sse[8];
#endif
	uint8_t c[64];
};

union AML_PREFIX(u16vec2) {
	uint16_t c[2];
};


union AML_PREFIX(u16vec3) {
	uint16_t c[3];
};


union AML_PREFIX(u16vec4) {
#if defined(USE_SSE)
	__m128i sse;
#endif
	uint16_t c[4];
};


union AML_PREFIX(u16vec8) {
#if defined(USE_SSE)
	__m128i sse[2];
#endif
	uint16_t c[8];
};


union AML_PREFIX(u16vec16) {
#if defined(USE_SSE)
	__m128i sse[4];
#endif
	uint16_t c[16];
};


union AML_PREFIX(u16vec32) {
#if defined(USE_AVX512)
	__m512i avx512;
#endif
#if defined(USE_AVX)
	__m256i avx[2];
#endif
#if defined(USE_SSE)
	__m128i sse[8];
#endif
	uint16_t c[32];
};


union AML_PREFIX(u16vec64) {

#if defined(USE_AVX512)
	__m512i avx512[2];
#endif
#if defined(USE_AVX)
	__m256i avx[4];
#endif
#if defined(USE_SSE)
	__m128i sse[16];
#endif
	uint16_t c[64];
};

namespace AML_PREFIX(AML) {

#if defined(USE_CONCEPTS)
	template<class T>
	concept Number = requires(T a, T b){
		a + a;
		a * a;
		a / a;
		pow(a, b);
	};
#endif


#if defined(USE_CONCEPTS)

	template<Number T>
#else
	template<class T>
#endif
	AML_FUNCTION T
	mapLinear(const T value, const T lowerInput, const T upperInput, const T lowerOutput,
			  const T upperOutput) {
		return ((value - lowerInput) * ((upperOutput - lowerOutput) / (upperInput - lowerInput))) + lowerOutput;
	}


#if defined(USE_CONCEPTS)

	template<Number T>
#else
	template<class T>
#endif
	AML_FUNCTION T
	mapNonLinear(const T value, const T lowerInput, const T upperInput,
				 const T lowerOutput, const T upperOutput, const T factor) {
		return (((pow(((value - lowerInput) / (upperInput - lowerInput)), factor)) * (upperOutput - lowerOutput)) +
				lowerOutput);
	}


#if defined(USE_CONCEPTS)

	template<Number T>
#else
	template<class T>
#endif
	AML_FUNCTION T interpolate(const T min, const T max, const T ratio) {
		return (ratio * (min - max) + max);
	}

#if defined(USE_CONCEPTS)

	template<Number T>
#else
	template<class T>
#endif
	AML_FUNCTION T interpolate(const T val1, const T val2, const T val3, const T ratio) {
		return ratio * (ratio * (val1 - val2 - val2 + val3) + val2 + val2 - val3 - val3) + val3;
	}

#if defined(USE_CONCEPTS)

	template<Number T>
#else
	template<class T>
#endif
	AML_FUNCTION T arithmeticMean(T a1, T a2) {
#if defined(AML_SAFE_MATH)
		return a1 + (a2 - a1) / 2;
#else
		return (a1 + a2) / 2;
#endif
	}
}


class AML_PREFIX(VectorU16_2D) {
public:
	AML_PREFIX(u16vec2) v{};

	AML_FUNCTION AML_PREFIX(VectorU16_2D)() {
		v.c[0] = 0;
		v.c[1] = 0;
	}


	AML_FUNCTION AML_PREFIX(VectorU16_2D)(uint16_t a, uint16_t b) {
		v.c[0] = a;
		v.c[1] = b;
	}
};

class AML_PREFIX(VectorU8_1D) {
public:
	AML_PREFIX(u8vec1) v{};

	AML_FUNCTION AML_PREFIX(VectorU8_1D)() {
		v.c = 0;
	}

	AML_FUNCTION explicit AML_PREFIX(VectorU8_1D)(const AML_PREFIX(u8vec1) &vec) {
		v = vec;
	}


	AML_FUNCTION AML_PREFIX(VectorU8_1D)(uint8_t a) {
		v.c = a;
	}

	AML_FUNCTION uint8_t &operator[]([[maybe_unused]]uint32_t location) {
		return v.c;
	}

	AML_FUNCTION bool anyTrue() {
		if (v.c) {
			return true;
		}
		return false;
	}

	AML_FUNCTION bool anyTrue(AML_PREFIX(VectorU8_1D) mask) {
		if (v.c && mask.v.c) {
			return true;
		}
		return false;
	}

	AML_FUNCTION bool allTrue() {
		if (!v.c) {
			return false;
		}
		return true;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_1D) operator!() {
		AML_PREFIX(VectorU8_1D) ret;
		ret.v.c = !v.c;
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_1D) *bitNot() {
		v.c = !v.c;
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_1D) operator&(AML_PREFIX(VectorU8_1D) o) {
		AML_PREFIX(VectorU8_1D) ret;
		ret.v.c = v.c & o.v.c;
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_1D) *bitAnd(AML_PREFIX(VectorU8_1D) o) {
		v.c = v.c & o.v.c;
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_1D) operator&&(AML_PREFIX(VectorU8_1D) o) {
		AML_PREFIX(VectorU8_1D) ret;
		ret.v.c = v.c && o.v.c;
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_1D) *boolAnd(AML_PREFIX(VectorU8_1D) o) {
		v.c = v.c && o.v.c;
		return this;
	}
};

class AML_PREFIX(VectorU8_2D) {
public:
	AML_PREFIX(u8vec2) v{};

	AML_FUNCTION AML_PREFIX(VectorU8_2D)() {
		v.c[0] = 0;
		v.c[1] = 0;
	}

	AML_FUNCTION explicit AML_PREFIX(VectorU8_2D)(const AML_PREFIX(u8vec2) &vec) {
		v = vec;
	}


	AML_FUNCTION AML_PREFIX(VectorU8_2D)(const uint8_t a, const uint8_t b) {
		v.c[0] = a;
		v.c[1] = b;
	}

	AML_FUNCTION uint8_t &operator[](const uint32_t location) {
		return v.c[location];
	}

	AML_FUNCTION bool anyTrue() {
		if (v.c[0]) {
			return true;
		}
		if (v.c[1]) {
			return true;
		}
		return false;
	}

	AML_FUNCTION bool anyTrue(const AML_PREFIX(VectorU8_2D) mask) {
		if (v.c[0] && mask.v.c[0]) {
			return true;
		}
		if (v.c[1] && mask.v.c[1]) {
			return true;
		}
		return false;
	}

	AML_FUNCTION bool allTrue() {
		if (!v.c[0]) {
			return false;
		}
		if (!v.c[1]) {
			return false;
		}
		return true;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_2D) operator!() {
		AML_PREFIX(VectorU8_2D) ret;
		ret.v.c[0] = !v.c[0];
		ret.v.c[1] = !v.c[1];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_2D) *bitNot() {
		v.c[0] = !v.c[0];
		v.c[1] = !v.c[1];
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_2D) operator&(const AML_PREFIX(VectorU8_2D) o) {
		AML_PREFIX(VectorU8_2D) ret;
		ret.v.c[0] = v.c[0] & o.v.c[0];
		ret.v.c[1] = v.c[1] & o.v.c[1];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_2D) *bitAnd(AML_PREFIX(VectorU8_2D) o) {
		v.c[0] = v.c[0] & o.v.c[0];
		v.c[1] = v.c[1] & o.v.c[1];
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_2D) operator&&(const AML_PREFIX(VectorU8_2D) o) {
		AML_PREFIX(VectorU8_2D) ret;
		ret.v.c[0] = v.c[0] && o.v.c[0];
		ret.v.c[1] = v.c[1] && o.v.c[1];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_2D) *boolAnd(const AML_PREFIX(VectorU8_2D) o) {
		v.c[0] = v.c[0] && o.v.c[0];
		v.c[1] = v.c[1] && o.v.c[1];
		return this;
	}
};

class AML_PREFIX(VectorU8_4D) {
public:
	AML_PREFIX(u8vec4) v;

	AML_FUNCTION AML_PREFIX(VectorU8_4D)() {
		v.c[0] = 0;
		v.c[1] = 0;
	}

	AML_FUNCTION explicit AML_PREFIX(VectorU8_4D)(const AML_PREFIX(u8vec4) &vec) {
		v = vec;
	}


	AML_FUNCTION AML_PREFIX(VectorU8_4D)(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
		v.c[0] = a;
		v.c[1] = b;
		v.c[2] = c;
		v.c[3] = d;
	}

	AML_FUNCTION uint8_t &operator[](uint32_t location) {
		return v.c[location];
	}

	AML_FUNCTION bool anyTrue() {
		if (v.c[0]) {
			return true;
		}
		if (v.c[1]) {
			return true;
		}
		if (v.c[2]) {
			return true;
		}
		if (v.c[3]) {
			return true;
		}
		return false;
	}

	AML_FUNCTION bool anyTrue(AML_PREFIX(VectorU8_4D) mask) {
		if (v.c[0] && mask.v.c[0]) {
			return true;
		}
		if (v.c[1] && mask.v.c[1]) {
			return true;
		}
		if (v.c[2] && mask.v.c[2]) {
			return true;
		}
		if (v.c[3] && mask.v.c[3]) {
			return true;
		}
		return false;
	}

	AML_FUNCTION bool allTrue() {
		if (!v.c[0]) {
			return false;
		}
		if (!v.c[1]) {
			return false;
		}
		if (!v.c[2]) {
			return false;
		}
		if (!v.c[3]) {
			return false;
		}
		return true;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_4D) operator!() {
		AML_PREFIX(VectorU8_4D) ret;
		ret.v.c[0] = !v.c[0];
		ret.v.c[1] = !v.c[1];
		ret.v.c[2] = !v.c[2];
		ret.v.c[3] = !v.c[3];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_4D) *bitNot() {
		v.c[0] = !v.c[0];
		v.c[1] = !v.c[1];
		v.c[2] = !v.c[2];
		v.c[3] = !v.c[3];
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_4D) operator&(const AML_PREFIX(VectorU8_4D) o) {
		AML_PREFIX(VectorU8_4D) ret;
		ret.v.c[0] = v.c[0] & o.v.c[0];
		ret.v.c[1] = v.c[1] & o.v.c[1];
		ret.v.c[2] = v.c[2] & o.v.c[2];
		ret.v.c[3] = v.c[3] & o.v.c[3];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_4D) *bitAnd(const AML_PREFIX(VectorU8_4D) o) {
		v.c[0] = v.c[0] & o.v.c[0];
		v.c[1] = v.c[1] & o.v.c[1];
		v.c[2] = v.c[2] & o.v.c[2];
		v.c[3] = v.c[3] & o.v.c[3];
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_4D) operator&&(const AML_PREFIX(VectorU8_4D) o) {
		AML_PREFIX(VectorU8_4D) ret;
		ret.v.c[0] = v.c[0] && o.v.c[0];
		ret.v.c[1] = v.c[1] && o.v.c[1];
		ret.v.c[2] = v.c[2] && o.v.c[2];
		ret.v.c[3] = v.c[3] && o.v.c[3];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_4D) *boolAnd(const AML_PREFIX(VectorU8_4D) o) {
		v.c[0] = v.c[0] && o.v.c[0];
		v.c[1] = v.c[1] && o.v.c[1];
		v.c[2] = v.c[2] && o.v.c[2];
		v.c[3] = v.c[3] && o.v.c[3];
		return this;
	}
};

class AML_PREFIX(VectorU8_8D) {
public:
	AML_PREFIX(u8vec8) v{};

	AML_FUNCTION AML_PREFIX(VectorU8_8D)() {
		v.c[0] = 0;
		v.c[1] = 0;
		v.c[2] = 0;
		v.c[3] = 0;
		v.c[4] = 0;
		v.c[5] = 0;
		v.c[6] = 0;
		v.c[7] = 0;
	}

	AML_FUNCTION explicit AML_PREFIX(VectorU8_8D)(const AML_PREFIX(u8vec8) &vec) {
		v = vec;
	}


	AML_FUNCTION AML_PREFIX(VectorU8_8D)(uint8_t a, uint8_t b, uint8_t c, uint8_t d, uint8_t e, uint8_t f, uint8_t g,
										 uint8_t h) {
		v.c[0] = a;
		v.c[1] = b;
		v.c[2] = c;
		v.c[3] = d;
		v.c[4] = e;
		v.c[5] = f;
		v.c[6] = g;
		v.c[7] = h;
	}

	AML_FUNCTION uint8_t &operator[](uint32_t location) {
		return v.c[location];
	}

	AML_FUNCTION bool anyTrue() {
		if (v.c[0]) {
			return true;
		}
		if (v.c[1]) {
			return true;
		}
		if (v.c[2]) {
			return true;
		}
		if (v.c[3]) {
			return true;
		}
		if (v.c[4]) {
			return true;
		}
		if (v.c[5]) {
			return true;
		}
		if (v.c[6]) {
			return true;
		}
		if (v.c[7]) {
			return true;
		}
		return false;
	}

	AML_FUNCTION bool anyTrue(AML_PREFIX(VectorU8_8D) mask) {
		if (v.c[0] && mask.v.c[0]) {
			return true;
		}
		if (v.c[1] && mask.v.c[1]) {
			return true;
		}
		if (v.c[2] && mask.v.c[2]) {
			return true;
		}
		if (v.c[3] && mask.v.c[3]) {
			return true;
		}
		if (v.c[4] && mask.v.c[4]) {
			return true;
		}
		if (v.c[5] && mask.v.c[5]) {
			return true;
		}
		if (v.c[6] && mask.v.c[6]) {
			return true;
		}
		if (v.c[7] && mask.v.c[7]) {
			return true;
		}
		return false;
	}

	AML_FUNCTION bool allTrue() {
		if (!v.c[0]) {
			return false;
		}
		if (!v.c[1]) {
			return false;
		}
		if (!v.c[2]) {
			return false;
		}
		if (!v.c[3]) {
			return false;
		}
		if (!v.c[4]) {
			return false;
		}
		if (!v.c[5]) {
			return false;
		}
		if (!v.c[6]) {
			return false;
		}
		if (!v.c[7]) {
			return false;
		}
		return true;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_8D) operator!() {
		AML_PREFIX(VectorU8_8D) ret;
		ret.v.c[0] = !v.c[0];
		ret.v.c[1] = !v.c[1];
		ret.v.c[2] = !v.c[2];
		ret.v.c[3] = !v.c[3];
		ret.v.c[4] = !v.c[4];
		ret.v.c[5] = !v.c[5];
		ret.v.c[6] = !v.c[6];
		ret.v.c[7] = !v.c[7];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_8D) *bitNot() {
		v.c[0] = !v.c[0];
		v.c[1] = !v.c[1];
		v.c[2] = !v.c[2];
		v.c[3] = !v.c[3];
		v.c[4] = !v.c[4];
		v.c[5] = !v.c[5];
		v.c[6] = !v.c[6];
		v.c[7] = !v.c[7];
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_8D) operator&(const AML_PREFIX(VectorU8_8D) o) {
		AML_PREFIX(VectorU8_8D) ret;
		ret.v.c[0] = v.c[0] & o.v.c[0];
		ret.v.c[1] = v.c[1] & o.v.c[1];
		ret.v.c[2] = v.c[2] & o.v.c[2];
		ret.v.c[3] = v.c[3] & o.v.c[3];
		ret.v.c[4] = v.c[4] & o.v.c[4];
		ret.v.c[5] = v.c[5] & o.v.c[5];
		ret.v.c[6] = v.c[6] & o.v.c[6];
		ret.v.c[7] = v.c[7] & o.v.c[7];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_8D) *bitAnd(const AML_PREFIX(VectorU8_8D) o) {
		v.c[0] = v.c[0] & o.v.c[0];
		v.c[1] = v.c[1] & o.v.c[1];
		v.c[2] = v.c[2] & o.v.c[2];
		v.c[3] = v.c[3] & o.v.c[3];
		v.c[4] = v.c[4] & o.v.c[4];
		v.c[5] = v.c[5] & o.v.c[5];
		v.c[6] = v.c[6] & o.v.c[6];
		v.c[7] = v.c[7] & o.v.c[7];
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_8D) operator&&(const AML_PREFIX(VectorU8_8D) o) {
		AML_PREFIX(VectorU8_8D) ret;
		ret.v.c[0] = v.c[0] && o.v.c[0];
		ret.v.c[1] = v.c[1] && o.v.c[1];
		ret.v.c[2] = v.c[2] && o.v.c[2];
		ret.v.c[3] = v.c[3] && o.v.c[3];
		ret.v.c[4] = v.c[4] && o.v.c[4];
		ret.v.c[5] = v.c[5] && o.v.c[5];
		ret.v.c[6] = v.c[6] && o.v.c[6];
		ret.v.c[7] = v.c[7] && o.v.c[7];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_8D) *boolAnd(AML_PREFIX(VectorU8_8D) o) {
		v.c[0] = v.c[0] && o.v.c[0];
		v.c[1] = v.c[1] && o.v.c[1];
		v.c[2] = v.c[2] && o.v.c[2];
		v.c[3] = v.c[3] && o.v.c[3];
		v.c[4] = v.c[4] && o.v.c[4];
		v.c[5] = v.c[5] && o.v.c[5];
		v.c[6] = v.c[6] && o.v.c[6];
		v.c[7] = v.c[7] && o.v.c[7];
		return this;
	}
};

class AML_PREFIX(VectorDouble1D) {
public:
	AML_PREFIX(doublevec1) v{};

	AML_FUNCTION AML_PREFIX(VectorDouble1D) *set(double value, AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) { v.c = value; }
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble1D)(const double *const values) {
		v.c = values[0];
	}

	AML_FUNCTION AML_PREFIX(VectorDouble1D)() {
		v.c = 0.0f;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble1D)(double value) {
		v.c = value;
	}

	AML_FUNCTION double &operator[]([[maybe_unused]]uint32_t location) {
		return v.c;
	}

};

class AML_PREFIX(VectorFloat1D) {
public:
	AML_PREFIX(floatvec1) v{};

	AML_FUNCTION AML_PREFIX(VectorFloat1D) *set(float value, AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) { v.c = value; }
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat1D)(const float *const values) {
		v.c = values[0];
	}

	AML_FUNCTION AML_PREFIX(VectorFloat1D)() {
		v.c = 0.0f;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat1D)(float value) {
		v.c = value;
	}

	AML_FUNCTION float &operator[]([[maybe_unused]]uint32_t location) {
		return v.c;
	}

};

class AML_PREFIX(VectorDouble2D) {
public:
	AML_PREFIX(doublevec2) v{};

	AML_FUNCTION AML_PREFIX(VectorDouble2D) *set(double value, const AML_PREFIX(VectorU8_2D) mask) {
		if (mask.v.c[0]) { v.c[0] = value; }
		if (mask.v.c[1]) { v.c[1] = value; }
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble2D)(const double *const values) {
		v.c[0] = values[0];
		v.c[1] = values[1];
	}

	AML_FUNCTION AML_PREFIX(VectorDouble2D)() {
		v.c[0] = 0.0f;
		v.c[1] = 0.0f;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble2D)(double value) {
		v.c[0] = value;
		v.c[1] = value;
	}

	AML_FUNCTION double &operator[](uint32_t location) {
		return v.c[location];
	}

#if defined(USE_SSE)

	AML_FUNCTION VectorDouble2D(const __m128d value) {
		v.sse = value;
	}

#endif

};

class AML_PREFIX(VectorFloat2D) {
public:
	AML_PREFIX(floatvec2) v{};

	AML_FUNCTION AML_PREFIX(VectorFloat2D) *set(float value, AML_PREFIX(VectorU8_2D) mask) {
		if (mask.v.c[0]) { v.c[0] = value; }
		if (mask.v.c[1]) { v.c[1] = value; }
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat2D)(const float *const values) {
		v.c[0] = values[0];
		v.c[1] = values[1];
	}

	AML_FUNCTION AML_PREFIX(VectorFloat2D)() {
		v.c[0] = 0.0f;
		v.c[1] = 0.0f;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat2D)(float value) {
		v.c[0] = value;
		v.c[1] = value;
	}

	AML_FUNCTION float &operator[](uint32_t location) {
		return v.c[location];
	}

};

class AML_PREFIX(VectorDouble4D) {
private:

public:
	AML_PREFIX(doublevec4) v{};

	AML_FUNCTION double &operator[](uint32_t position) {
		return v.c[position];
	}

	AML_FUNCTION void operator+=(AML_PREFIX(VectorDouble4D) vec2) {
#if defined(USE_AVX)
		v.avx = _mm256_add_pd(v.avx, vec2.v.avx);
#elif defined(USE_SSE) // SSE2
		v.sse[0] = _mm_add_pd(v.sse[0], vec2.v.sse[0]);
		v.sse[1] = _mm_add_pd(v.sse[1], vec2.v.sse[1]);
#elif defined(USE_NEON)
		v.neon[0] = vaddq_f64(v.neon[0], vec2.v.neon[0]);
		v.neon[1] = vaddq_f64(v.neon[1], vec2.v.neon[1]);
#else
		v.c[0] += vec2[0];
		v.c[1] += vec2[1];
		v.c[2] += vec2[2];
		v.c[3] += vec2[3];
#endif


	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) operator+(const AML_PREFIX(VectorDouble4D) vec2) {
#if defined(USE_AVX)
		AML_PREFIX(VectorDouble4D) ret;
		ret.v.avx = _mm256_add_pd(v.avx, vec2.v.avx);
		return ret;
#elif defined(USE_SSE2)
		AML_PREFIX(VectorDouble4D) ret;
		ret.v.sse[0] = _mm_add_pd(v.sse[0], vec2.v.sse[0]);
		ret.v.sse[1] = _mm_add_pd(v.sse[1], vec2.v.sse[1]);
		return ret;
#elif defined(USE_NEON)
		AML_PREFIX(VectorDouble4D) ret;
		ret.v.neon[0] = vaddq_f64(v.neon[0], vec2.v.neon[0]);
		ret.v.neon[1] = vaddq_f64(v.neon[1], vec2.v.neon[1]);
		return ret;
#else
		AML_PREFIX(VectorDouble4D) ret(v.c[0] + vec2.v.c[0], v.c[1] + vec2.v.c[1], v.c[2] + vec2.v.c[2],
									   v.c[3] + vec2.v.c[3]);
		return ret;
#endif


	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) operator+(double a) {
#if defined(USE_AVX)
		AML_PREFIX(VectorDouble4D) ret(a);
		ret.v.avx = _mm256_add_pd(v.avx, ret.v.avx);
		return ret;
#elif defined(USE_SSE2)
		AML_PREFIX(VectorDouble4D) ret(a);
		ret.v.sse[0] = _mm_add_pd(v.sse[0], ret.v.sse[0]);
		ret.v.sse[1] = _mm_add_pd(v.sse[1], ret.v.sse[1]);
		return ret;
#elif defined(USE_NEON)
		AML_PREFIX(VectorDouble4D) ret(a);
		ret.v.neon[0] = vaddq_f64(v.neon[0], ret.v.neon[0]);
		ret.v.neon[1] = vaddq_f64(v.neon[1], ret.v.neon[1]);
		return ret;
#else
		AML_PREFIX(VectorDouble4D) ret(v.c[0] + a, v.c[1] + a, v.c[2] + a, v.c[3] + a);
		return ret;
#endif


	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) *add(const AML_PREFIX(VectorDouble4D) a) {
#if defined(USE_AVX)
		v.avx = _mm256_add_pd(v.avx, a.v.avx);
#elif defined(USE_SSE) // SSE2
		v.sse[0] = _mm_add_pd(v.sse[0], a.v.sse[0]);
		v.sse[1] = _mm_add_pd(v.sse[1], a.v.sse[1]);
#elif defined(USE_NEON)
		v.neon[0] = vaddq_f64(v.neon[0], a.v.neon[0]);
		v.neon[1] = vaddq_f64(v.neon[1], a.v.neon[1]);
#else
		v.c[0] += a.v.c[0];
		v.c[1] += a.v.c[1];
		v.c[2] += a.v.c[2];
		v.c[3] += a.v.c[3];
#endif
		return this;
	}

	AML_FUNCTION void inverse() {

#if defined(USE_AVX)
		AML_PREFIX(doublevec4) a = {0.0f, 0.0f, 0.0f, 0.0f};
		v.avx = _mm256_sub_pd(a.avx, v.avx);
#elif defined(USE_SSE) // SSE2
		double a[2] = {0.0f, 0.0f};
		__m128d b = _mm_loadu_pd(a);
		v.sse[0] = _mm_sub_pd(b, v.sse[0]);
		v.sse[1] = _mm_sub_pd(b, v.sse[1]);
#else
		v.c[0] = 0 - v.c[0];
		v.c[1] = 0 - v.c[1];
		v.c[2] = 0 - v.c[2];
		v.c[3] = 0 - v.c[3];
#endif
	}

	AML_FUNCTION void operator-=(AML_PREFIX(VectorDouble4D) vec2) {
#if defined(USE_AVX)
		v.avx = _mm256_sub_pd(v.avx, vec2.v.avx);
#else
		v.c[0] -= vec2[0];
		v.c[1] -= vec2[1];
		v.c[2] -= vec2[2];
		v.c[3] -= vec2[3];
#endif


	}


	AML_FUNCTION AML_PREFIX(VectorDouble4D) operator-(const AML_PREFIX(VectorDouble4D) vec2) {
#if defined(USE_AVX)
		AML_PREFIX(VectorDouble4D) ret;
		ret.v.avx = _mm256_sub_pd(v.avx, vec2.v.avx);
		return ret;
#else
		AML_PREFIX(VectorDouble4D) ret(v.c[0] - vec2.v.c[0], v.c[1] - vec2.v.c[1], v.c[2] - vec2.v.c[2],
									   v.c[3] - vec2.v.c[3]);
		return ret;
#endif


	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) operator-(double a) {
		AML_PREFIX(VectorDouble4D) ret(a);
#if defined(USE_AVX)
		ret.v.avx = _mm256_sub_pd(v.avx, ret.v.avx);
		return ret;
#else
		ret = AML_PREFIX(VectorDouble4D)(v.c[0] - a, v.c[1] - a, v.c[2] - a, v.c[3] - a);
		return ret;
#endif


	}

	AML_FUNCTION double length() {
		return sqrt(v.c[0] * v.c[0] + v.c[1] * v.c[1] + v.c[2] * v.c[2] + v.c[3] * v.c[3]);
	}

	AML_FUNCTION void normalize() {
		double vecLength = 1 / length();
		v.c[0] *= vecLength;
		v.c[1] *= vecLength;
		v.c[2] *= vecLength;
		v.c[3] *= vecLength;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) *forEachSin() {
#if defined(USE_AVX) && defined(__INTEL_COMPILER)
		v.avx = _mm256_sin_pd(v.avx);
#else
		v.c[0] = sin(v.c[0]);
		v.c[1] = sin(v.c[1]);
		v.c[2] = sin(v.c[2]);
		v.c[3] = sin(v.c[3]);
#endif
		return this;
	}

	AML_FUNCTION void operator*=(double scalar) {
		v.c[0] *= scalar;
		v.c[1] *= scalar;
		v.c[2] *= scalar;
		v.c[3] *= scalar;
	}

	// for each multiply
	AML_FUNCTION void operator*=(AML_PREFIX(VectorDouble4D) vec2) {
#if defined(USE_AVX)
		v.avx = _mm256_mul_pd(v.avx, vec2.v.avx);
#elif defined(USE_NEON)
		v.neon[0] = vmulq_f64(v.neon[0], vec2.v.neon[0]);
		v.neon[1] = vmulq_f64(v.neon[1], vec2.v.neon[1]);
#else
		v.c[0] *= vec2.v.c[0];
		v.c[1] *= vec2.v.c[1];
		v.c[2] *= vec2.v.c[2];
		v.c[3] *= vec2.v.c[3];
#endif
	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) *forEachSqrt() {
#if defined(USE_AVX)
		v.avx = _mm256_sqrt_pd(v.avx);
#elif defined(USE_SSE)
		v.sse[0] = _mm_sqrt_pd(v.sse[0]);
		v.sse[1] = _mm_sqrt_pd(v.sse[1]);
#else
		v.c[0] = sqrt(v.c[0]);
		v.c[1] = sqrt(v.c[1]);
		v.c[2] = sqrt(v.c[2]);
		v.c[3] = sqrt(v.c[3]);
#endif
		return this;

	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) operator*(double a) {
		AML_PREFIX(VectorDouble4D) ret;
#if defined(USE_AVX)
		AML_PREFIX(doublevec4) b = {a, a, a, a};
		ret.v.avx = _mm256_mul_pd(v.avx, b.avx);
#else
		ret.v.c[0] *= v.c[0] * a;
		ret.v.c[1] *= v.c[1] * a;
		ret.v.c[2] *= v.c[2] * a;
		ret.v.c[3] *= v.c[3] * a;
#endif
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) operator/(double a) {
		AML_PREFIX(VectorDouble4D) ret;
#if defined(USE_AVX)
		AML_PREFIX(doublevec4) b = {a, a, a, a};
		ret.v.avx = _mm256_div_pd(v.avx, b.avx);
#else
		ret.v.c[0] /= v.c[0] * a;
		ret.v.c[1] /= v.c[1] * a;
		ret.v.c[2] /= v.c[2] * a;
		ret.v.c[3] /= v.c[3] * a;
#endif
		return ret;
	}


	AML_FUNCTION AML_PREFIX(VectorDouble4D) *capBetween1_0() {
#if defined(USE_AVX)
		AML_PREFIX(doublevec4) one = {1.0f, 1.0f, 1.0f, 1.0f};
		v.avx = _mm256_max_pd(v.avx, one.avx);
		v.avx = _mm256_min_pd(v.avx, _mm256_setzero_pd());
#else
		if (v.c[0] > 1) UNLIKELY {
			v.c[0] = 1;
		} else if (v.c[0] < 0) UNLIKELY {
			v.c[0] = 0;
		}
		if (v.c[1] > 1) UNLIKELY {
			v.c[1] = 1;
		} else if (v.c[1] < 0) UNLIKELY {
			v.c[1] = 0;
		}
		if (v.c[2] > 1) UNLIKELY {
			v.c[2] = 1;
		} else if (v.c[2] < 0) UNLIKELY {
			v.c[2] = 0;
		}
		if (v.c[3] > 1) UNLIKELY {
			v.c[3] = 1;
		} else if (v.c[3] < 0) UNLIKELY {
			v.c[3] = 0;
		}
#endif
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) *capBetweenX_Y(const double upperBoundary, const double lowerBoundary) {
#if defined(USE_AVX)
		AML_PREFIX(doublevec4) upper = {upperBoundary, upperBoundary, upperBoundary, upperBoundary};
		AML_PREFIX(doublevec4) lower = {lowerBoundary, lowerBoundary, lowerBoundary, lowerBoundary};
		v.avx = _mm256_max_pd(v.avx, upper.avx);
		v.avx = _mm256_min_pd(v.avx, lower.avx);
#else
		if (v.c[0] > upperBoundary) {
			v.c[0] = upperBoundary;
		} else if (v.c[0] < lowerBoundary) {
			v.c[0] = lowerBoundary;
		}
		if (v.c[1] > upperBoundary) {
			v.c[1] = upperBoundary;
		} else if (v.c[1] < lowerBoundary) {
			v.c[1] = lowerBoundary;
		}
		if (v.c[2] > upperBoundary) {
			v.c[2] = upperBoundary;
		} else if (v.c[2] < lowerBoundary) {
			v.c[2] = lowerBoundary;
		}
		if (v.c[3] > upperBoundary) {
			v.c[3] = upperBoundary;
		} else if (v.c[3] < lowerBoundary) {
			v.c[3] = lowerBoundary;
		}
#endif
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) *
	map(const double lowerInput, const double upperInput, const double lowerOutput, const double upperOutput) {
#if defined(USE_AVX)
		AML_PREFIX(doublevec4) a = {lowerInput, lowerInput, lowerInput, lowerInput};
		a.avx = _mm256_sub_pd(v.avx, a.avx);
		double factor = (upperOutput - lowerOutput) / (upperInput - lowerInput);
		AML_PREFIX(doublevec4) b = {factor, factor, factor, factor};
		a.avx = _mm256_mul_pd(a.avx, b.avx);
		AML_PREFIX(doublevec4) c = {lowerOutput, lowerOutput, lowerOutput, lowerOutput};
		v.avx = _mm256_add_pd(a.avx, c.avx);
#else
		v.c[0] = ((v.c[0] - lowerInput) * ((upperOutput - lowerOutput) / (upperInput - lowerInput))) + lowerOutput;
		v.c[1] = ((v.c[1] - lowerInput) * ((upperOutput - lowerOutput) / (upperInput - lowerInput))) + lowerOutput;
		v.c[2] = ((v.c[2] - lowerInput) * ((upperOutput - lowerOutput) / (upperInput - lowerInput))) + lowerOutput;
		v.c[3] = ((v.c[3] - lowerInput) * ((upperOutput - lowerOutput) / (upperInput - lowerInput))) + lowerOutput;
#endif
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) *mapNonLinear(const double lowerInput, const double upperInput,
														  const double lowerOutput, const double upperOutput,
														  const double factor) {
		v.c[0] = (((pow(((v.c[0] - lowerInput) / (upperInput - lowerInput)), factor)) * (upperOutput - lowerOutput)) +
				  lowerOutput);
		v.c[1] = (((pow(((v.c[1] - lowerInput) / (upperInput - lowerInput)), factor)) * (upperOutput - lowerOutput)) +
				  lowerOutput);
		v.c[2] = (((pow(((v.c[2] - lowerInput) / (upperInput - lowerInput)), factor)) * (upperOutput - lowerOutput)) +
				  lowerOutput);
		v.c[3] = (((pow(((v.c[3] - lowerInput) / (upperInput - lowerInput)), factor)) * (upperOutput - lowerOutput)) +
				  lowerOutput);
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) *forEachInterpolate(const double min, const double max) {
		v.c[0] = (v.c[0] * (min - max) + max);
		v.c[1] = (v.c[1] * (min - max) + max);
		v.c[2] = (v.c[2] * (min - max) + max);
		v.c[3] = (v.c[3] * (min - max) + max);
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) *interpolate(const AML_PREFIX(VectorDouble4D) max, double ratio) {
		v.c[0] = (ratio * (v.c[0] - max.v.c[0]) + max.v.c[0]);
		v.c[1] = (ratio * (v.c[1] - max.v.c[1]) + max.v.c[1]);
		v.c[2] = (ratio * (v.c[2] - max.v.c[2]) + max.v.c[2]);
		v.c[3] = (ratio * (v.c[3] - max.v.c[3]) + max.v.c[3]);
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) *
	interpolate(const AML_PREFIX(VectorDouble4D) val2, const AML_PREFIX(VectorDouble4D) val3, double ratio) {
		v.c[0] = ratio *
				 (ratio * (v.c[0] - val2.v.c[0] - val2.v.c[0] + val3.v.c[0]) + val2.v.c[0] + val2.v.c[0] - val3.v.c[0] -
				  val3.v.c[0]) + val3.v.c[0];
		v.c[1] = ratio *
				 (ratio * (v.c[1] - val2.v.c[1] - val2.v.c[1] + val3.v.c[1]) + val2.v.c[1] + val2.v.c[1] - val3.v.c[1] -
				  val3.v.c[1]) + val3.v.c[1];
		v.c[2] = ratio *
				 (ratio * (v.c[2] - val2.v.c[2] - val2.v.c[2] + val3.v.c[2]) + val2.v.c[2] + val2.v.c[2] - val3.v.c[2] -
				  val3.v.c[2]) + val3.v.c[2];
		v.c[3] = ratio *
				 (ratio * (v.c[3] - val2.v.c[3] - val2.v.c[3] + val3.v.c[3]) + val2.v.c[3] + val2.v.c[3] - val3.v.c[3] -
				  val3.v.c[3]) + val3.v.c[3];
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) *set(double value, AML_PREFIX(VectorU8_4D) mask) {
		if (mask.v.c[0]) { v.c[0] = value; }
		if (mask.v.c[1]) { v.c[1] = value; }
		if (mask.v.c[2]) { v.c[2] = value; }
		if (mask.v.c[3]) { v.c[3] = value; }
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) *set(const AML_PREFIX(VectorDouble4D) value, AML_PREFIX(VectorU8_4D) mask) {
		if (mask.v.c[0]) { v.c[0] = value.v.c[0]; }
		if (mask.v.c[1]) { v.c[1] = value.v.c[1]; }
		if (mask.v.c[2]) { v.c[2] = value.v.c[2]; }
		if (mask.v.c[3]) { v.c[3] = value.v.c[3]; }
		return this;
	}

	template<const int a, const int b, const int c, const int d>
	AML_FUNCTION AML_PREFIX(VectorDouble4D) *permutation() {
#if defined(USE_AVX2)
		v.avx = _mm256_permute4x64_pd(v.avx,a + (b << 2) + (c << 4) + (d << 6));
#else
		double a1 = v.c[a];
		double b1 = v.c[b];
		double c1 = v.c[c];
		double d1 = v.c[d];
		v.c[0] = a1;
		v.c[1] = b1;
		v.c[2] = c1;
		v.c[3] = d1;
#endif
		return this;
	}


	AML_FUNCTION AML_PREFIX(VectorDouble4D)(double a, double b, double c, double d) {
		v.c[0] = a;
		v.c[1] = b;
		v.c[2] = c;
		v.c[3] = d;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D)() {
		v.c[0] = 0;
		v.c[1] = 0;
		v.c[2] = 0;
		v.c[3] = 0;
	}

	AML_FUNCTION explicit AML_PREFIX(VectorDouble4D)(const double a) {
		v.c[0] = a;
		v.c[1] = a;
		v.c[2] = a;
		v.c[3] = a;
	}

#ifdef USE_AVX

	AML_FUNCTION explicit AML_PREFIX(VectorDouble4D)(const __m256d a) {
		v.avx = a;
	}

#endif

#if defined(USE_SSE)

	AML_FUNCTION explicit AML_PREFIX(VectorDouble4D)(const __m128d *const values) {
		v.sse[0] = values[0];
		v.sse[1] = values[1];
	}

#endif

	AML_FUNCTION explicit AML_PREFIX(VectorDouble4D)(const double *const values) {
		v.c[0] = values[0];
		v.c[1] = values[1];
		v.c[2] = values[2];
		v.c[3] = values[3];
	}

};

class AML_PREFIX(VectorFloat4D) {
public:
	AML_PREFIX(floatvec4) v{};

	AML_FUNCTION float &operator[](uint32_t position) {
		return v.c[position];
	}

	AML_FUNCTION void operator+=(const AML_PREFIX(VectorFloat4D) vec2) {
		v.c[0] += vec2.v.c[0];
		v.c[1] += vec2.v.c[1];
		v.c[2] += vec2.v.c[2];
		v.c[3] += vec2.v.c[3];
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) operator+(const AML_PREFIX(VectorFloat4D) vec2) {
		AML_PREFIX(VectorFloat4D) ret(v.c[0] + vec2.v.c[0], v.c[1] + vec2.v.c[1], v.c[2] + vec2.v.c[2],
									  v.c[3] + vec2.v.c[3]);
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) operator+(float a) {
		AML_PREFIX(VectorFloat4D) ret(v.c[0] + a, v.c[1] + a, v.c[2] + a, v.c[3] + a);
		return ret;


	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) *add(const AML_PREFIX(VectorFloat4D) a) {
		v.c[0] += a.v.c[0];
		v.c[1] += a.v.c[1];
		v.c[2] += a.v.c[2];
		v.c[3] += a.v.c[3];
		return this;
	}

	AML_FUNCTION void inverse() {
		v.c[0] = 0 - v.c[0];
		v.c[1] = 0 - v.c[1];
		v.c[2] = 0 - v.c[2];
		v.c[3] = 0 - v.c[3];
	}

	AML_FUNCTION void operator-=(const AML_PREFIX(VectorFloat4D) vec2) {
		v.c[0] -= vec2.v.c[0];
		v.c[1] -= vec2.v.c[1];
		v.c[2] -= vec2.v.c[2];
		v.c[3] -= vec2.v.c[3];

	}


	AML_FUNCTION AML_PREFIX(VectorFloat4D) operator-(const AML_PREFIX(VectorFloat4D) vec2) {
		AML_PREFIX(VectorFloat4D) ret(v.c[0] - vec2.v.c[0], v.c[1] - vec2.v.c[1], v.c[2] - vec2.v.c[2],
									  v.c[3] - vec2.v.c[3]);
		return ret;


	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) operator-(float a) {
		AML_PREFIX(VectorFloat4D) ret(a);
		ret = AML_PREFIX(VectorFloat4D)(v.c[0] - a, v.c[1] - a, v.c[2] - a, v.c[3] - a);
		return ret;


	}

	AML_FUNCTION double length() {
		return sqrt(v.c[0] * v.c[0] + v.c[1] * v.c[1] + v.c[2] * v.c[2] + v.c[3] * v.c[3]);
	}

	AML_FUNCTION void normalize() {
		float vecLength = 1 / length();
		v.c[0] *= vecLength;
		v.c[1] *= vecLength;
		v.c[2] *= vecLength;
		v.c[3] *= vecLength;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) *forEachSin() {
		v.c[0] = sin(v.c[0]);
		v.c[1] = sin(v.c[1]);
		v.c[2] = sin(v.c[2]);
		v.c[3] = sin(v.c[3]);
		return this;
	}

	AML_FUNCTION void operator*=(float scalar) {
		v.c[0] *= scalar;
		v.c[1] *= scalar;
		v.c[2] *= scalar;
		v.c[3] *= scalar;
	}

	// for each multiply
	AML_FUNCTION void operator*=(const AML_PREFIX(VectorFloat4D) vec2) {
		v.c[0] *= vec2.v.c[0];
		v.c[1] *= vec2.v.c[1];
		v.c[2] *= vec2.v.c[2];
		v.c[3] *= vec2.v.c[3];
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) *forEachSqrt() {
		v.c[0] = sqrt(v.c[0]);
		v.c[1] = sqrt(v.c[1]);
		v.c[2] = sqrt(v.c[2]);
		v.c[3] = sqrt(v.c[3]);
		return this;

	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) operator*(float a) {
		AML_PREFIX(VectorFloat4D) ret;
		ret.v.c[0] *= v.c[0] * a;
		ret.v.c[1] *= v.c[1] * a;
		ret.v.c[2] *= v.c[2] * a;
		ret.v.c[3] *= v.c[3] * a;
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) operator/(float a) {
		AML_PREFIX(VectorFloat4D) ret;
		ret.v.c[0] /= v.c[0] * a;
		ret.v.c[1] /= v.c[1] * a;
		ret.v.c[2] /= v.c[2] * a;
		ret.v.c[3] /= v.c[3] * a;
		return ret;
	}


	AML_FUNCTION AML_PREFIX(VectorFloat4D) *capBetween1_0() {
		if (v.c[0] > 1) UNLIKELY {
			v.c[0] = 1;
		} else if (v.c[0] < 0) UNLIKELY {
			v.c[0] = 0;
		}
		if (v.c[1] > 1) UNLIKELY {
			v.c[1] = 1;
		} else if (v.c[1] < 0) UNLIKELY {
			v.c[1] = 0;
		}
		if (v.c[2] > 1) UNLIKELY {
			v.c[2] = 1;
		} else if (v.c[2] < 0) UNLIKELY {
			v.c[2] = 0;
		}
		if (v.c[3] > 1) UNLIKELY {
			v.c[3] = 1;
		} else if (v.c[3] < 0) UNLIKELY {
			v.c[3] = 0;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) *capBetweenX_Y(const float upperBoundary, const float lowerBoundary) {
		if (v.c[0] > upperBoundary) {
			v.c[0] = upperBoundary;
		} else if (v.c[0] < lowerBoundary) {
			v.c[0] = lowerBoundary;
		}
		if (v.c[1] > upperBoundary) {
			v.c[1] = upperBoundary;
		} else if (v.c[1] < lowerBoundary) {
			v.c[1] = lowerBoundary;
		}
		if (v.c[2] > upperBoundary) {
			v.c[2] = upperBoundary;
		} else if (v.c[2] < lowerBoundary) {
			v.c[2] = lowerBoundary;
		}
		if (v.c[3] > upperBoundary) {
			v.c[3] = upperBoundary;
		} else if (v.c[3] < lowerBoundary) {
			v.c[3] = lowerBoundary;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) *
	map(const float lowerInput, const float upperInput, const float lowerOutput, const float upperOutput) {
		v.c[0] = ((v.c[0] - lowerInput) * ((upperOutput - lowerOutput) / (upperInput - lowerInput))) + lowerOutput;
		v.c[1] = ((v.c[1] - lowerInput) * ((upperOutput - lowerOutput) / (upperInput - lowerInput))) + lowerOutput;
		v.c[2] = ((v.c[2] - lowerInput) * ((upperOutput - lowerOutput) / (upperInput - lowerInput))) + lowerOutput;
		v.c[3] = ((v.c[3] - lowerInput) * ((upperOutput - lowerOutput) / (upperInput - lowerInput))) + lowerOutput;
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) *mapNonLinear(const float lowerInput, const float upperInput,
														 const float lowerOutput, const float upperOutput,
														 const float factor) {
		v.c[0] = (((pow(((v.c[0] - lowerInput) / (upperInput - lowerInput)), factor)) * (upperOutput - lowerOutput)) +
				  lowerOutput);
		v.c[1] = (((pow(((v.c[1] - lowerInput) / (upperInput - lowerInput)), factor)) * (upperOutput - lowerOutput)) +
				  lowerOutput);
		v.c[2] = (((pow(((v.c[2] - lowerInput) / (upperInput - lowerInput)), factor)) * (upperOutput - lowerOutput)) +
				  lowerOutput);
		v.c[3] = (((pow(((v.c[3] - lowerInput) / (upperInput - lowerInput)), factor)) * (upperOutput - lowerOutput)) +
				  lowerOutput);
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) *forEachInterpolate(const float min, const float max) {
		v.c[0] = (v.c[0] * (min - max) + max);
		v.c[1] = (v.c[1] * (min - max) + max);
		v.c[2] = (v.c[2] * (min - max) + max);
		v.c[3] = (v.c[3] * (min - max) + max);
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) *interpolate(const AML_PREFIX(VectorFloat4D) max, float ratio) {
		v.c[0] = (ratio * (v.c[0] - max.v.c[0]) + max.v.c[0]);
		v.c[1] = (ratio * (v.c[1] - max.v.c[1]) + max.v.c[1]);
		v.c[2] = (ratio * (v.c[2] - max.v.c[2]) + max.v.c[2]);
		v.c[3] = (ratio * (v.c[3] - max.v.c[3]) + max.v.c[3]);
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) *
	interpolate(const AML_PREFIX(VectorFloat4D) val2, const AML_PREFIX(VectorFloat4D) val3, float ratio) {
		v.c[0] = ratio *
				 (ratio * (v.c[0] - val2.v.c[0] - val2.v.c[0] + val3.v.c[0]) + val2.v.c[0] + val2.v.c[0] - val3.v.c[0] -
				  val3.v.c[0]) + val3.v.c[0];
		v.c[1] = ratio *
				 (ratio * (v.c[1] - val2.v.c[1] - val2.v.c[1] + val3.v.c[1]) + val2.v.c[1] + val2.v.c[1] - val3.v.c[1] -
				  val3.v.c[1]) + val3.v.c[1];
		v.c[2] = ratio *
				 (ratio * (v.c[2] - val2.v.c[2] - val2.v.c[2] + val3.v.c[2]) + val2.v.c[2] + val2.v.c[2] - val3.v.c[2] -
				  val3.v.c[2]) + val3.v.c[2];
		v.c[3] = ratio *
				 (ratio * (v.c[3] - val2.v.c[3] - val2.v.c[3] + val3.v.c[3]) + val2.v.c[3] + val2.v.c[3] - val3.v.c[3] -
				  val3.v.c[3]) + val3.v.c[3];
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) *set(float value, AML_PREFIX(VectorU8_4D) mask) {
		if (mask.v.c[0]) { v.c[0] = value; }
		if (mask.v.c[1]) { v.c[1] = value; }
		if (mask.v.c[2]) { v.c[2] = value; }
		if (mask.v.c[3]) { v.c[3] = value; }
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) *set(const AML_PREFIX(VectorFloat4D) value, AML_PREFIX(VectorU8_4D) mask) {
		if (mask.v.c[0]) { v.c[0] = value.v.c[0]; }
		if (mask.v.c[1]) { v.c[1] = value.v.c[1]; }
		if (mask.v.c[2]) { v.c[2] = value.v.c[2]; }
		if (mask.v.c[3]) { v.c[3] = value.v.c[3]; }
		return this;
	}

	template<const int a, const int b, const int c, const int d>
	AML_FUNCTION AML_PREFIX(VectorFloat4D) *permutation() {
		float a1 = v.c[a];
		float b1 = v.c[b];
		float c1 = v.c[c];
		float d1 = v.c[d];
		v.c[0] = a1;
		v.c[1] = b1;
		v.c[2] = c1;
		v.c[3] = d1;
		return this;
	}


	AML_FUNCTION AML_PREFIX(VectorFloat4D)(float a, float b, float c, float d) {
		v.c[0] = a;
		v.c[1] = b;
		v.c[2] = c;
		v.c[3] = d;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D)() {
		v.c[0] = 0;
		v.c[1] = 0;
		v.c[2] = 0;
		v.c[3] = 0;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D)(const float a) {
		v.c[0] = a;
		v.c[1] = a;
		v.c[2] = a;
		v.c[3] = a;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D)(const float *const values) {
		v.c[0] = values[0];
		v.c[1] = values[1];
		v.c[2] = values[2];
		v.c[3] = values[3];
	}

};

class AML_PREFIX(MatrixDouble4X4) {
public:
	AML_PREFIX(doublemat4x4) m{};

	AML_FUNCTION AML_PREFIX(VectorDouble4D) operator[](uint32_t column) {
		AML_PREFIX(VectorDouble4D) ret;
#if defined(USE_AVX)
		ret.v.avx = m.avx[column];
#else
		ret.v.c[0] = m.c[column * 4];
		ret.v.c[1] = m.c[column * 4 + 1];
		ret.v.c[2] = m.c[column * 4 + 2];
		ret.v.c[3] = m.c[column * 4 + 3];
#endif
		return ret;
	}

	AML_FUNCTION AML_PREFIX(MatrixDouble4X4) *identity() {
		m.c[0] = 1.0f;
		m.c[1] = 0.0f;
		m.c[2] = 0.0f;
		m.c[3] = 0.0f;
		m.c[4] = 0.0f;
		m.c[5] = 1.0f;
		m.c[6] = 0.0f;
		m.c[7] = 0.0f;
		m.c[8] = 0.0f;
		m.c[9] = 0.0f;
		m.c[10] = 1.0f;
		m.c[11] = 0.0f;
		m.c[12] = 0.0f;
		m.c[13] = 0.0f;
		m.c[14] = 0.0f;
		m.c[15] = 1.0f;
		return this;
	}

	AML_FUNCTION AML_PREFIX(MatrixDouble4X4) operator*(const AML_PREFIX(MatrixDouble4X4) &b) {
		AML_PREFIX(MatrixDouble4X4) ret;
#if defined(USE_AVX512F)
		__m512d O0 = (__m512d) {b.m.c[0], b.m.c[0], b.m.c[0], b.m.c[0], b.m.c[4], b.m.c[4], b.m.c[4], b.m.c[4]};
		__m512d O1 = (__m512d) {b.m.c[1], b.m.c[1], b.m.c[1], b.m.c[1], b.m.c[5], b.m.c[5], b.m.c[5], b.m.c[5]};
		__m512d O2 = (__m512d) {b.m.c[2], b.m.c[2], b.m.c[2], b.m.c[2], b.m.c[6], b.m.c[6], b.m.c[6], b.m.c[6]};
		__m512d O3 = (__m512d) {b.m.c[3], b.m.c[3], b.m.c[3], b.m.c[3], b.m.c[7], b.m.c[7], b.m.c[7], b.m.c[7]};

		__m512d T0 = _mm512_insertf64x4(_mm512_castpd256_pd512(m.avx[0]), m.avx[0], 1);
		__m512d T1 = _mm512_insertf64x4(_mm512_castpd256_pd512(m.avx[1]), m.avx[1], 1);
		__m512d T2 = _mm512_insertf64x4(_mm512_castpd256_pd512(m.avx[2]), m.avx[2], 1);
		__m512d T3 = _mm512_insertf64x4(_mm512_castpd256_pd512(m.avx[3]), m.avx[3], 1);
		ret.m.avx512[0] = _mm512_mul_pd(T0, O0);
		ret.m.avx512[0] = _mm512_fmadd_pd(T1, O1, ret.m.avx512[0]);
		ret.m.avx512[0] = _mm512_fmadd_pd(T2, O2, ret.m.avx512[0]);
		ret.m.avx512[0] = _mm512_fmadd_pd(T3, O3, ret.m.avx512[0]);

		__m512d O4 = (__m512d) {b.m.c[8], b.m.c[8], b.m.c[8], b.m.c[8], b.m.c[12], b.m.c[12], b.m.c[12], b.m.c[12]};
		__m512d O5 = (__m512d) {b.m.c[9], b.m.c[9], b.m.c[9], b.m.c[9], b.m.c[13], b.m.c[13], b.m.c[13], b.m.c[13]};
		__m512d O6 = (__m512d) {b.m.c[10], b.m.c[10], b.m.c[10], b.m.c[10], b.m.c[14], b.m.c[14], b.m.c[14], b.m.c[14]};
		__m512d O7 = (__m512d) {b.m.c[11], b.m.c[11], b.m.c[11], b.m.c[11], b.m.c[15], b.m.c[15], b.m.c[15], b.m.c[15]};
		ret.m.avx512[1] = _mm512_mul_pd(T0, O4);
		ret.m.avx512[1] = _mm512_fmadd_pd(T1, O5, ret.m.avx512[1]);
		ret.m.avx512[1] = _mm512_fmadd_pd(T2, O6, ret.m.avx512[1]);
		ret.m.avx512[1] = _mm512_fmadd_pd(T3, O7, ret.m.avx512[1]);
#elif defined(USE_FMA)
		/*
		 * m0 * bcst 0
		 * m0 * bcst 4
		 * m0 * bcst 8
		 * m0 * bcst 12
		 */
		__m256d O0 = _mm256_broadcastsd_pd((__m128d) {b.m.c[0], 0.0f});
		__m256d O1 = _mm256_broadcastsd_pd((__m128d) {b.m.c[1], 0.0f});
		__m256d O2 = _mm256_broadcastsd_pd((__m128d) {b.m.c[2], 0.0f});
		__m256d O3 = _mm256_broadcastsd_pd((__m128d) {b.m.c[3], 0.0f});

		ret.m.avx[0] = _mm256_mul_pd(m.avx[0], O0);
		ret.m.avx[0] = _mm256_fmadd_pd(m.avx[1], O1, ret.m.avx[0]);
		ret.m.avx[0] = _mm256_fmadd_pd(m.avx[2], O2, ret.m.avx[0]);
		ret.m.avx[0] = _mm256_fmadd_pd(m.avx[3], O3, ret.m.avx[0]);

		__m256d O4 = _mm256_broadcastsd_pd((__m128d) {b.m.c[4], 0.0f});
		__m256d O5 = _mm256_broadcastsd_pd((__m128d) {b.m.c[5], 0.0f});
		__m256d O6 = _mm256_broadcastsd_pd((__m128d) {b.m.c[6], 0.0f});
		__m256d O7 = _mm256_broadcastsd_pd((__m128d) {b.m.c[7], 0.0f});

		ret.m.avx[1] = _mm256_mul_pd(m.avx[0], O4);
		ret.m.avx[1] = _mm256_fmadd_pd(m.avx[1], O5, ret.m.avx[1]);
		ret.m.avx[1] = _mm256_fmadd_pd(m.avx[2], O6, ret.m.avx[1]);
		ret.m.avx[1] = _mm256_fmadd_pd(m.avx[3], O7, ret.m.avx[1]);

		__m256d O8 = _mm256_broadcastsd_pd((__m128d) {b.m.c[8], 0.0f});
		__m256d O9 = _mm256_broadcastsd_pd((__m128d) {b.m.c[9], 0.0f});
		__m256d O10 = _mm256_broadcastsd_pd((__m128d) {b.m.c[10], 0.0f});
		__m256d O11 = _mm256_broadcastsd_pd((__m128d) {b.m.c[11], 0.0f});

		ret.m.avx[2] = _mm256_mul_pd(m.avx[0], O8);
		ret.m.avx[2] = _mm256_fmadd_pd(m.avx[1], O9, ret.m.avx[2]);
		ret.m.avx[2] = _mm256_fmadd_pd(m.avx[2], O10, ret.m.avx[2]);
		ret.m.avx[2] = _mm256_fmadd_pd(m.avx[3], O11, ret.m.avx[2]);

		__m256d O12 = _mm256_broadcastsd_pd((__m128d) {b.m.c[12], 0.0f});
		__m256d O13 = _mm256_broadcastsd_pd((__m128d) {b.m.c[13], 0.0f});
		__m256d O14 = _mm256_broadcastsd_pd((__m128d) {b.m.c[14], 0.0f});
		__m256d O15 = _mm256_broadcastsd_pd((__m128d) {b.m.c[15], 0.0f});

		ret.m.avx[3] = _mm256_mul_pd(m.avx[0], O12);
		ret.m.avx[3] = _mm256_fmadd_pd(m.avx[1], O13, ret.m.avx[3]);
		ret.m.avx[3] = _mm256_fmadd_pd(m.avx[2], O14, ret.m.avx[3]);
		ret.m.avx[3] = _mm256_fmadd_pd(m.avx[3], O15, ret.m.avx[3]);

#elif defined(USE_SSE2)

		ret.m.sse[0] = _mm_mul_pd(m.sse[0], (__m128d) {b.m.c[0], b.m.c[0]});
		__m128d cache = _mm_mul_pd(m.sse[2], (__m128d) {b.m.c[1], b.m.c[1]});
		ret.m.sse[0] = _mm_add_pd(cache, ret.m.sse[0]);
		cache = _mm_mul_pd(m.sse[4], (__m128d) {b.m.c[2], b.m.c[2]});
		ret.m.sse[0] = _mm_add_pd(cache, ret.m.sse[0]);
		cache = _mm_mul_pd(m.sse[6], (__m128d) {b.m.c[3], b.m.c[3]});
		ret.m.sse[0] = _mm_add_pd(cache, ret.m.sse[0]);
		//
		ret.m.sse[1] = _mm_mul_pd(m.sse[1], (__m128d) {b.m.c[0], b.m.c[0]});
		cache = _mm_mul_pd(m.sse[3], (__m128d) {b.m.c[1], b.m.c[1]});
		ret.m.sse[1] = _mm_add_pd(cache, ret.m.sse[1]);
		cache = _mm_mul_pd(m.sse[5], (__m128d) {b.m.c[2], b.m.c[2]});
		ret.m.sse[1] = _mm_add_pd(cache, ret.m.sse[1]);
		cache = _mm_mul_pd(m.sse[7], (__m128d) {b.m.c[3], b.m.c[3]});
		ret.m.sse[1] = _mm_add_pd(cache, ret.m.sse[1]);
		//

		ret.m.sse[2] = _mm_mul_pd(m.sse[0], (__m128d) {b.m.c[4], b.m.c[4]});
		cache = _mm_mul_pd(m.sse[2], (__m128d) {b.m.c[5], b.m.c[5]});
		ret.m.sse[2] = _mm_add_pd(cache, ret.m.sse[2]);
		cache = _mm_mul_pd(m.sse[4], (__m128d) {b.m.c[6], b.m.c[6]});
		ret.m.sse[2] = _mm_add_pd(cache, ret.m.sse[2]);
		cache = _mm_mul_pd(m.sse[6], (__m128d) {b.m.c[7], b.m.c[7]});
		ret.m.sse[2] = _mm_add_pd(cache, ret.m.sse[2]);
		//
		ret.m.sse[3] = _mm_mul_pd(m.sse[1], (__m128d) {b.m.c[4], b.m.c[4]});
		cache = _mm_mul_pd(m.sse[3], (__m128d) {b.m.c[5], b.m.c[5]});
		ret.m.sse[3] = _mm_add_pd(cache, ret.m.sse[3]);
		cache = _mm_mul_pd(m.sse[5], (__m128d) {b.m.c[6], b.m.c[6]});
		ret.m.sse[3] = _mm_add_pd(cache, ret.m.sse[3]);
		cache = _mm_mul_pd(m.sse[7], (__m128d) {b.m.c[7], b.m.c[7]});
		ret.m.sse[3] = _mm_add_pd(cache, ret.m.sse[3]);
		//

		ret.m.sse[4] = _mm_mul_pd(m.sse[0], (__m128d) {b.m.c[8], b.m.c[8]});
		cache = _mm_mul_pd(m.sse[2], (__m128d) {b.m.c[9], b.m.c[9]});
		ret.m.sse[4] = _mm_add_pd(cache, ret.m.sse[4]);
		cache = _mm_mul_pd(m.sse[4], (__m128d) {b.m.c[10], b.m.c[10]});
		ret.m.sse[4] = _mm_add_pd(cache, ret.m.sse[4]);
		cache = _mm_mul_pd(m.sse[6], (__m128d) {b.m.c[11], b.m.c[11]});
		ret.m.sse[4] = _mm_add_pd(cache, ret.m.sse[4]);
		//
		ret.m.sse[5] = _mm_mul_pd(m.sse[1], (__m128d) {b.m.c[8], b.m.c[8]});
		cache = _mm_mul_pd(m.sse[3], (__m128d) {b.m.c[9], b.m.c[9]});
		ret.m.sse[5] = _mm_add_pd(cache, ret.m.sse[5]);
		cache = _mm_mul_pd(m.sse[5], (__m128d) {b.m.c[10], b.m.c[10]});
		ret.m.sse[5] = _mm_add_pd(cache, ret.m.sse[5]);
		cache = _mm_mul_pd(m.sse[7], (__m128d) {b.m.c[11], b.m.c[11]});
		ret.m.sse[5] = _mm_add_pd(cache, ret.m.sse[5]);
		//

		ret.m.sse[6] = _mm_mul_pd(m.sse[0], (__m128d) {b.m.c[12], b.m.c[12]});
		cache = _mm_mul_pd(m.sse[2], (__m128d) {b.m.c[13], b.m.c[13]});
		ret.m.sse[6] = _mm_add_pd(cache, ret.m.sse[6]);
		cache = _mm_mul_pd(m.sse[4], (__m128d) {b.m.c[14], b.m.c[14]});
		ret.m.sse[6] = _mm_add_pd(cache, ret.m.sse[6]);
		cache = _mm_mul_pd(m.sse[6], (__m128d) {b.m.c[15], b.m.c[15]});
		ret.m.sse[6] = _mm_add_pd(cache, ret.m.sse[6]);
		//
		ret.m.sse[7] = _mm_mul_pd(m.sse[1], (__m128d) {b.m.c[12], b.m.c[12]});
		cache = _mm_mul_pd(m.sse[3], (__m128d) {b.m.c[13], b.m.c[13]});
		ret.m.sse[7] = _mm_add_pd(cache, ret.m.sse[7]);
		cache = _mm_mul_pd(m.sse[5], (__m128d) {b.m.c[14], b.m.c[14]});
		ret.m.sse[7] = _mm_add_pd(cache, ret.m.sse[7]);
		cache = _mm_mul_pd(m.sse[7], (__m128d) {b.m.c[15], b.m.c[15]});
		ret.m.sse[7] = _mm_add_pd(cache, ret.m.sse[7]);
#elif defined(USE_NEON)
		ret.m.neon[0] = vmulq_f64(m.neon[0], (float64x2_t) {b.m.c[0], b.m.c[0]});
		ret.m.neon[0] = vfmaq_f64(ret.m.neon[0], m.neon[2], (float64x2_t) {b.m.c[1], b.m.c[1]});
		ret.m.neon[0] = vfmaq_f64(ret.m.neon[0], m.neon[4], (float64x2_t) {b.m.c[2], b.m.c[2]});
		ret.m.neon[0] = vfmaq_f64(ret.m.neon[0], m.neon[6], (float64x2_t) {b.m.c[3], b.m.c[3]});
		//
		ret.m.neon[1] = vmulq_f64(m.neon[1], (float64x2_t) {b.m.c[0], b.m.c[0]});
		ret.m.neon[1] = vfmaq_f64(ret.m.neon[1], m.neon[3], (float64x2_t) {b.m.c[1], b.m.c[1]});
		ret.m.neon[1] = vfmaq_f64(ret.m.neon[1], m.neon[5], (float64x2_t) {b.m.c[2], b.m.c[2]});
		ret.m.neon[1] = vfmaq_f64(ret.m.neon[1], m.neon[7], (float64x2_t) {b.m.c[3], b.m.c[3]});
		//
		ret.m.neon[2] = vmulq_f64(m.neon[0], (float64x2_t) {b.m.c[4], b.m.c[4]});
		ret.m.neon[2] = vfmaq_f64(ret.m.neon[2], m.neon[2], (float64x2_t) {b.m.c[5], b.m.c[5]});
		ret.m.neon[2] = vfmaq_f64(ret.m.neon[2], m.neon[4], (float64x2_t) {b.m.c[6], b.m.c[6]});
		ret.m.neon[2] = vfmaq_f64(ret.m.neon[2], m.neon[6], (float64x2_t) {b.m.c[7], b.m.c[7]});

		//
		ret.m.neon[3] = vmulq_f64(m.neon[1], (float64x2_t) {b.m.c[4], b.m.c[4]});
		ret.m.neon[3] = vfmaq_f64(ret.m.neon[3], m.neon[3], (float64x2_t) {b.m.c[5], b.m.c[5]});
		ret.m.neon[3] = vfmaq_f64(ret.m.neon[3], m.neon[5], (float64x2_t) {b.m.c[6], b.m.c[6]});
		ret.m.neon[3] = vfmaq_f64(ret.m.neon[3], m.neon[7], (float64x2_t) {b.m.c[7], b.m.c[7]});

		//
		ret.m.neon[4] = vmulq_f64(m.neon[0], (float64x2_t) {b.m.c[8], b.m.c[8]});
		ret.m.neon[4] = vfmaq_f64(ret.m.neon[4], m.neon[2], (float64x2_t) {b.m.c[9], b.m.c[9]});
		ret.m.neon[4] = vfmaq_f64(ret.m.neon[4], m.neon[4], (float64x2_t) {b.m.c[10], b.m.c[10]});
		ret.m.neon[4] = vfmaq_f64(ret.m.neon[4], m.neon[6], (float64x2_t) {b.m.c[11], b.m.c[11]});

		//
		ret.m.neon[5] = vmulq_f64(m.neon[1], (float64x2_t) {b.m.c[8], b.m.c[8]});
		ret.m.neon[5] = vfmaq_f64(ret.m.neon[5], m.neon[3], (float64x2_t) {b.m.c[9], b.m.c[9]});
		ret.m.neon[5] = vfmaq_f64(ret.m.neon[5], m.neon[5], (float64x2_t) {b.m.c[10], b.m.c[10]});
		ret.m.neon[5] = vfmaq_f64(ret.m.neon[5], m.neon[7], (float64x2_t) {b.m.c[11], b.m.c[11]});

		//

		ret.m.neon[6] = vmulq_f64(m.neon[0], (float64x2_t) {b.m.c[12], b.m.c[12]});
		ret.m.neon[6] = vfmaq_f64(ret.m.neon[6], m.neon[2], (float64x2_t) {b.m.c[13], b.m.c[13]});
		ret.m.neon[6] = vfmaq_f64(ret.m.neon[6], m.neon[4], (float64x2_t) {b.m.c[14], b.m.c[14]});
		ret.m.neon[6] = vfmaq_f64(ret.m.neon[6], m.neon[6], (float64x2_t) {b.m.c[15], b.m.c[15]});

		//
		ret.m.neon[7] = vmulq_f64(m.neon[1], (float64x2_t) {b.m.c[12], b.m.c[12]});
		ret.m.neon[7] = vfmaq_f64(ret.m.neon[7], m.neon[3], (float64x2_t) {b.m.c[13], b.m.c[13]});
		ret.m.neon[7] = vfmaq_f64(ret.m.neon[7], m.neon[5], (float64x2_t) {b.m.c[14], b.m.c[14]});
		ret.m.neon[7] = vfmaq_f64(ret.m.neon[7], m.neon[7], (float64x2_t) {b.m.c[15], b.m.c[15]});


#else
		ret.m.c[0] = m.c[0] * b.m.c[0] + m.c[4] * b.m.c[1] + m.c[8] * b.m.c[2] + m.c[12] * b.m.c[3];
		ret.m.c[1] = m.c[1] * b.m.c[0] + m.c[5] * b.m.c[1] + m.c[9] * b.m.c[2] + m.c[13] * b.m.c[3];
		ret.m.c[2] = m.c[2] * b.m.c[0] + m.c[6] * b.m.c[1] + m.c[10] * b.m.c[2] + m.c[14] * b.m.c[3];
		ret.m.c[3] = m.c[3] * b.m.c[0] + m.c[7] * b.m.c[1] + m.c[11] * b.m.c[2] + m.c[15] * b.m.c[3];
		ret.m.c[4] = m.c[0] * b.m.c[4] + m.c[4] * b.m.c[5] + m.c[8] * b.m.c[6] + m.c[12] * b.m.c[7];
		ret.m.c[5] = m.c[1] * b.m.c[4] + m.c[5] * b.m.c[5] + m.c[9] * b.m.c[6] + m.c[13] * b.m.c[7];
		ret.m.c[6] = m.c[2] * b.m.c[4] + m.c[6] * b.m.c[5] + m.c[10] * b.m.c[6] + m.c[14] * b.m.c[7];
		ret.m.c[7] = m.c[3] * b.m.c[4] + m.c[7] * b.m.c[5] + m.c[11] * b.m.c[6] + m.c[15] * b.m.c[7];
		ret.m.c[8] = m.c[0] * b.m.c[8] + m.c[4] * b.m.c[9] + m.c[8] * b.m.c[10] + m.c[12] * b.m.c[11];
		ret.m.c[9] = m.c[1] * b.m.c[8] + m.c[5] * b.m.c[9] + m.c[9] * b.m.c[10] + m.c[13] * b.m.c[11];
		ret.m.c[10] = m.c[2] * b.m.c[8] + m.c[6] * b.m.c[9] + m.c[10] * b.m.c[10] + m.c[14] * b.m.c[11];
		ret.m.c[11] = m.c[3] * b.m.c[8] + m.c[7] * b.m.c[9] + m.c[11] * b.m.c[10] + m.c[15] * b.m.c[11];
		ret.m.c[12] = m.c[0] * b.m.c[12] + m.c[4] * b.m.c[13] + m.c[8] * b.m.c[14] + m.c[12] * b.m.c[15];
		ret.m.c[13] = m.c[1] * b.m.c[12] + m.c[5] * b.m.c[13] + m.c[9] * b.m.c[14] + m.c[13] * b.m.c[15];
		ret.m.c[14] = m.c[2] * b.m.c[12] + m.c[6] * b.m.c[13] + m.c[10] * b.m.c[14] + m.c[14] * b.m.c[15];
		ret.m.c[15] = m.c[3] * b.m.c[12] + m.c[7] * b.m.c[13] + m.c[11] * b.m.c[14] + m.c[15] * b.m.c[15];
#endif
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) operator*(const AML_PREFIX(VectorDouble4D) b) {
		AML_PREFIX(VectorDouble4D) ret;
		ret.v.c[0] = m.c[0] * b.v.c[0] + m.c[4] * b.v.c[1] + m.c[8] * b.v.c[2] + m.c[12] * b.v.c[3];
		ret.v.c[1] = m.c[1] * b.v.c[0] + m.c[5] * b.v.c[1] + m.c[9] * b.v.c[2] + m.c[13] * b.v.c[3];
		ret.v.c[2] = m.c[2] * b.v.c[0] + m.c[6] * b.v.c[1] + m.c[10] * b.v.c[2] + m.c[14] * b.v.c[3];
		ret.v.c[3] = m.c[3] * b.v.c[0] + m.c[7] * b.v.c[1] + m.c[11] * b.v.c[2] + m.c[15] * b.v.c[3];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(MatrixDouble4X4)() {
		m.c[0] = 0.0f;
		m.c[1] = 0.0f;
		m.c[2] = 0.0f;
		m.c[3] = 0.0f;
		m.c[4] = 0.0f;
		m.c[5] = 0.0f;
		m.c[6] = 0.0f;
		m.c[7] = 0.0f;
		m.c[8] = 0.0f;
		m.c[9] = 0.0f;
		m.c[10] = 0.0f;
		m.c[11] = 0.0f;
		m.c[12] = 0.0f;
		m.c[13] = 0.0f;
		m.c[14] = 0.0f;
		m.c[15] = 0.0f;
	}

	AML_FUNCTION AML_PREFIX(MatrixDouble4X4)(const AML_PREFIX(VectorDouble4D) &a, const AML_PREFIX(VectorDouble4D) &b,
											 const AML_PREFIX(VectorDouble4D) &c,
											 const AML_PREFIX(VectorDouble4D) &d) {
#if defined(USE_AVX)
		m.avx[0] = a.v.avx;
		m.avx[1] = b.v.avx;
		m.avx[2] = c.v.avx;
		m.avx[3] = d.v.avx;
#else
		m.c[0] = a.v.c[0];
		m.c[1] = a.v.c[1];
		m.c[2] = a.v.c[2];
		m.c[3] = a.v.c[3];
		m.c[4] = b.v.c[0];
		m.c[5] = b.v.c[1];
		m.c[6] = b.v.c[2];
		m.c[7] = b.v.c[3];
		m.c[8] = c.v.c[0];
		m.c[9] = c.v.c[1];
		m.c[10] = c.v.c[2];
		m.c[11] = c.v.c[3];
		m.c[12] = d.v.c[0];
		m.c[13] = d.v.c[1];
		m.c[14] = d.v.c[2];
		m.c[15] = d.v.c[3];
#endif
	}

	AML_FUNCTION AML_PREFIX(MatrixDouble4X4)(const AML_PREFIX(MatrixDouble4X4) &b) {
#if defined(USE_AVX512)
		m.avx512[0] = b.m.avx512[0];
		m.avx512[1] = b.m.avx512[1];
#else
		m.c[0] = b.m.c[0];
		m.c[1] = b.m.c[1];
		m.c[2] = b.m.c[2];
		m.c[3] = b.m.c[3];
		m.c[4] = b.m.c[4];
		m.c[5] = b.m.c[5];
		m.c[6] = b.m.c[6];
		m.c[7] = b.m.c[7];
		m.c[8] = b.m.c[8];
		m.c[9] = b.m.c[9];
		m.c[10] = b.m.c[10];
		m.c[11] = b.m.c[11];
		m.c[12] = b.m.c[12];
		m.c[13] = b.m.c[13];
		m.c[14] = b.m.c[14];
		m.c[15] = b.m.c[15];
#endif
	}
};

class AML_PREFIX(VectorDouble8D) {
public:
	AML_PREFIX(doublevec8) v{};

	AML_FUNCTION double operator[](uint32_t position) {
		return v.c[position];
	}

	AML_FUNCTION void operator+=(const AML_PREFIX(VectorDouble8D) &vec2) {
#if defined(USE_AVX512F) || defined(KNCNI)
		v.avx512 = _mm512_add_pd(v.avx512, vec2.v.avx512);
#elif defined(USE_AVX)
		v.avx[0] = _mm256_add_pd(v.avx[0], vec2.v.avx[0]);
		v.avx[1] = _mm256_add_pd(v.avx[1], vec2.v.avx[1]);
#elif defined(USE_SSE) // SSE2
		v.sse[0] = _mm_add_pd(v.sse[0], vec2.v.sse[0]);
		v.sse[1] = _mm_add_pd(v.sse[1], vec2.v.sse[1]);
		v.sse[2] = _mm_add_pd(v.sse[2], vec2.v.sse[2]);
		v.sse[3] = _mm_add_pd(v.sse[3], vec2.v.sse[3]);
#else
		v.c[0] += vec2.v.c[0];
		v.c[1] += vec2.v.c[1];
		v.c[2] += vec2.v.c[2];
		v.c[3] += vec2.v.c[3];
		v.c[4] += vec2.v.c[4];
		v.c[5] += vec2.v.c[5];
		v.c[6] += vec2.v.c[6];
		v.c[7] += vec2.v.c[7];
#endif


	}

	AML_FUNCTION AML_PREFIX(VectorDouble8D)(double a, double b, double c, double d, double e, double f, double g,
											double h) {
		v.c[0] = a;
		v.c[1] = b;
		v.c[2] = c;
		v.c[3] = d;
		v.c[4] = e;
		v.c[5] = f;
		v.c[6] = g;
		v.c[7] = h;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble8D)(const AML_PREFIX(VectorDouble4D) a, const AML_PREFIX(VectorDouble4D) b) {
#if defined(USE_AVX)
		v.avx[0] = a.v.avx;
		v.avx[1] = b.v.avx;
#else
		v.c[0] = a.v.c[0];
		v.c[1] = a.v.c[1];
		v.c[2] = a.v.c[2];
		v.c[3] = a.v.c[3];
		v.c[4] = b.v.c[0];
		v.c[5] = b.v.c[1];
		v.c[6] = b.v.c[2];
		v.c[7] = b.v.c[3];
#endif
	}

	AML_FUNCTION AML_PREFIX(VectorDouble8D)() {
		v.c[0] = 0.0f;
		v.c[1] = 0.0f;
		v.c[2] = 0.0f;
		v.c[3] = 0.0f;
		v.c[4] = 0.0f;
		v.c[5] = 0.0f;
		v.c[6] = 0.0f;
		v.c[7] = 0.0f;
	}

#if defined(USE_AVX)

	AML_FUNCTION explicit AML_PREFIX(VectorDouble8D)(__m256d *values) {
		v.avx[0] = values[0];
		v.avx[1] = values[1];
	}

#endif
#if defined(USE_SSE)

	AML_FUNCTION explicit AML_PREFIX(VectorDouble8D)(__m128d *values) {
		v.sse[0] = values[0];
		v.sse[1] = values[1];
		v.sse[2] = values[2];
		v.sse[3] = values[3];
	}

#endif

#if defined(USE_AVX512)

	AML_FUNCTION AML_PREFIX(VectorDouble8D)(__m512d value) {
		v.avx512 = value;
	}

#endif

	AML_FUNCTION explicit AML_PREFIX(VectorDouble8D)(const double *const values) {
		v.c[0] = values[0];
		v.c[1] = values[1];
		v.c[2] = values[2];
		v.c[3] = values[3];
		v.c[4] = values[4];
		v.c[5] = values[5];
		v.c[6] = values[6];
		v.c[7] = values[7];
	}

	AML_FUNCTION AML_PREFIX(VectorDouble8D)(const double value) {
		v.c[0] = value;
		v.c[1] = value;
		v.c[2] = value;
		v.c[3] = value;
		v.c[4] = value;
		v.c[5] = value;
		v.c[6] = value;
		v.c[7] = value;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble8D) *set(double value, AML_PREFIX(VectorU8_8D) mask) {
		if (mask.v.c[0]) { v.c[0] = value; }
		if (mask.v.c[1]) { v.c[1] = value; }
		if (mask.v.c[2]) { v.c[2] = value; }
		if (mask.v.c[3]) { v.c[3] = value; }
		if (mask.v.c[4]) { v.c[4] = value; }
		if (mask.v.c[5]) { v.c[5] = value; }
		if (mask.v.c[6]) { v.c[6] = value; }
		if (mask.v.c[7]) { v.c[7] = value; }
		return this;
	}
};

class AML_PREFIX(VectorFloat8D) {
public:
	AML_PREFIX(floatvec8) v{};

	AML_FUNCTION float operator[](uint32_t position) {
		return v.c[position];
	}

	AML_FUNCTION void operator+=(const AML_PREFIX(VectorFloat8D) &vec2) {
		v.c[0] += vec2.v.c[0];
		v.c[1] += vec2.v.c[1];
		v.c[2] += vec2.v.c[2];
		v.c[3] += vec2.v.c[3];
		v.c[4] += vec2.v.c[4];
		v.c[5] += vec2.v.c[5];
		v.c[6] += vec2.v.c[6];
		v.c[7] += vec2.v.c[7];

	}

	AML_FUNCTION AML_PREFIX(VectorFloat8D)(float a, float b, float c, float d, float e, float f, float g, float h) {
		v.c[0] = a;
		v.c[1] = b;
		v.c[2] = c;
		v.c[3] = d;
		v.c[4] = e;
		v.c[5] = f;
		v.c[6] = g;
		v.c[7] = h;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat8D)(const AML_PREFIX(VectorDouble4D) a, const AML_PREFIX(VectorDouble4D) b) {
		v.c[0] = a.v.c[0];
		v.c[1] = a.v.c[1];
		v.c[2] = a.v.c[2];
		v.c[3] = a.v.c[3];
		v.c[4] = b.v.c[0];
		v.c[5] = b.v.c[1];
		v.c[6] = b.v.c[2];
		v.c[7] = b.v.c[3];
	}

	AML_FUNCTION AML_PREFIX(VectorFloat8D)() {
		v.c[0] = 0.0f;
		v.c[1] = 0.0f;
		v.c[2] = 0.0f;
		v.c[3] = 0.0f;
		v.c[4] = 0.0f;
		v.c[5] = 0.0f;
		v.c[6] = 0.0f;
		v.c[7] = 0.0f;
	}

	AML_FUNCTION explicit AML_PREFIX(VectorFloat8D)(const float *const values) {
		v.c[0] = values[0];
		v.c[1] = values[1];
		v.c[2] = values[2];
		v.c[3] = values[3];
		v.c[4] = values[4];
		v.c[5] = values[5];
		v.c[6] = values[6];
		v.c[7] = values[7];
	}

	AML_FUNCTION explicit AML_PREFIX(VectorFloat8D)(const float value) {
		v.c[0] = value;
		v.c[1] = value;
		v.c[2] = value;
		v.c[3] = value;
		v.c[4] = value;
		v.c[5] = value;
		v.c[6] = value;
		v.c[7] = value;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat8D) *set(const float value, const AML_PREFIX(VectorU8_8D) mask) {
		if (mask.v.c[0]) { v.c[0] = value; }
		if (mask.v.c[1]) { v.c[1] = value; }
		if (mask.v.c[2]) { v.c[2] = value; }
		if (mask.v.c[3]) { v.c[3] = value; }
		if (mask.v.c[4]) { v.c[4] = value; }
		if (mask.v.c[5]) { v.c[5] = value; }
		if (mask.v.c[6]) { v.c[6] = value; }
		if (mask.v.c[7]) { v.c[7] = value; }
		return this;
	}
};

#if defined(USE_AVX512)

#define MAX_COMPLEX_64_SIZE 8
#define MAX_COMPLEX_64_TYPE Array8Complex64
#define MAX_COMPLEX_64_MASK_TYPE VectorU8_8D
#define MAX_COMPLEX_64_VECTOR_TYPE VectorDouble8D
#define IDEAL_COMPLEX_64_SIZE 8
#define IDEAL_COMPLEX_64_TYPE Array8Complex64
#define IDEAL_COMPLEX_64_MASK_TYPE VectorU8_8D
#define IDEAL_COMPLEX_64_VECTOR_TYPE VectorDouble8D
#define MIN_COMPLEX_64_SIZE 4
#define MIN_COMPLEX_64_TYPE Array4Complex64
#define MIN_COMPLEX_64_MASK_TYPE VectorU8_4D
#define MIN_COMPLEX_64_VECTOR_TYPE VectorDouble4D

#define MAX_COMPLEX_32_SIZE 8
#define MAX_COMPLEX_32_TYPE Array8Complex32
#define MAX_COMPLEX_32_MASK_TYPE VectorU8_8D
#define MAX_COMPLEX_32_VECTOR_TYPE VectorFloat8D
#define IDEAL_COMPLEX_32_SIZE 8
#define IDEAL_COMPLEX_32_TYPE Array8Complex32
#define IDEAL_COMPLEX_32_MASK_TYPE VectorU8_8D
#define IDEAL_COMPLEX_32_VECTOR_TYPE VectorFloat8D
#define MIN_COMPLEX_32_SIZE 4
#define MIN_COMPLEX_32_TYPE Array4Complex32
#define MIN_COMPLEX_32_MASK_TYPE VectorU8_4D
#define MIN_COMPLEX_32_VECTOR_TYPE VectorDouble4D


#elif defined(USE_AVX)

#define MAX_COMPLEX_64_SIZE 8
#define MAX_COMPLEX_64_TYPE Array8Complex64
#define MAX_COMPLEX_64_MASK_TYPE VectorU8_8D
#define MAX_COMPLEX_64_VECTOR_TYPE VectorDouble8D
#define IDEAL_COMPLEX_64_SIZE 4
#define IDEAL_COMPLEX_64_TYPE Array4Complex64
#define IDEAL_COMPLEX_64_MASK_TYPE VectorU8_4D
#define IDEAL_COMPLEX_64_VECTOR_TYPE VectorDouble4D
#define MIN_COMPLEX_64_SIZE 2
#define MIN_COMPLEX_64_TYPE Array2Complex64
#define MIN_COMPLEX_64_MASK_TYPE VectorU8_2D
#define MIN_COMPLEX_64_VECTOR_TYPE VectorDouble2D

#define MAX_COMPLEX_32_SIZE 8
#define MAX_COMPLEX_32_TYPE Array8Complex32
#define MAX_COMPLEX_32_MASK_TYPE VectorU8_8D
#define MAX_COMPLEX_32_VECTOR_TYPE VectorFloat8D
#define IDEAL_COMPLEX_32_SIZE 8
#define IDEAL_COMPLEX_32_TYPE Array8Complex32
#define IDEAL_COMPLEX_32_MASK_TYPE VectorU8_8D
#define IDEAL_COMPLEX_32_VECTOR_TYPE VectorFloat8D
#define MIN_COMPLEX_32_SIZE 4
#define MIN_COMPLEX_32_TYPE Array4Complex32
#define MIN_COMPLEX_32_MASK_TYPE VectorU8_4D
#define MIN_COMPLEX_32_VECTOR_TYPE VectorFloat4D


#elif defined(USE_SSE) || defined(USE_NEON) || defined(USE_WASM_SIMD)

#define MAX_COMPLEX_64_SIZE 4
#define MAX_COMPLEX_64_TYPE Array4Complex64
#define MAX_COMPLEX_64_MASK_TYPE VectorU8_4D
#define MAX_COMPLEX_64_VECTOR_TYPE VectorDouble4D
#define IDEAL_COMPLEX_64_SIZE 2
#define IDEAL_COMPLEX_64_TYPE Array2Complex64
#define IDEAL_COMPLEX_64_MASK_TYPE VectorU8_2D
#define IDEAL_COMPLEX_64_VECTOR_TYPE VectorDouble2D
#define MIN_COMPLEX_64_SIZE 1
#define MIN_COMPLEX_64_TYPE Complex64
#define MIN_COMPLEX_64_MASK_TYPE VectorU8_1D
#define MIN_COMPLEX_64_VECTOR_TYPE VectorDouble1D


#define MAX_COMPLEX_32_SIZE 8
#define MAX_COMPLEX_32_TYPE Array8Complex32
#define MAX_COMPLEX_32_MASK_TYPE VectorU8_8D
#define MAX_COMPLEX_32_VECTOR_TYPE VectorFloat8D
#define IDEAL_COMPLEX_32_SIZE 4
#define IDEAL_COMPLEX_32_TYPE Array4Complex32
#define IDEAL_COMPLEX_32_MASK_TYPE VectorU8_4D
#define IDEAL_COMPLEX_32_VECTOR_TYPE VectorFloat4D
#define MIN_COMPLEX_32_SIZE 2
#define MIN_COMPLEX_32_TYPE Array2Complex32
#define MIN_COMPLEX_32_MASK_TYPE VectorU8_2D
#define MIN_COMPLEX_32_VECTOR_TYPE VectorFloat2D

#elif defined(AML_USE_ARRAY_STRICT)

#define MAX_COMPLEX_64_SIZE 2
#define MAX_COMPLEX_64_TYPE Array2Complex64
#define MAX_COMPLEX_64_MASK_TYPE VectorU8_2D
#define MAX_COMPLEX_64_VECTOR_TYPE VectorDouble2D
#define IDEAL_COMPLEX_64_SIZE 2
#define IDEAL_COMPLEX_64_TYPE Array2Complex64
#define IDEAL_COMPLEX_64_MASK_TYPE VectorU8_2D
#define IDEAL_COMPLEX_64_VECTOR_TYPE VectorDouble2D
#define MIN_COMPLEX_64_SIZE 2
#define MIN_COMPLEX_64_TYPE Array2Complex64
#define MIN_COMPLEX_64_MASK_TYPE VectorU8_2D
#define MIN_COMPLEX_64_VECTOR_TYPE VectorDouble2D


#define MAX_COMPLEX_32_SIZE 2
#define MAX_COMPLEX_32_TYPE Array2Complex32
#define MAX_COMPLEX_32_MASK_TYPE VectorU8_2D
#define MAX_COMPLEX_32_VECTOR_TYPE VectorFloat2D
#define IDEAL_COMPLEX_32_SIZE 2
#define IDEAL_COMPLEX_32_TYPE Array2Complex32
#define IDEAL_COMPLEX_32_MASK_TYPE VectorU8_2D
#define IDEAL_COMPLEX_32_VECTOR_TYPE VectorFloat2D
#define MIN_COMPLEX_32_SIZE 2
#define MIN_COMPLEX_32_TYPE Array2Complex32
#define MIN_COMPLEX_32_MASK_TYPE VectorU8_2D
#define MIN_COMPLEX_32_VECTOR_TYPE VectorFloat2D


#else


#define MAX_COMPLEX_64_SIZE 1
#define MAX_COMPLEX_64_TYPE Complex64
#define MAX_COMPLEX_64_MASK_TYPE VectorU8_1D
#define MAX_COMPLEX_64_VECTOR_TYPE VectorDouble1D
#define IDEAL_COMPLEX_64_SIZE 1
#define IDEAL_COMPLEX_64_TYPE Complex64
#define IDEAL_COMPLEX_64_MASK_TYPE VectorU8_1D
#define IDEAL_COMPLEX_64_VECTOR_TYPE VectorDouble1D
#define MIN_COMPLEX_64_SIZE 1
#define MIN_COMPLEX_64_TYPE Complex64
#define MIN_COMPLEX_64_MASK_TYPE VectorU8_1D
#define MIN_COMPLEX_64_VECTOR_TYPE VectorDouble1D

#define MAX_COMPLEX_32_SIZE 1
#define MAX_COMPLEX_32_TYPE Complex32
#define MAX_COMPLEX_32_MASK_TYPE VectorU8_1D
#define MAX_COMPLEX_32_VECTOR_TYPE VectorFloat1D
#define IDEAL_COMPLEX_32_SIZE 1
#define IDEAL_COMPLEX_32_TYPE Complex32
#define IDEAL_COMPLEX_32_MASK_TYPE VectorU8_1D
#define IDEAL_COMPLEX_32_VECTOR_TYPE VectorFloat1D
#define MIN_COMPLEX_32_SIZE 1
#define MIN_COMPLEX_32_TYPE Complex32
#define MIN_COMPLEX_32_MASK_TYPE VectorU8_1D
#define MIN_COMPLEX_32_VECTOR_TYPE VectorFloat1D


#endif

class AML_PREFIX(Complex64) {
public:
	AML_PREFIX(doublevec2) c{};

	AML_FUNCTION constexpr AML_PREFIX(Complex64)(const double real, const double img = 0.0) {
		c.c[0] = real;
		c.c[1] = img;
	}

	AML_FUNCTION explicit AML_PREFIX(Complex64)(double *values) {
		c.c[0] = values[0];
		c.c[1] = values[1];
	}

#if defined(AML_USE_STD_COMPLEX)

	AML_FUNCTION AML_PREFIX(Complex64)(std::complex<double> sc) {
		c.c[0] = sc.real();
		c.c[1] = sc.imag();
	}

#endif

	AML_FUNCTION AML_PREFIX(Complex64)() = default;

	AML_FUNCTION void set([[maybe_unused]]uint64_t location, AML_PREFIX(Complex64) value) {
		c.c[0] = value.c.c[0];
		c.c[1] = value.c.c[1];
	}

//add sub
	AML_FUNCTION AML_PREFIX(Complex64) *add(const AML_PREFIX(Complex64) a) {
		c.c[0] += a.c.c[0];
		c.c[1] += a.c.c[1];
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *add(const AML_PREFIX(Complex64) a, AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			c.c[1] += a.c.c[1];
			c.c[0] += a.c.c[0];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) operator+(const AML_PREFIX(Complex64) a) const {
		AML_PREFIX(Complex64) ret(c.c[0] + a.c.c[0], c.c[1] + a.c.c[1]);
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Complex64) operator-(const AML_PREFIX(Complex64) a) const {
		AML_PREFIX(Complex64) ret(c.c[0] - a.c.c[0], c.c[1] - a.c.c[1]);
		return ret;
	}


	AML_FUNCTION void operator+=(const AML_PREFIX(Complex64) a) {
		c.c[0] += a.c.c[0];
		c.c[1] += a.c.c[1];
	}

	AML_FUNCTION void operator-=(const AML_PREFIX(Complex64) a) {
		c.c[0] -= a.c.c[0];
		c.c[1] -= a.c.c[1];
	}

	AML_FUNCTION AML_PREFIX(Complex64) *subtract(const AML_PREFIX(Complex64) a) {
		c.c[0] -= a.c.c[0];
		c.c[1] -= a.c.c[1];
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *subtract(const AML_PREFIX(Complex64) a, AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			c.c[0] -= a.c.c[0];
			c.c[1] -= a.c.c[1];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *conjugate() {
		c.c[1] = -c.c[1];
		return this;
	}

//mul
	AML_FUNCTION AML_PREFIX(Complex64) *multiply(const AML_PREFIX(Complex64) a) {
		double d1 = c.c[0] * a.c.c[0] - c.c[1] * a.c.c[1];
		double d2 = c.c[0] * a.c.c[1] + c.c[1] * a.c.c[0];
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}


	AML_FUNCTION AML_PREFIX(Complex64) *multiply(const AML_PREFIX(Complex64) &a, const AML_PREFIX(VectorU8_1D) &mask) {
		double d1;
		double d2;
		if (mask.v.c) LIKELY {
			d1 = c.c[0] * a.c.c[0] - c.c[1] * a.c.c[1];
			d2 = c.c[0] * a.c.c[1] + c.c[1] * a.c.c[0];
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;


	}

	AML_FUNCTION AML_PREFIX(Complex64) operator*(const AML_PREFIX(Complex64) a) const {
		AML_PREFIX(Complex64) ret(c.c[0] * a.c.c[0] - c.c[1] * a.c.c[1], c.c[0] * a.c.c[1] + c.c[1] * a.c.c[0]);
		return ret;
	}

	AML_FUNCTION void operator*=(const AML_PREFIX(Complex64) a) {
		double d1 = c.c[0] * a.c.c[0] - c.c[1] * a.c.c[1];
		double d2 = c.c[0] * a.c.c[1] + c.c[1] * a.c.c[0];
		c.c[0] = d1;
		c.c[1] = d2;
	}

	AML_FUNCTION void operator*=(double a) {
		c.c[0] = c.c[0] * a;
		c.c[1] = c.c[1] * a;
	}


	AML_FUNCTION AML_PREFIX(Complex64) *square() {
		double d1 = c.c[0] * c.c[0] - c.c[1] * c.c[1];
		double d2 = c.c[0] * c.c[1] + c.c[1] * c.c[0];
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *square(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			double d1 = c.c[0] * c.c[0] - c.c[1] * c.c[1];
			double d2 = c.c[0] * c.c[1] + c.c[1] * c.c[0];
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

//division
	AML_FUNCTION AML_PREFIX(Complex64) operator/(const AML_PREFIX(Complex64) a) const {
		AML_PREFIX(Complex64) ret;
		ret.c.c[0] = (c.c[0] * a.c.c[0] + c.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.c.c[1] = (c.c[1] * a.c.c[0] - c.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		return ret;
	}

	AML_FUNCTION void operator/=(const AML_PREFIX(Complex64) a) {
		double d1 = (c.c[0] * a.c.c[0] + c.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		double d2 = (c.c[1] * a.c.c[0] - c.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		c.c[0] = d1;
		c.c[1] = d2;
	}

	AML_FUNCTION void operator/=(double a) {
		double d1 = c.c[0] / a;
		double d2 = c.c[1] / a;
		c.c[0] = d1;
		c.c[1] = d2;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *divide(const AML_PREFIX(Complex64) a) {
		double d1 = (c.c[0] * a.c.c[0] + c.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		double d2 = (c.c[1] * a.c.c[0] - c.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) operator/(double a) {
		AML_PREFIX(Complex64) ret;
		ret.c.c[0] = c.c[0] / a;
		ret.c.c[1] = c.c[1] / a;
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *divide(double a) {
		double d1 = c.c[0] / a;
		double d2 = c.c[1] / a;
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *divide(const double a, const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			double d1 = c.c[0] / a;
			double d2 = c.c[1] / a;

			c.c[0] = d1;
			c.c[1] = d2;

		}

		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *divide(const AML_PREFIX(Complex64) a, const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			double d1 = (c.c[0] * a.c.c[0] + c.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			double d2 = (c.c[1] * a.c.c[0] - c.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);

			c.c[0] = d1;
			c.c[1] = d2;

		}

		return this;
	}

//sqrt
	AML_FUNCTION AML_PREFIX(Complex64) *sqrt() {
		double d2 = ::sqrt((-c.c[0] + ::sqrt(c.c[0] * c.c[0] + c.c[1] * c.c[1])) / (2));
		double d1;
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(c.c[0]);
		} else LIKELY {
			d1 = c.c[1] / (2 * d2);
		}
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *sqrt(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			double d2 = ::sqrt((-c.c[0] + ::sqrt(c.c[0] * c.c[0] + c.c[1] * c.c[1])) / (2));
			double d1;
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(c.c[0]);
			} else LIKELY {
				d1 = c.c[1] / (2 * d2);
			}
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *sin() {
		double d1 = ::sin(c.c[0]) * ::cosh(c.c[1]);
		double d2 = ::cos(c.c[1]) * ::sinh(c.c[0]);

		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *cos() {
		double d1 = ::cos(c.c[0]) * ::cosh(c.c[1]);
		double d2 = -::sin(c.c[1]) * ::sinh(c.c[0]);

		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *tan() {
		double d1 = ::sin(c.c[0] + c.c[0]) / (::cos(c.c[0] + c.c[0]) * ::cosh(c.c[1] + c.c[1]));
		double d2 = ::sinh(c.c[1] + c.c[1]) / (::cos(c.c[0] + c.c[0]) * ::cosh(c.c[1] + c.c[1]));

		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *sin(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) {
			double d1 = ::sin(c.c[0]) * ::cosh(c.c[1]);
			double d2 = ::cos(c.c[1]) * ::sinh(c.c[0]);
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *cos(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) {
			double d1 = ::cos(c.c[0]) * ::cosh(c.c[1]);
			double d2 = -::sin(c.c[1]) * ::sinh(c.c[0]);
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *tan(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) {
			double d1 = ::sin(c.c[0] + c.c[0]) / (::cos(c.c[0] + c.c[0]) * ::cosh(c.c[1] + c.c[1]));
			double d2 = ::sinh(c.c[1] + c.c[1]) / (::cos(c.c[0] + c.c[0]) * ::cosh(c.c[1] + c.c[1]));
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Complex64) *exp() {
		double d1 = ::exp(c.c[0]) * ::cos(c.c[1]);
		double d2 = ::exp(c.c[0]) * ::sin(c.c[1]);


		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *exp(double n) {
		double d1 = ::pow(n, c.c[0]) * ::cos(c.c[1] * ::log(n));
		double d2 = ::pow(n, c.c[0]) * ::sin(c.c[1] * ::log(n));
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *pow(double n) {
		double d1 = ::pow(c.c[0] * c.c[0] + c.c[1] * c.c[1], n / 2) * ::cos(n * atan2(c.c[1], c.c[0]));
		double d2 = ::pow(c.c[0] * c.c[0] + c.c[1] * c.c[1], n / 2) * ::sin(n * atan2(c.c[1], c.c[0]));
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *pow(const AML_PREFIX(Complex64) n) {
		double d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
		double d2 = ::atan2(c.c[1], c.c[0]);
		double d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		double d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		double d5 = d3 * ::cos(d4);
		double d6 = d3 * ::sin(d4);
		c.c[0] = d5;
		c.c[1] = d6;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *pow(double n, const AML_PREFIX(VectorU8_1D) &mask) {
		if (mask.v.c) LIKELY {
			double d1 = ::pow(c.c[0] * c.c[0] + c.c[1] * c.c[1], n / 2) * ::cos(n * atan2(c.c[1], c.c[0]));
			double d2 = ::pow(c.c[0] * c.c[0] + c.c[1] * c.c[1], n / 2) * ::sin(n * atan2(c.c[1], c.c[0]));
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *pow(const AML_PREFIX(Complex64) n, const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			double d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
			double d2 = ::atan2(c.c[1], c.c[0]);
			double d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			double d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			double d5 = d3 * ::cos(d4);
			double d6 = d3 * ::sin(d4);
			c.c[0] = d5;
			c.c[1] = d6;
		}
		return this;
	}

	AML_FUNCTION double abs() {
		return ::sqrt(c.c[0] * c.c[0] + c.c[1] * c.c[1]);
	}

	AML_FUNCTION bool abs_gt(double a) {
		return a * a < c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}

	AML_FUNCTION bool abs_lt(double a) {
		return a * a > c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}

	AML_FUNCTION bool abs_eq(double a) {
		return a * a == c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}

	AML_FUNCTION bool abs_gt(const AML_PREFIX(Complex64) a) {
		return a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1] < c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}

	AML_FUNCTION bool abs_lt(const AML_PREFIX(Complex64) a) {
		return a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1] > c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}

	AML_FUNCTION bool abs_eq(const AML_PREFIX(Complex64) a) {
		return a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1] == c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}


	AML_FUNCTION AML_PREFIX(Complex64) *ln() {
		double d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
		double d2 = ::atan2(c.c[1], c.c[0]);
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *log() {
		double d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
		double d2 = ::atan2(c.c[1], c.c[0]);
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *log10() {
		double d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / (2 * AML_LN10);
		double d2 = ::atan2(c.c[1], c.c[0]) / AML_LN10;
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *ln(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			double d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
			double d2 = ::atan2(c.c[1], c.c[0]);
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *log(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			double d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
			double d2 = ::atan2(c.c[1], c.c[0]);
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) *log10(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			double d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / (2 * AML_LN10);
			double d2 = ::atan2(c.c[1], c.c[0]) / AML_LN10;
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}


	AML_FUNCTION double imaginary() {
		return c.c[1];
	}

	AML_FUNCTION double real() {
		return c.c[0];
	}

	AML_FUNCTION double angle() {
		return ::atan2(c.c[1], c.c[0]);
	}

	AML_FUNCTION double length() {
		return ::sqrt(c.c[0] * c.c[0] + c.c[1] * c.c[1]);
	}

	AML_FUNCTION AML_PREFIX(Complex64) *polar(double length, double angle) {
		c.c[0] = length * ::cos(angle);
		c.c[1] = length * ::sin(angle);
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex64) operator[]([[maybe_unused]]uint64_t location) {
		return *this;
	}


};


class AML_PREFIX(Complex64Ptr) : public AML_PREFIX(Complex64) {
	double *r;
	double *i;
	uint32_t index = 0;

	AML_FUNCTION void update() {
		*r = c.c[0];
		*i = c.c[1];
	}

public:
	AML_FUNCTION AML_PREFIX(Complex64Ptr)(double *real, double *imag) : AML_PREFIX(Complex64)(*real, *imag) {
		r = real;
		i = imag;
	}

	AML_FUNCTION AML_PREFIX(Complex64Ptr)(double *real, double *imag, uint32_t position) : AML_PREFIX(Complex64)(*real,
																												 *imag) {
		r = real;
		i = imag;
		index = position;
	}

	AML_FUNCTION AML_PREFIX(Complex64) &operator*() {
		return *this;
	}

	AML_FUNCTION ~AML_PREFIX(Complex64Ptr)() {
		*r = c.c[0];
		*i = c.c[1];
	}

	AML_FUNCTION void operator()() {
		*r = c.c[0];
		*i = c.c[1];
	}

	AML_FUNCTION uint32_t getIndex() {
		return index;
	}

	AML_FUNCTION void operator=(const AML_PREFIX(Complex64) newVal) {
		c.c[0] = newVal.c.c[0];
		c.c[1] = newVal.c.c[1];
		update();
	}

};

#if !defined(AML_NO_STRING)

AML_FUNCTION std::string operator<<(std::string &lhs, const AML_PREFIX(Complex64) &rhs) {
	std::ostringstream string;
	string << lhs << rhs.c.c[0] << " + " << rhs.c.c[1] << "i";
	return string.str();
}

AML_FUNCTION std::string operator<<(const char *lhs, const AML_PREFIX(Complex64) &rhs) {
	std::ostringstream string;
	string << lhs << rhs.c.c[0] << " + " << rhs.c.c[1] << "i";
	return string.str();
}

template<class charT, class traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &o, const AML_PREFIX(Complex64) &x) {
	std::basic_ostringstream<charT, traits> s;
	s.flags(o.flags());
	s.imbue(o.getloc());
	s.precision(o.precision());
	s << x.c.c[0] << " + " << x.c.c[1] << "i";
	return o << s.str();
}

#endif

AML_FUNCTION AML_PREFIX(Complex64) operator+(const double &lhs, const AML_PREFIX(Complex64) &rhs) {
	AML_PREFIX(Complex64) ret(lhs + rhs.c.c[0], 0.0 + rhs.c.c[1]);
	return ret;
}

AML_FUNCTION AML_PREFIX(Complex64) operator-(const double &lhs, const AML_PREFIX(Complex64) &rhs) {
	AML_PREFIX(Complex64) ret(lhs - rhs.c.c[0], 0.0 - rhs.c.c[1]);
	return ret;
}

AML_FUNCTION AML_PREFIX(Complex64) operator*(const double &lhs, const AML_PREFIX(Complex64) &rhs) {
	AML_PREFIX(Complex64) ret(lhs * rhs.c.c[0], lhs * rhs.c.c[1]);
	return ret;
}

AML_FUNCTION AML_PREFIX(Complex64) operator/(const double &lhs, const AML_PREFIX(Complex64) &rhs) {
	AML_PREFIX(Complex64) ret;
	ret.c.c[0] = (lhs * rhs.c.c[0]) / (rhs.c.c[0] * rhs.c.c[0] + rhs.c.c[1] * rhs.c.c[1]);
	ret.c.c[1] = (-lhs * rhs.c.c[1]) / (rhs.c.c[0] * rhs.c.c[0] + rhs.c.c[1] * rhs.c.c[1]);
	return ret;
}

#if defined(AML_USE_STD_COMPLEX)

AML_FUNCTION AML_PREFIX(Complex64) operator+(const std::complex<double> &lhs, const AML_PREFIX(Complex64) &rhs) {
	AML_PREFIX(Complex64) ret = lhs;
	return ret + rhs;
}

AML_FUNCTION AML_PREFIX(Complex64) operator-(const std::complex<double> &lhs, const AML_PREFIX(Complex64) &rhs) {
	AML_PREFIX(Complex64) ret = lhs;
	return ret - rhs;
}

AML_FUNCTION AML_PREFIX(Complex64) operator*(const std::complex<double> &lhs, const AML_PREFIX(Complex64) &rhs) {
	AML_PREFIX(Complex64) ret = lhs;
	return ret * rhs;
}

AML_FUNCTION AML_PREFIX(Complex64) operator/(const std::complex<double> &lhs, const AML_PREFIX(Complex64) &rhs) {
	AML_PREFIX(Complex64) ret = lhs;
	return ret / rhs;
}

class AML_PREFIX(STD_COMPLEX64_CAST) : public std::complex<double> {
public:
	AML_FUNCTION AML_PREFIX(STD_COMPLEX64_CAST)(const AML_PREFIX(Complex64) &other) : std::complex<double>(other.c.c[0],
																										   other.c.c[1]) {}
};

#endif

#if !defined(USE_CUDA)

constexpr Complex64 operator ""_i(long double d) {
	return Complex64(0.0f, (double) d);
}

constexpr Complex64 operator ""_i(unsigned long long d) {
	return Complex64(0.0f, (double) d);
}

#endif

#if defined(AML_USE_STD_COMPLEX)

AML_FUNCTION std::complex<double> toStdComplex(const AML_PREFIX(Complex64) a) {
	std::complex<double> ret(a.c.c[0], a.c.c[1]);
	return ret;
}

#endif

class AML_PREFIX(Array2Complex64) {
public:
	AML_PREFIX(doublevec2) r{};
	AML_PREFIX(doublevec2) i{};

	AML_FUNCTION AML_PREFIX(Array2Complex64)() {}

	AML_FUNCTION AML_PREFIX(Array2Complex64)(const AML_PREFIX(Complex64) value) {
		r.c[0] = value.c.c[0];
		i.c[0] = value.c.c[1];
		r.c[1] = value.c.c[0];
		i.c[1] = value.c.c[1];
	}


	AML_FUNCTION AML_PREFIX(VectorDouble2D) real() {
		return AML_PREFIX(VectorDouble2D)(r.c);
	}

	AML_FUNCTION AML_PREFIX(VectorDouble2D) complex() {
		return AML_PREFIX(VectorDouble2D)(i.c);
	}

	AML_FUNCTION AML_PREFIX(Complex64) operator[](uint64_t location) {
		return AML_PREFIX(Complex64)(r.c[location], i.c[location]);
	}

	AML_FUNCTION void set(uint64_t location, AML_PREFIX(Complex64) value) {
		r.c[location] = value.c.c[0];
		i.c[location] = value.c.c[1];
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *add(const AML_PREFIX(Array2Complex64) a) {
		i.c[0] += a.i.c[0];
		i.c[1] += a.i.c[1];
		r.c[0] += a.r.c[0];
		r.c[1] += a.r.c[1];
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *add(const AML_PREFIX(Complex64) a) {
		i.c[0] += a.c.c[1];
		i.c[1] += a.c.c[1];
		r.c[0] += a.c.c[0];
		r.c[1] += a.c.c[0];
		return this;
	}

	AML_FUNCTION void operator+=(const AML_PREFIX(Array2Complex64) a) {
		i.c[0] += a.i.c[0];
		i.c[1] += a.i.c[1];
		r.c[0] += a.r.c[0];
		r.c[1] += a.r.c[1];
	}

	AML_FUNCTION void operator+=(const AML_PREFIX(Complex64) a) {
		i.c[0] += a.c.c[1];
		i.c[1] += a.c.c[1];
		r.c[0] += a.c.c[0];
		r.c[1] += a.c.c[0];
	}


	AML_FUNCTION AML_PREFIX(Array2Complex64) operator+(const AML_PREFIX(Array2Complex64) a) const {
		AML_PREFIX(Array2Complex64) ret;
		ret.i.c[0] = i.c[0] + a.i.c[0];
		ret.i.c[1] = i.c[1] + a.i.c[1];
		ret.r.c[0] = r.c[0] + a.r.c[0];
		ret.r.c[1] = r.c[1] + a.r.c[1];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) operator+(const AML_PREFIX(Complex64) a) const {
		AML_PREFIX(Array2Complex64) ret;
		ret.i.c[0] = i.c[0] + a.c.c[1];
		ret.i.c[1] = i.c[1] + a.c.c[1];
		ret.r.c[0] = r.c[0] + a.c.c[0];
		ret.r.c[1] = r.c[1] + a.c.c[0];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *add(const AML_PREFIX(Array2Complex64) a, AML_PREFIX(VectorU8_2D) mask) {
		if (mask.v.c[0]) {
			i.c[0] += a.i.c[0];
			r.c[0] += a.r.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] += a.i.c[1];
			r.c[1] += a.r.c[1];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *add(const AML_PREFIX(Complex64) a, AML_PREFIX(VectorU8_2D) mask) {
		if (mask.v.c[0]) {
			i.c[0] += a.c.c[1];
			r.c[0] += a.c.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] += a.c.c[1];
			r.c[1] += a.c.c[0];
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array2Complex64) *subtract(const AML_PREFIX(Array2Complex64) a) {
		i.c[0] -= a.i.c[0];
		i.c[1] -= a.i.c[1];
		r.c[0] -= a.r.c[0];
		r.c[1] -= a.r.c[1];
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *subtract(const AML_PREFIX(Complex64) a) {
		i.c[0] -= a.c.c[1];
		i.c[1] -= a.c.c[1];
		r.c[0] -= a.c.c[0];
		r.c[1] -= a.c.c[0];
		return this;
	}

	AML_FUNCTION void operator-=(const AML_PREFIX(Array2Complex64) a) {
		i.c[0] -= a.i.c[0];
		i.c[1] -= a.i.c[1];
		r.c[0] -= a.r.c[0];
		r.c[1] -= a.r.c[1];
	}

	AML_FUNCTION void operator-=(const AML_PREFIX(Complex64) a) {
		i.c[0] -= a.c.c[1];
		i.c[1] -= a.c.c[1];
		r.c[0] -= a.c.c[0];
		r.c[1] -= a.c.c[0];
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) operator-(const AML_PREFIX(Array2Complex64) a) const {
		AML_PREFIX(Array2Complex64) ret;
		ret.i.c[0] = i.c[0] - a.i.c[0];
		ret.i.c[1] = i.c[1] - a.i.c[1];
		ret.r.c[0] = r.c[0] - a.r.c[0];
		ret.r.c[1] = r.c[1] - a.r.c[1];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) operator-(const AML_PREFIX(Complex64) a) const {
		AML_PREFIX(Array2Complex64) ret;
		ret.i.c[0] = i.c[0] - a.c.c[1];
		ret.i.c[1] = i.c[1] - a.c.c[1];
		ret.r.c[0] = r.c[0] - a.c.c[0];
		ret.r.c[1] = r.c[1] - a.c.c[0];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *
	subtract(const AML_PREFIX(Array2Complex64) a, AML_PREFIX(VectorU8_2D) mask) {
		if (mask.v.c[0]) {
			i.c[0] -= a.i.c[0];
			r.c[0] -= a.r.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] -= a.i.c[1];
			r.c[1] -= a.r.c[1];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *subtract(const AML_PREFIX(Complex64) a, AML_PREFIX(VectorU8_2D) mask) {
		if (mask.v.c[0]) {
			i.c[0] -= a.c.c[1];
			r.c[0] -= a.c.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] -= a.c.c[1];
			r.c[1] -= a.c.c[0];
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array2Complex64) operator*(const AML_PREFIX(Array2Complex64) &a) const {
		AML_PREFIX(Array2Complex64) ret;
		ret.r.c[0] = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		ret.i.c[0] = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		ret.r.c[1] = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		ret.i.c[1] = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];


		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *multiply(const AML_PREFIX(Array2Complex64) &a) {
		double d1;
		double d2;
		d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;

		d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) operator*(const AML_PREFIX(Complex64) &a) const {
		AML_PREFIX(Array2Complex64) ret;
		ret.r.c[0] = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		ret.i.c[0] = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		ret.r.c[1] = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		ret.i.c[1] = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];

		return ret;
	}

	AML_FUNCTION void operator*=(const AML_PREFIX(Array2Complex64) &a) {
		double d1;
		double d2;
		d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;

		d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
	}

	AML_FUNCTION void operator*=(const AML_PREFIX(Complex64) &a) {
		double d1;
		double d2;
		d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
		r.c[1] = d1;
		i.c[1] = d2;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *multiply(const AML_PREFIX(Complex64) &a) {
		double d1;
		double d2;
		d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *
	multiply(const AML_PREFIX(Complex64) &a, const AML_PREFIX(VectorU8_2D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
			d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
			d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;


	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *
	multiply(const AML_PREFIX(Array2Complex64) &a, const AML_PREFIX(VectorU8_2D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
			d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
			d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *square() {
		double d1;
		double d2;
		d1 = r.c[0] * r.c[0] - i.c[0] * i.c[0];
		d2 = r.c[0] * i.c[0] + i.c[0] * r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * r.c[1] - i.c[1] * i.c[1];
		d2 = r.c[1] * i.c[1] + i.c[1] * r.c[1];
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *square(const AML_PREFIX(VectorU8_2D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * r.c[0] - i.c[0] * i.c[0];
			d2 = r.c[0] * i.c[0] + i.c[0] * r.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * r.c[1] - i.c[1] * i.c[1];
			d2 = r.c[1] * i.c[1] + i.c[1] * r.c[1];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *divide(const AML_PREFIX(Complex64) a) {
		double d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		double d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *
	divide(const AML_PREFIX(Complex64) a, const AML_PREFIX(VectorU8_2D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *divide(const AML_PREFIX(Array2Complex64) &a) {
		double d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		double d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *
	divide(const AML_PREFIX(Array2Complex64) &a, const AML_PREFIX(VectorU8_2D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = (r.c[0] * a.i.c[0] + i.c[0] * a.i.c[1]) / (a.i.c[0] * a.i.c[0] + a.i.c[1] * a.i.c[1]);
			d2 = (i.c[0] * a.i.c[0] - r.c[0] * a.i.c[1]) / (a.i.c[0] * a.i.c[0] + a.i.c[1] * a.i.c[1]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
			d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) operator/(const AML_PREFIX(Complex64) &a) const {
		AML_PREFIX(Array2Complex64) ret;
		double d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		double d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[0] = d1;
		ret.i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[1] = d1;
		ret.i.c[1] = d2;
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) operator/(const AML_PREFIX(Array2Complex64) &a) const {
		AML_PREFIX(Array2Complex64) ret;
		double d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		double d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		ret.r.c[0] = d1;
		ret.i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		ret.r.c[1] = d1;
		ret.i.c[1] = d2;
		return ret;
	}

	AML_FUNCTION void operator/=(const AML_PREFIX(Complex64) &a) {
		double d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		double d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
	}

	AML_FUNCTION void operator/=(const AML_PREFIX(Array2Complex64) &a) {
		double d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		double d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *sqrt() {
		double d1;
		double d2;
		d2 = ::sqrt((-r.c[0] + ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[0]);
		} else LIKELY {
			d1 = i.c[0] / (2 * d2);
		}
		r.c[0] = d1;
		i.c[0] = d2;
		d2 = ::sqrt((-r.c[1] + ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[1]);
		} else LIKELY {
			d1 = i.c[1] / (2 * d2);
		}
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *sqrt(const AML_PREFIX(VectorU8_2D) mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d2 = ::sqrt((-r.c[0] + ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[0]);
			} else LIKELY {
				d1 = i.c[0] / (2 * d2);
			}
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d2 = ::sqrt((-r.c[1] + ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[1]);
			} else LIKELY {
				d1 = i.c[1] / (2 * d2);
			}
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *sin() {
		double d1;
		double d2;
		d1 = ::sin(r.c[0]) * ::cosh(i.c[0]);
		d2 = ::cos(i.c[0]) * ::sinh(r.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::sin(r.c[1]) * ::cosh(i.c[1]);
		d2 = ::cos(i.c[1]) * ::sinh(r.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *cos() {
		double d1;
		double d2;
		d1 = ::cos(r.c[0]) * ::cosh(i.c[0]);
		d2 = -::sin(i.c[0]) * ::sinh(r.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::cos(r.c[1]) * ::cosh(i.c[1]);
		d2 = -::sin(i.c[1]) * ::sinh(r.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *tan() {
		double d1;
		double d2;
		d1 = ::sin(r.c[0] + r.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
		d2 = ::sinh(i.c[0] + i.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::sin(r.c[1] + r.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
		d2 = ::sinh(i.c[1] + i.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *sin(const AML_PREFIX(VectorU8_2D) mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::sin(r.c[0]) * ::cosh(i.c[0]);
			d2 = ::cos(i.c[0]) * ::sinh(r.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::sin(r.c[1]) * ::cosh(i.c[1]);
			d2 = ::cos(i.c[1]) * ::sinh(r.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *cos(const AML_PREFIX(VectorU8_2D) mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::cos(r.c[0]) * ::cosh(i.c[0]);
			d2 = -::sin(i.c[0]) * ::sinh(r.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::cos(r.c[1]) * ::cosh(i.c[1]);
			d2 = -::sin(i.c[1]) * ::sinh(r.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *tan(const AML_PREFIX(VectorU8_2D) mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::sin(r.c[0] + r.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
			d2 = ::sinh(i.c[0] + i.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::sin(r.c[1] + r.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
			d2 = ::sinh(i.c[1] + i.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *exp() {
		double d1 = ::exp(r.c[0]) * ::cos(i.c[0]);
		double d2 = ::exp(r.c[0]) * ::sin(i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::exp(r.c[1]) * ::cos(i.c[1]);
		d2 = ::exp(r.c[1]) * ::sin(i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *exp(double n) {
		double d1 = ::pow(n, r.c[0]) * ::cos(i.c[0] * ::log(n));
		double d2 = ::pow(n, r.c[0]) * ::sin(i.c[0] * ::log(n));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::pow(n, r.c[1]) * ::cos(i.c[1] * ::log(n));
		d2 = ::pow(n, r.c[1]) * ::sin(i.c[1] * ::log(n));
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *exp(const AML_PREFIX(VectorU8_2D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::exp(r.c[0]) * ::cos(i.c[0]);
			d2 = ::exp(r.c[0]) * ::sin(i.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::exp(r.c[1]) * ::cos(i.c[1]);
			d2 = ::exp(r.c[1]) * ::sin(i.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *exp(double n, const AML_PREFIX(VectorU8_2D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::pow(n, r.c[0]) * ::cos(i.c[0] * ::log(n));
			d2 = ::pow(n, r.c[0]) * ::sin(i.c[0] * ::log(n));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::pow(n, r.c[1]) * ::cos(i.c[1] * ::log(n));
			d2 = ::pow(n, r.c[1]) * ::sin(i.c[1] * ::log(n));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *pow(const AML_PREFIX(Array2Complex64) n) {
		double d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		double d2 = ::atan2(r.c[0], i.c[0]);
		double d3 = ::exp(d1 * n.i.c[0] - d2 * n.r.c[0]);
		double d4 = d1 * n.r.c[0] + d2 * n.i.c[0];
		double d5 = d3 * ::cos(d4);
		double d6 = d3 * ::sin(d4);
		i.c[0] = d5;
		r.c[0] = d6;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		d3 = ::exp(d1 * n.i.c[1] - d2 * n.r.c[1]);
		d4 = d1 * n.r.c[1] + d2 * n.i.c[1];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[1] = d5;
		r.c[1] = d6;
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array2Complex64) *pow(const AML_PREFIX(Complex64) n) {
		double d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		double d2 = ::atan2(r.c[0], i.c[0]);
		double d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		double d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		double d5 = d3 * ::cos(d4);
		double d6 = d3 * ::sin(d4);
		i.c[0] = d5;
		r.c[0] = d6;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[1] = d5;
		r.c[1] = d6;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *
	pow(const AML_PREFIX(Array2Complex64) n, const AML_PREFIX(VectorU8_2D) &mask) {
		double d1;
		double d2;
		double d3;
		double d4;
		double d5;
		double d6;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			d3 = ::exp(d1 * n.i.c[0] - d2 * n.r.c[0]);
			d4 = d1 * n.r.c[0] + d2 * n.i.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[0] = d5;
			r.c[0] = d6;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			d3 = ::exp(d1 * n.i.c[1] - d2 * n.r.c[1]);
			d4 = d1 * n.r.c[1] + d2 * n.i.c[1];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[1] = d5;
			r.c[1] = d6;
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array2Complex64) *pow(const AML_PREFIX(Complex64) n, const AML_PREFIX(VectorU8_2D) &mask) {
		double d1;
		double d2;
		double d3;
		double d4;
		double d5;
		double d6;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[0] = d5;
			r.c[0] = d6;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[1] = d5;
			r.c[1] = d6;
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array2Complex64) *pow(double n) {
		double d1;
		double d2;
		d1 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::cos(n * atan2(i.c[0], r.c[0]));
		d2 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::sin(n * atan2(i.c[0], r.c[0]));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::cos(n * atan2(i.c[1], r.c[1]));
		d2 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::sin(n * atan2(i.c[1], r.c[1]));
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *pow(double n, const AML_PREFIX(VectorU8_2D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::cos(n * atan2(i.c[0], r.c[0]));
			d2 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::sin(n * atan2(i.c[0], r.c[0]));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::cos(n * atan2(i.c[1], r.c[1]));
			d2 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::sin(n * atan2(i.c[1], r.c[1]));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble2D) abs() {
		AML_PREFIX(VectorDouble2D) ret;
		ret.v.c[0] = ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0]);
		ret.v.c[1] = ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1]);
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_2D) abs_gt(double a) {
		AML_PREFIX(VectorU8_2D) ret;
		ret.v.c[0] = a * a < r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a < r.c[1] * r.c[1] + i.c[1] * i.c[1];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_2D) abs_lt(double a) {
		AML_PREFIX(VectorU8_2D) ret;
		ret.v.c[0] = a * a > r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a > r.c[1] * r.c[1] + i.c[1] * i.c[1];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_2D) abs_eq(double a) {
		AML_PREFIX(VectorU8_2D) ret;
		ret.v.c[0] = a * a == r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a == r.c[1] * r.c[1] + i.c[1] * i.c[1];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *ln(const AML_PREFIX(VectorU8_2D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			i.c[1] = d1;
			r.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *log(const AML_PREFIX(VectorU8_2D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			i.c[1] = d1;
			r.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *log10(const AML_PREFIX(VectorU8_2D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[0], i.c[0]) / AML_LN10;
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[1], i.c[1]) / AML_LN10;
			i.c[1] = d1;
			r.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *ln() {
		double d1;
		double d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		d2 = ::atan2(r.c[0], i.c[0]);
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		i.c[1] = d1;
		r.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *log() {
		double d1;
		double d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		d2 = ::atan2(r.c[0], i.c[0]);
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		i.c[1] = d1;
		r.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex64) *log10() {
		double d1;
		double d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[0], i.c[0]) / AML_LN10;
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[1], i.c[1]) / AML_LN10;
		i.c[1] = d1;
		r.c[1] = d2;
		return this;
	}

	class AML_PREFIX(Complex64_2_Itr) : public std::iterator<
			std::input_iterator_tag,   // iterator_category
			AML_PREFIX(Complex64Ptr),                      // value_type
			long,                      // difference_type
			const AML_PREFIX(Complex64Ptr) *,               // pointer
			AML_PREFIX(Complex64Ptr)                       // reference
	> {

		AML_PREFIX(Array2Complex64) *a;
		int position;

	public:
		AML_FUNCTION explicit AML_PREFIX(Complex64_2_Itr)(AML_PREFIX(Array2Complex64) *array, int length) : a(array),
																											position(
																													length) {

		}

		AML_FUNCTION AML_PREFIX(Complex64_2_Itr) &operator++() {
			position++;
			return *this;
		}

		AML_FUNCTION bool operator==(const AML_PREFIX(Complex64_2_Itr) other) const {
			return position == other.position;
		}

		AML_FUNCTION bool operator!=(const AML_PREFIX(Complex64_2_Itr) other) const { return !(*this == other); }

		AML_FUNCTION reference operator*() const {
			return AML_PREFIX(Complex64Ptr)(&a->r.c[position], &a->i.c[position], position);
		}


	};

	AML_FUNCTION AML_PREFIX(Complex64_2_Itr) begin() {
		return AML_PREFIX(Complex64_2_Itr)(this, 0);
	}

	AML_FUNCTION AML_PREFIX(Complex64_2_Itr) end() {
		return AML_PREFIX(Complex64_2_Itr)(this, 2);
	}
};


#if !defined(AML_NO_STRING)


AML_FUNCTION std::string operator<<(std::string &lhs, const AML_PREFIX(Array2Complex64) &rhs) {
	std::ostringstream string;
	string << lhs << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1] << "i }";
	return string.str();
}

AML_FUNCTION std::string operator<<(const char *lhs, const AML_PREFIX(Array2Complex64) &rhs) {
	std::ostringstream string;
	string << lhs << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1] << "i }";
	return string.str();
}

template<class charT, class traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &o, const AML_PREFIX(Array2Complex64) &rhs) {
	std::basic_ostringstream<charT, traits> s;
	s.flags(o.flags());
	s.imbue(o.getloc());
	s.precision(o.precision());
	s << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1] << "i }";
	return o << s.str();
}

#endif

AML_FUNCTION AML_PREFIX(Array2Complex64)
operator+(const AML_PREFIX(Complex64) &lhs, const AML_PREFIX(Array2Complex64) &rhs) {
	return rhs + lhs;
}

AML_FUNCTION AML_PREFIX(Array2Complex64)
operator-(const AML_PREFIX(Complex64) &lhs, const AML_PREFIX(Array2Complex64) &rhs) {
	AML_PREFIX(Array2Complex64) ret;
	ret.i.c[0] = lhs.c.c[1] - rhs.i.c[0];
	ret.i.c[1] = lhs.c.c[1] - rhs.i.c[1];
	ret.r.c[0] = lhs.c.c[0] - rhs.r.c[0];
	ret.r.c[1] = lhs.c.c[0] - rhs.r.c[1];
	return ret;
}

AML_FUNCTION AML_PREFIX(Array2Complex64)
operator*(const AML_PREFIX(Complex64) &lhs, const AML_PREFIX(Array2Complex64) &rhs) {
	return rhs * lhs;
}

AML_FUNCTION AML_PREFIX(Array2Complex64)
operator/(const AML_PREFIX(Complex64) &lhs, const AML_PREFIX(Array2Complex64) &rhs) {
	AML_PREFIX(Array2Complex64) ret;
	double d1 =
			(lhs.c.c[0] * rhs.r.c[0] + lhs.c.c[1] * rhs.i.c[0]) / (rhs.r.c[0] * rhs.r.c[0] + rhs.i.c[0] * rhs.i.c[0]);
	double d2 =
			(lhs.c.c[1] * rhs.r.c[0] - lhs.c.c[0] * rhs.i.c[0]) / (rhs.r.c[0] * rhs.r.c[0] + rhs.i.c[0] * rhs.i.c[0]);
	ret.r.c[0] = d1;
	ret.i.c[0] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[1] + lhs.c.c[1] * rhs.i.c[1]) / (rhs.r.c[1] * rhs.r.c[1] + rhs.i.c[1] * rhs.i.c[1]);
	d2 = (lhs.c.c[1] * rhs.r.c[1] - lhs.c.c[0] * rhs.i.c[1]) / (rhs.r.c[1] * rhs.r.c[1] + rhs.i.c[1] * rhs.i.c[1]);
	ret.r.c[1] = d1;
	ret.i.c[1] = d2;
	return ret;
}

class AML_PREFIX(Array4Complex64) {
public:
	AML_PREFIX(doublevec4) r{};
	AML_PREFIX(doublevec4) i{};

	AML_FUNCTION AML_PREFIX(Array4Complex64)() {}

	AML_FUNCTION AML_PREFIX(Array4Complex64)(const AML_PREFIX(Complex64) value) {
		r.c[0] = value.c.c[0];
		i.c[0] = value.c.c[1];
		r.c[1] = value.c.c[0];
		i.c[1] = value.c.c[1];
		r.c[2] = value.c.c[0];
		i.c[2] = value.c.c[1];
		r.c[3] = value.c.c[0];
		i.c[3] = value.c.c[1];
	}


	AML_FUNCTION AML_PREFIX(VectorDouble4D) real() {
		return AML_PREFIX(VectorDouble4D)(r.c);
	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) complex() {
		return AML_PREFIX(VectorDouble4D)(i.c);
	}

	AML_FUNCTION AML_PREFIX(Complex64) operator[](const uint64_t location) {
		return AML_PREFIX(Complex64)(r.c[location], i.c[location]);
	}

	AML_FUNCTION void set(const uint64_t location, const AML_PREFIX(Complex64) value) {
		r.c[location] = value.c.c[0];
		i.c[location] = value.c.c[1];
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *add(const AML_PREFIX(Array4Complex64) &a) {
#if defined(USE_AVX)
		i.avx = _mm256_add_pd(i.avx, a.i.avx);
		r.avx = _mm256_add_pd(r.avx, a.r.avx);
#else
		i.c[0] += a.i.c[0];
		i.c[1] += a.i.c[1];
		i.c[2] += a.i.c[2];
		i.c[3] += a.i.c[3];
		r.c[0] += a.r.c[0];
		r.c[1] += a.r.c[1];
		r.c[2] += a.r.c[2];
		r.c[3] += a.r.c[3];
#endif
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *add(const AML_PREFIX(Complex64) a) {
#if defined(USE_AVX)
		__m256d a_i = {a.c.c[1], a.c.c[1], a.c.c[1], a.c.c[1]};
		i.avx = _mm256_add_pd(i.avx, a_i);
		__m256d a_r = {a.c.c[0], a.c.c[0], a.c.c[0], a.c.c[0]};
		r.avx = _mm256_add_pd(r.avx, a_r);
#else
		i.c[0] += a.c.c[1];
		i.c[1] += a.c.c[1];
		i.c[2] += a.c.c[1];
		i.c[3] += a.c.c[1];
		r.c[0] += a.c.c[0];
		r.c[1] += a.c.c[0];
		r.c[2] += a.c.c[0];
		r.c[3] += a.c.c[0];
#endif
		return this;
	}

	AML_FUNCTION void operator+=(const AML_PREFIX(Array4Complex64) &a) {
#if defined(USE_AVX)
		i.avx = _mm256_add_pd(i.avx, a.i.avx);
		r.avx = _mm256_add_pd(r.avx, a.r.avx);
#else
		i.c[0] += a.i.c[0];
		i.c[1] += a.i.c[1];
		i.c[2] += a.i.c[2];
		i.c[3] += a.i.c[3];
		r.c[0] += a.r.c[0];
		r.c[1] += a.r.c[1];
		r.c[2] += a.r.c[2];
		r.c[3] += a.r.c[3];
#endif
	}

	AML_FUNCTION void operator+=(const AML_PREFIX(Complex64) a) {
#if defined(USE_AVX)
		__m256d a_i = {a.c.c[1], a.c.c[1], a.c.c[1], a.c.c[1]};
		i.avx = _mm256_add_pd(i.avx, a_i);
		__m256d a_r = {a.c.c[0], a.c.c[0], a.c.c[0], a.c.c[0]};
		r.avx = _mm256_add_pd(r.avx, a_r);
#else
		i.c[0] += a.c.c[1];
		i.c[1] += a.c.c[1];
		i.c[2] += a.c.c[1];
		i.c[3] += a.c.c[1];
		r.c[0] += a.c.c[0];
		r.c[1] += a.c.c[0];
		r.c[2] += a.c.c[0];
		r.c[3] += a.c.c[0];
#endif
	}


	AML_FUNCTION AML_PREFIX(Array4Complex64) operator+(const AML_PREFIX(Array4Complex64) &a) const {
		AML_PREFIX(Array4Complex64) ret{};
#if defined(USE_AVX)
		ret.i.avx = _mm256_add_pd(i.avx, a.i.avx);
		ret.r.avx = _mm256_add_pd(r.avx, a.r.avx);
#else
		ret.i.c[0] = i.c[0] + a.i.c[0];
		ret.i.c[1] = i.c[1] + a.i.c[1];
		ret.i.c[2] = i.c[2] + a.i.c[2];
		ret.i.c[3] = i.c[3] + a.i.c[3];
		ret.r.c[0] = r.c[0] + a.r.c[0];
		ret.r.c[1] = r.c[1] + a.r.c[1];
		ret.r.c[2] = r.c[2] + a.r.c[2];
		ret.r.c[3] = r.c[3] + a.r.c[3];
#endif
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) operator+(const AML_PREFIX(Complex64) a) const {
		AML_PREFIX(Array4Complex64) ret{};
#if defined(USE_AVX)
		__m256d a_i = {a.c.c[1], a.c.c[1], a.c.c[1], a.c.c[1]};
		ret.i.avx = _mm256_add_pd(i.avx, a_i);
		__m256d a_r = {a.c.c[0], a.c.c[0], a.c.c[0], a.c.c[0]};
		ret.r.avx = _mm256_add_pd(r.avx, a_r);
#else
		ret.i.c[0] = i.c[0] + a.c.c[1];
		ret.i.c[1] = i.c[1] + a.c.c[1];
		ret.i.c[2] = i.c[2] + a.c.c[1];
		ret.i.c[3] = i.c[3] + a.c.c[1];
		ret.r.c[0] = r.c[0] + a.c.c[0];
		ret.r.c[1] = r.c[1] + a.c.c[0];
		ret.r.c[2] = r.c[2] + a.c.c[0];
		ret.r.c[3] = r.c[3] + a.c.c[0];
#endif
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *
	add(const AML_PREFIX(Array4Complex64) &a, const AML_PREFIX(VectorU8_4D) mask) {
		if (mask.v.c[0]) {
			i.c[0] += a.i.c[0];
			r.c[0] += a.r.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] += a.i.c[1];
			r.c[1] += a.r.c[1];
		}
		if (mask.v.c[2]) {
			i.c[2] += a.i.c[2];
			r.c[2] += a.r.c[2];
		}
		if (mask.v.c[3]) {
			i.c[3] += a.i.c[3];
			r.c[3] += a.r.c[3];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *add(const AML_PREFIX(Complex64) a, const AML_PREFIX(VectorU8_4D) mask) {
		if (mask.v.c[0]) {
			i.c[0] += a.c.c[1];
			r.c[0] += a.c.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] += a.c.c[1];
			r.c[1] += a.c.c[0];
		}
		if (mask.v.c[2]) {
			i.c[2] += a.c.c[1];
			r.c[2] += a.c.c[0];
		}
		if (mask.v.c[3]) {
			i.c[3] += a.c.c[1];
			r.c[3] += a.c.c[0];
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array4Complex64) *subtract(const AML_PREFIX(Array4Complex64) &a) {
#if defined(USE_AVX)
		i.avx = _mm256_sub_pd(i.avx, a.i.avx);
		r.avx = _mm256_sub_pd(r.avx, a.r.avx);
#else
		i.c[0] -= a.i.c[0];
		i.c[1] -= a.i.c[1];
		i.c[2] -= a.i.c[2];
		i.c[3] -= a.i.c[3];
		r.c[0] -= a.r.c[0];
		r.c[1] -= a.r.c[1];
		r.c[2] -= a.r.c[2];
		r.c[3] -= a.r.c[3];
#endif
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *subtract(const AML_PREFIX(Complex64) a) {
#if defined(USE_AVX)
		__m256d a_i = {a.c.c[1], a.c.c[1], a.c.c[1], a.c.c[1]};
		i.avx = _mm256_sub_pd(i.avx, a_i);
		__m256d a_r = {a.c.c[0], a.c.c[0], a.c.c[0], a.c.c[0]};
		r.avx = _mm256_sub_pd(r.avx, a_r);
#else
		i.c[0] -= a.c.c[1];
		i.c[1] -= a.c.c[1];
		i.c[2] -= a.c.c[1];
		i.c[3] -= a.c.c[1];
		r.c[0] -= a.c.c[0];
		r.c[1] -= a.c.c[0];
		r.c[2] -= a.c.c[0];
		r.c[3] -= a.c.c[0];
#endif
		return this;
	}

	AML_FUNCTION void operator-=(const AML_PREFIX(Array4Complex64) &a) {
#if defined(USE_AVX)
		i.avx = _mm256_sub_pd(i.avx, a.i.avx);
		r.avx = _mm256_sub_pd(r.avx, a.r.avx);
#else
		i.c[0] -= a.i.c[0];
		i.c[1] -= a.i.c[1];
		i.c[2] -= a.i.c[2];
		i.c[3] -= a.i.c[3];
		r.c[0] -= a.r.c[0];
		r.c[1] -= a.r.c[1];
		r.c[2] -= a.r.c[2];
		r.c[3] -= a.r.c[3];
#endif
	}

	AML_FUNCTION void operator-=(const AML_PREFIX(Complex64) a) {
#if defined(USE_AVX)
		__m256d a_i = {a.c.c[1], a.c.c[1], a.c.c[1], a.c.c[1]};
		i.avx = _mm256_sub_pd(i.avx, a_i);
		__m256d a_r = {a.c.c[0], a.c.c[0], a.c.c[0], a.c.c[0]};
		r.avx = _mm256_sub_pd(r.avx, a_r);
#else
		i.c[0] -= a.c.c[1];
		i.c[1] -= a.c.c[1];
		i.c[2] -= a.c.c[1];
		i.c[3] -= a.c.c[1];
		r.c[0] -= a.c.c[0];
		r.c[1] -= a.c.c[0];
		r.c[2] -= a.c.c[0];
		r.c[3] -= a.c.c[0];
#endif
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) operator-(const AML_PREFIX(Array4Complex64) &a) const {
		AML_PREFIX(Array4Complex64) ret;
#if defined(USE_AVX)
		ret.i.avx = _mm256_sub_pd(i.avx, a.i.avx);
		ret.r.avx = _mm256_sub_pd(r.avx, a.r.avx);
#else
		ret.i.c[0] = i.c[0] - a.i.c[0];
		ret.i.c[1] = i.c[1] - a.i.c[1];
		ret.i.c[2] = i.c[2] - a.i.c[2];
		ret.i.c[3] = i.c[3] - a.i.c[3];
		ret.r.c[0] = r.c[0] - a.r.c[0];
		ret.r.c[1] = r.c[1] - a.r.c[1];
		ret.r.c[2] = r.c[2] - a.r.c[2];
		ret.r.c[3] = r.c[3] - a.r.c[3];
#endif
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) operator-(const AML_PREFIX(Complex64) a) const {
		AML_PREFIX(Array4Complex64) ret;
#if defined(USE_AVX)
		__m256d a_i = {a.c.c[1], a.c.c[1], a.c.c[1], a.c.c[1]};
		ret.i.avx = _mm256_sub_pd(i.avx, a_i);
		__m256d a_r = {a.c.c[0], a.c.c[0], a.c.c[0], a.c.c[0]};
		ret.r.avx = _mm256_sub_pd(r.avx, a_r);
#else
		ret.i.c[0] = i.c[0] - a.c.c[1];
		ret.i.c[1] = i.c[1] - a.c.c[1];
		ret.i.c[2] = i.c[2] - a.c.c[1];
		ret.i.c[3] = i.c[3] - a.c.c[1];
		ret.r.c[0] = r.c[0] - a.c.c[0];
		ret.r.c[1] = r.c[1] - a.c.c[0];
		ret.r.c[2] = r.c[2] - a.c.c[0];
		ret.r.c[3] = r.c[3] - a.c.c[0];
#endif
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *
	subtract(const AML_PREFIX(Array4Complex64) &a, const AML_PREFIX(VectorU8_4D) &mask) {
		if (mask.v.c[0]) {
			i.c[0] -= a.i.c[0];
			r.c[0] -= a.r.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] -= a.i.c[1];
			r.c[1] -= a.r.c[1];
		}
		if (mask.v.c[2]) {
			i.c[2] -= a.i.c[2];
			r.c[2] -= a.r.c[2];
		}
		if (mask.v.c[3]) {
			i.c[3] -= a.i.c[3];
			r.c[3] -= a.r.c[3];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *
	subtract(const AML_PREFIX(Complex64) a, const AML_PREFIX(VectorU8_4D) &mask) {
		if (mask.v.c[0]) {
			i.c[0] -= a.c.c[1];
			r.c[0] -= a.c.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] -= a.c.c[1];
			r.c[1] -= a.c.c[0];
		}
		if (mask.v.c[2]) {
			i.c[2] -= a.c.c[1];
			r.c[2] -= a.c.c[0];
		}
		if (mask.v.c[3]) {
			i.c[3] -= a.c.c[1];
			r.c[3] -= a.c.c[0];
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array4Complex64) operator*(const AML_PREFIX(Array4Complex64) &a) const {
		AML_PREFIX(Array4Complex64) ret;
#if defined(USE_FMA)
		__m256d c_0 = _mm256_mul_pd(i.avx, a.i.avx);
		ret.r.avx = _mm256_fmsub_pd(r.avx, a.r.avx,c_0);
		__m256d c_2 = _mm256_mul_pd(i.avx, a.r.avx);
		ret.i.avx = _mm256_fmadd_pd(r.avx, a.i.avx,c_2);
#else
		ret.r.c[0] = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		ret.i.c[0] = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		ret.r.c[1] = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		ret.i.c[1] = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
		ret.r.c[2] = r.c[2] * a.r.c[2] - i.c[2] * a.i.c[2];
		ret.i.c[2] = r.c[2] * a.i.c[2] + i.c[2] * a.r.c[2];
		ret.r.c[3] = r.c[3] * a.r.c[3] - i.c[3] * a.i.c[3];
		ret.i.c[3] = r.c[3] * a.i.c[3] + i.c[3] * a.r.c[3];
#endif


		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *multiply(const AML_PREFIX(Array4Complex64) &a) {
		double d1;
		double d2;
		d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;

		d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * a.r.c[2] - i.c[2] * a.i.c[2];
		d2 = r.c[2] * a.i.c[2] + i.c[2] * a.r.c[2];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * a.r.c[3] - i.c[3] * a.i.c[3];
		d2 = r.c[3] * a.i.c[3] + i.c[3] * a.r.c[3];
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) operator*(const AML_PREFIX(Complex64) a) const {
		AML_PREFIX(Array4Complex64) ret;
		ret.r.c[0] = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		ret.i.c[0] = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		ret.r.c[1] = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		ret.i.c[1] = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
		ret.r.c[2] = r.c[2] * a.c.c[0] - i.c[2] * a.c.c[1];
		ret.i.c[2] = r.c[2] * a.c.c[1] + i.c[2] * a.c.c[0];
		ret.r.c[3] = r.c[3] * a.c.c[0] - i.c[3] * a.c.c[1];
		ret.i.c[3] = r.c[3] * a.c.c[1] + i.c[3] * a.c.c[0];


		return ret;
	}

	AML_FUNCTION void operator*=(const AML_PREFIX(Array4Complex64) &a) {
		double d1;
		double d2;
		d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;

		d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * a.r.c[2] - i.c[2] * a.i.c[2];
		d2 = r.c[2] * a.i.c[2] + i.c[2] * a.r.c[2];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * a.r.c[3] - i.c[3] * a.i.c[3];
		d2 = r.c[3] * a.i.c[3] + i.c[3] * a.r.c[3];
		r.c[3] = d1;
		i.c[3] = d2;
	}

	AML_FUNCTION void operator*=(const AML_PREFIX(Complex64) a) {
		double d1;
		double d2;
		d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * a.c.c[0] - i.c[2] * a.c.c[1];
		d2 = r.c[2] * a.c.c[1] + i.c[2] * a.c.c[0];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * a.c.c[0] - i.c[3] * a.c.c[1];
		d2 = r.c[3] * a.c.c[1] + i.c[3] * a.c.c[0];
		r.c[3] = d1;
		i.c[3] = d2;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *multiply(const AML_PREFIX(Complex64) a) {
		double d1;
		double d2;
		d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * a.c.c[0] - i.c[2] * a.c.c[1];
		d2 = r.c[2] * a.c.c[1] + i.c[2] * a.c.c[0];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * a.c.c[0] - i.c[3] * a.c.c[1];
		d2 = r.c[3] * a.c.c[1] + i.c[3] * a.c.c[0];
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *
	multiply(const AML_PREFIX(Complex64) a, const AML_PREFIX(VectorU8_4D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
			d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
			d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = r.c[2] * a.c.c[0] - i.c[2] * a.c.c[1];
			d2 = r.c[2] * a.c.c[1] + i.c[2] * a.c.c[0];
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = r.c[3] * a.c.c[0] - i.c[3] * a.c.c[1];
			d2 = r.c[3] * a.c.c[1] + i.c[3] * a.c.c[0];
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;


	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *
	multiply(const AML_PREFIX(Array4Complex64) &a, const AML_PREFIX(VectorU8_4D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
			d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
			d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = r.c[2] * a.r.c[2] - i.c[2] * a.i.c[2];
			d2 = r.c[2] * a.i.c[2] + i.c[2] * a.r.c[2];
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = r.c[3] * a.r.c[3] - i.c[3] * a.i.c[3];
			d2 = r.c[3] * a.i.c[3] + i.c[3] * a.r.c[3];
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *square() {
		double d1;
		double d2;
		d1 = r.c[0] * r.c[0] - i.c[0] * i.c[0];
		d2 = r.c[0] * i.c[0] + i.c[0] * r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * r.c[1] - i.c[1] * i.c[1];
		d2 = r.c[1] * i.c[1] + i.c[1] * r.c[1];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * r.c[2] - i.c[2] * i.c[2];
		d2 = r.c[2] * i.c[2] + i.c[2] * r.c[2];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * r.c[3] - i.c[3] * i.c[3];
		d2 = r.c[3] * i.c[3] + i.c[3] * r.c[3];
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *square(const AML_PREFIX(VectorU8_4D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * r.c[0] - i.c[0] * i.c[0];
			d2 = r.c[0] * i.c[0] + i.c[0] * r.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * r.c[1] - i.c[1] * i.c[1];
			d2 = r.c[1] * i.c[1] + i.c[1] * r.c[1];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = r.c[2] * r.c[2] - i.c[2] * i.c[2];
			d2 = r.c[2] * i.c[2] + i.c[2] * r.c[2];
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = r.c[3] * r.c[3] - i.c[3] * i.c[3];
			d2 = r.c[3] * i.c[3] + i.c[3] * r.c[3];
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *divide(const AML_PREFIX(Complex64) a) {
		double d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		double d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = (r.c[2] * a.c.c[0] + i.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[2] * a.c.c[0] - r.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = (r.c[3] * a.c.c[0] + i.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[3] * a.c.c[0] - r.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *
	divide(const AML_PREFIX(Complex64) a, const AML_PREFIX(VectorU8_4D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = (r.c[2] * a.c.c[0] + i.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[2] * a.c.c[0] - r.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = (r.c[3] * a.c.c[0] + i.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[3] * a.c.c[0] - r.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *divide(const AML_PREFIX(Array4Complex64) &a) {
		double d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		double d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = (r.c[2] * a.r.c[2] + i.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		d2 = (i.c[2] * a.r.c[2] - r.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = (r.c[3] * a.r.c[3] + i.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		d2 = (i.c[3] * a.r.c[3] - r.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *
	divide(const AML_PREFIX(Array4Complex64) &a, const AML_PREFIX(VectorU8_4D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = (r.c[0] * a.i.c[0] + i.c[0] * a.i.c[1]) / (a.i.c[0] * a.i.c[0] + a.i.c[1] * a.i.c[1]);
			d2 = (i.c[0] * a.i.c[0] - r.c[0] * a.i.c[1]) / (a.i.c[0] * a.i.c[0] + a.i.c[1] * a.i.c[1]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
			d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = (r.c[2] * a.r.c[2] + i.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
			d2 = (i.c[2] * a.r.c[2] - r.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = (r.c[3] * a.r.c[3] + i.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
			d2 = (i.c[3] * a.r.c[3] - r.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) operator/(const AML_PREFIX(Complex64) a) const {
		AML_PREFIX(Array4Complex64) ret;
		double d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		double d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[0] = d1;
		ret.i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[1] = d1;
		ret.i.c[1] = d2;
		d1 = (r.c[2] * a.c.c[0] + i.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[2] * a.c.c[0] - r.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[2] = d1;
		ret.i.c[2] = d2;
		d1 = (r.c[3] * a.c.c[0] + i.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[3] * a.c.c[0] - r.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[3] = d1;
		ret.i.c[3] = d2;
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) operator/(const AML_PREFIX(Array4Complex64) &a) const {
		AML_PREFIX(Array4Complex64) ret;
		double d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		double d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		ret.r.c[0] = d1;
		ret.i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		ret.r.c[1] = d1;
		ret.i.c[1] = d2;
		d1 = (r.c[2] * a.r.c[2] + i.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		d2 = (i.c[2] * a.r.c[2] - r.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		ret.r.c[2] = d1;
		ret.i.c[2] = d2;
		d1 = (r.c[3] * a.r.c[3] + i.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		d2 = (i.c[3] * a.r.c[3] - r.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		ret.r.c[3] = d1;
		ret.i.c[3] = d2;
		return ret;
	}

	AML_FUNCTION void operator/=(const AML_PREFIX(Complex64) a) {
		double d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		double d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = (r.c[2] * a.c.c[0] + i.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[2] * a.c.c[0] - r.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = (r.c[3] * a.c.c[0] + i.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[3] * a.c.c[0] - r.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[3] = d1;
		i.c[3] = d2;
	}

	AML_FUNCTION void operator/=(const AML_PREFIX(Array4Complex64) &a) {
		double d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		double d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = (r.c[2] * a.r.c[2] + i.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		d2 = (i.c[2] * a.r.c[2] - r.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = (r.c[3] * a.r.c[3] + i.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		d2 = (i.c[3] * a.r.c[3] - r.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *sqrt() {
		double d1;
		double d2;
		d2 = ::sqrt((-r.c[0] + ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[0]);
		} else LIKELY {
			d1 = i.c[0] / (2 * d2);
		}
		r.c[0] = d1;
		i.c[0] = d2;
		d2 = ::sqrt((-r.c[1] + ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[1]);
		} else LIKELY {
			d1 = i.c[1] / (2 * d2);
		}
		r.c[1] = d1;
		i.c[1] = d2;
		d2 = ::sqrt((-r.c[2] + ::sqrt(r.c[2] * r.c[2] + i.c[2] * i.c[2])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[2]);
		} else LIKELY {
			d1 = i.c[2] / (2 * d2);
		}
		r.c[2] = d1;
		i.c[2] = d2;
		d2 = ::sqrt((-r.c[3] + ::sqrt(r.c[3] * r.c[3] + i.c[3] * i.c[3])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[3]);
		} else LIKELY {
			d1 = i.c[3] / (2 * d2);
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *sqrt(const AML_PREFIX(VectorU8_4D) mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d2 = ::sqrt((-r.c[0] + ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[0]);
			} else LIKELY {
				d1 = i.c[0] / (2 * d2);
			}
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d2 = ::sqrt((-r.c[1] + ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[1]);
			} else LIKELY {
				d1 = i.c[1] / (2 * d2);
			}
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d2 = ::sqrt((-r.c[2] + ::sqrt(r.c[2] * r.c[2] + i.c[2] * i.c[2])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[2]);
			} else LIKELY {
				d1 = i.c[2] / (2 * d2);
			}
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d2 = ::sqrt((-r.c[3] + ::sqrt(r.c[3] * r.c[3] + i.c[3] * i.c[3])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[3]);
			} else LIKELY {
				d1 = i.c[3] / (2 * d2);
			}
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *sin() {
		double d1;
		double d2;
		d1 = ::sin(r.c[0]) * ::cosh(i.c[0]);
		d2 = ::cos(i.c[0]) * ::sinh(r.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::sin(r.c[1]) * ::cosh(i.c[1]);
		d2 = ::cos(i.c[1]) * ::sinh(r.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::sin(r.c[2]) * ::cosh(i.c[2]);
		d2 = ::cos(i.c[2]) * ::sinh(r.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::sin(r.c[3]) * ::cosh(i.c[3]);
		d2 = ::cos(i.c[3]) * ::sinh(r.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *cos() {
		double d1;
		double d2;
		d1 = ::cos(r.c[0]) * ::cosh(i.c[0]);
		d2 = -::sin(i.c[0]) * ::sinh(r.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::cos(r.c[1]) * ::cosh(i.c[1]);
		d2 = -::sin(i.c[1]) * ::sinh(r.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::cos(r.c[2]) * ::cosh(i.c[2]);
		d2 = -::sin(i.c[2]) * ::sinh(r.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::cos(r.c[3]) * ::cosh(i.c[3]);
		d2 = -::sin(i.c[3]) * ::sinh(r.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *tan() {
		double d1;
		double d2;
		d1 = ::sin(r.c[0] + r.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
		d2 = ::sinh(i.c[0] + i.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::sin(r.c[1] + r.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
		d2 = ::sinh(i.c[1] + i.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::sin(r.c[2] + r.c[2]) / (::cos(r.c[2] + r.c[2]) * ::cosh(i.c[2] + i.c[2]));
		d2 = ::sinh(i.c[2] + i.c[2]) / (::cos(r.c[2] + r.c[2]) * ::cosh(i.c[2] + i.c[2]));
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::sin(r.c[3] + r.c[3]) / (::cos(r.c[3] + r.c[3]) * ::cosh(i.c[3] + i.c[3]));
		d2 = ::sinh(i.c[3] + i.c[3]) / (::cos(r.c[3] + r.c[3]) * ::cosh(i.c[3] + i.c[3]));
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *sin(const AML_PREFIX(VectorU8_4D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::sin(r.c[0]) * ::cosh(i.c[0]);
			d2 = ::cos(i.c[0]) * ::sinh(r.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::sin(r.c[1]) * ::cosh(i.c[1]);
			d2 = ::cos(i.c[1]) * ::sinh(r.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::sin(r.c[2]) * ::cosh(i.c[2]);
			d2 = ::cos(i.c[2]) * ::sinh(r.c[2]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::sin(r.c[3]) * ::cosh(i.c[3]);
			d2 = ::cos(i.c[3]) * ::sinh(r.c[3]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *cos(const AML_PREFIX(VectorU8_4D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::cos(r.c[0]) * ::cosh(i.c[0]);
			d2 = -::sin(i.c[0]) * ::sinh(r.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::cos(r.c[1]) * ::cosh(i.c[1]);
			d2 = -::sin(i.c[1]) * ::sinh(r.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::cos(r.c[2]) * ::cosh(i.c[2]);
			d2 = -::sin(i.c[2]) * ::sinh(r.c[2]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::cos(r.c[3]) * ::cosh(i.c[3]);
			d2 = -::sin(i.c[3]) * ::sinh(r.c[3]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *tan(const AML_PREFIX(VectorU8_4D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::sin(r.c[0] + r.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
			d2 = ::sinh(i.c[0] + i.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::sin(r.c[1] + r.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
			d2 = ::sinh(i.c[1] + i.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::sin(r.c[2] + r.c[2]) / (::cos(r.c[2] + r.c[2]) * ::cosh(i.c[2] + i.c[2]));
			d2 = ::sinh(i.c[2] + i.c[2]) / (::cos(r.c[2] + r.c[2]) * ::cosh(i.c[2] + i.c[2]));
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::sin(r.c[3] + r.c[3]) / (::cos(r.c[3] + r.c[3]) * ::cosh(i.c[3] + i.c[3]));
			d2 = ::sinh(i.c[3] + i.c[3]) / (::cos(r.c[3] + r.c[3]) * ::cosh(i.c[3] + i.c[3]));
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array4Complex64) *exp() {
		double d1 = ::exp(r.c[0]) * ::cos(i.c[0]);
		double d2 = ::exp(r.c[0]) * ::sin(i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::exp(r.c[1]) * ::cos(i.c[1]);
		d2 = ::exp(r.c[1]) * ::sin(i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::exp(r.c[2]) * ::cos(i.c[2]);
		d2 = ::exp(r.c[2]) * ::sin(i.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::exp(r.c[3]) * ::cos(i.c[3]);
		d2 = ::exp(r.c[3]) * ::sin(i.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *exp(double n) {
		double d1 = ::pow(n, r.c[0]) * ::cos(i.c[0] * ::log(n));
		double d2 = ::pow(n, r.c[0]) * ::sin(i.c[0] * ::log(n));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::pow(n, r.c[1]) * ::cos(i.c[1] * ::log(n));
		d2 = ::pow(n, r.c[1]) * ::sin(i.c[1] * ::log(n));
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::pow(n, r.c[2]) * ::cos(i.c[2] * ::log(n));
		d2 = ::pow(n, r.c[2]) * ::sin(i.c[2] * ::log(n));
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::pow(n, r.c[3]) * ::cos(i.c[3] * ::log(n));
		d2 = ::pow(n, r.c[3]) * ::sin(i.c[3] * ::log(n));
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *exp(const AML_PREFIX(VectorU8_4D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::exp(r.c[0]) * ::cos(i.c[0]);
			d2 = ::exp(r.c[0]) * ::sin(i.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::exp(r.c[1]) * ::cos(i.c[1]);
			d2 = ::exp(r.c[1]) * ::sin(i.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::exp(r.c[2]) * ::cos(i.c[2]);
			d2 = ::exp(r.c[2]) * ::sin(i.c[2]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::exp(r.c[3]) * ::cos(i.c[3]);
			d2 = ::exp(r.c[3]) * ::sin(i.c[3]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *pow(const AML_PREFIX(Array4Complex64) &n) {
		double d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		double d2 = ::atan2(r.c[0], i.c[0]);
		double d3 = ::exp(d1 * n.i.c[0] - d2 * n.r.c[0]);
		double d4 = d1 * n.r.c[0] + d2 * n.i.c[0];
		double d5 = d3 * ::cos(d4);
		double d6 = d3 * ::sin(d4);
		i.c[0] = d5;
		r.c[0] = d6;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		d3 = ::exp(d1 * n.i.c[1] - d2 * n.r.c[1]);
		d4 = d1 * n.r.c[1] + d2 * n.i.c[1];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[1] = d5;
		r.c[1] = d6;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
		d2 = ::atan2(r.c[2], i.c[2]);
		d3 = ::exp(d1 * n.i.c[2] - d2 * n.r.c[2]);
		d4 = d1 * n.r.c[2] + d2 * n.i.c[2];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[2] = d5;
		r.c[2] = d6;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
		d2 = ::atan2(r.c[3], i.c[3]);
		d3 = ::exp(d1 * n.i.c[3] - d2 * n.r.c[3]);
		d4 = d1 * n.r.c[3] + d2 * n.i.c[3];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[3] = d5;
		r.c[3] = d6;
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array4Complex64) *pow(const AML_PREFIX(Complex64) n) {
		double d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		double d2 = ::atan2(r.c[0], i.c[0]);
		double d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		double d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		double d5 = d3 * ::cos(d4);
		double d6 = d3 * ::sin(d4);
		i.c[0] = d5;
		r.c[0] = d6;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[1] = d5;
		r.c[1] = d6;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
		d2 = ::atan2(r.c[2], i.c[2]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[2] = d5;
		r.c[2] = d6;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
		d2 = ::atan2(r.c[3], i.c[3]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[3] = d5;
		r.c[3] = d6;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *
	pow(const AML_PREFIX(Array4Complex64) &n, const AML_PREFIX(VectorU8_4D) &mask) {
		double d1;
		double d2;
		double d3;
		double d4;
		double d5;
		double d6;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			d3 = ::exp(d1 * n.i.c[0] - d2 * n.r.c[0]);
			d4 = d1 * n.r.c[0] + d2 * n.i.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[0] = d5;
			r.c[0] = d6;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			d3 = ::exp(d1 * n.i.c[1] - d2 * n.r.c[1]);
			d4 = d1 * n.r.c[1] + d2 * n.i.c[1];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[1] = d5;
			r.c[1] = d6;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
			d2 = ::atan2(r.c[2], i.c[2]);
			d3 = ::exp(d1 * n.i.c[2] - d2 * n.r.c[2]);
			d4 = d1 * n.r.c[2] + d2 * n.i.c[2];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[2] = d5;
			r.c[2] = d6;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
			d2 = ::atan2(r.c[3], i.c[3]);
			d3 = ::exp(d1 * n.i.c[3] - d2 * n.r.c[3]);
			d4 = d1 * n.r.c[3] + d2 * n.i.c[3];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[3] = d5;
			r.c[3] = d6;
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array4Complex64) *pow(const AML_PREFIX(Complex64) n, const AML_PREFIX(VectorU8_4D) &mask) {
		double d1;
		double d2;
		double d3;
		double d4;
		double d5;
		double d6;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[0] = d5;
			r.c[0] = d6;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[1] = d5;
			r.c[1] = d6;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
			d2 = ::atan2(r.c[2], i.c[2]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[2] = d5;
			r.c[2] = d6;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
			d2 = ::atan2(r.c[3], i.c[3]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[3] = d5;
			r.c[3] = d6;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *exp(double n, const AML_PREFIX(VectorU8_4D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::pow(n, r.c[0]) * ::cos(i.c[0] * ::log(n));
			d2 = ::pow(n, r.c[0]) * ::sin(i.c[0] * ::log(n));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::pow(n, r.c[1]) * ::cos(i.c[1] * ::log(n));
			d2 = ::pow(n, r.c[1]) * ::sin(i.c[1] * ::log(n));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::pow(n, r.c[2]) * ::cos(i.c[2] * ::log(n));
			d2 = ::pow(n, r.c[2]) * ::sin(i.c[2] * ::log(n));
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::pow(n, r.c[3]) * ::cos(i.c[3] * ::log(n));
			d2 = ::pow(n, r.c[3]) * ::sin(i.c[3] * ::log(n));
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *pow(double n) {
		double d1;
		double d2;
		d1 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::cos(n * atan2(i.c[0], r.c[0]));
		d2 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::sin(n * atan2(i.c[0], r.c[0]));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::cos(n * atan2(i.c[1], r.c[1]));
		d2 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::sin(n * atan2(i.c[1], r.c[1]));
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::pow(r.c[2] * r.c[2] + i.c[2] * i.c[2], n / 2) * ::cos(n * atan2(i.c[2], r.c[2]));
		d2 = ::pow(r.c[2] * r.c[2] + i.c[2] * i.c[2], n / 2) * ::sin(n * atan2(i.c[2], r.c[2]));
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::pow(r.c[3] * r.c[3] + i.c[3] * i.c[3], n / 2) * ::cos(n * atan2(i.c[3], r.c[3]));
		d2 = ::pow(r.c[3] * r.c[3] + i.c[3] * i.c[3], n / 2) * ::sin(n * atan2(i.c[3], r.c[3]));
		r.c[3] = d1;
		i.c[3] = d2;

		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *pow(double n, const AML_PREFIX(VectorU8_4D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::cos(n * atan2(i.c[0], r.c[0]));
			d2 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::sin(n * atan2(i.c[0], r.c[0]));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::cos(n * atan2(i.c[1], r.c[1]));
			d2 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::sin(n * atan2(i.c[1], r.c[1]));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::pow(r.c[2] * r.c[2] + i.c[2] * i.c[2], n / 2) * ::cos(n * atan2(i.c[2], r.c[2]));
			d2 = ::pow(r.c[2] * r.c[2] + i.c[2] * i.c[2], n / 2) * ::sin(n * atan2(i.c[2], r.c[2]));
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::pow(r.c[3] * r.c[3] + i.c[3] * i.c[3], n / 2) * ::cos(n * atan2(i.c[3], r.c[3]));
			d2 = ::pow(r.c[3] * r.c[3] + i.c[3] * i.c[3], n / 2) * ::sin(n * atan2(i.c[3], r.c[3]));
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble4D) abs() {
		AML_PREFIX(VectorDouble4D) ret;
		ret.v.c[0] = ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0]);
		ret.v.c[1] = ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1]);
		ret.v.c[2] = ::sqrt(r.c[2] * r.c[2] + i.c[2] * i.c[2]);
		ret.v.c[3] = ::sqrt(r.c[3] * r.c[3] + i.c[3] * i.c[3]);
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_4D) abs_gt(double a) {
		AML_PREFIX(VectorU8_4D) ret;
		ret.v.c[0] = a * a < r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a < r.c[1] * r.c[1] + i.c[1] * i.c[1];
		ret.v.c[2] = a * a < r.c[2] * r.c[2] + i.c[2] * i.c[2];
		ret.v.c[3] = a * a < r.c[3] * r.c[3] + i.c[3] * i.c[3];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_4D) abs_lt(double a) {
		AML_PREFIX(VectorU8_4D) ret;
		ret.v.c[0] = a * a > r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a > r.c[1] * r.c[1] + i.c[1] * i.c[1];
		ret.v.c[2] = a * a > r.c[2] * r.c[2] + i.c[2] * i.c[2];
		ret.v.c[3] = a * a > r.c[3] * r.c[3] + i.c[3] * i.c[3];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_4D) abs_eq(double a) {
		AML_PREFIX(VectorU8_4D) ret;
		ret.v.c[0] = a * a == r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a == r.c[1] * r.c[1] + i.c[1] * i.c[1];
		ret.v.c[2] = a * a == r.c[2] * r.c[2] + i.c[2] * i.c[2];
		ret.v.c[3] = a * a == r.c[3] * r.c[3] + i.c[3] * i.c[3];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *ln(const AML_PREFIX(VectorU8_4D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			i.c[1] = d1;
			r.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
			d2 = ::atan2(r.c[2], i.c[2]);
			i.c[2] = d1;
			r.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
			d2 = ::atan2(r.c[3], i.c[3]);
			i.c[3] = d1;
			r.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *log(const AML_PREFIX(VectorU8_4D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			i.c[1] = d1;
			r.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
			d2 = ::atan2(r.c[2], i.c[2]);
			i.c[2] = d1;
			r.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
			d2 = ::atan2(r.c[3], i.c[3]);
			i.c[3] = d1;
			r.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *log10(const AML_PREFIX(VectorU8_4D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[0], i.c[0]) / AML_LN10;
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[1], i.c[1]) / AML_LN10;
			i.c[1] = d1;
			r.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[2], i.c[2]) / AML_LN10;
			i.c[2] = d1;
			r.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[3], i.c[3]) / AML_LN10;
			i.c[3] = d1;
			r.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *ln() {
		double d1;
		double d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		d2 = ::atan2(r.c[0], i.c[0]);
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		i.c[1] = d1;
		r.c[1] = d2;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
		d2 = ::atan2(r.c[2], i.c[2]);
		i.c[2] = d1;
		r.c[2] = d2;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
		d2 = ::atan2(r.c[3], i.c[3]);
		i.c[3] = d1;
		r.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *log() {
		double d1;
		double d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		d2 = ::atan2(r.c[0], i.c[0]);
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		i.c[1] = d1;
		r.c[1] = d2;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
		d2 = ::atan2(r.c[2], i.c[2]);
		i.c[2] = d1;
		r.c[2] = d2;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
		d2 = ::atan2(r.c[3], i.c[3]);
		i.c[3] = d1;
		r.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex64) *log10() {
		double d1;
		double d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[0], i.c[0]) / AML_LN10;
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[1], i.c[1]) / AML_LN10;
		i.c[1] = d1;
		r.c[1] = d2;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[2], i.c[2]) / AML_LN10;
		i.c[2] = d1;
		r.c[2] = d2;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[3], i.c[3]) / AML_LN10;
		i.c[3] = d1;
		r.c[3] = d2;
		return this;
	}

	class AML_PREFIX(Complex64_4_Itr) : public std::iterator<
			std::input_iterator_tag,   // iterator_category
			AML_PREFIX(Complex64Ptr),                      // value_type
			long,                      // difference_type
			const AML_PREFIX(Complex64Ptr) *,               // pointer
			AML_PREFIX(Complex64Ptr)                       // reference
	> {

		AML_PREFIX(Array4Complex64) *a;
		int position;

	public:
		AML_FUNCTION explicit AML_PREFIX(Complex64_4_Itr)(AML_PREFIX(Array4Complex64) *array, int length) : a(array),
																											position(
																													length) {}

		AML_FUNCTION AML_PREFIX(Complex64_4_Itr) &operator++() {
			position++;
			return *this;
		}

		AML_FUNCTION bool operator==(const AML_PREFIX(Complex64_4_Itr) other) const {
			return position == other.position;
		}

		AML_FUNCTION bool operator!=(const AML_PREFIX(Complex64_4_Itr) other) const { return !(*this == other); }

		AML_FUNCTION reference operator*() const {
			return AML_PREFIX(Complex64Ptr)(&a->r.c[position], &a->i.c[position], position);
		}


	};

	AML_FUNCTION AML_PREFIX(Complex64_4_Itr) begin() {
		return AML_PREFIX(Complex64_4_Itr)(this, 0);
	}

	AML_FUNCTION AML_PREFIX(Complex64_4_Itr) end() {
		return AML_PREFIX(Complex64_4_Itr)(this, 4);
	}
};


#if !defined(AML_NO_STRING)


AML_FUNCTION std::string operator<<(std::string &lhs, const AML_PREFIX(Array4Complex64) &rhs) {
	std::ostringstream string;
	string << lhs << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1]
		   << "i ,  " << rhs.r.c[2] << " + " << rhs.i.c[2] << "i ,  " << rhs.r.c[3] << " + " << rhs.i.c[3] << "i }";
	return string.str();
}

AML_FUNCTION std::string operator<<(const char *lhs, const AML_PREFIX(Array4Complex64) &rhs) {
	std::ostringstream string;
	string << lhs << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1]
		   << "i ,  " << rhs.r.c[2] << " + " << rhs.i.c[2] << "i ,  " << rhs.r.c[3] << " + " << rhs.i.c[3] << "i }";
	return string.str();
}

template<class charT, class traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &o, const AML_PREFIX(Array4Complex64) &rhs) {
	std::basic_ostringstream<charT, traits> s;
	s.flags(o.flags());
	s.imbue(o.getloc());
	s.precision(o.precision());
	s << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1]
	  << "i ,  " << rhs.r.c[2] << " + " << rhs.i.c[2] << "i ,  " << rhs.r.c[3] << " + " << rhs.i.c[3] << "i }";
	return o << s.str();
}

#endif

AML_FUNCTION AML_PREFIX(Array4Complex64)
operator+(const AML_PREFIX(Complex64) &lhs, const AML_PREFIX(Array4Complex64) &rhs) {
	return rhs + lhs;
}

AML_FUNCTION AML_PREFIX(Array4Complex64)
operator-(const AML_PREFIX(Complex64) &lhs, const AML_PREFIX(Array4Complex64) &rhs) {
	AML_PREFIX(Array4Complex64) ret;
	ret.i.c[0] = lhs.c.c[1] - rhs.i.c[0];
	ret.i.c[1] = lhs.c.c[1] - rhs.i.c[1];
	ret.i.c[2] = lhs.c.c[1] - rhs.i.c[2];
	ret.i.c[3] = lhs.c.c[1] - rhs.i.c[3];
	ret.r.c[0] = lhs.c.c[0] - rhs.r.c[0];
	ret.r.c[1] = lhs.c.c[0] - rhs.r.c[1];
	ret.r.c[2] = lhs.c.c[0] - rhs.r.c[2];
	ret.r.c[3] = lhs.c.c[0] - rhs.r.c[3];
	return ret;
}

AML_FUNCTION AML_PREFIX(Array4Complex64)
operator*(const AML_PREFIX(Complex64) &lhs, const AML_PREFIX(Array4Complex64) &rhs) {
	return rhs * lhs;
}

AML_FUNCTION AML_PREFIX(Array4Complex64)
operator/(const AML_PREFIX(Complex64) &lhs, const AML_PREFIX(Array4Complex64) &rhs) {
	AML_PREFIX(Array4Complex64) ret;
	double d1 =
			(lhs.c.c[0] * rhs.r.c[0] + lhs.c.c[1] * rhs.i.c[0]) / (rhs.r.c[0] * rhs.r.c[0] + rhs.i.c[0] * rhs.i.c[0]);
	double d2 =
			(lhs.c.c[1] * rhs.r.c[0] - lhs.c.c[0] * rhs.i.c[0]) / (rhs.r.c[0] * rhs.r.c[0] + rhs.i.c[0] * rhs.i.c[0]);
	ret.r.c[0] = d1;
	ret.i.c[0] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[1] + lhs.c.c[1] * rhs.i.c[1]) / (rhs.r.c[1] * rhs.r.c[1] + rhs.i.c[1] * rhs.i.c[1]);
	d2 = (lhs.c.c[1] * rhs.r.c[1] - lhs.c.c[0] * rhs.i.c[1]) / (rhs.r.c[1] * rhs.r.c[1] + rhs.i.c[1] * rhs.i.c[1]);
	ret.r.c[1] = d1;
	ret.i.c[1] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[2] + lhs.c.c[1] * rhs.i.c[2]) / (rhs.r.c[2] * rhs.r.c[2] + rhs.i.c[2] * rhs.i.c[2]);
	d2 = (lhs.c.c[1] * rhs.r.c[2] - lhs.c.c[0] * rhs.i.c[2]) / (rhs.r.c[2] * rhs.r.c[2] + rhs.i.c[2] * rhs.i.c[2]);
	ret.r.c[2] = d1;
	ret.i.c[2] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[3] + lhs.c.c[1] * rhs.i.c[3]) / (rhs.r.c[3] * rhs.r.c[3] + rhs.i.c[3] * rhs.i.c[3]);
	d2 = (lhs.c.c[1] * rhs.r.c[3] - lhs.c.c[0] * rhs.i.c[3]) / (rhs.r.c[3] * rhs.r.c[3] + rhs.i.c[3] * rhs.i.c[3]);
	ret.r.c[3] = d1;
	ret.i.c[3] = d2;
	return ret;
}

class AML_PREFIX(Array8Complex64) {
public:
	AML_PREFIX(doublevec8) r{};
	AML_PREFIX(doublevec8) i{};

	AML_FUNCTION AML_PREFIX(Array8Complex64)() {}

	AML_FUNCTION AML_PREFIX(Array8Complex64)(const AML_PREFIX(Complex64) value) {
#if defined(USE_AVX)
		r.avx[0] = _mm256_set1_pd(value.c.c[0]);
		r.avx[1] = _mm256_set1_pd(value.c.c[0]);
		i.avx[0] = _mm256_set1_pd(value.c.c[1]);
		i.avx[1] = _mm256_set1_pd(value.c.c[1]);
#else
		r.c[0] = value.c.c[0];
		i.c[0] = value.c.c[1];
		r.c[1] = value.c.c[0];
		i.c[1] = value.c.c[1];
		r.c[2] = value.c.c[0];
		i.c[2] = value.c.c[1];
		r.c[3] = value.c.c[0];
		i.c[3] = value.c.c[1];
		r.c[4] = value.c.c[0];
		i.c[4] = value.c.c[1];
		r.c[5] = value.c.c[0];
		i.c[5] = value.c.c[1];
		r.c[6] = value.c.c[0];
		i.c[6] = value.c.c[1];
		r.c[7] = value.c.c[0];
		i.c[7] = value.c.c[1];
#endif
	}

	AML_FUNCTION AML_PREFIX(VectorDouble8D) real() {
		return AML_PREFIX(VectorDouble8D)(r.c);
	}

	AML_FUNCTION AML_PREFIX(VectorDouble8D) complex() {
		return AML_PREFIX(VectorDouble8D)(i.c);
	}

	AML_FUNCTION AML_PREFIX(Complex64) operator[](uint64_t location) {
		return AML_PREFIX(Complex64)(r.c[location], i.c[location]);
	}

	AML_FUNCTION void set(uint64_t location, AML_PREFIX(Complex64) value) {
		r.c[location] = value.c.c[0];
		i.c[location] = value.c.c[1];
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *add(const AML_PREFIX(Array8Complex64) &a) {
#if defined(USE_AVX)
		i.avx[0] = _mm256_add_pd(i.avx[0], a.i.avx[0]);
		i.avx[1] = _mm256_add_pd(i.avx[1], a.i.avx[1]);
		r.avx[0] = _mm256_add_pd(r.avx[0], a.r.avx[0]);
		r.avx[1] = _mm256_add_pd(r.avx[1], a.r.avx[1]);
#else
		r.c[0] += a.r.c[0];
		r.c[1] += a.r.c[1];
		r.c[2] += a.r.c[2];
		r.c[3] += a.r.c[3];
		r.c[4] += a.r.c[4];
		r.c[5] += a.r.c[5];
		r.c[6] += a.r.c[6];
		r.c[7] += a.r.c[7];
		i.c[0] += a.i.c[0];
		i.c[1] += a.i.c[1];
		i.c[2] += a.i.c[2];
		i.c[3] += a.i.c[3];
		i.c[4] += a.i.c[4];
		i.c[5] += a.i.c[5];
		i.c[6] += a.i.c[6];
		i.c[7] += a.i.c[7];
#endif
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *add(const AML_PREFIX(Complex64) &a) {
#if defined(USE_AVX)
		__m256d a_i = {a.c.c[1], a.c.c[1], a.c.c[1], a.c.c[1]};
		i.avx[0] = _mm256_add_pd(i.avx[0], a_i);
		i.avx[1] = _mm256_add_pd(i.avx[1], a_i);
		__m256d a_r = {a.c.c[0], a.c.c[0], a.c.c[0], a.c.c[0]};
		r.avx[0] = _mm256_add_pd(r.avx[0], a_r);
		r.avx[1] = _mm256_add_pd(r.avx[1], a_r);
#else
		i.c[0] += a.c.c[1];
		i.c[1] += a.c.c[1];
		i.c[2] += a.c.c[1];
		i.c[3] += a.c.c[1];
		i.c[4] += a.c.c[1];
		i.c[5] += a.c.c[1];
		i.c[6] += a.c.c[1];
		i.c[7] += a.c.c[1];
		r.c[0] += a.c.c[0];
		r.c[1] += a.c.c[0];
		r.c[2] += a.c.c[0];
		r.c[3] += a.c.c[0];
		r.c[4] += a.c.c[0];
		r.c[5] += a.c.c[0];
		r.c[6] += a.c.c[0];
		r.c[7] += a.c.c[0];
#endif
		return this;
	}

	AML_FUNCTION void operator+=(const AML_PREFIX(Array8Complex64) &a) {
#if defined(USE_AVX)
		i.avx[0] = _mm256_add_pd(i.avx[0], a.i.avx[0]);
		i.avx[1] = _mm256_add_pd(i.avx[1], a.i.avx[1]);
		r.avx[0] = _mm256_add_pd(r.avx[0], a.r.avx[0]);
		r.avx[1] = _mm256_add_pd(r.avx[1], a.r.avx[1]);
#else
		i.c[0] += a.i.c[0];
		i.c[1] += a.i.c[1];
		i.c[2] += a.i.c[2];
		i.c[3] += a.i.c[3];
		i.c[4] += a.i.c[4];
		i.c[5] += a.i.c[5];
		i.c[6] += a.i.c[6];
		i.c[7] += a.i.c[7];
		r.c[0] += a.r.c[0];
		r.c[1] += a.r.c[1];
		r.c[2] += a.r.c[2];
		r.c[3] += a.r.c[3];
		r.c[4] += a.r.c[4];
		r.c[5] += a.r.c[5];
		r.c[6] += a.r.c[6];
		r.c[7] += a.r.c[7];
#endif
	}

	AML_FUNCTION void operator+=(const AML_PREFIX(Complex64) a) {
#if defined(USE_AVX)
		__m256d a_i = {a.c.c[1], a.c.c[1], a.c.c[1], a.c.c[1]};
		i.avx[0] = _mm256_add_pd(i.avx[0], a_i);
		i.avx[1] = _mm256_add_pd(i.avx[1], a_i);
		__m256d a_r = {a.c.c[0], a.c.c[0], a.c.c[0], a.c.c[0]};
		r.avx[0] = _mm256_add_pd(r.avx[0], a_r);
		r.avx[1] = _mm256_add_pd(r.avx[1], a_r);
#else
		i.c[0] += a.c.c[1];
		i.c[1] += a.c.c[1];
		i.c[2] += a.c.c[1];
		i.c[3] += a.c.c[1];
		i.c[4] += a.c.c[1];
		i.c[5] += a.c.c[1];
		i.c[6] += a.c.c[1];
		i.c[7] += a.c.c[1];
		r.c[0] += a.c.c[0];
		r.c[1] += a.c.c[0];
		r.c[2] += a.c.c[0];
		r.c[3] += a.c.c[0];
		r.c[4] += a.c.c[0];
		r.c[5] += a.c.c[0];
		r.c[6] += a.c.c[0];
		r.c[7] += a.c.c[0];
#endif
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) operator+(const AML_PREFIX(Array8Complex64) &a) const {
		AML_PREFIX(Array8Complex64) ret;
#if defined(USE_AVX)
		ret.i.avx[0] = _mm256_add_pd(i.avx[0], a.i.avx[0]);
		ret.i.avx[1] = _mm256_add_pd(i.avx[1], a.i.avx[1]);
		ret.r.avx[0] = _mm256_add_pd(r.avx[0], a.r.avx[0]);
		ret.r.avx[1] = _mm256_add_pd(r.avx[1], a.r.avx[1]);
#else
		ret.r.c[0] = r.c[0] + a.r.c[0];
		ret.r.c[1] = r.c[1] + a.r.c[1];
		ret.r.c[2] = r.c[2] + a.r.c[2];
		ret.r.c[3] = r.c[3] + a.r.c[3];
		ret.r.c[4] = r.c[4] + a.r.c[4];
		ret.r.c[5] = r.c[5] + a.r.c[5];
		ret.r.c[6] = r.c[6] + a.r.c[6];
		ret.r.c[7] = r.c[7] + a.r.c[7];
		ret.i.c[0] = i.c[0] + a.i.c[0];
		ret.i.c[1] = i.c[1] + a.i.c[1];
		ret.i.c[2] = i.c[2] + a.i.c[2];
		ret.i.c[3] = i.c[3] + a.i.c[3];
		ret.i.c[4] = i.c[4] + a.i.c[4];
		ret.i.c[5] = i.c[5] + a.i.c[5];
		ret.i.c[6] = i.c[6] + a.i.c[6];
		ret.i.c[7] = i.c[7] + a.i.c[7];
#endif
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) operator+(const AML_PREFIX(Complex64) a) const {
		AML_PREFIX(Array8Complex64) ret{};
#if defined(USE_AVX)
		__m256d a_i = {a.c.c[1], a.c.c[1], a.c.c[1], a.c.c[1]};
		ret.i.avx[0] = _mm256_add_pd(i.avx[0], a_i);
		ret.i.avx[1] = _mm256_add_pd(i.avx[1], a_i);
		__m256d a_r = {a.c.c[0], a.c.c[0], a.c.c[0], a.c.c[0]};
		ret.r.avx[0] = _mm256_add_pd(r.avx[0], a_r);
		ret.r.avx[1] = _mm256_add_pd(r.avx[1], a_r);
#else
		ret.i.c[0] = i.c[0] + a.c.c[1];
		ret.i.c[1] = i.c[1] + a.c.c[1];
		ret.i.c[2] = i.c[2] + a.c.c[1];
		ret.i.c[3] = i.c[3] + a.c.c[1];
		ret.i.c[4] = i.c[4] + a.c.c[1];
		ret.i.c[5] = i.c[5] + a.c.c[1];
		ret.i.c[6] = i.c[6] + a.c.c[1];
		ret.i.c[7] = i.c[7] + a.c.c[1];
		ret.r.c[0] = r.c[0] + a.c.c[0];
		ret.r.c[1] = r.c[1] + a.c.c[0];
		ret.r.c[2] = r.c[2] + a.c.c[0];
		ret.r.c[3] = r.c[3] + a.c.c[0];
		ret.r.c[4] = r.c[4] + a.c.c[0];
		ret.r.c[5] = r.c[5] + a.c.c[0];
		ret.r.c[6] = r.c[6] + a.c.c[0];
		ret.r.c[7] = r.c[7] + a.c.c[0];
#endif
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *add(const AML_PREFIX(Array8Complex64) &a, AML_PREFIX(VectorU8_8D) mask) {
		if (mask.v.c[0]) {
			i.c[0] += a.i.c[0];
			r.c[0] += a.r.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] += a.i.c[1];
			r.c[1] += a.r.c[1];
		}
		if (mask.v.c[2]) {
			i.c[2] += a.i.c[2];
			r.c[2] += a.r.c[2];
		}
		if (mask.v.c[3]) {
			i.c[3] += a.i.c[3];
			r.c[3] += a.r.c[3];
		}
		if (mask.v.c[4]) {
			i.c[4] += a.i.c[4];
			r.c[4] += a.r.c[4];
		}
		if (mask.v.c[5]) {
			i.c[5] += a.i.c[5];
			r.c[5] += a.r.c[5];
		}
		if (mask.v.c[6]) {
			i.c[6] += a.i.c[6];
			r.c[6] += a.r.c[6];
		}
		if (mask.v.c[7]) {
			i.c[7] += a.i.c[7];
			r.c[7] += a.r.c[7];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *add(const AML_PREFIX(Complex64) a, AML_PREFIX(VectorU8_8D) mask) {
		if (mask.v.c[0]) {
			i.c[0] += a.c.c[1];
			r.c[0] += a.c.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] += a.c.c[1];
			r.c[1] += a.c.c[0];
		}
		if (mask.v.c[2]) {
			i.c[2] += a.c.c[1];
			r.c[2] += a.c.c[0];
		}
		if (mask.v.c[3]) {
			i.c[3] += a.c.c[1];
			r.c[3] += a.c.c[0];
		}
		if (mask.v.c[4]) {
			i.c[4] += a.c.c[1];
			r.c[4] += a.c.c[0];
		}
		if (mask.v.c[5]) {
			i.c[5] += a.c.c[1];
			r.c[5] += a.c.c[0];
		}
		if (mask.v.c[6]) {
			i.c[6] += a.c.c[1];
			r.c[6] += a.c.c[0];
		}
		if (mask.v.c[7]) {
			i.c[7] += a.c.c[1];
			r.c[7] += a.c.c[0];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *subtract(const AML_PREFIX(Array8Complex64) a) {
#if defined(USE_AVX)
		i.avx[0] = _mm256_sub_pd(i.avx[0], a.i.avx[0]);
		i.avx[1] = _mm256_sub_pd(i.avx[1], a.i.avx[1]);
		r.avx[0] = _mm256_sub_pd(r.avx[0], a.r.avx[0]);
		r.avx[1] = _mm256_sub_pd(r.avx[1], a.r.avx[1]);
#else
		i.c[0] -= a.i.c[0];
		i.c[1] -= a.i.c[1];
		i.c[2] -= a.i.c[2];
		i.c[3] -= a.i.c[3];
		i.c[4] -= a.i.c[4];
		i.c[5] -= a.i.c[5];
		i.c[6] -= a.i.c[6];
		i.c[7] -= a.i.c[7];
		r.c[0] -= a.r.c[0];
		r.c[1] -= a.r.c[1];
		r.c[2] -= a.r.c[2];
		r.c[3] -= a.r.c[3];
		r.c[4] -= a.r.c[4];
		r.c[5] -= a.r.c[5];
		r.c[6] -= a.r.c[6];
		r.c[7] -= a.r.c[7];
#endif
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *subtract(const AML_PREFIX(Complex64) a) {
#if defined(USE_AVX)
		__m256d a_i = {a.c.c[1], a.c.c[1], a.c.c[1], a.c.c[1]};
		i.avx[0] = _mm256_sub_pd(i.avx[0], a_i);
		i.avx[1] = _mm256_sub_pd(i.avx[1], a_i);
		__m256d a_r = {a.c.c[0], a.c.c[0], a.c.c[0], a.c.c[0]};
		r.avx[0] = _mm256_sub_pd(r.avx[0], a_r);
		r.avx[1] = _mm256_sub_pd(r.avx[1], a_r);
#else
		i.c[0] -= a.c.c[1];
		i.c[1] -= a.c.c[1];
		i.c[2] -= a.c.c[1];
		i.c[3] -= a.c.c[1];
		i.c[4] -= a.c.c[1];
		i.c[5] -= a.c.c[1];
		i.c[6] -= a.c.c[1];
		i.c[7] -= a.c.c[1];
		r.c[0] -= a.c.c[0];
		r.c[1] -= a.c.c[0];
		r.c[2] -= a.c.c[0];
		r.c[3] -= a.c.c[0];
		r.c[4] -= a.c.c[0];
		r.c[5] -= a.c.c[0];
		r.c[6] -= a.c.c[0];
		r.c[7] -= a.c.c[0];
#endif
		return this;
	}

	AML_FUNCTION void operator-=(const AML_PREFIX(Array8Complex64) a) {
#if defined(USE_AVX)
		i.avx[0] = _mm256_sub_pd(i.avx[0], a.i.avx[0]);
		i.avx[1] = _mm256_sub_pd(i.avx[1], a.i.avx[1]);
		r.avx[0] = _mm256_sub_pd(r.avx[0], a.r.avx[0]);
		r.avx[1] = _mm256_sub_pd(r.avx[1], a.r.avx[1]);
#else
		i.c[0] -= a.i.c[0];
		i.c[1] -= a.i.c[1];
		i.c[2] -= a.i.c[2];
		i.c[3] -= a.i.c[3];
		i.c[4] -= a.i.c[4];
		i.c[5] -= a.i.c[5];
		i.c[6] -= a.i.c[6];
		i.c[7] -= a.i.c[7];
		r.c[0] -= a.r.c[0];
		r.c[1] -= a.r.c[1];
		r.c[2] -= a.r.c[2];
		r.c[3] -= a.r.c[3];
		r.c[4] -= a.r.c[4];
		r.c[5] -= a.r.c[5];
		r.c[6] -= a.r.c[6];
		r.c[7] -= a.r.c[7];
#endif
	}

	AML_FUNCTION void operator-=(const AML_PREFIX(Complex64) a) {
#if defined(USE_AVX)
		__m256d a_i = {a.c.c[1], a.c.c[1], a.c.c[1], a.c.c[1]};
		i.avx[0] = _mm256_sub_pd(i.avx[0], a_i);
		i.avx[1] = _mm256_sub_pd(i.avx[1], a_i);
		__m256d a_r = {a.c.c[0], a.c.c[0], a.c.c[0], a.c.c[0]};
		r.avx[0] = _mm256_sub_pd(r.avx[0], a_r);
		r.avx[1] = _mm256_sub_pd(r.avx[1], a_r);
#else
		i.c[0] -= a.c.c[1];
		i.c[1] -= a.c.c[1];
		i.c[2] -= a.c.c[1];
		i.c[3] -= a.c.c[1];
		i.c[4] -= a.c.c[1];
		i.c[5] -= a.c.c[1];
		i.c[6] -= a.c.c[1];
		i.c[7] -= a.c.c[1];
		r.c[0] -= a.c.c[0];
		r.c[1] -= a.c.c[0];
		r.c[2] -= a.c.c[0];
		r.c[3] -= a.c.c[0];
		r.c[4] -= a.c.c[0];
		r.c[5] -= a.c.c[0];
		r.c[6] -= a.c.c[0];
		r.c[7] -= a.c.c[0];
#endif
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) operator-(const AML_PREFIX(Array8Complex64) &a) const {
		AML_PREFIX(Array8Complex64) ret;
#if defined(USE_AVX)
		ret.i.avx[0] = _mm256_sub_pd(i.avx[0], a.i.avx[0]);
		ret.i.avx[1] = _mm256_sub_pd(i.avx[1], a.i.avx[1]);
		ret.r.avx[0] = _mm256_sub_pd(r.avx[0], a.r.avx[0]);
		ret.r.avx[1] = _mm256_sub_pd(r.avx[1], a.r.avx[1]);
#else
		ret.i.c[0] = i.c[0] - a.i.c[0];
		ret.i.c[1] = i.c[1] - a.i.c[1];
		ret.i.c[2] = i.c[2] - a.i.c[2];
		ret.i.c[3] = i.c[3] - a.i.c[3];
		ret.i.c[4] = i.c[4] - a.i.c[4];
		ret.i.c[5] = i.c[5] - a.i.c[5];
		ret.i.c[6] = i.c[6] - a.i.c[6];
		ret.i.c[7] = i.c[7] - a.i.c[7];
		ret.r.c[0] = r.c[0] - a.r.c[0];
		ret.r.c[1] = r.c[1] - a.r.c[1];
		ret.r.c[2] = r.c[2] - a.r.c[2];
		ret.r.c[3] = r.c[3] - a.r.c[3];
		ret.r.c[4] = r.c[4] - a.r.c[4];
		ret.r.c[5] = r.c[5] - a.r.c[5];
		ret.r.c[6] = r.c[6] - a.r.c[6];
		ret.r.c[7] = r.c[7] - a.r.c[7];
#endif
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) operator-(const AML_PREFIX(Complex64) a) const {
		AML_PREFIX(Array8Complex64) ret;
#if defined(USE_AVX)
		__m256d a_i = {a.c.c[1], a.c.c[1], a.c.c[1], a.c.c[1]};
		ret.i.avx[0] = _mm256_sub_pd(i.avx[0], a_i);
		ret.i.avx[1] = _mm256_sub_pd(i.avx[1], a_i);
		__m256d a_r = {a.c.c[0], a.c.c[0], a.c.c[0], a.c.c[0]};
		ret.r.avx[0] = _mm256_sub_pd(r.avx[0], a_r);
		ret.r.avx[1] = _mm256_sub_pd(r.avx[1], a_r);
#else
		ret.i.c[0] = i.c[0] - a.c.c[1];
		ret.i.c[1] = i.c[1] - a.c.c[1];
		ret.i.c[2] = i.c[2] - a.c.c[1];
		ret.i.c[3] = i.c[3] - a.c.c[1];
		ret.i.c[4] = i.c[4] - a.c.c[1];
		ret.i.c[5] = i.c[5] - a.c.c[1];
		ret.i.c[6] = i.c[6] - a.c.c[1];
		ret.i.c[7] = i.c[7] - a.c.c[1];
		ret.r.c[0] = r.c[0] - a.c.c[0];
		ret.r.c[1] = r.c[1] - a.c.c[0];
		ret.r.c[2] = r.c[2] - a.c.c[0];
		ret.r.c[3] = r.c[3] - a.c.c[0];
		ret.r.c[4] = r.c[4] - a.c.c[0];
		ret.r.c[5] = r.c[5] - a.c.c[0];
		ret.r.c[6] = r.c[6] - a.c.c[0];
		ret.r.c[7] = r.c[7] - a.c.c[0];
#endif
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *
	subtract(const AML_PREFIX(Array8Complex64) &a, AML_PREFIX(VectorU8_8D) mask) {
		if (mask.v.c[0]) {
			i.c[0] -= a.i.c[0];
			r.c[0] -= a.r.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] -= a.i.c[1];
			r.c[1] -= a.r.c[1];
		}
		if (mask.v.c[2]) {
			i.c[2] -= a.i.c[2];
			r.c[2] -= a.r.c[2];
		}
		if (mask.v.c[3]) {
			i.c[3] -= a.i.c[3];
			r.c[3] -= a.r.c[3];
		}
		if (mask.v.c[4]) {
			i.c[4] -= a.i.c[4];
			r.c[4] -= a.r.c[4];
		}
		if (mask.v.c[5]) {
			i.c[5] -= a.i.c[5];
			r.c[5] -= a.r.c[5];
		}
		if (mask.v.c[6]) {
			i.c[6] -= a.i.c[6];
			r.c[6] -= a.r.c[6];
		}
		if (mask.v.c[7]) {
			i.c[7] -= a.i.c[7];
			r.c[7] -= a.r.c[7];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *subtract(const AML_PREFIX(Complex64) a, AML_PREFIX(VectorU8_8D) mask) {
		if (mask.v.c[0]) {
			i.c[0] -= a.c.c[1];
			r.c[0] -= a.c.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] -= a.c.c[1];
			r.c[1] -= a.c.c[0];
		}
		if (mask.v.c[2]) {
			i.c[2] -= a.c.c[1];
			r.c[2] -= a.c.c[0];
		}
		if (mask.v.c[3]) {
			i.c[3] -= a.c.c[1];
			r.c[3] -= a.c.c[0];
		}
		if (mask.v.c[4]) {
			i.c[4] -= a.c.c[1];
			r.c[4] -= a.c.c[0];
		}
		if (mask.v.c[5]) {
			i.c[5] -= a.c.c[1];
			r.c[5] -= a.c.c[0];
		}
		if (mask.v.c[6]) {
			i.c[6] -= a.c.c[1];
			r.c[6] -= a.c.c[0];
		}
		if (mask.v.c[7]) {
			i.c[7] -= a.c.c[1];
			r.c[7] -= a.c.c[0];
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array8Complex64) operator*(const AML_PREFIX(Array8Complex64) &a) const {
		AML_PREFIX(Array8Complex64) ret;
#if defined(USE_FMA)
		__m256d c_0 = _mm256_mul_pd(i.avx[0], a.i.avx[0]);
		ret.r.avx[0] = _mm256_fmsub_pd(r.avx[0], a.r.avx[0],c_0);
		__m256d c_1 = _mm256_mul_pd(i.avx[1], a.i.avx[1]);
		ret.r.avx[1] = _mm256_fmsub_pd(r.avx[1], a.r.avx[1],c_1);
		__m256d c_2 = _mm256_mul_pd(i.avx[0], a.r.avx[0]);
		ret.i.avx[0] = _mm256_fmadd_pd(r.avx[0], a.i.avx[0],c_2);
		__m256d c_3 = _mm256_mul_pd(i.avx[1], a.r.avx[1]);
		ret.i.avx[1] = _mm256_fmadd_pd(r.avx[1], a.i.avx[1],c_3);
#else
		ret.r.c[0] = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		ret.i.c[0] = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		ret.r.c[1] = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		ret.i.c[1] = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
		ret.r.c[2] = r.c[2] * a.r.c[2] - i.c[2] * a.i.c[2];
		ret.i.c[2] = r.c[2] * a.i.c[2] + i.c[2] * a.r.c[2];
		ret.r.c[3] = r.c[3] * a.r.c[3] - i.c[3] * a.i.c[3];
		ret.i.c[3] = r.c[3] * a.i.c[3] + i.c[3] * a.r.c[3];
		ret.r.c[4] = r.c[4] * a.r.c[4] - i.c[4] * a.i.c[4];
		ret.i.c[4] = r.c[4] * a.i.c[4] + i.c[4] * a.r.c[4];
		ret.r.c[5] = r.c[5] * a.r.c[5] - i.c[5] * a.i.c[5];
		ret.i.c[5] = r.c[5] * a.i.c[5] + i.c[5] * a.r.c[5];
		ret.r.c[6] = r.c[6] * a.r.c[6] - i.c[6] * a.i.c[6];
		ret.i.c[6] = r.c[6] * a.i.c[6] + i.c[6] * a.r.c[6];
		ret.r.c[7] = r.c[7] * a.r.c[7] - i.c[7] * a.i.c[7];
		ret.i.c[7] = r.c[7] * a.i.c[7] + i.c[7] * a.r.c[7];
#endif

		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *multiply(const AML_PREFIX(Array8Complex64) &a) {
		double d1;
		double d2;
		d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;

		d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * a.r.c[2] - i.c[2] * a.i.c[2];
		d2 = r.c[2] * a.i.c[2] + i.c[2] * a.r.c[2];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * a.r.c[3] - i.c[3] * a.i.c[3];
		d2 = r.c[3] * a.i.c[3] + i.c[3] * a.r.c[3];
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = r.c[4] * a.r.c[4] - i.c[4] * a.i.c[4];
		d2 = r.c[4] * a.i.c[4] + i.c[4] * a.r.c[4];
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = r.c[5] * a.r.c[5] - i.c[5] * a.i.c[5];
		d2 = r.c[5] * a.i.c[5] + i.c[5] * a.r.c[5];
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = r.c[6] * a.r.c[6] - i.c[6] * a.i.c[6];
		d2 = r.c[6] * a.i.c[6] + i.c[6] * a.r.c[6];
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = r.c[7] * a.r.c[7] - i.c[7] * a.i.c[7];
		d2 = r.c[7] * a.i.c[7] + i.c[7] * a.r.c[7];
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION void operator*=(const AML_PREFIX(Array8Complex64) &a) {
		double d1;
		double d2;
		d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;

		d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * a.r.c[2] - i.c[2] * a.i.c[2];
		d2 = r.c[2] * a.i.c[2] + i.c[2] * a.r.c[2];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * a.r.c[3] - i.c[3] * a.i.c[3];
		d2 = r.c[3] * a.i.c[3] + i.c[3] * a.r.c[3];
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = r.c[4] * a.r.c[4] - i.c[4] * a.i.c[4];
		d2 = r.c[4] * a.i.c[4] + i.c[4] * a.r.c[4];
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = r.c[5] * a.r.c[5] - i.c[5] * a.i.c[5];
		d2 = r.c[5] * a.i.c[5] + i.c[5] * a.r.c[5];
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = r.c[6] * a.r.c[6] - i.c[6] * a.i.c[6];
		d2 = r.c[6] * a.i.c[6] + i.c[6] * a.r.c[6];
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = r.c[7] * a.r.c[7] - i.c[7] * a.i.c[7];
		d2 = r.c[7] * a.i.c[7] + i.c[7] * a.r.c[7];
		r.c[7] = d1;
		i.c[7] = d2;
	}

	AML_FUNCTION void operator*=(const AML_PREFIX(Complex64) &a) {
		double d1;
		double d2;
		d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * a.c.c[0] - i.c[2] * a.c.c[1];
		d2 = r.c[2] * a.c.c[1] + i.c[2] * a.c.c[0];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * a.c.c[0] - i.c[3] * a.c.c[1];
		d2 = r.c[3] * a.c.c[1] + i.c[3] * a.c.c[0];
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = r.c[4] * a.c.c[0] - i.c[4] * a.c.c[1];
		d2 = r.c[4] * a.c.c[1] + i.c[4] * a.c.c[0];
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = r.c[5] * a.c.c[0] - i.c[5] * a.c.c[1];
		d2 = r.c[5] * a.c.c[1] + i.c[5] * a.c.c[0];
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = r.c[6] * a.c.c[0] - i.c[6] * a.c.c[1];
		d2 = r.c[6] * a.c.c[1] + i.c[6] * a.c.c[0];
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = r.c[7] * a.c.c[0] - i.c[7] * a.c.c[1];
		d2 = r.c[7] * a.c.c[1] + i.c[7] * a.c.c[0];
		r.c[7] = d1;
		i.c[7] = d2;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) operator*(const AML_PREFIX(Complex64) &a) const {
		AML_PREFIX(Array8Complex64) ret;
		ret.r.c[0] = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		ret.i.c[0] = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		ret.r.c[1] = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		ret.i.c[1] = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
		ret.r.c[2] = r.c[2] * a.c.c[0] - i.c[2] * a.c.c[1];
		ret.i.c[2] = r.c[2] * a.c.c[1] + i.c[2] * a.c.c[0];
		ret.r.c[3] = r.c[3] * a.c.c[0] - i.c[3] * a.c.c[1];
		ret.i.c[3] = r.c[3] * a.c.c[1] + i.c[3] * a.c.c[0];
		ret.r.c[4] = r.c[4] * a.c.c[0] - i.c[4] * a.c.c[1];
		ret.i.c[4] = r.c[4] * a.c.c[1] + i.c[4] * a.c.c[0];
		ret.r.c[5] = r.c[5] * a.c.c[0] - i.c[5] * a.c.c[1];
		ret.i.c[5] = r.c[5] * a.c.c[1] + i.c[5] * a.c.c[0];
		ret.r.c[6] = r.c[6] * a.c.c[0] - i.c[6] * a.c.c[1];
		ret.i.c[6] = r.c[6] * a.c.c[1] + i.c[6] * a.c.c[0];
		ret.r.c[7] = r.c[7] * a.c.c[0] - i.c[7] * a.c.c[1];
		ret.i.c[7] = r.c[7] * a.c.c[1] + i.c[7] * a.c.c[0];


		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *multiply(const AML_PREFIX(Complex64) &a) {
		double d1;
		double d2;
		d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * a.c.c[0] - i.c[2] * a.c.c[1];
		d2 = r.c[2] * a.c.c[1] + i.c[2] * a.c.c[0];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * a.c.c[0] - i.c[3] * a.c.c[1];
		d2 = r.c[3] * a.c.c[1] + i.c[3] * a.c.c[0];
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = r.c[4] * a.c.c[0] - i.c[4] * a.c.c[1];
		d2 = r.c[4] * a.c.c[1] + i.c[4] * a.c.c[0];
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = r.c[5] * a.c.c[0] - i.c[5] * a.c.c[1];
		d2 = r.c[5] * a.c.c[1] + i.c[5] * a.c.c[0];
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = r.c[6] * a.c.c[0] - i.c[6] * a.c.c[1];
		d2 = r.c[6] * a.c.c[1] + i.c[6] * a.c.c[0];
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = r.c[7] * a.c.c[0] - i.c[7] * a.c.c[1];
		d2 = r.c[7] * a.c.c[1] + i.c[7] * a.c.c[0];
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *
	multiply(const AML_PREFIX(Complex64) &a, const AML_PREFIX(VectorU8_8D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
			d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
			d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = r.c[2] * a.c.c[0] - i.c[2] * a.c.c[1];
			d2 = r.c[2] * a.c.c[1] + i.c[2] * a.c.c[0];
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = r.c[3] * a.c.c[0] - i.c[3] * a.c.c[1];
			d2 = r.c[3] * a.c.c[1] + i.c[3] * a.c.c[0];
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = r.c[4] * a.c.c[0] - i.c[4] * a.c.c[1];
			d2 = r.c[4] * a.c.c[1] + i.c[4] * a.c.c[0];
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = r.c[5] * a.c.c[0] - i.c[5] * a.c.c[1];
			d2 = r.c[5] * a.c.c[1] + i.c[5] * a.c.c[0];
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = r.c[6] * a.c.c[0] - i.c[6] * a.c.c[1];
			d2 = r.c[6] * a.c.c[1] + i.c[6] * a.c.c[0];
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = r.c[7] * a.c.c[0] - i.c[7] * a.c.c[1];
			d2 = r.c[7] * a.c.c[1] + i.c[7] * a.c.c[0];
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;


	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *
	multiply(const AML_PREFIX(Array8Complex64) &a, const AML_PREFIX(VectorU8_8D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
			d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
			d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = r.c[2] * a.r.c[2] - i.c[2] * a.i.c[2];
			d2 = r.c[2] * a.i.c[2] + i.c[2] * a.r.c[2];
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = r.c[3] * a.r.c[3] - i.c[3] * a.i.c[3];
			d2 = r.c[3] * a.i.c[3] + i.c[3] * a.r.c[3];
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = r.c[4] * a.r.c[4] - i.c[4] * a.i.c[4];
			d2 = r.c[4] * a.i.c[4] + i.c[4] * a.r.c[4];
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = r.c[5] * a.r.c[5] - i.c[5] * a.i.c[5];
			d2 = r.c[5] * a.i.c[5] + i.c[5] * a.r.c[5];
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = r.c[6] * a.r.c[6] - i.c[6] * a.i.c[6];
			d2 = r.c[6] * a.i.c[6] + i.c[6] * a.r.c[6];
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = r.c[7] * a.r.c[7] - i.c[7] * a.i.c[7];
			d2 = r.c[7] * a.i.c[7] + i.c[7] * a.r.c[7];
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *square() {
		double d1;
		double d2;
		d1 = r.c[0] * r.c[0] - i.c[0] * i.c[0];
		d2 = r.c[0] * i.c[0] + i.c[0] * r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * r.c[1] - i.c[1] * i.c[1];
		d2 = r.c[1] * i.c[1] + i.c[1] * r.c[1];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * r.c[2] - i.c[2] * i.c[2];
		d2 = r.c[2] * i.c[2] + i.c[2] * r.c[2];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * r.c[3] - i.c[3] * i.c[3];
		d2 = r.c[3] * i.c[3] + i.c[3] * r.c[3];
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = r.c[4] * r.c[4] - i.c[4] * i.c[4];
		d2 = r.c[4] * i.c[4] + i.c[4] * r.c[4];
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = r.c[5] * r.c[5] - i.c[5] * i.c[5];
		d2 = r.c[5] * i.c[5] + i.c[5] * r.c[5];
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = r.c[6] * r.c[6] - i.c[6] * i.c[6];
		d2 = r.c[6] * i.c[6] + i.c[6] * r.c[6];
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = r.c[7] * r.c[7] - i.c[7] * i.c[7];
		d2 = r.c[7] * i.c[7] + i.c[7] * r.c[7];
		r.c[7] = d1;
		i.c[7] =
				d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *square(const AML_PREFIX(VectorU8_8D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * r.c[0] - i.c[0] * i.c[0];
			d2 = r.c[0] * i.c[0] + i.c[0] * r.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * r.c[1] - i.c[1] * i.c[1];
			d2 = r.c[1] * i.c[1] + i.c[1] * r.c[1];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = r.c[2] * r.c[2] - i.c[2] * i.c[2];
			d2 = r.c[2] * i.c[2] + i.c[2] * r.c[2];
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = r.c[3] * r.c[3] - i.c[3] * i.c[3];
			d2 = r.c[3] * i.c[3] + i.c[3] * r.c[3];
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = r.c[4] * r.c[4] - i.c[4] * i.c[4];
			d2 = r.c[4] * i.c[4] + i.c[4] * r.c[4];
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = r.c[5] * r.c[5] - i.c[5] * i.c[5];
			d2 = r.c[5] * i.c[5] + i.c[5] * r.c[5];
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = r.c[6] * r.c[6] - i.c[6] * i.c[6];
			d2 = r.c[6] * i.c[6] + i.c[6] * r.c[6];
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = r.c[7] * r.c[7] - i.c[7] * i.c[7];
			d2 = r.c[7] * i.c[7] + i.c[7] * r.c[7];
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *divide(const AML_PREFIX(Complex64) a) {
		double d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		double d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = (r.c[2] * a.c.c[0] + i.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[2] * a.c.c[0] - r.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = (r.c[3] * a.c.c[0] + i.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[3] * a.c.c[0] - r.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = (r.c[4] * a.c.c[0] + i.c[4] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[4] * a.c.c[0] - r.c[4] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = (r.c[5] * a.c.c[0] + i.c[5] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[5] * a.c.c[0] - r.c[5] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = (r.c[6] * a.c.c[0] + i.c[6] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[6] * a.c.c[0] - r.c[6] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = (r.c[7] * a.c.c[0] + i.c[7] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[7] * a.c.c[0] - r.c[7] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *
	divide(const AML_PREFIX(Complex64) a, const AML_PREFIX(VectorU8_8D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = (r.c[2] * a.c.c[0] + i.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[2] * a.c.c[0] - r.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = (r.c[3] * a.c.c[0] + i.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[3] * a.c.c[0] - r.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = (r.c[4] * a.c.c[0] + i.c[4] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[4] * a.c.c[0] - r.c[4] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = (r.c[5] * a.c.c[0] + i.c[5] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[5] * a.c.c[0] - r.c[5] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = (r.c[6] * a.c.c[0] + i.c[6] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[6] * a.c.c[0] - r.c[6] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = (r.c[7] * a.c.c[0] + i.c[7] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[7] * a.c.c[0] - r.c[7] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *divide(const AML_PREFIX(Array8Complex64) &a) {
		double d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		double d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = (r.c[2] * a.r.c[2] + i.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		d2 = (i.c[2] * a.r.c[2] - r.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = (r.c[3] * a.r.c[3] + i.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		d2 = (i.c[3] * a.r.c[3] - r.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = (r.c[4] * a.r.c[4] + i.c[4] * a.i.c[4]) / (a.r.c[4] * a.r.c[4] + a.i.c[4] * a.i.c[4]);
		d2 = (i.c[4] * a.r.c[4] - r.c[4] * a.i.c[4]) / (a.r.c[4] * a.r.c[4] + a.i.c[4] * a.i.c[4]);
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = (r.c[5] * a.r.c[5] + i.c[5] * a.i.c[5]) / (a.r.c[5] * a.r.c[5] + a.i.c[5] * a.i.c[5]);
		d2 = (i.c[5] * a.r.c[5] - r.c[5] * a.i.c[5]) / (a.r.c[5] * a.r.c[5] + a.i.c[5] * a.i.c[5]);
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = (r.c[6] * a.r.c[6] + i.c[6] * a.i.c[6]) / (a.r.c[6] * a.r.c[6] + a.i.c[6] * a.i.c[6]);
		d2 = (i.c[6] * a.r.c[6] - r.c[6] * a.i.c[6]) / (a.r.c[6] * a.r.c[6] + a.i.c[6] * a.i.c[6]);
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = (r.c[7] * a.r.c[7] + i.c[7] * a.i.c[7]) / (a.r.c[7] * a.r.c[7] + a.i.c[7] * a.i.c[7]);
		d2 = (i.c[7] * a.r.c[7] - r.c[7] * a.i.c[7]) / (a.r.c[7] * a.r.c[7] + a.i.c[7] * a.i.c[7]);
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *
	divide(const AML_PREFIX(Array8Complex64) &a, const AML_PREFIX(VectorU8_8D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = (r.c[0] * a.i.c[0] + i.c[0] * a.i.c[1]) / (a.i.c[0] * a.i.c[0] + a.i.c[1] * a.i.c[1]);
			d2 = (i.c[0] * a.i.c[0] - r.c[0] * a.i.c[1]) / (a.i.c[0] * a.i.c[0] + a.i.c[1] * a.i.c[1]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
			d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = (r.c[2] * a.r.c[2] + i.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
			d2 = (i.c[2] * a.r.c[2] - r.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = (r.c[3] * a.r.c[3] + i.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
			d2 = (i.c[3] * a.r.c[3] - r.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = (r.c[4] * a.r.c[4] + i.c[4] * a.i.c[4]) / (a.r.c[4] * a.r.c[4] + a.i.c[4] * a.i.c[4]);
			d2 = (i.c[4] * a.r.c[4] - r.c[4] * a.i.c[4]) / (a.r.c[4] * a.r.c[4] + a.i.c[4] * a.i.c[4]);
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = (r.c[5] * a.r.c[5] + i.c[5] * a.i.c[5]) / (a.r.c[5] * a.r.c[5] + a.i.c[5] * a.i.c[5]);
			d2 = (i.c[5] * a.r.c[5] - r.c[5] * a.i.c[5]) / (a.r.c[5] * a.r.c[5] + a.i.c[5] * a.i.c[5]);
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = (r.c[6] * a.r.c[6] + i.c[6] * a.i.c[6]) / (a.r.c[6] * a.r.c[6] + a.i.c[6] * a.i.c[6]);
			d2 = (i.c[6] * a.r.c[6] - r.c[6] * a.i.c[6]) / (a.r.c[6] * a.r.c[6] + a.i.c[6] * a.i.c[6]);
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = (r.c[7] * a.r.c[7] + i.c[7] * a.i.c[7]) / (a.r.c[7] * a.r.c[7] + a.i.c[7] * a.i.c[7]);
			d2 = (i.c[7] * a.r.c[7] - r.c[7] * a.i.c[7]) / (a.r.c[7] * a.r.c[7] + a.i.c[7] * a.i.c[7]);
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) operator/(const AML_PREFIX(Complex64) &a) const {
		AML_PREFIX(Array8Complex64) ret;
		double d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		double d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[0] = d1;
		ret.i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[1] = d1;
		ret.i.c[1] = d2;
		d1 = (r.c[2] * a.c.c[0] + i.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[2] * a.c.c[0] - r.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[2] = d1;
		ret.i.c[2] = d2;
		d1 = (r.c[3] * a.c.c[0] + i.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[3] * a.c.c[0] - r.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[3] = d1;
		ret.i.c[3] = d2;
		d1 = (r.c[4] * a.c.c[0] + i.c[4] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[4] * a.c.c[0] - r.c[4] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[4] = d1;
		ret.i.c[4] = d2;
		d1 = (r.c[5] * a.c.c[0] + i.c[5] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[5] * a.c.c[0] - r.c[5] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[5] = d1;
		ret.i.c[5] = d2;
		d1 = (r.c[6] * a.c.c[0] + i.c[6] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[6] * a.c.c[0] - r.c[6] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[6] = d1;
		ret.i.c[6] = d2;
		d1 = (r.c[7] * a.c.c[0] + i.c[7] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[7] * a.c.c[0] - r.c[7] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[7] = d1;
		ret.i.c[7] = d2;
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) operator/(const AML_PREFIX(Array8Complex64) &a) const {
		AML_PREFIX(Array8Complex64) ret;
		double d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		double d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		ret.r.c[0] = d1;
		ret.i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		ret.r.c[1] = d1;
		ret.i.c[1] = d2;
		d1 = (r.c[2] * a.r.c[2] + i.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		d2 = (i.c[2] * a.r.c[2] - r.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		ret.r.c[2] = d1;
		ret.i.c[2] = d2;
		d1 = (r.c[3] * a.r.c[3] + i.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		d2 = (i.c[3] * a.r.c[3] - r.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		ret.r.c[3] = d1;
		ret.i.c[3] = d2;
		d1 = (r.c[4] * a.r.c[4] + i.c[4] * a.i.c[4]) / (a.r.c[4] * a.r.c[4] + a.i.c[4] * a.i.c[4]);
		d2 = (i.c[4] * a.r.c[4] - r.c[4] * a.i.c[4]) / (a.r.c[4] * a.r.c[4] + a.i.c[4] * a.i.c[4]);
		ret.r.c[4] = d1;
		ret.i.c[4] = d2;
		d1 = (r.c[5] * a.r.c[5] + i.c[5] * a.i.c[5]) / (a.r.c[5] * a.r.c[5] + a.i.c[5] * a.i.c[5]);
		d2 = (i.c[5] * a.r.c[5] - r.c[5] * a.i.c[5]) / (a.r.c[5] * a.r.c[5] + a.i.c[5] * a.i.c[5]);
		ret.r.c[5] = d1;
		ret.i.c[5] = d2;
		d1 = (r.c[6] * a.r.c[6] + i.c[6] * a.i.c[6]) / (a.r.c[6] * a.r.c[6] + a.i.c[6] * a.i.c[6]);
		d2 = (i.c[6] * a.r.c[6] - r.c[6] * a.i.c[6]) / (a.r.c[6] * a.r.c[6] + a.i.c[6] * a.i.c[6]);
		ret.r.c[6] = d1;
		ret.i.c[6] = d2;
		d1 = (r.c[7] * a.r.c[7] + i.c[7] * a.i.c[7]) / (a.r.c[7] * a.r.c[7] + a.i.c[7] * a.i.c[7]);
		d2 = (i.c[7] * a.r.c[7] - r.c[7] * a.i.c[7]) / (a.r.c[7] * a.r.c[7] + a.i.c[7] * a.i.c[7]);
		ret.r.c[7] = d1;
		ret.i.c[7] = d2;
		return ret;
	}

	AML_FUNCTION void operator/=(const AML_PREFIX(Complex64) &a) {
		double d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		double d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = (r.c[2] * a.c.c[0] + i.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[2] * a.c.c[0] - r.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = (r.c[3] * a.c.c[0] + i.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[3] * a.c.c[0] - r.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = (r.c[4] * a.c.c[0] + i.c[4] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[4] * a.c.c[0] - r.c[4] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = (r.c[5] * a.c.c[0] + i.c[5] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[5] * a.c.c[0] - r.c[5] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = (r.c[6] * a.c.c[0] + i.c[6] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[6] * a.c.c[0] - r.c[6] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = (r.c[7] * a.c.c[0] + i.c[7] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[7] * a.c.c[0] - r.c[7] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[7] = d1;
		i.c[7] = d2;
	}

	AML_FUNCTION void operator/=(const AML_PREFIX(Array8Complex64) &a) {
		double d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		double d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = (r.c[2] * a.r.c[2] + i.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		d2 = (i.c[2] * a.r.c[2] - r.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = (r.c[3] * a.r.c[3] + i.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		d2 = (i.c[3] * a.r.c[3] - r.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = (r.c[4] * a.r.c[4] + i.c[4] * a.i.c[4]) / (a.r.c[4] * a.r.c[4] + a.i.c[4] * a.i.c[4]);
		d2 = (i.c[4] * a.r.c[4] - r.c[4] * a.i.c[4]) / (a.r.c[4] * a.r.c[4] + a.i.c[4] * a.i.c[4]);
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = (r.c[5] * a.r.c[5] + i.c[5] * a.i.c[5]) / (a.r.c[5] * a.r.c[5] + a.i.c[5] * a.i.c[5]);
		d2 = (i.c[5] * a.r.c[5] - r.c[5] * a.i.c[5]) / (a.r.c[5] * a.r.c[5] + a.i.c[5] * a.i.c[5]);
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = (r.c[6] * a.r.c[6] + i.c[6] * a.i.c[6]) / (a.r.c[6] * a.r.c[6] + a.i.c[6] * a.i.c[6]);
		d2 = (i.c[6] * a.r.c[6] - r.c[6] * a.i.c[6]) / (a.r.c[6] * a.r.c[6] + a.i.c[6] * a.i.c[6]);
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = (r.c[7] * a.r.c[7] + i.c[7] * a.i.c[7]) / (a.r.c[7] * a.r.c[7] + a.i.c[7] * a.i.c[7]);
		d2 = (i.c[7] * a.r.c[7] - r.c[7] * a.i.c[7]) / (a.r.c[7] * a.r.c[7] + a.i.c[7] * a.i.c[7]);
		r.c[7] = d1;
		i.c[7] = d2;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *sqrt() {
		double d1;
		double d2;
		d2 = ::sqrt((-r.c[0] + ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[0]);
		} else LIKELY {
			d1 = i.c[0] / (2 * d2);
		}
		r.c[0] = d1;
		i.c[0] = d2;
		d2 = ::sqrt((-r.c[1] + ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[1]);
		} else LIKELY {
			d1 = i.c[1] / (2 * d2);
		}
		r.c[1] = d1;
		i.c[1] = d2;
		d2 = ::sqrt((-r.c[2] + ::sqrt(r.c[2] * r.c[2] + i.c[2] * i.c[2])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[2]);
		} else LIKELY {
			d1 = i.c[2] / (2 * d2);
		}
		r.c[2] = d1;
		i.c[2] = d2;
		d2 = ::sqrt((-r.c[3] + ::sqrt(r.c[3] * r.c[3] + i.c[3] * i.c[3])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[3]);
		} else LIKELY {
			d1 = i.c[3] / (2 * d2);
		}
		r.c[3] = d1;
		i.c[3] = d2;
		d2 = ::sqrt((-r.c[4] + ::sqrt(r.c[4] * r.c[4] + i.c[4] * i.c[4])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[4]);
		} else LIKELY {
			d1 = i.c[4] / (2 * d2);
		}
		r.c[4] = d1;
		i.c[4] = d2;
		d2 = ::sqrt((-r.c[5] + ::sqrt(r.c[5] * r.c[5] + i.c[5] * i.c[5])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[5]);
		} else LIKELY {
			d1 = i.c[5] / (2 * d2);
		}
		r.c[5] = d1;
		i.c[5] = d2;
		d2 = ::sqrt((-r.c[6] + ::sqrt(r.c[6] * r.c[6] + i.c[6] * i.c[6])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[6]);
		} else LIKELY {
			d1 = i.c[6] / (2 * d2);
		}
		r.c[6] = d1;
		i.c[6] = d2;
		d2 = ::sqrt((-r.c[7] + ::sqrt(r.c[7] * r.c[7] + i.c[7] * i.c[7])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[7]);
		} else LIKELY {
			d1 = i.c[7] / (2 * d2);
		}
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *sqrt(const AML_PREFIX(VectorU8_8D) mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d2 = ::sqrt((-r.c[0] + ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[0]);
			} else LIKELY {
				d1 = i.c[0] / (2 * d2);
			}
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d2 = ::sqrt((-r.c[1] + ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[1]);
			} else LIKELY {
				d1 = i.c[1] / (2 * d2);
			}
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d2 = ::sqrt((-r.c[2] + ::sqrt(r.c[2] * r.c[2] + i.c[2] * i.c[2])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[2]);
			} else LIKELY {
				d1 = i.c[2] / (2 * d2);
			}
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d2 = ::sqrt((-r.c[3] + ::sqrt(r.c[3] * r.c[3] + i.c[3] * i.c[3])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[3]);
			} else LIKELY {
				d1 = i.c[3] / (2 * d2);
			}
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d2 = ::sqrt((-r.c[4] + ::sqrt(r.c[4] * r.c[4] + i.c[4] * i.c[4])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[4]);
			} else LIKELY {
				d1 = i.c[4] / (2 * d2);
			}
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d2 = ::sqrt((-r.c[5] + ::sqrt(r.c[5] * r.c[5] + i.c[5] * i.c[5])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[5]);
			} else LIKELY {
				d1 = i.c[5] / (2 * d2);
			}
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d2 = ::sqrt((-r.c[6] + ::sqrt(r.c[6] * r.c[6] + i.c[6] * i.c[6])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[6]);
			} else LIKELY {
				d1 = i.c[6] / (2 * d2);
			}
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d2 = ::sqrt((-r.c[7] + ::sqrt(r.c[7] * r.c[7] + i.c[7] * i.c[7])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[7]);
			} else LIKELY {
				d1 = i.c[7] / (2 * d2);
			}
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *sin() {
		double d1;
		double d2;
		d1 = ::sin(r.c[0]) * ::cosh(i.c[0]);
		d2 = ::cos(i.c[0]) * ::sinh(r.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::sin(r.c[1]) * ::cosh(i.c[1]);
		d2 = ::cos(i.c[1]) * ::sinh(r.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::sin(r.c[2]) * ::cosh(i.c[2]);
		d2 = ::cos(i.c[2]) * ::sinh(r.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::sin(r.c[3]) * ::cosh(i.c[3]);
		d2 = ::cos(i.c[3]) * ::sinh(r.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = ::sin(r.c[4]) * ::cosh(i.c[4]);
		d2 = ::cos(i.c[4]) * ::sinh(r.c[4]);
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = ::sin(r.c[5]) * ::cosh(i.c[5]);
		d2 = ::cos(i.c[5]) * ::sinh(r.c[5]);
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = ::sin(r.c[6]) * ::cosh(i.c[6]);
		d2 = ::cos(i.c[6]) * ::sinh(r.c[6]);
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = ::sin(r.c[7]) * ::cosh(i.c[7]);
		d2 = ::cos(i.c[7]) * ::sinh(r.c[7]);
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *cos() {
		double d1;
		double d2;
		d1 = ::cos(r.c[0]) * ::cosh(i.c[0]);
		d2 = -::sin(i.c[0]) * ::sinh(r.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::cos(r.c[1]) * ::cosh(i.c[1]);
		d2 = -::sin(i.c[1]) * ::sinh(r.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::cos(r.c[2]) * ::cosh(i.c[2]);
		d2 = -::sin(i.c[2]) * ::sinh(r.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::cos(r.c[3]) * ::cosh(i.c[3]);
		d2 = -::sin(i.c[3]) * ::sinh(r.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = ::cos(r.c[4]) * ::cosh(i.c[4]);
		d2 = -::sin(i.c[4]) * ::sinh(r.c[4]);
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = ::cos(r.c[5]) * ::cosh(i.c[5]);
		d2 = -::sin(i.c[5]) * ::sinh(r.c[5]);
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = ::cos(r.c[6]) * ::cosh(i.c[6]);
		d2 = -::sin(i.c[6]) * ::sinh(r.c[6]);
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = ::cos(r.c[7]) * ::cosh(i.c[7]);
		d2 = -::sin(i.c[7]) * ::sinh(r.c[7]);
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *tan() {
		double d1;
		double d2;
		d1 = ::sin(r.c[0] + r.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
		d2 = ::sinh(i.c[0] + i.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::sin(r.c[1] + r.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
		d2 = ::sinh(i.c[1] + i.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::sin(r.c[2] + r.c[2]) / (::cos(r.c[2] + r.c[2]) * ::cosh(i.c[2] + i.c[2]));
		d2 = ::sinh(i.c[2] + i.c[2]) / (::cos(r.c[2] + r.c[2]) * ::cosh(i.c[2] + i.c[2]));
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::sin(r.c[3] + r.c[3]) / (::cos(r.c[3] + r.c[3]) * ::cosh(i.c[3] + i.c[3]));
		d2 = ::sinh(i.c[3] + i.c[3]) / (::cos(r.c[3] + r.c[3]) * ::cosh(i.c[3] + i.c[3]));
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = ::sin(r.c[4] + r.c[4]) / (::cos(r.c[4] + r.c[4]) * ::cosh(i.c[4] + i.c[4]));
		d2 = ::sinh(i.c[4] + i.c[4]) / (::cos(r.c[4] + r.c[4]) * ::cosh(i.c[4] + i.c[4]));
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = ::sin(r.c[5] + r.c[5]) / (::cos(r.c[5] + r.c[5]) * ::cosh(i.c[5] + i.c[5]));
		d2 = ::sinh(i.c[5] + i.c[5]) / (::cos(r.c[5] + r.c[5]) * ::cosh(i.c[5] + i.c[5]));
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = ::sin(r.c[6] + r.c[6]) / (::cos(r.c[6] + r.c[6]) * ::cosh(i.c[6] + i.c[6]));
		d2 = ::sinh(i.c[6] + i.c[6]) / (::cos(r.c[6] + r.c[6]) * ::cosh(i.c[6] + i.c[6]));
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = ::sin(r.c[7] + r.c[7]) / (::cos(r.c[7] + r.c[7]) * ::cosh(i.c[7] + i.c[7]));
		d2 = ::sinh(i.c[7] + i.c[7]) / (::cos(r.c[7] + r.c[7]) * ::cosh(i.c[7] + i.c[7]));
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *sin(const AML_PREFIX(VectorU8_8D) mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::sin(r.c[0]) * ::cosh(i.c[0]);
			d2 = ::cos(i.c[0]) * ::sinh(r.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::sin(r.c[1]) * ::cosh(i.c[1]);
			d2 = ::cos(i.c[1]) * ::sinh(r.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::sin(r.c[2]) * ::cosh(i.c[2]);
			d2 = ::cos(i.c[2]) * ::sinh(r.c[2]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::sin(r.c[3]) * ::cosh(i.c[3]);
			d2 = ::cos(i.c[3]) * ::sinh(r.c[3]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::sin(r.c[4]) * ::cosh(i.c[4]);
			d2 = ::cos(i.c[4]) * ::sinh(r.c[4]);
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::sin(r.c[5]) * ::cosh(i.c[5]);
			d2 = ::cos(i.c[5]) * ::sinh(r.c[5]);
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::sin(r.c[6]) * ::cosh(i.c[6]);
			d2 = ::cos(i.c[6]) * ::sinh(r.c[6]);
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::sin(r.c[7]) * ::cosh(i.c[7]);
			d2 = ::cos(i.c[7]) * ::sinh(r.c[7]);
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *cos(const AML_PREFIX(VectorU8_8D) mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::cos(r.c[0]) * ::cosh(i.c[0]);
			d2 = -::sin(i.c[0]) * ::sinh(r.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::cos(r.c[1]) * ::cosh(i.c[1]);
			d2 = -::sin(i.c[1]) * ::sinh(r.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::cos(r.c[2]) * ::cosh(i.c[2]);
			d2 = -::sin(i.c[2]) * ::sinh(r.c[2]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::cos(r.c[3]) * ::cosh(i.c[3]);
			d2 = -::sin(i.c[3]) * ::sinh(r.c[3]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::cos(r.c[4]) * ::cosh(i.c[4]);
			d2 = -::sin(i.c[4]) * ::sinh(r.c[4]);
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::cos(r.c[5]) * ::cosh(i.c[5]);
			d2 = -::sin(i.c[5]) * ::sinh(r.c[5]);
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::cos(r.c[6]) * ::cosh(i.c[6]);
			d2 = -::sin(i.c[6]) * ::sinh(r.c[6]);
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::cos(r.c[7]) * ::cosh(i.c[7]);
			d2 = -::sin(i.c[7]) * ::sinh(r.c[7]);
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *tan(const AML_PREFIX(VectorU8_8D) mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::sin(r.c[0] + r.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
			d2 = ::sinh(i.c[0] + i.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::sin(r.c[1] + r.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
			d2 = ::sinh(i.c[1] + i.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::sin(r.c[2] + r.c[2]) / (::cos(r.c[2] + r.c[2]) * ::cosh(i.c[2] + i.c[2]));
			d2 = ::sinh(i.c[2] + i.c[2]) / (::cos(r.c[2] + r.c[2]) * ::cosh(i.c[2] + i.c[2]));
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::sin(r.c[3] + r.c[3]) / (::cos(r.c[3] + r.c[3]) * ::cosh(i.c[3] + i.c[3]));
			d2 = ::sinh(i.c[3] + i.c[3]) / (::cos(r.c[3] + r.c[3]) * ::cosh(i.c[3] + i.c[3]));
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::sin(r.c[4] + r.c[4]) / (::cos(r.c[4] + r.c[4]) * ::cosh(i.c[4] + i.c[4]));
			d2 = ::sinh(i.c[4] + i.c[4]) / (::cos(r.c[4] + r.c[4]) * ::cosh(i.c[4] + i.c[4]));
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::sin(r.c[5] + r.c[5]) / (::cos(r.c[5] + r.c[5]) * ::cosh(i.c[5] + i.c[5]));
			d2 = ::sinh(i.c[5] + i.c[5]) / (::cos(r.c[5] + r.c[5]) * ::cosh(i.c[5] + i.c[5]));
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::sin(r.c[6] + r.c[6]) / (::cos(r.c[6] + r.c[6]) * ::cosh(i.c[6] + i.c[6]));
			d2 = ::sinh(i.c[6] + i.c[6]) / (::cos(r.c[6] + r.c[6]) * ::cosh(i.c[6] + i.c[6]));
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::sin(r.c[7] + r.c[7]) / (::cos(r.c[7] + r.c[7]) * ::cosh(i.c[7] + i.c[7]));
			d2 = ::sinh(i.c[7] + i.c[7]) / (::cos(r.c[7] + r.c[7]) * ::cosh(i.c[7] + i.c[7]));
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array8Complex64) *exp() {
		double d1 = ::exp(r.c[0]) * ::cos(i.c[0]);
		double d2 = ::exp(r.c[0]) * ::sin(i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::exp(r.c[1]) * ::cos(i.c[1]);
		d2 = ::exp(r.c[1]) * ::sin(i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::exp(r.c[2]) * ::cos(i.c[2]);
		d2 = ::exp(r.c[2]) * ::sin(i.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::exp(r.c[3]) * ::cos(i.c[3]);
		d2 = ::exp(r.c[3]) * ::sin(i.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = ::exp(r.c[4]) * ::cos(i.c[4]);
		d2 = ::exp(r.c[4]) * ::sin(i.c[4]);
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = ::exp(r.c[5]) * ::cos(i.c[5]);
		d2 = ::exp(r.c[5]) * ::sin(i.c[5]);
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = ::exp(r.c[6]) * ::cos(i.c[6]);
		d2 = ::exp(r.c[6]) * ::sin(i.c[6]);
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = ::exp(r.c[7]) * ::cos(i.c[7]);
		d2 = ::exp(r.c[7]) * ::sin(i.c[7]);
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *exp(double n) {
		double d1 = ::pow(n, r.c[0]) * ::cos(i.c[0] * ::log(n));
		double d2 = ::pow(n, r.c[0]) * ::sin(i.c[0] * ::log(n));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::pow(n, r.c[1]) * ::cos(i.c[1] * ::log(n));
		d2 = ::pow(n, r.c[1]) * ::sin(i.c[1] * ::log(n));
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::pow(n, r.c[2]) * ::cos(i.c[2] * ::log(n));
		d2 = ::pow(n, r.c[2]) * ::sin(i.c[2] * ::log(n));
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::pow(n, r.c[3]) * ::cos(i.c[3] * ::log(n));
		d2 = ::pow(n, r.c[3]) * ::sin(i.c[3] * ::log(n));
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = ::pow(n, r.c[4]) * ::cos(i.c[4] * ::log(n));
		d2 = ::pow(n, r.c[4]) * ::sin(i.c[4] * ::log(n));
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = ::pow(n, r.c[5]) * ::cos(i.c[5] * ::log(n));
		d2 = ::pow(n, r.c[5]) * ::sin(i.c[5] * ::log(n));
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = ::pow(n, r.c[6]) * ::cos(i.c[6] * ::log(n));
		d2 = ::pow(n, r.c[6]) * ::sin(i.c[6] * ::log(n));
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = ::pow(n, r.c[7]) * ::cos(i.c[7] * ::log(n));
		d2 = ::pow(n, r.c[7]) * ::sin(i.c[7] * ::log(n));
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *pow(const AML_PREFIX(Array8Complex64) &n) {
		double d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		double d2 = ::atan2(r.c[0], i.c[0]);
		double d3 = ::exp(d1 * n.i.c[0] - d2 * n.r.c[0]);
		double d4 = d1 * n.r.c[0] + d2 * n.i.c[0];
		double d5 = d3 * ::cos(d4);
		double d6 = d3 * ::sin(d4);
		i.c[0] = d5;
		r.c[0] = d6;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		d3 = ::exp(d1 * n.i.c[1] - d2 * n.r.c[1]);
		d4 = d1 * n.r.c[1] + d2 * n.i.c[1];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[1] = d5;
		r.c[1] = d6;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
		d2 = ::atan2(r.c[2], i.c[2]);
		d3 = ::exp(d1 * n.i.c[2] - d2 * n.r.c[2]);
		d4 = d1 * n.r.c[2] + d2 * n.i.c[2];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[2] = d5;
		r.c[2] = d6;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
		d2 = ::atan2(r.c[3], i.c[3]);
		d3 = ::exp(d1 * n.i.c[3] - d2 * n.r.c[3]);
		d4 = d1 * n.r.c[3] + d2 * n.i.c[3];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[3] = d5;
		r.c[3] = d6;
		d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / 2;
		d2 = ::atan2(r.c[4], i.c[4]);
		d3 = ::exp(d1 * n.i.c[4] - d2 * n.r.c[4]);
		d4 = d1 * n.r.c[4] + d2 * n.i.c[4];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[4] = d5;
		r.c[4] = d6;
		d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / 2;
		d2 = ::atan2(r.c[5], i.c[5]);
		d3 = ::exp(d1 * n.i.c[5] - d2 * n.r.c[5]);
		d4 = d1 * n.r.c[5] + d2 * n.i.c[5];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[5] = d5;
		r.c[5] = d6;
		d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / 2;
		d2 = ::atan2(r.c[6], i.c[6]);
		d3 = ::exp(d1 * n.i.c[6] - d2 * n.r.c[6]);
		d4 = d1 * n.r.c[6] + d2 * n.i.c[6];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[6] = d5;
		r.c[6] = d6;
		d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / 2;
		d2 = ::atan2(r.c[7], i.c[7]);
		d3 = ::exp(d1 * n.i.c[7] - d2 * n.r.c[7]);
		d4 = d1 * n.r.c[7] + d2 * n.i.c[7];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[7] = d5;
		r.c[7] = d6;
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array8Complex64) *pow(const AML_PREFIX(Complex64) n) {
		double d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		double d2 = ::atan2(r.c[0], i.c[0]);
		double d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		double d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		double d5 = d3 * ::cos(d4);
		double d6 = d3 * ::sin(d4);
		i.c[0] = d5;
		r.c[0] = d6;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[1] = d5;
		r.c[1] = d6;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
		d2 = ::atan2(r.c[2], i.c[2]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[2] = d5;
		r.c[2] = d6;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
		d2 = ::atan2(r.c[3], i.c[3]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[3] = d5;
		r.c[3] = d6;
		d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / 2;
		d2 = ::atan2(r.c[4], i.c[4]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[4] = d5;
		r.c[4] = d6;
		d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / 2;
		d2 = ::atan2(r.c[5], i.c[5]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[5] = d5;
		r.c[5] = d6;
		d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / 2;
		d2 = ::atan2(r.c[6], i.c[6]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[6] = d5;
		r.c[6] = d6;
		d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / 2;
		d2 = ::atan2(r.c[7], i.c[7]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[7] = d5;
		r.c[7] = d6;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *
	pow(const AML_PREFIX(Array8Complex64) &n, const AML_PREFIX(VectorU8_8D) &mask) {
		double d1;
		double d2;
		double d3;
		double d4;
		double d5;
		double d6;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			d3 = ::exp(d1 * n.i.c[0] - d2 * n.r.c[0]);
			d4 = d1 * n.r.c[0] + d2 * n.i.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[0] = d5;
			r.c[0] = d6;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			d3 = ::exp(d1 * n.i.c[1] - d2 * n.r.c[1]);
			d4 = d1 * n.r.c[1] + d2 * n.i.c[1];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[1] = d5;
			r.c[1] = d6;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
			d2 = ::atan2(r.c[2], i.c[2]);
			d3 = ::exp(d1 * n.i.c[2] - d2 * n.r.c[2]);
			d4 = d1 * n.r.c[2] + d2 * n.i.c[2];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[2] = d5;
			r.c[2] = d6;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
			d2 = ::atan2(r.c[3], i.c[3]);
			d3 = ::exp(d1 * n.i.c[3] - d2 * n.r.c[3]);
			d4 = d1 * n.r.c[3] + d2 * n.i.c[3];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[3] = d5;
			r.c[3] = d6;
		}
		if (mask.v.c[4]) {
			d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / 2;
			d2 = ::atan2(r.c[4], i.c[4]);
			d3 = ::exp(d1 * n.i.c[4] - d2 * n.r.c[4]);
			d4 = d1 * n.r.c[4] + d2 * n.i.c[4];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[4] = d5;
			r.c[4] = d6;
		}
		if (mask.v.c[5]) {
			d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / 2;
			d2 = ::atan2(r.c[5], i.c[5]);
			d3 = ::exp(d1 * n.i.c[5] - d2 * n.r.c[5]);
			d4 = d1 * n.r.c[5] + d2 * n.i.c[5];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[5] = d5;
			r.c[5] = d6;
		}
		if (mask.v.c[6]) {
			d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / 2;
			d2 = ::atan2(r.c[6], i.c[6]);
			d3 = ::exp(d1 * n.i.c[6] - d2 * n.r.c[6]);
			d4 = d1 * n.r.c[6] + d2 * n.i.c[6];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[6] = d5;
			r.c[6] = d6;
		}
		if (mask.v.c[7]) {
			d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / 2;
			d2 = ::atan2(r.c[7], i.c[7]);
			d3 = ::exp(d1 * n.i.c[7] - d2 * n.r.c[7]);
			d4 = d1 * n.r.c[7] + d2 * n.i.c[7];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[7] = d5;
			r.c[7] = d6;
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array8Complex64) *pow(const AML_PREFIX(Complex64) n, const AML_PREFIX(VectorU8_8D) &mask) {
		double d1;
		double d2;
		double d3;
		double d4;
		double d5;
		double d6;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[0] = d5;
			r.c[0] = d6;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[1] = d5;
			r.c[1] = d6;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
			d2 = ::atan2(r.c[2], i.c[2]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[2] = d5;
			r.c[2] = d6;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
			d2 = ::atan2(r.c[3], i.c[3]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[3] = d5;
			r.c[3] = d6;
		}
		if (mask.v.c[4]) {
			d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / 2;
			d2 = ::atan2(r.c[4], i.c[4]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[4] = d5;
			r.c[4] = d6;
		}
		if (mask.v.c[5]) {
			d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / 2;
			d2 = ::atan2(r.c[5], i.c[5]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[5] = d5;
			r.c[5] = d6;
		}
		if (mask.v.c[6]) {
			d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / 2;
			d2 = ::atan2(r.c[6], i.c[6]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[6] = d5;
			r.c[6] = d6;
		}
		if (mask.v.c[7]) {
			d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / 2;
			d2 = ::atan2(r.c[7], i.c[7]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[7] = d5;
			r.c[7] = d6;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *exp(const AML_PREFIX(VectorU8_8D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::exp(r.c[0]) * ::cos(i.c[0]);
			d2 = ::exp(r.c[0]) * ::sin(i.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::exp(r.c[1]) * ::cos(i.c[1]);
			d2 = ::exp(r.c[1]) * ::sin(i.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::exp(r.c[2]) * ::cos(i.c[2]);
			d2 = ::exp(r.c[2]) * ::sin(i.c[2]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::exp(r.c[3]) * ::cos(i.c[3]);
			d2 = ::exp(r.c[3]) * ::sin(i.c[3]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::exp(r.c[4]) * ::cos(i.c[4]);
			d2 = ::exp(r.c[4]) * ::sin(i.c[4]);
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::exp(r.c[5]) * ::cos(i.c[5]);
			d2 = ::exp(r.c[5]) * ::sin(i.c[5]);
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::exp(r.c[6]) * ::cos(i.c[6]);
			d2 = ::exp(r.c[6]) * ::sin(i.c[6]);
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::exp(r.c[7]) * ::cos(i.c[7]);
			d2 = ::exp(r.c[7]) * ::sin(i.c[7]);
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *exp(double n, const AML_PREFIX(VectorU8_8D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::pow(n, r.c[0]) * ::cos(i.c[0] * ::log(n));
			d2 = ::pow(n, r.c[0]) * ::sin(i.c[0] * ::log(n));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::pow(n, r.c[1]) * ::cos(i.c[1] * ::log(n));
			d2 = ::pow(n, r.c[1]) * ::sin(i.c[1] * ::log(n));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::pow(n, r.c[2]) * ::cos(i.c[2] * ::log(n));
			d2 = ::pow(n, r.c[2]) * ::sin(i.c[2] * ::log(n));
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::pow(n, r.c[3]) * ::cos(i.c[3] * ::log(n));
			d2 = ::pow(n, r.c[3]) * ::sin(i.c[3] * ::log(n));
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::pow(n, r.c[4]) * ::cos(i.c[4] * ::log(n));
			d2 = ::pow(n, r.c[4]) * ::sin(i.c[4] * ::log(n));
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::pow(n, r.c[5]) * ::cos(i.c[5] * ::log(n));
			d2 = ::pow(n, r.c[5]) * ::sin(i.c[5] * ::log(n));
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::pow(n, r.c[6]) * ::cos(i.c[6] * ::log(n));
			d2 = ::pow(n, r.c[6]) * ::sin(i.c[6] * ::log(n));
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::pow(n, r.c[7]) * ::cos(i.c[7] * ::log(n));
			d2 = ::pow(n, r.c[7]) * ::sin(i.c[7] * ::log(n));
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *pow(double n) {
		double d1;
		double d2;
		d1 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::cos(n * atan2(i.c[0], r.c[0]));
		d2 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::sin(n * atan2(i.c[0], r.c[0]));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::cos(n * atan2(i.c[1], r.c[1]));
		d2 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::sin(n * atan2(i.c[1], r.c[1]));
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::pow(r.c[2] * r.c[2] + i.c[2] * i.c[2], n / 2) * ::cos(n * atan2(i.c[2], r.c[2]));
		d2 = ::pow(r.c[2] * r.c[2] + i.c[2] * i.c[2], n / 2) * ::sin(n * atan2(i.c[2], r.c[2]));
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::pow(r.c[3] * r.c[3] + i.c[3] * i.c[3], n / 2) * ::cos(n * atan2(i.c[3], r.c[3]));
		d2 = ::pow(r.c[3] * r.c[3] + i.c[3] * i.c[3], n / 2) * ::sin(n * atan2(i.c[3], r.c[3]));
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = ::pow(r.c[4] * r.c[4] + i.c[4] * i.c[4], n / 2) * ::cos(n * atan2(i.c[4], r.c[4]));
		d2 = ::pow(r.c[4] * r.c[4] + i.c[4] * i.c[4], n / 2) * ::sin(n * atan2(i.c[4], r.c[4]));
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = ::pow(r.c[5] * r.c[5] + i.c[5] * i.c[5], n / 2) * ::cos(n * atan2(i.c[5], r.c[5]));
		d2 = ::pow(r.c[5] * r.c[5] + i.c[5] * i.c[5], n / 2) * ::sin(n * atan2(i.c[5], r.c[5]));
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = ::pow(r.c[6] * r.c[6] + i.c[6] * i.c[6], n / 2) * ::cos(n * atan2(i.c[6], r.c[6]));
		d2 = ::pow(r.c[6] * r.c[6] + i.c[6] * i.c[6], n / 2) * ::sin(n * atan2(i.c[6], r.c[6]));
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = ::pow(r.c[7] * r.c[7] + i.c[7] * i.c[7], n / 2) * ::cos(n * atan2(i.c[7], r.c[7]));
		d2 = ::pow(r.c[7] * r.c[7] + i.c[7] * i.c[7], n / 2) * ::sin(n * atan2(i.c[7], r.c[7]));
		r.c[7] = d1;
		i.c[7] = d2;

		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *pow(double n, const AML_PREFIX(VectorU8_8D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::cos(n * atan2(i.c[0], r.c[0]));
			d2 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::sin(n * atan2(i.c[0], r.c[0]));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::cos(n * atan2(i.c[1], r.c[1]));
			d2 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::sin(n * atan2(i.c[1], r.c[1]));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::pow(r.c[2] * r.c[2] + i.c[2] * i.c[2], n / 2) * ::cos(n * atan2(i.c[2], r.c[2]));
			d2 = ::pow(r.c[2] * r.c[2] + i.c[2] * i.c[2], n / 2) * ::sin(n * atan2(i.c[2], r.c[2]));
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::pow(r.c[3] * r.c[3] + i.c[3] * i.c[3], n / 2) * ::cos(n * atan2(i.c[3], r.c[3]));
			d2 = ::pow(r.c[3] * r.c[3] + i.c[3] * i.c[3], n / 2) * ::sin(n * atan2(i.c[3], r.c[3]));
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::pow(r.c[4] * r.c[4] + i.c[4] * i.c[4], n / 2) * ::cos(n * atan2(i.c[4], r.c[4]));
			d2 = ::pow(r.c[4] * r.c[4] + i.c[4] * i.c[4], n / 2) * ::sin(n * atan2(i.c[4], r.c[4]));
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::pow(r.c[5] * r.c[5] + i.c[5] * i.c[5], n / 2) * ::cos(n * atan2(i.c[5], r.c[5]));
			d2 = ::pow(r.c[5] * r.c[5] + i.c[5] * i.c[5], n / 2) * ::sin(n * atan2(i.c[5], r.c[5]));
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::pow(r.c[6] * r.c[6] + i.c[6] * i.c[6], n / 2) * ::cos(n * atan2(i.c[6], r.c[6]));
			d2 = ::pow(r.c[6] * r.c[6] + i.c[6] * i.c[6], n / 2) * ::sin(n * atan2(i.c[6], r.c[6]));
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::pow(r.c[7] * r.c[7] + i.c[7] * i.c[7], n / 2) * ::cos(n * atan2(i.c[7], r.c[7]));
			d2 = ::pow(r.c[7] * r.c[7] + i.c[7] * i.c[7], n / 2) * ::sin(n * atan2(i.c[7], r.c[7]));
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorDouble8D) abs() {
		AML_PREFIX(VectorDouble8D) ret;
		ret.v.c[0] = ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0]);
		ret.v.c[1] = ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1]);
		ret.v.c[2] = ::sqrt(r.c[2] * r.c[2] + i.c[2] * i.c[2]);
		ret.v.c[3] = ::sqrt(r.c[3] * r.c[3] + i.c[3] * i.c[3]);
		ret.v.c[4] = ::sqrt(r.c[4] * r.c[4] + i.c[4] * i.c[4]);
		ret.v.c[5] = ::sqrt(r.c[5] * r.c[5] + i.c[5] * i.c[5]);
		ret.v.c[6] = ::sqrt(r.c[6] * r.c[6] + i.c[6] * i.c[6]);
		ret.v.c[7] = ::sqrt(r.c[7] * r.c[7] + i.c[7] * i.c[7]);
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_8D) abs_gt(double a) {
		AML_PREFIX(VectorU8_8D) ret;
		ret.v.c[0] = a * a < r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a < r.c[1] * r.c[1] + i.c[1] * i.c[1];
		ret.v.c[2] = a * a < r.c[2] * r.c[2] + i.c[2] * i.c[2];
		ret.v.c[3] = a * a < r.c[3] * r.c[3] + i.c[3] * i.c[3];
		ret.v.c[4] = a * a < r.c[4] * r.c[4] + i.c[4] * i.c[4];
		ret.v.c[5] = a * a < r.c[5] * r.c[5] + i.c[5] * i.c[5];
		ret.v.c[6] = a * a < r.c[6] * r.c[6] + i.c[6] * i.c[6];
		ret.v.c[7] = a * a < r.c[7] * r.c[7] + i.c[7] * i.c[7];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_8D) abs_lt(double a) {
		AML_PREFIX(VectorU8_8D) ret;
		ret.v.c[0] = a * a > r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a > r.c[1] * r.c[1] + i.c[1] * i.c[1];
		ret.v.c[2] = a * a > r.c[2] * r.c[2] + i.c[2] * i.c[2];
		ret.v.c[3] = a * a > r.c[3] * r.c[3] + i.c[3] * i.c[3];
		ret.v.c[4] = a * a > r.c[4] * r.c[4] + i.c[4] * i.c[4];
		ret.v.c[5] = a * a > r.c[5] * r.c[5] + i.c[5] * i.c[5];
		ret.v.c[6] = a * a > r.c[6] * r.c[6] + i.c[6] * i.c[6];
		ret.v.c[7] = a * a > r.c[7] * r.c[7] + i.c[7] * i.c[7];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_8D) abs_eq(double a) {
		AML_PREFIX(VectorU8_8D) ret;
		ret.v.c[0] = a * a == r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a == r.c[1] * r.c[1] + i.c[1] * i.c[1];
		ret.v.c[2] = a * a == r.c[2] * r.c[2] + i.c[2] * i.c[2];
		ret.v.c[3] = a * a == r.c[3] * r.c[3] + i.c[3] * i.c[3];
		ret.v.c[4] = a * a == r.c[4] * r.c[4] + i.c[4] * i.c[4];
		ret.v.c[5] = a * a == r.c[5] * r.c[5] + i.c[5] * i.c[5];
		ret.v.c[6] = a * a == r.c[6] * r.c[6] + i.c[6] * i.c[6];
		ret.v.c[7] = a * a == r.c[7] * r.c[7] + i.c[7] * i.c[7];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *ln(const AML_PREFIX(VectorU8_8D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			i.c[1] = d1;
			r.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
			d2 = ::atan2(r.c[2], i.c[2]);
			i.c[2] = d1;
			r.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
			d2 = ::atan2(r.c[3], i.c[3]);
			i.c[3] = d1;
			r.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / 2;
			d2 = ::atan2(r.c[4], i.c[4]);
			i.c[4] = d1;
			r.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / 2;
			d2 = ::atan2(r.c[5], i.c[5]);
			i.c[5] = d1;
			r.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / 2;
			d2 = ::atan2(r.c[6], i.c[6]);
			i.c[6] = d1;
			r.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / 2;
			d2 = ::atan2(r.c[7], i.c[7]);
			i.c[7] = d1;
			r.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *log(const AML_PREFIX(VectorU8_8D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			i.c[1] = d1;
			r.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
			d2 = ::atan2(r.c[2], i.c[2]);
			i.c[2] = d1;
			r.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
			d2 = ::atan2(r.c[3], i.c[3]);
			i.c[3] = d1;
			r.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / 2;
			d2 = ::atan2(r.c[4], i.c[4]);
			i.c[4] = d1;
			r.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / 2;
			d2 = ::atan2(r.c[5], i.c[5]);
			i.c[5] = d1;
			r.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / 2;
			d2 = ::atan2(r.c[6], i.c[6]);
			i.c[6] = d1;
			r.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / 2;
			d2 = ::atan2(r.c[7], i.c[7]);
			i.c[7] = d1;
			r.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *log10(const AML_PREFIX(VectorU8_8D) &mask) {
		double d1;
		double d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[0], i.c[0]) / AML_LN10;
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[1], i.c[1]) / AML_LN10;
			i.c[1] = d1;
			r.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[2], i.c[2]) / AML_LN10;
			i.c[2] = d1;
			r.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[3], i.c[3]) / AML_LN10;
			i.c[3] = d1;
			r.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[4], i.c[4]) / AML_LN10;
			i.c[4] = d1;
			r.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[5], i.c[5]) / AML_LN10;
			i.c[5] = d1;
			r.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[6], i.c[6]) / AML_LN10;
			i.c[6] = d1;
			r.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[7], i.c[7]) / AML_LN10;
			i.c[7] = d1;
			r.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *ln() {
		double d1;
		double d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		d2 = ::atan2(r.c[0], i.c[0]);
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		i.c[1] = d1;
		r.c[1] = d2;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
		d2 = ::atan2(r.c[2], i.c[2]);
		i.c[2] = d1;
		r.c[2] = d2;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
		d2 = ::atan2(r.c[3], i.c[3]);
		i.c[3] = d1;
		r.c[3] = d2;
		d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / 2;
		d2 = ::atan2(r.c[4], i.c[4]);
		i.c[4] = d1;
		r.c[4] = d2;
		d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / 2;
		d2 = ::atan2(r.c[5], i.c[5]);
		i.c[5] = d1;
		r.c[5] = d2;
		d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / 2;
		d2 = ::atan2(r.c[6], i.c[6]);
		i.c[6] = d1;
		r.c[6] = d2;
		d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / 2;
		d2 = ::atan2(r.c[7], i.c[7]);
		i.c[7] = d1;
		r.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *log() {
		double d1;
		double d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		d2 = ::atan2(r.c[0], i.c[0]);
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		i.c[1] = d1;
		r.c[1] = d2;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
		d2 = ::atan2(r.c[2], i.c[2]);
		i.c[2] = d1;
		r.c[2] = d2;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
		d2 = ::atan2(r.c[3], i.c[3]);
		i.c[3] = d1;
		r.c[3] = d2;
		d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / 2;
		d2 = ::atan2(r.c[4], i.c[4]);
		i.c[4] = d1;
		r.c[4] = d2;
		d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / 2;
		d2 = ::atan2(r.c[5], i.c[5]);
		i.c[5] = d1;
		r.c[5] = d2;
		d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / 2;
		d2 = ::atan2(r.c[6], i.c[6]);
		i.c[6] = d1;
		r.c[6] = d2;
		d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / 2;
		d2 = ::atan2(r.c[7], i.c[7]);
		i.c[7] = d1;
		r.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex64) *log10() {
		double d1;
		double d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[0], i.c[0]) / AML_LN10;
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[1], i.c[1]) / AML_LN10;
		i.c[1] = d1;
		r.c[1] = d2;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[2], i.c[2]) / AML_LN10;
		i.c[2] = d1;
		r.c[2] = d2;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[3], i.c[3]) / AML_LN10;
		i.c[3] = d1;
		r.c[3] = d2;
		d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[4], i.c[4]) / AML_LN10;
		i.c[4] = d1;
		r.c[4] = d2;
		d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[5], i.c[5]) / AML_LN10;
		i.c[5] = d1;
		r.c[5] = d2;
		d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[6], i.c[6]) / AML_LN10;
		i.c[6] = d1;
		r.c[6] = d2;
		d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[7], i.c[7]) / AML_LN10;
		i.c[7] = d1;
		r.c[7] = d2;
		return this;
	}

	class AML_PREFIX(Complex64_8_Itr) : public std::iterator<
			std::input_iterator_tag,   // iterator_category
			AML_PREFIX(Complex64Ptr),                      // value_type
			long,                      // difference_type
			const AML_PREFIX(Complex64Ptr) *,               // pointer
			AML_PREFIX(Complex64Ptr)                       // reference
	> {

		AML_PREFIX(Array8Complex64) *a;
		int position;

	public:
		AML_FUNCTION explicit AML_PREFIX(Complex64_8_Itr)(AML_PREFIX(Array8Complex64) *array, int length) : a(array),
																											position(
																													length) {

		}

		AML_FUNCTION AML_PREFIX(Complex64_8_Itr) &operator++() {
			position++;
			return *this;
		}

		AML_FUNCTION bool operator==(const AML_PREFIX(Complex64_8_Itr) other) const {
			return position == other.position;
		}

		AML_FUNCTION bool operator!=(const AML_PREFIX(Complex64_8_Itr) other) const { return !(*this == other); }

		AML_FUNCTION reference operator*() const {
			return AML_PREFIX(Complex64Ptr)(&a->r.c[position], &a->i.c[position], position);
		}


	};

	AML_FUNCTION AML_PREFIX(Complex64_8_Itr) begin() {
		return AML_PREFIX(Complex64_8_Itr)(this, 0);
	}

	AML_FUNCTION AML_PREFIX(Complex64_8_Itr) end() {
		return AML_PREFIX(Complex64_8_Itr)(this, 8);
	}

};


#if !defined(AML_NO_STRING)


AML_FUNCTION std::string operator<<(std::string &lhs, const AML_PREFIX(Array8Complex64) &rhs) {
	std::ostringstream string;
	string << lhs << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1]
		   << "i ,  " << rhs.r.c[2] << " + " << rhs.i.c[2] << "i ,  " << rhs.r.c[3] << " + " << rhs.i.c[3] << "i ,  "
		   << rhs.r.c[4] << " + " << rhs.i.c[4] << "i ,  " << rhs.r.c[5] << " + " << rhs.i.c[5] << "i ,  " << rhs.r.c[6]
		   << " + " << rhs.i.c[6] << "i ,  " << rhs.r.c[7] << " + " << rhs.i.c[7] << "i }";
	return string.str();
}

AML_FUNCTION std::string operator<<(const char *lhs, const AML_PREFIX(Array8Complex64) &rhs) {
	std::ostringstream string;
	string << lhs << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1]
		   << "i ,  " << rhs.r.c[2] << " + " << rhs.i.c[2] << "i ,  " << rhs.r.c[3] << " + " << rhs.i.c[3] << "i ,  "
		   << rhs.r.c[4] << " + " << rhs.i.c[4] << "i ,  " << rhs.r.c[5] << " + " << rhs.i.c[5] << "i ,  " << rhs.r.c[6]
		   << " + " << rhs.i.c[6] << "i ,  " << rhs.r.c[7] << " + " << rhs.i.c[7] << "i }";
	return string.str();
}

template<class charT, class traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &o, const AML_PREFIX(Array8Complex64) &rhs) {
	std::basic_ostringstream<charT, traits> s;
	s.flags(o.flags());
	s.imbue(o.getloc());
	s.precision(o.precision());
	s << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1]
	  << "i ,  " << rhs.r.c[2] << " + " << rhs.i.c[2] << "i ,  " << rhs.r.c[3] << " + " << rhs.i.c[3] << "i ,  "
	  << rhs.r.c[4] << " + " << rhs.i.c[4] << "i ,  " << rhs.r.c[5] << " + " << rhs.i.c[5] << "i ,  " << rhs.r.c[6]
	  << " + " << rhs.i.c[6] << "i ,  " << rhs.r.c[7] << " + " << rhs.i.c[7] << "i }";
	return o << s.str();
}

#endif

AML_FUNCTION AML_PREFIX(Array8Complex64)
operator+(const AML_PREFIX(Complex64) &lhs, const AML_PREFIX(Array8Complex64) &rhs) {
	return rhs + lhs;
}

AML_FUNCTION AML_PREFIX(Array8Complex64)
operator-(const AML_PREFIX(Complex64) &lhs, const AML_PREFIX(Array8Complex64) &rhs) {
	AML_PREFIX(Array8Complex64) ret;
	ret.i.c[0] = lhs.c.c[1] - rhs.i.c[0];
	ret.i.c[1] = lhs.c.c[1] - rhs.i.c[1];
	ret.i.c[2] = lhs.c.c[1] - rhs.i.c[2];
	ret.i.c[3] = lhs.c.c[1] - rhs.i.c[3];
	ret.i.c[4] = lhs.c.c[1] - rhs.i.c[4];
	ret.i.c[5] = lhs.c.c[1] - rhs.i.c[5];
	ret.i.c[6] = lhs.c.c[1] - rhs.i.c[6];
	ret.i.c[7] = lhs.c.c[1] - rhs.i.c[7];
	ret.r.c[0] = lhs.c.c[0] - rhs.r.c[0];
	ret.r.c[1] = lhs.c.c[0] - rhs.r.c[1];
	ret.r.c[2] = lhs.c.c[0] - rhs.r.c[2];
	ret.r.c[3] = lhs.c.c[0] - rhs.r.c[3];
	ret.r.c[4] = lhs.c.c[0] - rhs.r.c[4];
	ret.r.c[5] = lhs.c.c[0] - rhs.r.c[5];
	ret.r.c[6] = lhs.c.c[0] - rhs.r.c[6];
	ret.r.c[7] = lhs.c.c[0] - rhs.r.c[7];
	return ret;
}

AML_FUNCTION AML_PREFIX(Array8Complex64)
operator*(const AML_PREFIX(Complex64) &lhs, const AML_PREFIX(Array8Complex64) &rhs) {
	return rhs * lhs;
}

AML_FUNCTION AML_PREFIX(Array8Complex64)
operator/(const AML_PREFIX(Complex64) &lhs, const AML_PREFIX(Array8Complex64) &rhs) {
	AML_PREFIX(Array8Complex64) ret;
	double d1 =
			(lhs.c.c[0] * rhs.r.c[0] + lhs.c.c[1] * rhs.i.c[0]) / (rhs.r.c[0] * rhs.r.c[0] + rhs.i.c[0] * rhs.i.c[0]);
	double d2 =
			(lhs.c.c[1] * rhs.r.c[0] - lhs.c.c[0] * rhs.i.c[0]) / (rhs.r.c[0] * rhs.r.c[0] + rhs.i.c[0] * rhs.i.c[0]);
	ret.r.c[0] = d1;
	ret.i.c[0] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[1] + lhs.c.c[1] * rhs.i.c[1]) / (rhs.r.c[1] * rhs.r.c[1] + rhs.i.c[1] * rhs.i.c[1]);
	d2 = (lhs.c.c[1] * rhs.r.c[1] - lhs.c.c[0] * rhs.i.c[1]) / (rhs.r.c[1] * rhs.r.c[1] + rhs.i.c[1] * rhs.i.c[1]);
	ret.r.c[1] = d1;
	ret.i.c[1] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[2] + lhs.c.c[1] * rhs.i.c[2]) / (rhs.r.c[2] * rhs.r.c[2] + rhs.i.c[2] * rhs.i.c[2]);
	d2 = (lhs.c.c[1] * rhs.r.c[2] - lhs.c.c[0] * rhs.i.c[2]) / (rhs.r.c[2] * rhs.r.c[2] + rhs.i.c[2] * rhs.i.c[2]);
	ret.r.c[2] = d1;
	ret.i.c[2] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[3] + lhs.c.c[1] * rhs.i.c[3]) / (rhs.r.c[3] * rhs.r.c[3] + rhs.i.c[3] * rhs.i.c[3]);
	d2 = (lhs.c.c[1] * rhs.r.c[3] - lhs.c.c[0] * rhs.i.c[3]) / (rhs.r.c[3] * rhs.r.c[3] + rhs.i.c[3] * rhs.i.c[3]);
	ret.r.c[3] = d1;
	ret.i.c[3] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[4] + lhs.c.c[1] * rhs.i.c[4]) / (rhs.r.c[4] * rhs.r.c[4] + rhs.i.c[4] * rhs.i.c[4]);
	d2 = (lhs.c.c[1] * rhs.r.c[4] - lhs.c.c[0] * rhs.i.c[4]) / (rhs.r.c[4] * rhs.r.c[4] + rhs.i.c[4] * rhs.i.c[4]);
	ret.r.c[4] = d1;
	ret.i.c[4] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[5] + lhs.c.c[1] * rhs.i.c[5]) / (rhs.r.c[5] * rhs.r.c[5] + rhs.i.c[5] * rhs.i.c[5]);
	d2 = (lhs.c.c[1] * rhs.r.c[5] - lhs.c.c[0] * rhs.i.c[5]) / (rhs.r.c[5] * rhs.r.c[5] + rhs.i.c[5] * rhs.i.c[5]);
	ret.r.c[5] = d1;
	ret.i.c[5] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[6] + lhs.c.c[1] * rhs.i.c[6]) / (rhs.r.c[6] * rhs.r.c[6] + rhs.i.c[6] * rhs.i.c[6]);
	d2 = (lhs.c.c[1] * rhs.r.c[6] - lhs.c.c[0] * rhs.i.c[6]) / (rhs.r.c[6] * rhs.r.c[6] + rhs.i.c[6] * rhs.i.c[6]);
	ret.r.c[6] = d1;
	ret.i.c[6] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[7] + lhs.c.c[1] * rhs.i.c[7]) / (rhs.r.c[7] * rhs.r.c[7] + rhs.i.c[7] * rhs.i.c[7]);
	d2 = (lhs.c.c[1] * rhs.r.c[7] - lhs.c.c[0] * rhs.i.c[7]) / (rhs.r.c[7] * rhs.r.c[7] + rhs.i.c[7] * rhs.i.c[7]);
	ret.r.c[7] = d1;
	ret.i.c[7] = d2;
	return ret;
}

class AML_PREFIX(Complex32) {
public:
	AML_PREFIX(floatvec2) c{};


	AML_FUNCTION constexpr AML_PREFIX(Complex32)(const float real, const float img) {
		c.c[0] = real;
		c.c[1] = img;
	}

	AML_FUNCTION explicit AML_PREFIX(Complex32)(float *values) {
		c.c[0] = values[0];
		c.c[1] = values[1];
	}

	AML_FUNCTION constexpr AML_PREFIX(Complex32)(float real) {
		c.c[0] = real;
		c.c[1] = 0.0;
	}

#if defined(AML_USE_STD_COMPLEX)

	AML_FUNCTION AML_PREFIX(Complex32)(std::complex<float> sc) {
		c.c[0] = sc.real();
		c.c[1] = sc.imag();
	}

#endif

	AML_FUNCTION AML_PREFIX(Complex32)() {
		c.c[0] = 0;
		c.c[1] = 0;
	}

	AML_FUNCTION void set([[maybe_unused]]uint64_t location, const AML_PREFIX(Complex32) value) {
		c.c[0] = value.c.c[0];
		c.c[1] = value.c.c[1];
	}

//add sub
	AML_FUNCTION AML_PREFIX(Complex32) *add(const AML_PREFIX(Complex32) a) {
		c.c[0] += a.c.c[0];
		c.c[1] += a.c.c[1];
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *add(const AML_PREFIX(Complex32) a, const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			c.c[1] += a.c.c[1];
			c.c[0] += a.c.c[0];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) operator+(const AML_PREFIX(Complex32) a) {
		AML_PREFIX(Complex32) ret(c.c[0] + a.c.c[0], c.c[1] + a.c.c[1]);
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Complex32) operator-(AML_PREFIX(Complex32) a) {
		AML_PREFIX(Complex32) ret(c.c[0] - a.c.c[0], c.c[1] - a.c.c[1]);
		return ret;
	}


	AML_FUNCTION void operator+=(const AML_PREFIX(Complex32) a) {
		c.c[0] += a.c.c[0];
		c.c[1] += a.c.c[1];
	}

	AML_FUNCTION void operator-=(const AML_PREFIX(Complex32) a) {
		c.c[0] -= a.c.c[0];
		c.c[1] -= a.c.c[1];
	}

	AML_FUNCTION AML_PREFIX(Complex32) *subtract(const AML_PREFIX(Complex32) a) {
		c.c[0] -= a.c.c[0];
		c.c[1] -= a.c.c[1];
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *subtract(const AML_PREFIX(Complex32) a, const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			c.c[0] -= a.c.c[0];
			c.c[1] -= a.c.c[1];

		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *conjugate() {
		c.c[1] = -c.c[1];
		return this;
	}

//mul
	AML_FUNCTION AML_PREFIX(Complex32) *multiply(const AML_PREFIX(Complex32) a) {
		float d1 = c.c[0] * a.c.c[0] - c.c[1] * a.c.c[1];
		float d2 = c.c[0] * a.c.c[1] + c.c[1] * a.c.c[0];
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}


	AML_FUNCTION AML_PREFIX(Complex32) *multiply(const AML_PREFIX(Complex32) &a, const AML_PREFIX(VectorU8_1D) &mask) {
		float d1;
		float d2;
		if (mask.v.c) LIKELY {
			d1 = c.c[0] * a.c.c[0] - c.c[1] * a.c.c[1];
			d2 = c.c[0] * a.c.c[1] + c.c[1] * a.c.c[0];
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;


	}

	AML_FUNCTION AML_PREFIX(Complex32) operator*(const AML_PREFIX(Complex32) a) {
		AML_PREFIX(Complex32) ret(c.c[0] * a.c.c[0] - c.c[1] * a.c.c[1], c.c[0] * a.c.c[1] + c.c[1] * a.c.c[0]);
		return ret;
	}

	AML_FUNCTION void operator*=(const AML_PREFIX(Complex32) a) {
		float d1 = c.c[0] * a.c.c[0] - c.c[1] * a.c.c[1];
		float d2 = c.c[0] * a.c.c[1] + c.c[1] * a.c.c[0];
		c.c[0] = d1;
		c.c[1] = d2;
	}

	AML_FUNCTION void operator*=(float a) {
		c.c[0] = c.c[0] * a;
		c.c[1] = c.c[1] * a;
	}


	AML_FUNCTION AML_PREFIX(Complex32) *square() {
		float d1 = c.c[0] * c.c[0] - c.c[1] * c.c[1];
		float d2 = c.c[0] * c.c[1] + c.c[1] * c.c[0];
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *square(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			float d1 = c.c[0] * c.c[0] - c.c[1] * c.c[1];
			float d2 = c.c[0] * c.c[1] + c.c[1] * c.c[0];
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

//division
	AML_FUNCTION AML_PREFIX(Complex32) operator/(const AML_PREFIX(Complex32) a) {
		AML_PREFIX(Complex32) ret;
		ret.c.c[0] = (c.c[0] * a.c.c[0] + c.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.c.c[1] = (c.c[1] * a.c.c[0] - c.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		return ret;
	}

	AML_FUNCTION void operator/=(const AML_PREFIX(Complex32) a) {
		float d1 = (c.c[0] * a.c.c[0] + c.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		float d2 = (c.c[1] * a.c.c[0] - c.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		c.c[0] = d1;
		c.c[1] = d2;
	}

	AML_FUNCTION void operator/=(float a) {
		float d1 = c.c[0] / a;
		float d2 = c.c[1] / a;
		c.c[0] = d1;
		c.c[1] = d2;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *divide(const AML_PREFIX(Complex32) a) {
		float d1 = (c.c[0] * a.c.c[0] + c.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		float d2 = (c.c[1] * a.c.c[0] - c.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) operator/(float a) {
		AML_PREFIX(Complex32) ret;
		ret.c.c[0] = c.c[0] / a;
		ret.c.c[1] = c.c[1] / a;
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *divide(float a) {
		float d1 = c.c[0] / a;
		float d2 = c.c[1] / a;
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *divide(const float a, const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			float d1 = c.c[0] / a;
			float d2 = c.c[1] / a;
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *divide(const AML_PREFIX(Complex32) a, const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			float d1 = (c.c[0] * a.c.c[0] + c.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			float d2 = (c.c[1] * a.c.c[0] - c.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

//sqrt
	AML_FUNCTION AML_PREFIX(Complex32) *sqrt() {
		float d2 = ::sqrt((-c.c[0] + ::sqrt(c.c[0] * c.c[0] + c.c[1] * c.c[1])) / (2));
		float d1;
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(c.c[0]);
		} else LIKELY {
			d1 = c.c[1] / (2 * d2);
		}
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *sqrt(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			float d2 = ::sqrt((-c.c[0] + ::sqrt(c.c[0] * c.c[0] + c.c[1] * c.c[1])) / (2));
			float d1;
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(c.c[0]);
			} else LIKELY {
				d1 = c.c[1] / (2 * d2);
			}
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *sin() {
		float d1 = ::sin(c.c[0]) * ::cosh(c.c[1]);
		float d2 = ::cos(c.c[1]) * ::sinh(c.c[0]);

		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *cos() {
		float d1 = ::cos(c.c[0]) * ::cosh(c.c[1]);
		float d2 = -::sin(c.c[1]) * ::sinh(c.c[0]);

		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *tan() {
		float d1 = ::sin(c.c[0] + c.c[0]) / (::cos(c.c[0] + c.c[0]) * ::cosh(c.c[1] + c.c[1]));
		float d2 = ::sinh(c.c[1] + c.c[1]) / (::cos(c.c[0] + c.c[0]) * ::cosh(c.c[1] + c.c[1]));

		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *sin(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) {
			float d1 = ::sin(c.c[0]) * ::cosh(c.c[1]);
			float d2 = ::cos(c.c[1]) * ::sinh(c.c[0]);
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *cos(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) {
			float d1 = ::cos(c.c[0]) * ::cosh(c.c[1]);
			float d2 = -::sin(c.c[1]) * ::sinh(c.c[0]);
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *tan(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) {
			float d1 = ::sin(c.c[0] + c.c[0]) / (::cos(c.c[0] + c.c[0]) * ::cosh(c.c[1] + c.c[1]));
			float d2 = ::sinh(c.c[1] + c.c[1]) / (::cos(c.c[0] + c.c[0]) * ::cosh(c.c[1] + c.c[1]));
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Complex32) *exp() {
		float d1 = ::exp(c.c[0]) * ::cos(c.c[1]);
		float d2 = ::exp(c.c[0]) * ::sin(c.c[1]);


		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *exp(float n) {
		float d1 = ::pow(n, c.c[0]) * ::cos(c.c[1] * ::log(n));
		float d2 = ::pow(n, c.c[0]) * ::sin(c.c[1] * ::log(n));
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *pow(float n) {
		float d1 = ::pow(c.c[0] * c.c[0] + c.c[1] * c.c[1], n / 2) * ::cos(n * atan2(c.c[1], c.c[0]));
		float d2 = ::pow(c.c[0] * c.c[0] + c.c[1] * c.c[1], n / 2) * ::sin(n * atan2(c.c[1], c.c[0]));
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *pow(const AML_PREFIX(Complex32) n) {
		float d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
		float d2 = ::atan2(c.c[1], c.c[0]);
		float d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		float d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		float d5 = d3 * ::cos(d4);
		float d6 = d3 * ::sin(d4);
		c.c[0] = d5;
		c.c[1] = d6;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *pow(float n, const AML_PREFIX(VectorU8_1D) &mask) {
		if (mask.v.c) LIKELY {
			float d1 = ::pow(c.c[0] * c.c[0] + c.c[1] * c.c[1], n / 2) * ::cos(n * atan2(c.c[1], c.c[0]));
			float d2 = ::pow(c.c[0] * c.c[0] + c.c[1] * c.c[1], n / 2) * ::sin(n * atan2(c.c[1], c.c[0]));
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *pow(const AML_PREFIX(Complex32) n, const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			float d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
			float d2 = ::atan2(c.c[1], c.c[0]);
			float d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			float d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			float d5 = d3 * ::cos(d4);
			float d6 = d3 * ::sin(d4);
			c.c[0] = d5;
			c.c[1] = d6;
		}
		return this;
	}

	AML_FUNCTION float abs() {
		return ::sqrt(c.c[0] * c.c[0] + c.c[1] * c.c[1]);
	}

	AML_FUNCTION bool abs_gt(float a) {
		return a * a < c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}

	AML_FUNCTION bool abs_lt(float a) {
		return a * a > c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}

	AML_FUNCTION bool abs_eq(float a) {
		return a * a == c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}

	AML_FUNCTION bool abs_gt(const AML_PREFIX(Complex32) a) {
		return a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1] < c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}

	AML_FUNCTION bool abs_lt(const AML_PREFIX(Complex32) a) {
		return a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1] > c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}

	AML_FUNCTION bool abs_eq(const AML_PREFIX(Complex32) a) {
		return a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1] == c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}


	AML_FUNCTION AML_PREFIX(Complex32) *ln() {
		float d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
		float d2 = ::atan2(c.c[1], c.c[0]);
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *log() {
		float d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
		float d2 = ::atan2(c.c[1], c.c[0]);
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *log10() {
		float d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / (2 * AML_LN10);
		float d2 = ::atan2(c.c[1], c.c[0]) / AML_LN10;
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *ln(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			float d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
			float d2 = ::atan2(c.c[1], c.c[0]);
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *log(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			float d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
			float d2 = ::atan2(c.c[1], c.c[0]);
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) *log10(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			float d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / (2 * AML_LN10);
			float d2 = ::atan2(c.c[1], c.c[0]) / AML_LN10;
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}


	AML_FUNCTION float imaginary() {
		return c.c[1];
	}

	AML_FUNCTION float real() {
		return c.c[0];
	}

	AML_FUNCTION float angle() {
		return ::atan2(c.c[1], c.c[0]);
	}

	AML_FUNCTION float length() {
		return ::sqrt(c.c[0] * c.c[0] + c.c[1] * c.c[1]);
	}

	AML_FUNCTION AML_PREFIX(Complex32) *polar(float length, float angle) {
		c.c[0] = length * ::cos(angle);
		c.c[1] = length * ::sin(angle);
		return this;
	}

	AML_FUNCTION AML_PREFIX(Complex32) operator[]([[maybe_unused]]uint64_t location) {
		return *this;
	}


};


#if !defined(AML_NO_STRING)


AML_FUNCTION std::string operator<<(std::string &lhs, const AML_PREFIX(Complex32) &rhs) {
	std::ostringstream string;
	string << lhs << rhs.c.c[0] << " + " << rhs.c.c[1] << "i";
	return string.str();
}

AML_FUNCTION std::string operator<<(const char *lhs, const AML_PREFIX(Complex32) &rhs) {
	std::ostringstream string;
	string << lhs << rhs.c.c[0] << " + " << rhs.c.c[1] << "i";
	return string.str();
}

template<class charT, class traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &o, const AML_PREFIX(Complex32) &x) {
	std::basic_ostringstream<charT, traits> s;
	s.flags(o.flags());
	s.imbue(o.getloc());
	s.precision(o.precision());
	s << x.c.c[0] << " + " << x.c.c[1] << "i";
	return o << s.str();
}

#endif

AML_FUNCTION AML_PREFIX(Complex32) operator+(const float &lhs, const AML_PREFIX(Complex32) &rhs) {
	AML_PREFIX(Complex32) ret(lhs + rhs.c.c[0], 0.0 + rhs.c.c[1]);
	return ret;
}

AML_FUNCTION AML_PREFIX(Complex32) operator-(const float &lhs, const AML_PREFIX(Complex32) &rhs) {
	AML_PREFIX(Complex32) ret(lhs - rhs.c.c[0], 0.0 - rhs.c.c[1]);
	return ret;
}

AML_FUNCTION AML_PREFIX(Complex32) operator*(const float &lhs, const AML_PREFIX(Complex32) &rhs) {
	AML_PREFIX(Complex32) ret(lhs * rhs.c.c[0], lhs * rhs.c.c[1]);
	return ret;
}

AML_FUNCTION AML_PREFIX(Complex32) operator/(const float &lhs, const AML_PREFIX(Complex32) &rhs) {
	AML_PREFIX(Complex32) ret;
	ret.c.c[0] = (lhs * rhs.c.c[0]) / (rhs.c.c[0] * rhs.c.c[0] + rhs.c.c[1] * rhs.c.c[1]);
	ret.c.c[1] = (-lhs * rhs.c.c[1]) / (rhs.c.c[0] * rhs.c.c[0] + rhs.c.c[1] * rhs.c.c[1]);
	return ret;
}

#if defined(AML_USE_STD_COMPLEX)

AML_FUNCTION AML_PREFIX(Complex32) operator+(const std::complex<float> &lhs, const AML_PREFIX(Complex32) &rhs) {
	AML_PREFIX(Complex32) ret = lhs;
	return ret + rhs;
}

AML_FUNCTION AML_PREFIX(Complex32) operator-(const std::complex<float> &lhs, const AML_PREFIX(Complex32) &rhs) {
	AML_PREFIX(Complex32) ret = lhs;
	return ret - rhs;
}

AML_FUNCTION AML_PREFIX(Complex32) operator*(const std::complex<float> &lhs, const AML_PREFIX(Complex32) &rhs) {
	AML_PREFIX(Complex32) ret = lhs;
	return ret * rhs;
}

AML_FUNCTION AML_PREFIX(Complex32) operator/(const std::complex<float> &lhs, const AML_PREFIX(Complex32) &rhs) {
	AML_PREFIX(Complex32) ret = lhs;
	return ret / rhs;
}

class STD_COMPLEX32_CAST : public std::complex<float> {
public:
	AML_FUNCTION STD_COMPLEX32_CAST(const AML_PREFIX(Complex32) &other) : std::complex<float>(other.c.c[0],
																							  other.c.c[1]) {}
};

#endif

#if !defined(USE_CUDA)

constexpr Complex32 operator ""_fi(long double d) {
	return Complex32(0.0f, (float) d);
}

constexpr Complex32 operator ""_fi(unsigned long long d) {
	return Complex32(0.0f, (float) d);
}

#endif

#if defined(AML_USE_STD_COMPLEX)

AML_FUNCTION std::complex<float> toStdComplex(Complex32 a) {
	std::complex<float> ret(a.c.c[0], a.c.c[1]);
	return ret;
}

#endif


class AML_PREFIX(Array2Complex32) {
public:
	AML_PREFIX(floatvec2) r{};
	AML_PREFIX(floatvec2) i{};

	AML_FUNCTION AML_PREFIX(Array2Complex32)() {}

	AML_FUNCTION AML_PREFIX(Array2Complex32)(const AML_PREFIX(Complex32) value) {
		r.c[0] = value.c.c[0];
		i.c[0] = value.c.c[1];
		r.c[1] = value.c.c[0];
		i.c[1] = value.c.c[1];
	}


	AML_FUNCTION AML_PREFIX(VectorFloat2D) real() {
		return AML_PREFIX(VectorFloat2D)(r.c);
	}

	AML_FUNCTION AML_PREFIX(VectorFloat2D) complex() {
		return AML_PREFIX(VectorFloat2D)(i.c);
	}

	AML_FUNCTION AML_PREFIX(Complex32) operator[](uint64_t location) {
		return AML_PREFIX(Complex32)(r.c[location], i.c[location]);
	}

	AML_FUNCTION void set(uint64_t location, AML_PREFIX(Complex32) value) {
		r.c[location] = value.c.c[0];
		i.c[location] = value.c.c[1];
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *add(const AML_PREFIX(Array2Complex32) a) {
		i.c[0] += a.i.c[0];
		i.c[1] += a.i.c[1];
		r.c[0] += a.r.c[0];
		r.c[1] += a.r.c[1];
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *add(const AML_PREFIX(Complex32) a) {
		i.c[0] += a.c.c[1];
		i.c[1] += a.c.c[1];
		r.c[0] += a.c.c[0];
		r.c[1] += a.c.c[0];
		return this;
	}

	AML_FUNCTION void operator+=(const AML_PREFIX(Array2Complex32) a) {
		i.c[0] += a.i.c[0];
		i.c[1] += a.i.c[1];
		r.c[0] += a.r.c[0];
		r.c[1] += a.r.c[1];
	}

	AML_FUNCTION void operator+=(const AML_PREFIX(Complex32) a) {
		i.c[0] += a.c.c[1];
		i.c[1] += a.c.c[1];
		r.c[0] += a.c.c[0];
		r.c[1] += a.c.c[0];
	}


	AML_FUNCTION AML_PREFIX(Array2Complex32) operator+(const AML_PREFIX(Array2Complex32) a) const {
		AML_PREFIX(Array2Complex32) ret;
		ret.i.c[0] = i.c[0] + a.i.c[0];
		ret.i.c[1] = i.c[1] + a.i.c[1];
		ret.r.c[0] = r.c[0] + a.r.c[0];
		ret.r.c[1] = r.c[1] + a.r.c[1];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) operator+(const AML_PREFIX(Complex32) a) const {
		AML_PREFIX(Array2Complex32) ret;
		ret.i.c[0] = i.c[0] + a.c.c[1];
		ret.i.c[1] = i.c[1] + a.c.c[1];
		ret.r.c[0] = r.c[0] + a.c.c[0];
		ret.r.c[1] = r.c[1] + a.c.c[0];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *add(const AML_PREFIX(Array2Complex32) a, AML_PREFIX(VectorU8_2D) mask) {
		if (mask.v.c[0]) {
			i.c[0] += a.i.c[0];
			r.c[0] += a.r.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] += a.i.c[1];
			r.c[1] += a.r.c[1];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *add(const AML_PREFIX(Complex32) a, AML_PREFIX(VectorU8_2D) mask) {
		if (mask.v.c[0]) {
			i.c[0] += a.c.c[1];
			r.c[0] += a.c.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] += a.c.c[1];
			r.c[1] += a.c.c[0];
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array2Complex32) *subtract(const AML_PREFIX(Array2Complex32) a) {
		i.c[0] -= a.i.c[0];
		i.c[1] -= a.i.c[1];
		r.c[0] -= a.r.c[0];
		r.c[1] -= a.r.c[1];
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *subtract(const AML_PREFIX(Complex32) a) {
		i.c[0] -= a.c.c[1];
		i.c[1] -= a.c.c[1];
		r.c[0] -= a.c.c[0];
		r.c[1] -= a.c.c[0];
		return this;
	}

	AML_FUNCTION void operator-=(const AML_PREFIX(Array2Complex32) a) {
		i.c[0] -= a.i.c[0];
		i.c[1] -= a.i.c[1];
		r.c[0] -= a.r.c[0];
		r.c[1] -= a.r.c[1];
	}

	AML_FUNCTION void operator-=(const AML_PREFIX(Complex32) a) {
		i.c[0] -= a.c.c[1];
		i.c[1] -= a.c.c[1];
		r.c[0] -= a.c.c[0];
		r.c[1] -= a.c.c[0];
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) operator-(const AML_PREFIX(Array2Complex32) a) const {
		AML_PREFIX(Array2Complex32) ret;
		ret.i.c[0] = i.c[0] - a.i.c[0];
		ret.i.c[1] = i.c[1] - a.i.c[1];
		ret.r.c[0] = r.c[0] - a.r.c[0];
		ret.r.c[1] = r.c[1] - a.r.c[1];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) operator-(const AML_PREFIX(Complex32) a) const {
		AML_PREFIX(Array2Complex32) ret;
		ret.i.c[0] = i.c[0] - a.c.c[1];
		ret.i.c[1] = i.c[1] - a.c.c[1];
		ret.r.c[0] = r.c[0] - a.c.c[0];
		ret.r.c[1] = r.c[1] - a.c.c[0];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *
	subtract(const AML_PREFIX(Array2Complex32) a, AML_PREFIX(VectorU8_2D) mask) {
		if (mask.v.c[0]) {
			i.c[0] -= a.i.c[0];
			r.c[0] -= a.r.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] -= a.i.c[1];
			r.c[1] -= a.r.c[1];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *subtract(const AML_PREFIX(Complex32) a, AML_PREFIX(VectorU8_2D) mask) {
		if (mask.v.c[0]) {
			i.c[0] -= a.c.c[1];
			r.c[0] -= a.c.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] -= a.c.c[1];
			r.c[1] -= a.c.c[0];
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array2Complex32) operator*(const AML_PREFIX(Array2Complex32) &a) const {
		AML_PREFIX(Array2Complex32) ret;
		ret.r.c[0] = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		ret.i.c[0] = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		ret.r.c[1] = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		ret.i.c[1] = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];


		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *multiply(const AML_PREFIX(Array2Complex32) &a) {
		float d1;
		float d2;
		d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;

		d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) operator*(const AML_PREFIX(Complex32) &a) const {
		AML_PREFIX(Array2Complex32) ret;
		ret.r.c[0] = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		ret.i.c[0] = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		ret.r.c[1] = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		ret.i.c[1] = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];

		return ret;
	}

	AML_FUNCTION void operator*=(const AML_PREFIX(Array2Complex32) &a) {
		float d1;
		float d2;
		d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;

		d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
	}

	AML_FUNCTION void operator*=(const AML_PREFIX(Complex32) &a) {
		float d1;
		float d2;
		d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
		r.c[1] = d1;
		i.c[1] = d2;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *multiply(const AML_PREFIX(Complex32) &a) {
		float d1;
		float d2;
		d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *
	multiply(const AML_PREFIX(Complex32) &a, const AML_PREFIX(VectorU8_2D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
			d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
			d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;


	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *
	multiply(const AML_PREFIX(Array2Complex32) &a, const AML_PREFIX(VectorU8_2D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
			d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
			d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *square() {
		float d1;
		float d2;
		d1 = r.c[0] * r.c[0] - i.c[0] * i.c[0];
		d2 = r.c[0] * i.c[0] + i.c[0] * r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * r.c[1] - i.c[1] * i.c[1];
		d2 = r.c[1] * i.c[1] + i.c[1] * r.c[1];
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *square(const AML_PREFIX(VectorU8_2D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * r.c[0] - i.c[0] * i.c[0];
			d2 = r.c[0] * i.c[0] + i.c[0] * r.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * r.c[1] - i.c[1] * i.c[1];
			d2 = r.c[1] * i.c[1] + i.c[1] * r.c[1];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *divide(const AML_PREFIX(Complex32) a) {
		float d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		float d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *
	divide(const AML_PREFIX(Complex32) a, const AML_PREFIX(VectorU8_2D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *divide(const AML_PREFIX(Array2Complex32) &a) {
		float d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		float d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *
	divide(const AML_PREFIX(Array2Complex32) &a, const AML_PREFIX(VectorU8_2D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = (r.c[0] * a.i.c[0] + i.c[0] * a.i.c[1]) / (a.i.c[0] * a.i.c[0] + a.i.c[1] * a.i.c[1]);
			d2 = (i.c[0] * a.i.c[0] - r.c[0] * a.i.c[1]) / (a.i.c[0] * a.i.c[0] + a.i.c[1] * a.i.c[1]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
			d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) operator/(const AML_PREFIX(Complex32) &a) const {
		AML_PREFIX(Array2Complex32) ret;
		float d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		float d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[0] = d1;
		ret.i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[1] = d1;
		ret.i.c[1] = d2;
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) operator/(const AML_PREFIX(Array2Complex32) &a) const {
		AML_PREFIX(Array2Complex32) ret;
		float d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		float d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		ret.r.c[0] = d1;
		ret.i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		ret.r.c[1] = d1;
		ret.i.c[1] = d2;
		return ret;
	}

	AML_FUNCTION void operator/=(const AML_PREFIX(Complex32) &a) {
		float d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		float d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
	}

	AML_FUNCTION void operator/=(const AML_PREFIX(Array2Complex32) &a) {
		float d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		float d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *sqrt() {
		float d1;
		float d2;
		d2 = ::sqrt((-r.c[0] + ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[0]);
		} else LIKELY {
			d1 = i.c[0] / (2 * d2);
		}
		r.c[0] = d1;
		i.c[0] = d2;
		d2 = ::sqrt((-r.c[1] + ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[1]);
		} else LIKELY {
			d1 = i.c[1] / (2 * d2);
		}
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *sqrt(const AML_PREFIX(VectorU8_2D) mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d2 = ::sqrt((-r.c[0] + ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[0]);
			} else LIKELY {
				d1 = i.c[0] / (2 * d2);
			}
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d2 = ::sqrt((-r.c[1] + ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[1]);
			} else LIKELY {
				d1 = i.c[1] / (2 * d2);
			}
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *sin() {
		float d1;
		float d2;
		d1 = ::sin(r.c[0]) * ::cosh(i.c[0]);
		d2 = ::cos(i.c[0]) * ::sinh(r.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::sin(r.c[1]) * ::cosh(i.c[1]);
		d2 = ::cos(i.c[1]) * ::sinh(r.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *cos() {
		float d1;
		float d2;
		d1 = ::cos(r.c[0]) * ::cosh(i.c[0]);
		d2 = -::sin(i.c[0]) * ::sinh(r.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::cos(r.c[1]) * ::cosh(i.c[1]);
		d2 = -::sin(i.c[1]) * ::sinh(r.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *tan() {
		float d1;
		float d2;
		d1 = ::sin(r.c[0] + r.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
		d2 = ::sinh(i.c[0] + i.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::sin(r.c[1] + r.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
		d2 = ::sinh(i.c[1] + i.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *sin(const AML_PREFIX(VectorU8_2D) mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::sin(r.c[0]) * ::cosh(i.c[0]);
			d2 = ::cos(i.c[0]) * ::sinh(r.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::sin(r.c[1]) * ::cosh(i.c[1]);
			d2 = ::cos(i.c[1]) * ::sinh(r.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *cos(const AML_PREFIX(VectorU8_2D) mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::cos(r.c[0]) * ::cosh(i.c[0]);
			d2 = -::sin(i.c[0]) * ::sinh(r.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::cos(r.c[1]) * ::cosh(i.c[1]);
			d2 = -::sin(i.c[1]) * ::sinh(r.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *tan(const AML_PREFIX(VectorU8_2D) mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::sin(r.c[0] + r.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
			d2 = ::sinh(i.c[0] + i.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::sin(r.c[1] + r.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
			d2 = ::sinh(i.c[1] + i.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *exp() {
		float d1 = ::exp(r.c[0]) * ::cos(i.c[0]);
		float d2 = ::exp(r.c[0]) * ::sin(i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::exp(r.c[1]) * ::cos(i.c[1]);
		d2 = ::exp(r.c[1]) * ::sin(i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *exp(float n) {
		float d1 = ::pow(n, r.c[0]) * ::cos(i.c[0] * ::log(n));
		float d2 = ::pow(n, r.c[0]) * ::sin(i.c[0] * ::log(n));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::pow(n, r.c[1]) * ::cos(i.c[1] * ::log(n));
		d2 = ::pow(n, r.c[1]) * ::sin(i.c[1] * ::log(n));
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *exp(const AML_PREFIX(VectorU8_2D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::exp(r.c[0]) * ::cos(i.c[0]);
			d2 = ::exp(r.c[0]) * ::sin(i.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::exp(r.c[1]) * ::cos(i.c[1]);
			d2 = ::exp(r.c[1]) * ::sin(i.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *exp(float n, const AML_PREFIX(VectorU8_2D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::pow(n, r.c[0]) * ::cos(i.c[0] * ::log(n));
			d2 = ::pow(n, r.c[0]) * ::sin(i.c[0] * ::log(n));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::pow(n, r.c[1]) * ::cos(i.c[1] * ::log(n));
			d2 = ::pow(n, r.c[1]) * ::sin(i.c[1] * ::log(n));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *pow(const AML_PREFIX(Array2Complex32) n) {
		float d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		float d2 = ::atan2(r.c[0], i.c[0]);
		float d3 = ::exp(d1 * n.i.c[0] - d2 * n.r.c[0]);
		float d4 = d1 * n.r.c[0] + d2 * n.i.c[0];
		float d5 = d3 * ::cos(d4);
		float d6 = d3 * ::sin(d4);
		i.c[0] = d5;
		r.c[0] = d6;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		d3 = ::exp(d1 * n.i.c[1] - d2 * n.r.c[1]);
		d4 = d1 * n.r.c[1] + d2 * n.i.c[1];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[1] = d5;
		r.c[1] = d6;
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array2Complex32) *pow(const AML_PREFIX(Complex32) n) {
		float d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		float d2 = ::atan2(r.c[0], i.c[0]);
		float d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		float d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		float d5 = d3 * ::cos(d4);
		float d6 = d3 * ::sin(d4);
		i.c[0] = d5;
		r.c[0] = d6;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[1] = d5;
		r.c[1] = d6;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *
	pow(const AML_PREFIX(Array2Complex32) n, const AML_PREFIX(VectorU8_2D) &mask) {
		float d1;
		float d2;
		float d3;
		float d4;
		float d5;
		float d6;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			d3 = ::exp(d1 * n.i.c[0] - d2 * n.r.c[0]);
			d4 = d1 * n.r.c[0] + d2 * n.i.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[0] = d5;
			r.c[0] = d6;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			d3 = ::exp(d1 * n.i.c[1] - d2 * n.r.c[1]);
			d4 = d1 * n.r.c[1] + d2 * n.i.c[1];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[1] = d5;
			r.c[1] = d6;
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array2Complex32) *pow(const AML_PREFIX(Complex32) n, const AML_PREFIX(VectorU8_2D) &mask) {
		float d1;
		float d2;
		float d3;
		float d4;
		float d5;
		float d6;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[0] = d5;
			r.c[0] = d6;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[1] = d5;
			r.c[1] = d6;
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array2Complex32) *pow(float n) {
		float d1;
		float d2;
		d1 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::cos(n * atan2(i.c[0], r.c[0]));
		d2 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::sin(n * atan2(i.c[0], r.c[0]));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::cos(n * atan2(i.c[1], r.c[1]));
		d2 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::sin(n * atan2(i.c[1], r.c[1]));
		r.c[1] = d1;
		i.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *pow(float n, const AML_PREFIX(VectorU8_2D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::cos(n * atan2(i.c[0], r.c[0]));
			d2 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::sin(n * atan2(i.c[0], r.c[0]));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::cos(n * atan2(i.c[1], r.c[1]));
			d2 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::sin(n * atan2(i.c[1], r.c[1]));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat2D) abs() {
		AML_PREFIX(VectorFloat2D) ret;
		ret.v.c[0] = ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0]);
		ret.v.c[1] = ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1]);
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_2D) abs_gt(float a) {
		AML_PREFIX(VectorU8_2D) ret;
		ret.v.c[0] = a * a < r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a < r.c[1] * r.c[1] + i.c[1] * i.c[1];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_2D) abs_lt(float a) {
		AML_PREFIX(VectorU8_2D) ret;
		ret.v.c[0] = a * a > r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a > r.c[1] * r.c[1] + i.c[1] * i.c[1];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_2D) abs_eq(float a) {
		AML_PREFIX(VectorU8_2D) ret;
		ret.v.c[0] = a * a == r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a == r.c[1] * r.c[1] + i.c[1] * i.c[1];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *ln(const AML_PREFIX(VectorU8_2D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			i.c[1] = d1;
			r.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *log(const AML_PREFIX(VectorU8_2D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			i.c[1] = d1;
			r.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *log10(const AML_PREFIX(VectorU8_2D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[0], i.c[0]) / AML_LN10;
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[1], i.c[1]) / AML_LN10;
			i.c[1] = d1;
			r.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *ln() {
		float d1;
		float d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		d2 = ::atan2(r.c[0], i.c[0]);
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		i.c[1] = d1;
		r.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *log() {
		float d1;
		float d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		d2 = ::atan2(r.c[0], i.c[0]);
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		i.c[1] = d1;
		r.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array2Complex32) *log10() {
		float d1;
		float d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[0], i.c[0]) / AML_LN10;
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[1], i.c[1]) / AML_LN10;
		i.c[1] = d1;
		r.c[1] = d2;
		return this;
	}
};


#if !defined(AML_NO_STRING)

AML_FUNCTION std::string operator<<(std::string &lhs, const AML_PREFIX(Array2Complex32) &rhs) {
	std::ostringstream string;
	string << lhs << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1] << "i }";
	return string.str();
}

AML_FUNCTION std::string operator<<(const char *lhs, const AML_PREFIX(Array2Complex32) &rhs) {
	std::ostringstream string;
	string << lhs << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1] << "i }";
	return string.str();
}

template<class charT, class traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &o, const AML_PREFIX(Array2Complex32) &rhs) {
	std::basic_ostringstream<charT, traits> s;
	s.flags(o.flags());
	s.imbue(o.getloc());
	s.precision(o.precision());
	s << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1] << "i }";
	return o << s.str();
}

#endif

AML_FUNCTION AML_PREFIX(Array2Complex32)
operator+(const AML_PREFIX(Complex32) &lhs, const AML_PREFIX(Array2Complex32) &rhs) {
	return rhs + lhs;
}

AML_FUNCTION AML_PREFIX(Array2Complex32)
operator-(const AML_PREFIX(Complex32) &lhs, const AML_PREFIX(Array2Complex32) &rhs) {
	AML_PREFIX(Array2Complex32) ret;
	ret.i.c[0] = lhs.c.c[1] - rhs.i.c[0];
	ret.i.c[1] = lhs.c.c[1] - rhs.i.c[1];
	ret.r.c[0] = lhs.c.c[0] - rhs.r.c[0];
	ret.r.c[1] = lhs.c.c[0] - rhs.r.c[1];
	return ret;
}

AML_FUNCTION AML_PREFIX(Array2Complex32)
operator*(const AML_PREFIX(Complex32) &lhs, const AML_PREFIX(Array2Complex32) &rhs) {
	return rhs * lhs;
}

AML_FUNCTION AML_PREFIX(Array2Complex32)
operator/(const AML_PREFIX(Complex32) &lhs, const AML_PREFIX(Array2Complex32) &rhs) {
	AML_PREFIX(Array2Complex32) ret;
	float d1 =
			(lhs.c.c[0] * rhs.r.c[0] + lhs.c.c[1] * rhs.i.c[0]) / (rhs.r.c[0] * rhs.r.c[0] + rhs.i.c[0] * rhs.i.c[0]);
	float d2 =
			(lhs.c.c[1] * rhs.r.c[0] - lhs.c.c[0] * rhs.i.c[0]) / (rhs.r.c[0] * rhs.r.c[0] + rhs.i.c[0] * rhs.i.c[0]);
	ret.r.c[0] = d1;
	ret.i.c[0] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[1] + lhs.c.c[1] * rhs.i.c[1]) / (rhs.r.c[1] * rhs.r.c[1] + rhs.i.c[1] * rhs.i.c[1]);
	d2 = (lhs.c.c[1] * rhs.r.c[1] - lhs.c.c[0] * rhs.i.c[1]) / (rhs.r.c[1] * rhs.r.c[1] + rhs.i.c[1] * rhs.i.c[1]);
	ret.r.c[1] = d1;
	ret.i.c[1] = d2;
	return ret;
}

class AML_PREFIX(Array4Complex32) {
public:
	AML_PREFIX(floatvec4) r{};
	AML_PREFIX(floatvec4) i{};

	AML_FUNCTION AML_PREFIX(Array4Complex32)() {}

	AML_FUNCTION AML_PREFIX(Array4Complex32)(AML_PREFIX(Complex32) value) {
		r.c[0] = value.c.c[0];
		i.c[0] = value.c.c[1];
		r.c[1] = value.c.c[0];
		i.c[1] = value.c.c[1];
		r.c[2] = value.c.c[0];
		i.c[2] = value.c.c[1];
		r.c[3] = value.c.c[0];
		i.c[3] = value.c.c[1];
	}


	AML_FUNCTION AML_PREFIX(VectorFloat4D) real() {
		return AML_PREFIX(VectorFloat4D)(r.c);
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) complex() {
		return AML_PREFIX(VectorFloat4D)(i.c);
	}

	AML_FUNCTION AML_PREFIX(Complex32) operator[](uint64_t location) {
		return AML_PREFIX(Complex32)(r.c[location], i.c[location]);
	}

	AML_FUNCTION void set(uint64_t location, AML_PREFIX(Complex32) value) {
		r.c[location] = value.c.c[0];
		i.c[location] = value.c.c[1];
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *add(const AML_PREFIX(Array4Complex32) a) {
		i.c[0] += a.i.c[0];
		i.c[1] += a.i.c[1];
		i.c[2] += a.i.c[2];
		i.c[3] += a.i.c[3];
		r.c[0] += a.r.c[0];
		r.c[1] += a.r.c[1];
		r.c[2] += a.r.c[2];
		r.c[3] += a.r.c[3];
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *add(const AML_PREFIX(Complex32) a) {
		i.c[0] += a.c.c[1];
		i.c[1] += a.c.c[1];
		i.c[2] += a.c.c[1];
		i.c[3] += a.c.c[1];
		r.c[0] += a.c.c[0];
		r.c[1] += a.c.c[0];
		r.c[2] += a.c.c[0];
		r.c[3] += a.c.c[0];
		return this;
	}

	AML_FUNCTION void operator+=(const AML_PREFIX(Array4Complex32) a) {
		i.c[0] += a.i.c[0];
		i.c[1] += a.i.c[1];
		i.c[2] += a.i.c[2];
		i.c[3] += a.i.c[3];
		r.c[0] += a.r.c[0];
		r.c[1] += a.r.c[1];
		r.c[2] += a.r.c[2];
		r.c[3] += a.r.c[3];
	}

	AML_FUNCTION void operator+=(const AML_PREFIX(Complex32) a) {
		i.c[0] += a.c.c[1];
		i.c[1] += a.c.c[1];
		i.c[2] += a.c.c[1];
		i.c[3] += a.c.c[1];
		r.c[0] += a.c.c[0];
		r.c[1] += a.c.c[0];
		r.c[2] += a.c.c[0];
		r.c[3] += a.c.c[0];
	}


	AML_FUNCTION AML_PREFIX(Array4Complex32) operator+(const AML_PREFIX(Array4Complex32) a) const {
		AML_PREFIX(Array4Complex32) ret{};
		ret.i.c[0] = i.c[0] + a.i.c[0];
		ret.i.c[1] = i.c[1] + a.i.c[1];
		ret.i.c[2] = i.c[2] + a.i.c[2];
		ret.i.c[3] = i.c[3] + a.i.c[3];
		ret.r.c[0] = r.c[0] + a.r.c[0];
		ret.r.c[1] = r.c[1] + a.r.c[1];
		ret.r.c[2] = r.c[2] + a.r.c[2];
		ret.r.c[3] = r.c[3] + a.r.c[3];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) operator+(const AML_PREFIX(Complex32) a) const {
		AML_PREFIX(Array4Complex32) ret{};
		ret.i.c[0] = i.c[0] + a.c.c[1];
		ret.i.c[1] = i.c[1] + a.c.c[1];
		ret.i.c[2] = i.c[2] + a.c.c[1];
		ret.i.c[3] = i.c[3] + a.c.c[1];
		ret.r.c[0] = r.c[0] + a.c.c[0];
		ret.r.c[1] = r.c[1] + a.c.c[0];
		ret.r.c[2] = r.c[2] + a.c.c[0];
		ret.r.c[3] = r.c[3] + a.c.c[0];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *add(const AML_PREFIX(Array4Complex32) a, AML_PREFIX(VectorU8_4D) mask) {
		if (mask.v.c[0]) {
			i.c[0] += a.i.c[0];
			r.c[0] += a.r.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] += a.i.c[1];
			r.c[1] += a.r.c[1];
		}
		if (mask.v.c[2]) {
			i.c[2] += a.i.c[2];
			r.c[2] += a.r.c[2];
		}
		if (mask.v.c[3]) {
			i.c[3] += a.i.c[3];
			r.c[3] += a.r.c[3];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *add(const AML_PREFIX(Complex32) a, AML_PREFIX(VectorU8_4D) mask) {
		if (mask.v.c[0]) {
			i.c[0] += a.c.c[1];
			r.c[0] += a.c.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] += a.c.c[1];
			r.c[1] += a.c.c[0];
		}
		if (mask.v.c[2]) {
			i.c[2] += a.c.c[1];
			r.c[2] += a.c.c[0];
		}
		if (mask.v.c[3]) {
			i.c[3] += a.c.c[1];
			r.c[3] += a.c.c[0];
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array4Complex32) *subtract(const AML_PREFIX(Array4Complex32) a) {
		i.c[0] -= a.i.c[0];
		i.c[1] -= a.i.c[1];
		i.c[2] -= a.i.c[2];
		i.c[3] -= a.i.c[3];
		r.c[0] -= a.r.c[0];
		r.c[1] -= a.r.c[1];
		r.c[2] -= a.r.c[2];
		r.c[3] -= a.r.c[3];
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *subtract(const AML_PREFIX(Complex32) a) {
		i.c[0] -= a.c.c[1];
		i.c[1] -= a.c.c[1];
		i.c[2] -= a.c.c[1];
		i.c[3] -= a.c.c[1];
		r.c[0] -= a.c.c[0];
		r.c[1] -= a.c.c[0];
		r.c[2] -= a.c.c[0];
		r.c[3] -= a.c.c[0];
		return this;
	}

	AML_FUNCTION void operator-=(const AML_PREFIX(Array4Complex32) a) {
		i.c[0] -= a.i.c[0];
		i.c[1] -= a.i.c[1];
		i.c[2] -= a.i.c[2];
		i.c[3] -= a.i.c[3];
		r.c[0] -= a.r.c[0];
		r.c[1] -= a.r.c[1];
		r.c[2] -= a.r.c[2];
		r.c[3] -= a.r.c[3];
	}

	AML_FUNCTION void operator-=(const AML_PREFIX(Complex32) a) {
		i.c[0] -= a.c.c[1];
		i.c[1] -= a.c.c[1];
		i.c[2] -= a.c.c[1];
		i.c[3] -= a.c.c[1];
		r.c[0] -= a.c.c[0];
		r.c[1] -= a.c.c[0];
		r.c[2] -= a.c.c[0];
		r.c[3] -= a.c.c[0];
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) operator-(const AML_PREFIX(Array4Complex32) a) const {
		AML_PREFIX(Array4Complex32) ret;
		ret.i.c[0] = i.c[0] - a.i.c[0];
		ret.i.c[1] = i.c[1] - a.i.c[1];
		ret.i.c[2] = i.c[2] - a.i.c[2];
		ret.i.c[3] = i.c[3] - a.i.c[3];
		ret.r.c[0] = r.c[0] - a.r.c[0];
		ret.r.c[1] = r.c[1] - a.r.c[1];
		ret.r.c[2] = r.c[2] - a.r.c[2];
		ret.r.c[3] = r.c[3] - a.r.c[3];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) operator-(const AML_PREFIX(Complex32) a) const {
		AML_PREFIX(Array4Complex32) ret;
		ret.i.c[0] = i.c[0] - a.c.c[1];
		ret.i.c[1] = i.c[1] - a.c.c[1];
		ret.i.c[2] = i.c[2] - a.c.c[1];
		ret.i.c[3] = i.c[3] - a.c.c[1];
		ret.r.c[0] = r.c[0] - a.c.c[0];
		ret.r.c[1] = r.c[1] - a.c.c[0];
		ret.r.c[2] = r.c[2] - a.c.c[0];
		ret.r.c[3] = r.c[3] - a.c.c[0];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *
	subtract(const AML_PREFIX(Array4Complex32) a, AML_PREFIX(VectorU8_4D) mask) {
		if (mask.v.c[0]) {
			i.c[0] -= a.i.c[0];
			r.c[0] -= a.r.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] -= a.i.c[1];
			r.c[1] -= a.r.c[1];
		}
		if (mask.v.c[2]) {
			i.c[2] -= a.i.c[2];
			r.c[2] -= a.r.c[2];
		}
		if (mask.v.c[3]) {
			i.c[3] -= a.i.c[3];
			r.c[3] -= a.r.c[3];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *subtract(const AML_PREFIX(Complex32) a, AML_PREFIX(VectorU8_4D) mask) {
		if (mask.v.c[0]) {
			i.c[0] -= a.c.c[1];
			r.c[0] -= a.c.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] -= a.c.c[1];
			r.c[1] -= a.c.c[0];
		}
		if (mask.v.c[2]) {
			i.c[2] -= a.c.c[1];
			r.c[2] -= a.c.c[0];
		}
		if (mask.v.c[3]) {
			i.c[3] -= a.c.c[1];
			r.c[3] -= a.c.c[0];
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array4Complex32) operator*(const AML_PREFIX(Array4Complex32) &a) const {
		AML_PREFIX(Array4Complex32) ret;
		ret.r.c[0] = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		ret.i.c[0] = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		ret.r.c[1] = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		ret.i.c[1] = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
		ret.r.c[2] = r.c[2] * a.r.c[2] - i.c[2] * a.i.c[2];
		ret.i.c[2] = r.c[2] * a.i.c[2] + i.c[2] * a.r.c[2];
		ret.r.c[3] = r.c[3] * a.r.c[3] - i.c[3] * a.i.c[3];
		ret.i.c[3] = r.c[3] * a.i.c[3] + i.c[3] * a.r.c[3];


		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *multiply(const AML_PREFIX(Array4Complex32) &a) {
		float d1;
		float d2;
		d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;

		d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * a.r.c[2] - i.c[2] * a.i.c[2];
		d2 = r.c[2] * a.i.c[2] + i.c[2] * a.r.c[2];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * a.r.c[3] - i.c[3] * a.i.c[3];
		d2 = r.c[3] * a.i.c[3] + i.c[3] * a.r.c[3];
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) operator*(const AML_PREFIX(Complex32) &a) const {
		AML_PREFIX(Array4Complex32) ret;
		ret.r.c[0] = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		ret.i.c[0] = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		ret.r.c[1] = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		ret.i.c[1] = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
		ret.r.c[2] = r.c[2] * a.c.c[0] - i.c[2] * a.c.c[1];
		ret.i.c[2] = r.c[2] * a.c.c[1] + i.c[2] * a.c.c[0];
		ret.r.c[3] = r.c[3] * a.c.c[0] - i.c[3] * a.c.c[1];
		ret.i.c[3] = r.c[3] * a.c.c[1] + i.c[3] * a.c.c[0];
		return ret;
	}

	AML_FUNCTION void operator*=(const AML_PREFIX(Array4Complex32) &a) {
		float d1;
		float d2;
		d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;

		d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * a.r.c[2] - i.c[2] * a.i.c[2];
		d2 = r.c[2] * a.i.c[2] + i.c[2] * a.r.c[2];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * a.r.c[3] - i.c[3] * a.i.c[3];
		d2 = r.c[3] * a.i.c[3] + i.c[3] * a.r.c[3];
		r.c[3] = d1;
		i.c[3] = d2;
	}

	AML_FUNCTION void operator*=(const AML_PREFIX(Complex32) &a) {
		float d1;
		float d2;
		d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * a.c.c[0] - i.c[2] * a.c.c[1];
		d2 = r.c[2] * a.c.c[1] + i.c[2] * a.c.c[0];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * a.c.c[0] - i.c[3] * a.c.c[1];
		d2 = r.c[3] * a.c.c[1] + i.c[3] * a.c.c[0];
		r.c[3] = d1;
		i.c[3] = d2;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *multiply(const AML_PREFIX(Complex32) &a) {
		float d1;
		float d2;
		d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * a.c.c[0] - i.c[2] * a.c.c[1];
		d2 = r.c[2] * a.c.c[1] + i.c[2] * a.c.c[0];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * a.c.c[0] - i.c[3] * a.c.c[1];
		d2 = r.c[3] * a.c.c[1] + i.c[3] * a.c.c[0];
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *
	multiply(const AML_PREFIX(Complex32) &a, const AML_PREFIX(VectorU8_4D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
			d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
			d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = r.c[2] * a.c.c[0] - i.c[2] * a.c.c[1];
			d2 = r.c[2] * a.c.c[1] + i.c[2] * a.c.c[0];
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = r.c[3] * a.c.c[0] - i.c[3] * a.c.c[1];
			d2 = r.c[3] * a.c.c[1] + i.c[3] * a.c.c[0];
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;


	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *
	multiply(const AML_PREFIX(Array4Complex32) &a, const AML_PREFIX(VectorU8_4D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
			d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
			d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = r.c[2] * a.r.c[2] - i.c[2] * a.i.c[2];
			d2 = r.c[2] * a.i.c[2] + i.c[2] * a.r.c[2];
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = r.c[3] * a.r.c[3] - i.c[3] * a.i.c[3];
			d2 = r.c[3] * a.i.c[3] + i.c[3] * a.r.c[3];
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *square() {
		float d1;
		float d2;
		d1 = r.c[0] * r.c[0] - i.c[0] * i.c[0];
		d2 = r.c[0] * i.c[0] + i.c[0] * r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * r.c[1] - i.c[1] * i.c[1];
		d2 = r.c[1] * i.c[1] + i.c[1] * r.c[1];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * r.c[2] - i.c[2] * i.c[2];
		d2 = r.c[2] * i.c[2] + i.c[2] * r.c[2];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * r.c[3] - i.c[3] * i.c[3];
		d2 = r.c[3] * i.c[3] + i.c[3] * r.c[3];
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *square(const AML_PREFIX(VectorU8_4D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * r.c[0] - i.c[0] * i.c[0];
			d2 = r.c[0] * i.c[0] + i.c[0] * r.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * r.c[1] - i.c[1] * i.c[1];
			d2 = r.c[1] * i.c[1] + i.c[1] * r.c[1];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = r.c[2] * r.c[2] - i.c[2] * i.c[2];
			d2 = r.c[2] * i.c[2] + i.c[2] * r.c[2];
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = r.c[3] * r.c[3] - i.c[3] * i.c[3];
			d2 = r.c[3] * i.c[3] + i.c[3] * r.c[3];
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *divide(const AML_PREFIX(Complex32) a) {
		float d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		float d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = (r.c[2] * a.c.c[0] + i.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[2] * a.c.c[0] - r.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = (r.c[3] * a.c.c[0] + i.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[3] * a.c.c[0] - r.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *
	divide(const AML_PREFIX(Complex32) a, const AML_PREFIX(VectorU8_4D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = (r.c[2] * a.c.c[0] + i.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[2] * a.c.c[0] - r.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = (r.c[3] * a.c.c[0] + i.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[3] * a.c.c[0] - r.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *divide(const AML_PREFIX(Array4Complex32) &a) {
		float d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		float d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = (r.c[2] * a.r.c[2] + i.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		d2 = (i.c[2] * a.r.c[2] - r.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = (r.c[3] * a.r.c[3] + i.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		d2 = (i.c[3] * a.r.c[3] - r.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *
	divide(const AML_PREFIX(Array4Complex32) &a, const AML_PREFIX(VectorU8_4D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = (r.c[0] * a.i.c[0] + i.c[0] * a.i.c[1]) / (a.i.c[0] * a.i.c[0] + a.i.c[1] * a.i.c[1]);
			d2 = (i.c[0] * a.i.c[0] - r.c[0] * a.i.c[1]) / (a.i.c[0] * a.i.c[0] + a.i.c[1] * a.i.c[1]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
			d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = (r.c[2] * a.r.c[2] + i.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
			d2 = (i.c[2] * a.r.c[2] - r.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = (r.c[3] * a.r.c[3] + i.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
			d2 = (i.c[3] * a.r.c[3] - r.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) operator/(const AML_PREFIX(Complex32) &a) const {
		AML_PREFIX(Array4Complex32) ret;
		float d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		float d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[0] = d1;
		ret.i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[1] = d1;
		ret.i.c[1] = d2;
		d1 = (r.c[2] * a.c.c[0] + i.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[2] * a.c.c[0] - r.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[2] = d1;
		ret.i.c[2] = d2;
		d1 = (r.c[3] * a.c.c[0] + i.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[3] * a.c.c[0] - r.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[3] = d1;
		ret.i.c[3] = d2;
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) operator/(const AML_PREFIX(Array4Complex32) &a) const {
		AML_PREFIX(Array4Complex32) ret;
		float d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		float d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		ret.r.c[0] = d1;
		ret.i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		ret.r.c[1] = d1;
		ret.i.c[1] = d2;
		d1 = (r.c[2] * a.r.c[2] + i.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		d2 = (i.c[2] * a.r.c[2] - r.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		ret.r.c[2] = d1;
		ret.i.c[2] = d2;
		d1 = (r.c[3] * a.r.c[3] + i.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		d2 = (i.c[3] * a.r.c[3] - r.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		ret.r.c[3] = d1;
		ret.i.c[3] = d2;
		return ret;
	}

	AML_FUNCTION void operator/=(const AML_PREFIX(Complex32) &a) {
		float d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		float d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = (r.c[2] * a.c.c[0] + i.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[2] * a.c.c[0] - r.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = (r.c[3] * a.c.c[0] + i.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[3] * a.c.c[0] - r.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[3] = d1;
		i.c[3] = d2;
	}

	AML_FUNCTION void operator/=(const AML_PREFIX(Array4Complex32) &a) {
		float d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		float d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = (r.c[2] * a.r.c[2] + i.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		d2 = (i.c[2] * a.r.c[2] - r.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = (r.c[3] * a.r.c[3] + i.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		d2 = (i.c[3] * a.r.c[3] - r.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *sqrt() {
		float d1;
		float d2;
		d2 = ::sqrt((-r.c[0] + ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[0]);
		} else LIKELY {
			d1 = i.c[0] / (2 * d2);
		}
		r.c[0] = d1;
		i.c[0] = d2;
		d2 = ::sqrt((-r.c[1] + ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[1]);
		} else LIKELY {
			d1 = i.c[1] / (2 * d2);
		}
		r.c[1] = d1;
		i.c[1] = d2;
		d2 = ::sqrt((-r.c[2] + ::sqrt(r.c[2] * r.c[2] + i.c[2] * i.c[2])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[2]);
		} else LIKELY {
			d1 = i.c[2] / (2 * d2);
		}
		r.c[2] = d1;
		i.c[2] = d2;
		d2 = ::sqrt((-r.c[3] + ::sqrt(r.c[3] * r.c[3] + i.c[3] * i.c[3])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[3]);
		} else LIKELY {
			d1 = i.c[3] / (2 * d2);
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *sqrt(const AML_PREFIX(VectorU8_4D) mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d2 = ::sqrt((-r.c[0] + ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[0]);
			} else LIKELY {
				d1 = i.c[0] / (2 * d2);
			}
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d2 = ::sqrt((-r.c[1] + ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[1]);
			} else LIKELY {
				d1 = i.c[1] / (2 * d2);
			}
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d2 = ::sqrt((-r.c[2] + ::sqrt(r.c[2] * r.c[2] + i.c[2] * i.c[2])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[2]);
			} else LIKELY {
				d1 = i.c[2] / (2 * d2);
			}
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d2 = ::sqrt((-r.c[3] + ::sqrt(r.c[3] * r.c[3] + i.c[3] * i.c[3])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[3]);
			} else LIKELY {
				d1 = i.c[3] / (2 * d2);
			}
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *sin() {
		float d1;
		float d2;
		d1 = ::sin(r.c[0]) * ::cosh(i.c[0]);
		d2 = ::cos(i.c[0]) * ::sinh(r.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::sin(r.c[1]) * ::cosh(i.c[1]);
		d2 = ::cos(i.c[1]) * ::sinh(r.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::sin(r.c[2]) * ::cosh(i.c[2]);
		d2 = ::cos(i.c[2]) * ::sinh(r.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::sin(r.c[3]) * ::cosh(i.c[3]);
		d2 = ::cos(i.c[3]) * ::sinh(r.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *cos() {
		float d1;
		float d2;
		d1 = ::cos(r.c[0]) * ::cosh(i.c[0]);
		d2 = -::sin(i.c[0]) * ::sinh(r.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::cos(r.c[1]) * ::cosh(i.c[1]);
		d2 = -::sin(i.c[1]) * ::sinh(r.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::cos(r.c[2]) * ::cosh(i.c[2]);
		d2 = -::sin(i.c[2]) * ::sinh(r.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::cos(r.c[3]) * ::cosh(i.c[3]);
		d2 = -::sin(i.c[3]) * ::sinh(r.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *tan() {
		float d1;
		float d2;
		d1 = ::sin(r.c[0] + r.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
		d2 = ::sinh(i.c[0] + i.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::sin(r.c[1] + r.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
		d2 = ::sinh(i.c[1] + i.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::sin(r.c[2] + r.c[2]) / (::cos(r.c[2] + r.c[2]) * ::cosh(i.c[2] + i.c[2]));
		d2 = ::sinh(i.c[2] + i.c[2]) / (::cos(r.c[2] + r.c[2]) * ::cosh(i.c[2] + i.c[2]));
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::sin(r.c[3] + r.c[3]) / (::cos(r.c[3] + r.c[3]) * ::cosh(i.c[3] + i.c[3]));
		d2 = ::sinh(i.c[3] + i.c[3]) / (::cos(r.c[3] + r.c[3]) * ::cosh(i.c[3] + i.c[3]));
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *sin(const AML_PREFIX(VectorU8_4D) mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::sin(r.c[0]) * ::cosh(i.c[0]);
			d2 = ::cos(i.c[0]) * ::sinh(r.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::sin(r.c[1]) * ::cosh(i.c[1]);
			d2 = ::cos(i.c[1]) * ::sinh(r.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::sin(r.c[2]) * ::cosh(i.c[2]);
			d2 = ::cos(i.c[2]) * ::sinh(r.c[2]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::sin(r.c[3]) * ::cosh(i.c[3]);
			d2 = ::cos(i.c[3]) * ::sinh(r.c[3]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *cos(const AML_PREFIX(VectorU8_4D) mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::cos(r.c[0]) * ::cosh(i.c[0]);
			d2 = -::sin(i.c[0]) * ::sinh(r.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::cos(r.c[1]) * ::cosh(i.c[1]);
			d2 = -::sin(i.c[1]) * ::sinh(r.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::cos(r.c[2]) * ::cosh(i.c[2]);
			d2 = -::sin(i.c[2]) * ::sinh(r.c[2]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::cos(r.c[3]) * ::cosh(i.c[3]);
			d2 = -::sin(i.c[3]) * ::sinh(r.c[3]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *tan(const AML_PREFIX(VectorU8_4D) mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::sin(r.c[0] + r.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
			d2 = ::sinh(i.c[0] + i.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::sin(r.c[1] + r.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
			d2 = ::sinh(i.c[1] + i.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::sin(r.c[2] + r.c[2]) / (::cos(r.c[2] + r.c[2]) * ::cosh(i.c[2] + i.c[2]));
			d2 = ::sinh(i.c[2] + i.c[2]) / (::cos(r.c[2] + r.c[2]) * ::cosh(i.c[2] + i.c[2]));
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::sin(r.c[3] + r.c[3]) / (::cos(r.c[3] + r.c[3]) * ::cosh(i.c[3] + i.c[3]));
			d2 = ::sinh(i.c[3] + i.c[3]) / (::cos(r.c[3] + r.c[3]) * ::cosh(i.c[3] + i.c[3]));
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array4Complex32) *exp() {
		float d1 = ::exp(r.c[0]) * ::cos(i.c[0]);
		float d2 = ::exp(r.c[0]) * ::sin(i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::exp(r.c[1]) * ::cos(i.c[1]);
		d2 = ::exp(r.c[1]) * ::sin(i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::exp(r.c[2]) * ::cos(i.c[2]);
		d2 = ::exp(r.c[2]) * ::sin(i.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::exp(r.c[3]) * ::cos(i.c[3]);
		d2 = ::exp(r.c[3]) * ::sin(i.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *exp(float n) {
		float d1 = ::pow(n, r.c[0]) * ::cos(i.c[0] * ::log(n));
		float d2 = ::pow(n, r.c[0]) * ::sin(i.c[0] * ::log(n));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::pow(n, r.c[1]) * ::cos(i.c[1] * ::log(n));
		d2 = ::pow(n, r.c[1]) * ::sin(i.c[1] * ::log(n));
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::pow(n, r.c[2]) * ::cos(i.c[2] * ::log(n));
		d2 = ::pow(n, r.c[2]) * ::sin(i.c[2] * ::log(n));
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::pow(n, r.c[3]) * ::cos(i.c[3] * ::log(n));
		d2 = ::pow(n, r.c[3]) * ::sin(i.c[3] * ::log(n));
		r.c[3] = d1;
		i.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *exp(const AML_PREFIX(VectorU8_4D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::exp(r.c[0]) * ::cos(i.c[0]);
			d2 = ::exp(r.c[0]) * ::sin(i.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::exp(r.c[1]) * ::cos(i.c[1]);
			d2 = ::exp(r.c[1]) * ::sin(i.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::exp(r.c[2]) * ::cos(i.c[2]);
			d2 = ::exp(r.c[2]) * ::sin(i.c[2]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::exp(r.c[3]) * ::cos(i.c[3]);
			d2 = ::exp(r.c[3]) * ::sin(i.c[3]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *pow(const AML_PREFIX(Array4Complex32) n) {
		float d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		float d2 = ::atan2(r.c[0], i.c[0]);
		float d3 = ::exp(d1 * n.i.c[0] - d2 * n.r.c[0]);
		float d4 = d1 * n.r.c[0] + d2 * n.i.c[0];
		float d5 = d3 * ::cos(d4);
		float d6 = d3 * ::sin(d4);
		i.c[0] = d5;
		r.c[0] = d6;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		d3 = ::exp(d1 * n.i.c[1] - d2 * n.r.c[1]);
		d4 = d1 * n.r.c[1] + d2 * n.i.c[1];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[1] = d5;
		r.c[1] = d6;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
		d2 = ::atan2(r.c[2], i.c[2]);
		d3 = ::exp(d1 * n.i.c[2] - d2 * n.r.c[2]);
		d4 = d1 * n.r.c[2] + d2 * n.i.c[2];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[2] = d5;
		r.c[2] = d6;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
		d2 = ::atan2(r.c[3], i.c[3]);
		d3 = ::exp(d1 * n.i.c[3] - d2 * n.r.c[3]);
		d4 = d1 * n.r.c[3] + d2 * n.i.c[3];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[3] = d5;
		r.c[3] = d6;
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array4Complex32) *pow(const AML_PREFIX(Complex32) n) {
		float d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		float d2 = ::atan2(r.c[0], i.c[0]);
		float d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		float d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		float d5 = d3 * ::cos(d4);
		float d6 = d3 * ::sin(d4);
		i.c[0] = d5;
		r.c[0] = d6;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[1] = d5;
		r.c[1] = d6;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
		d2 = ::atan2(r.c[2], i.c[2]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[2] = d5;
		r.c[2] = d6;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
		d2 = ::atan2(r.c[3], i.c[3]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[3] = d5;
		r.c[3] = d6;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *
	pow(const AML_PREFIX(Array4Complex32) n, const AML_PREFIX(VectorU8_4D) &mask) {
		float d1;
		float d2;
		float d3;
		float d4;
		float d5;
		float d6;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			d3 = ::exp(d1 * n.i.c[0] - d2 * n.r.c[0]);
			d4 = d1 * n.r.c[0] + d2 * n.i.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[0] = d5;
			r.c[0] = d6;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			d3 = ::exp(d1 * n.i.c[1] - d2 * n.r.c[1]);
			d4 = d1 * n.r.c[1] + d2 * n.i.c[1];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[1] = d5;
			r.c[1] = d6;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
			d2 = ::atan2(r.c[2], i.c[2]);
			d3 = ::exp(d1 * n.i.c[2] - d2 * n.r.c[2]);
			d4 = d1 * n.r.c[2] + d2 * n.i.c[2];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[2] = d5;
			r.c[2] = d6;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
			d2 = ::atan2(r.c[3], i.c[3]);
			d3 = ::exp(d1 * n.i.c[3] - d2 * n.r.c[3]);
			d4 = d1 * n.r.c[3] + d2 * n.i.c[3];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[3] = d5;
			r.c[3] = d6;
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array4Complex32) *pow(AML_PREFIX(Complex32) n, const AML_PREFIX(VectorU8_4D) &mask) {
		float d1;
		float d2;
		float d3;
		float d4;
		float d5;
		float d6;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[0] = d5;
			r.c[0] = d6;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[1] = d5;
			r.c[1] = d6;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
			d2 = ::atan2(r.c[2], i.c[2]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[2] = d5;
			r.c[2] = d6;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
			d2 = ::atan2(r.c[3], i.c[3]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[3] = d5;
			r.c[3] = d6;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *exp(float n, const AML_PREFIX(VectorU8_4D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::pow(n, r.c[0]) * ::cos(i.c[0] * ::log(n));
			d2 = ::pow(n, r.c[0]) * ::sin(i.c[0] * ::log(n));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::pow(n, r.c[1]) * ::cos(i.c[1] * ::log(n));
			d2 = ::pow(n, r.c[1]) * ::sin(i.c[1] * ::log(n));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::pow(n, r.c[2]) * ::cos(i.c[2] * ::log(n));
			d2 = ::pow(n, r.c[2]) * ::sin(i.c[2] * ::log(n));
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::pow(n, r.c[3]) * ::cos(i.c[3] * ::log(n));
			d2 = ::pow(n, r.c[3]) * ::sin(i.c[3] * ::log(n));
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *pow(float n) {
		float d1;
		float d2;
		d1 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::cos(n * atan2(i.c[0], r.c[0]));
		d2 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::sin(n * atan2(i.c[0], r.c[0]));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::cos(n * atan2(i.c[1], r.c[1]));
		d2 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::sin(n * atan2(i.c[1], r.c[1]));
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::pow(r.c[2] * r.c[2] + i.c[2] * i.c[2], n / 2) * ::cos(n * atan2(i.c[2], r.c[2]));
		d2 = ::pow(r.c[2] * r.c[2] + i.c[2] * i.c[2], n / 2) * ::sin(n * atan2(i.c[2], r.c[2]));
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::pow(r.c[3] * r.c[3] + i.c[3] * i.c[3], n / 2) * ::cos(n * atan2(i.c[3], r.c[3]));
		d2 = ::pow(r.c[3] * r.c[3] + i.c[3] * i.c[3], n / 2) * ::sin(n * atan2(i.c[3], r.c[3]));
		r.c[3] = d1;
		i.c[3] = d2;

		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *pow(float n, const AML_PREFIX(VectorU8_4D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::cos(n * atan2(i.c[0], r.c[0]));
			d2 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::sin(n * atan2(i.c[0], r.c[0]));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::cos(n * atan2(i.c[1], r.c[1]));
			d2 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::sin(n * atan2(i.c[1], r.c[1]));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::pow(r.c[2] * r.c[2] + i.c[2] * i.c[2], n / 2) * ::cos(n * atan2(i.c[2], r.c[2]));
			d2 = ::pow(r.c[2] * r.c[2] + i.c[2] * i.c[2], n / 2) * ::sin(n * atan2(i.c[2], r.c[2]));
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::pow(r.c[3] * r.c[3] + i.c[3] * i.c[3], n / 2) * ::cos(n * atan2(i.c[3], r.c[3]));
			d2 = ::pow(r.c[3] * r.c[3] + i.c[3] * i.c[3], n / 2) * ::sin(n * atan2(i.c[3], r.c[3]));
			r.c[3] = d1;
			i.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat4D) abs() {
		AML_PREFIX(VectorFloat4D) ret;
		ret.v.c[0] = ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0]);
		ret.v.c[1] = ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1]);
		ret.v.c[2] = ::sqrt(r.c[2] * r.c[2] + i.c[2] * i.c[2]);
		ret.v.c[3] = ::sqrt(r.c[3] * r.c[3] + i.c[3] * i.c[3]);
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_4D) abs_gt(float a) {
		AML_PREFIX(VectorU8_4D) ret;
		ret.v.c[0] = a * a < r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a < r.c[1] * r.c[1] + i.c[1] * i.c[1];
		ret.v.c[2] = a * a < r.c[2] * r.c[2] + i.c[2] * i.c[2];
		ret.v.c[3] = a * a < r.c[3] * r.c[3] + i.c[3] * i.c[3];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_4D) abs_lt(float a) {
		AML_PREFIX(VectorU8_4D) ret;
		ret.v.c[0] = a * a > r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a > r.c[1] * r.c[1] + i.c[1] * i.c[1];
		ret.v.c[2] = a * a > r.c[2] * r.c[2] + i.c[2] * i.c[2];
		ret.v.c[3] = a * a > r.c[3] * r.c[3] + i.c[3] * i.c[3];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_4D) abs_eq(float a) {
		AML_PREFIX(VectorU8_4D) ret;
		ret.v.c[0] = a * a == r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a == r.c[1] * r.c[1] + i.c[1] * i.c[1];
		ret.v.c[2] = a * a == r.c[2] * r.c[2] + i.c[2] * i.c[2];
		ret.v.c[3] = a * a == r.c[3] * r.c[3] + i.c[3] * i.c[3];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *ln(const AML_PREFIX(VectorU8_4D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			i.c[1] = d1;
			r.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
			d2 = ::atan2(r.c[2], i.c[2]);
			i.c[2] = d1;
			r.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
			d2 = ::atan2(r.c[3], i.c[3]);
			i.c[3] = d1;
			r.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *log(const AML_PREFIX(VectorU8_4D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			i.c[1] = d1;
			r.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
			d2 = ::atan2(r.c[2], i.c[2]);
			i.c[2] = d1;
			r.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
			d2 = ::atan2(r.c[3], i.c[3]);
			i.c[3] = d1;
			r.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *log10(const AML_PREFIX(VectorU8_4D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[0], i.c[0]) / AML_LN10;
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[1], i.c[1]) / AML_LN10;
			i.c[1] = d1;
			r.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[2], i.c[2]) / AML_LN10;
			i.c[2] = d1;
			r.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[3], i.c[3]) / AML_LN10;
			i.c[3] = d1;
			r.c[3] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *ln() {
		float d1;
		float d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		d2 = ::atan2(r.c[0], i.c[0]);
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		i.c[1] = d1;
		r.c[1] = d2;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
		d2 = ::atan2(r.c[2], i.c[2]);
		i.c[2] = d1;
		r.c[2] = d2;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
		d2 = ::atan2(r.c[3], i.c[3]);
		i.c[3] = d1;
		r.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *log() {
		float d1;
		float d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		d2 = ::atan2(r.c[0], i.c[0]);
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		i.c[1] = d1;
		r.c[1] = d2;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
		d2 = ::atan2(r.c[2], i.c[2]);
		i.c[2] = d1;
		r.c[2] = d2;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
		d2 = ::atan2(r.c[3], i.c[3]);
		i.c[3] = d1;
		r.c[3] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array4Complex32) *log10() {
		float d1;
		float d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[0], i.c[0]) / AML_LN10;
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[1], i.c[1]) / AML_LN10;
		i.c[1] = d1;
		r.c[1] = d2;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[2], i.c[2]) / AML_LN10;
		i.c[2] = d1;
		r.c[2] = d2;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[3], i.c[3]) / AML_LN10;
		i.c[3] = d1;
		r.c[3] = d2;
		return this;
	}
};


#if !defined(AML_NO_STRING)

AML_FUNCTION std::string operator<<(std::string &lhs, const AML_PREFIX(Array4Complex32) &rhs) {
	std::ostringstream string;
	string << lhs << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1]
		   << "i ,  " << rhs.r.c[2] << " + " << rhs.i.c[2] << "i ,  " << rhs.r.c[3] << " + " << rhs.i.c[3] << "i }";
	return string.str();
}

AML_FUNCTION std::string operator<<(const char *lhs, const AML_PREFIX(Array4Complex32) &rhs) {
	std::ostringstream string;
	string << lhs << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1]
		   << "i ,  " << rhs.r.c[2] << " + " << rhs.i.c[2] << "i ,  " << rhs.r.c[3] << " + " << rhs.i.c[3] << "i }";
	return string.str();
}

template<class charT, class traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &o, const AML_PREFIX(Array4Complex32) &rhs) {
	std::basic_ostringstream<charT, traits> s;
	s.flags(o.flags());
	s.imbue(o.getloc());
	s.precision(o.precision());
	s << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1]
	  << "i ,  " << rhs.r.c[2] << " + " << rhs.i.c[2] << "i ,  " << rhs.r.c[3] << " + " << rhs.i.c[3] << "i }";
	return o << s.str();
}

#endif

AML_FUNCTION AML_PREFIX(Array4Complex32)
operator+(const AML_PREFIX(Complex32) &lhs, const AML_PREFIX(Array4Complex32) &rhs) {
	return rhs + lhs;
}


AML_FUNCTION AML_PREFIX(Array4Complex32)
operator-(const AML_PREFIX(Complex32) &lhs, const AML_PREFIX(Array4Complex32) &rhs) {
	AML_PREFIX(Array4Complex32) ret;
	ret.i.c[0] = lhs.c.c[1] - rhs.i.c[0];
	ret.i.c[1] = lhs.c.c[1] - rhs.i.c[1];
	ret.i.c[2] = lhs.c.c[1] - rhs.i.c[2];
	ret.i.c[3] = lhs.c.c[1] - rhs.i.c[3];
	ret.r.c[0] = lhs.c.c[0] - rhs.r.c[0];
	ret.r.c[1] = lhs.c.c[0] - rhs.r.c[1];
	ret.r.c[2] = lhs.c.c[0] - rhs.r.c[2];
	ret.r.c[3] = lhs.c.c[0] - rhs.r.c[3];
	return ret;
}

AML_FUNCTION AML_PREFIX(Array4Complex32)
operator*(const AML_PREFIX(Complex32) &lhs, const AML_PREFIX(Array4Complex32) &rhs) {
	return rhs * lhs;
}

AML_FUNCTION AML_PREFIX(Array4Complex32)
operator/(const AML_PREFIX(Complex32) &lhs, const AML_PREFIX(Array4Complex32) &rhs) {
	AML_PREFIX(Array4Complex32) ret;
	float d1 =
			(lhs.c.c[0] * rhs.r.c[0] + lhs.c.c[1] * rhs.i.c[0]) / (rhs.r.c[0] * rhs.r.c[0] + rhs.i.c[0] * rhs.i.c[0]);
	float d2 =
			(lhs.c.c[1] * rhs.r.c[0] - lhs.c.c[0] * rhs.i.c[0]) / (rhs.r.c[0] * rhs.r.c[0] + rhs.i.c[0] * rhs.i.c[0]);
	ret.r.c[0] = d1;
	ret.i.c[0] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[1] + lhs.c.c[1] * rhs.i.c[1]) / (rhs.r.c[1] * rhs.r.c[1] + rhs.i.c[1] * rhs.i.c[1]);
	d2 = (lhs.c.c[1] * rhs.r.c[1] - lhs.c.c[0] * rhs.i.c[1]) / (rhs.r.c[1] * rhs.r.c[1] + rhs.i.c[1] * rhs.i.c[1]);
	ret.r.c[1] = d1;
	ret.i.c[1] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[2] + lhs.c.c[1] * rhs.i.c[2]) / (rhs.r.c[2] * rhs.r.c[2] + rhs.i.c[2] * rhs.i.c[2]);
	d2 = (lhs.c.c[1] * rhs.r.c[2] - lhs.c.c[0] * rhs.i.c[2]) / (rhs.r.c[2] * rhs.r.c[2] + rhs.i.c[2] * rhs.i.c[2]);
	ret.r.c[2] = d1;
	ret.i.c[2] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[3] + lhs.c.c[1] * rhs.i.c[3]) / (rhs.r.c[3] * rhs.r.c[3] + rhs.i.c[3] * rhs.i.c[3]);
	d2 = (lhs.c.c[1] * rhs.r.c[3] - lhs.c.c[0] * rhs.i.c[3]) / (rhs.r.c[3] * rhs.r.c[3] + rhs.i.c[3] * rhs.i.c[3]);
	ret.r.c[3] = d1;
	ret.i.c[3] = d2;
	return ret;
}


class AML_PREFIX(Array8Complex32) {
public:
	AML_PREFIX(floatvec8) r{};
	AML_PREFIX(floatvec8) i{};

	AML_FUNCTION AML_PREFIX(Array8Complex32)() {}

	AML_FUNCTION AML_PREFIX(Array8Complex32)(const AML_PREFIX(Complex32) value) {
		r.c[0] = value.c.c[0];
		i.c[0] = value.c.c[1];
		r.c[1] = value.c.c[0];
		i.c[1] = value.c.c[1];
		r.c[2] = value.c.c[0];
		i.c[2] = value.c.c[1];
		r.c[3] = value.c.c[0];
		i.c[3] = value.c.c[1];
		r.c[4] = value.c.c[0];
		i.c[4] = value.c.c[1];
		r.c[5] = value.c.c[0];
		i.c[5] = value.c.c[1];
		r.c[6] = value.c.c[0];
		i.c[6] = value.c.c[1];
		r.c[7] = value.c.c[0];
		i.c[7] = value.c.c[1];
	}

	AML_FUNCTION AML_PREFIX(VectorFloat8D) real() {
		return AML_PREFIX(VectorFloat8D)(r.c);
	}

	AML_FUNCTION AML_PREFIX(VectorFloat8D) complex() {
		return AML_PREFIX(VectorFloat8D)(i.c);
	}

	AML_FUNCTION AML_PREFIX(Complex32) operator[](uint64_t location) {
		return AML_PREFIX(Complex32)(r.c[location], i.c[location]);
	}

	AML_FUNCTION void set(uint64_t location, AML_PREFIX(Complex32) value) {
		r.c[location] = value.c.c[0];
		i.c[location] = value.c.c[1];
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *add(const AML_PREFIX(Array8Complex32) &a) {
		i.c[0] += a.i.c[0];
		i.c[1] += a.i.c[1];
		i.c[2] += a.i.c[2];
		i.c[3] += a.i.c[3];
		i.c[4] += a.i.c[4];
		i.c[5] += a.i.c[5];
		i.c[6] += a.i.c[6];
		i.c[7] += a.i.c[7];
		r.c[0] += a.r.c[0];
		r.c[1] += a.r.c[1];
		r.c[2] += a.r.c[2];
		r.c[3] += a.r.c[3];
		r.c[4] += a.r.c[4];
		r.c[5] += a.r.c[5];
		r.c[6] += a.r.c[6];
		r.c[7] += a.r.c[7];
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *add(const AML_PREFIX(Complex32) a) {
		i.c[0] += a.c.c[1];
		i.c[1] += a.c.c[1];
		i.c[2] += a.c.c[1];
		i.c[3] += a.c.c[1];
		i.c[4] += a.c.c[1];
		i.c[5] += a.c.c[1];
		i.c[6] += a.c.c[1];
		i.c[7] += a.c.c[1];
		r.c[0] += a.c.c[0];
		r.c[1] += a.c.c[0];
		r.c[2] += a.c.c[0];
		r.c[3] += a.c.c[0];
		r.c[4] += a.c.c[0];
		r.c[5] += a.c.c[0];
		r.c[6] += a.c.c[0];
		r.c[7] += a.c.c[0];
		return this;
	}


	AML_FUNCTION void operator+=(const AML_PREFIX(Array8Complex32) &a) {
		i.c[0] += a.i.c[0];
		i.c[1] += a.i.c[1];
		i.c[2] += a.i.c[2];
		i.c[3] += a.i.c[3];
		i.c[4] += a.i.c[4];
		i.c[5] += a.i.c[5];
		i.c[6] += a.i.c[6];
		i.c[7] += a.i.c[7];
		r.c[0] += a.r.c[0];
		r.c[1] += a.r.c[1];
		r.c[2] += a.r.c[2];
		r.c[3] += a.r.c[3];
		r.c[4] += a.r.c[4];
		r.c[5] += a.r.c[5];
		r.c[6] += a.r.c[6];
		r.c[7] += a.r.c[7];
	}

	AML_FUNCTION void operator+=(const AML_PREFIX(Complex32) a) {
		i.c[0] += a.c.c[1];
		i.c[1] += a.c.c[1];
		i.c[2] += a.c.c[1];
		i.c[3] += a.c.c[1];
		i.c[4] += a.c.c[1];
		i.c[5] += a.c.c[1];
		i.c[6] += a.c.c[1];
		i.c[7] += a.c.c[1];
		r.c[0] += a.c.c[0];
		r.c[1] += a.c.c[0];
		r.c[2] += a.c.c[0];
		r.c[3] += a.c.c[0];
		r.c[4] += a.c.c[0];
		r.c[5] += a.c.c[0];
		r.c[6] += a.c.c[0];
		r.c[7] += a.c.c[0];
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) operator+(const AML_PREFIX(Array8Complex32) a) const {
		AML_PREFIX(Array8Complex32) ret;
		ret.i.c[0] = i.c[0] + a.i.c[0];
		ret.i.c[1] = i.c[1] + a.i.c[1];
		ret.i.c[2] = i.c[2] + a.i.c[2];
		ret.i.c[3] = i.c[3] + a.i.c[3];
		ret.i.c[4] = i.c[4] + a.i.c[4];
		ret.i.c[5] = i.c[5] + a.i.c[5];
		ret.i.c[6] = i.c[6] + a.i.c[6];
		ret.i.c[7] = i.c[7] + a.i.c[7];
		ret.r.c[0] = r.c[0] + a.r.c[0];
		ret.r.c[1] = r.c[1] + a.r.c[1];
		ret.r.c[2] = r.c[2] + a.r.c[2];
		ret.r.c[3] = r.c[3] + a.r.c[3];
		ret.r.c[4] = r.c[4] + a.r.c[4];
		ret.r.c[5] = r.c[5] + a.r.c[5];
		ret.r.c[6] = r.c[6] + a.r.c[6];
		ret.r.c[7] = r.c[7] + a.r.c[7];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) operator+(AML_PREFIX(Complex32) a) const {
		AML_PREFIX(Array8Complex32) ret;
		ret.i.c[0] = i.c[0] + a.c.c[1];
		ret.i.c[1] = i.c[1] + a.c.c[1];
		ret.i.c[2] = i.c[2] + a.c.c[1];
		ret.i.c[3] = i.c[3] + a.c.c[1];
		ret.i.c[4] = i.c[4] + a.c.c[1];
		ret.i.c[5] = i.c[5] + a.c.c[1];
		ret.i.c[6] = i.c[6] + a.c.c[1];
		ret.i.c[7] = i.c[7] + a.c.c[1];
		ret.r.c[0] = r.c[0] + a.c.c[0];
		ret.r.c[1] = r.c[1] + a.c.c[0];
		ret.r.c[2] = r.c[2] + a.c.c[0];
		ret.r.c[3] = r.c[3] + a.c.c[0];
		ret.r.c[4] = r.c[4] + a.c.c[0];
		ret.r.c[5] = r.c[5] + a.c.c[0];
		ret.r.c[6] = r.c[6] + a.c.c[0];
		ret.r.c[7] = r.c[7] + a.c.c[0];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *add(const AML_PREFIX(Array8Complex32) &a, AML_PREFIX(VectorU8_8D) mask) {
		if (mask.v.c[0]) {
			i.c[0] += a.i.c[0];
			r.c[0] += a.r.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] += a.i.c[1];
			r.c[1] += a.r.c[1];
		}
		if (mask.v.c[2]) {
			i.c[2] += a.i.c[2];
			r.c[2] += a.r.c[2];
		}
		if (mask.v.c[3]) {
			i.c[3] += a.i.c[3];
			r.c[3] += a.r.c[3];
		}
		if (mask.v.c[4]) {
			i.c[4] += a.i.c[4];
			r.c[4] += a.r.c[4];
		}
		if (mask.v.c[5]) {
			i.c[5] += a.i.c[5];
			r.c[5] += a.r.c[5];
		}
		if (mask.v.c[6]) {
			i.c[6] += a.i.c[6];
			r.c[6] += a.r.c[6];
		}
		if (mask.v.c[7]) {
			i.c[7] += a.i.c[7];
			r.c[7] += a.r.c[7];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *add(AML_PREFIX(Complex32) a, AML_PREFIX(VectorU8_8D) mask) {
		if (mask.v.c[0]) {
			i.c[0] += a.c.c[1];
			r.c[0] += a.c.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] += a.c.c[1];
			r.c[1] += a.c.c[0];
		}
		if (mask.v.c[2]) {
			i.c[2] += a.c.c[1];
			r.c[2] += a.c.c[0];
		}
		if (mask.v.c[3]) {
			i.c[3] += a.c.c[1];
			r.c[3] += a.c.c[0];
		}
		if (mask.v.c[4]) {
			i.c[4] += a.c.c[1];
			r.c[4] += a.c.c[0];
		}
		if (mask.v.c[5]) {
			i.c[5] += a.c.c[1];
			r.c[5] += a.c.c[0];
		}
		if (mask.v.c[6]) {
			i.c[6] += a.c.c[1];
			r.c[6] += a.c.c[0];
		}
		if (mask.v.c[7]) {
			i.c[7] += a.c.c[1];
			r.c[7] += a.c.c[0];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *subtract(const AML_PREFIX(Array8Complex32) &a) {
		i.c[0] -= a.i.c[0];
		i.c[1] -= a.i.c[1];
		i.c[2] -= a.i.c[2];
		i.c[3] -= a.i.c[3];
		i.c[4] -= a.i.c[4];
		i.c[5] -= a.i.c[5];
		i.c[6] -= a.i.c[6];
		i.c[7] -= a.i.c[7];
		r.c[0] -= a.r.c[0];
		r.c[1] -= a.r.c[1];
		r.c[2] -= a.r.c[2];
		r.c[3] -= a.r.c[3];
		r.c[4] -= a.r.c[4];
		r.c[5] -= a.r.c[5];
		r.c[6] -= a.r.c[6];
		r.c[7] -= a.r.c[7];
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *subtract(const AML_PREFIX(Complex32) a) {
		i.c[0] -= a.c.c[1];
		i.c[1] -= a.c.c[1];
		i.c[2] -= a.c.c[1];
		i.c[3] -= a.c.c[1];
		i.c[4] -= a.c.c[1];
		i.c[5] -= a.c.c[1];
		i.c[6] -= a.c.c[1];
		i.c[7] -= a.c.c[1];
		r.c[0] -= a.c.c[0];
		r.c[1] -= a.c.c[0];
		r.c[2] -= a.c.c[0];
		r.c[3] -= a.c.c[0];
		r.c[4] -= a.c.c[0];
		r.c[5] -= a.c.c[0];
		r.c[6] -= a.c.c[0];
		r.c[7] -= a.c.c[0];
		return this;
	}

	AML_FUNCTION void operator-=(const AML_PREFIX(Array8Complex32) &a) {
		i.c[0] -= a.i.c[0];
		i.c[1] -= a.i.c[1];
		i.c[2] -= a.i.c[2];
		i.c[3] -= a.i.c[3];
		i.c[4] -= a.i.c[4];
		i.c[5] -= a.i.c[5];
		i.c[6] -= a.i.c[6];
		i.c[7] -= a.i.c[7];
		r.c[0] -= a.r.c[0];
		r.c[1] -= a.r.c[1];
		r.c[2] -= a.r.c[2];
		r.c[3] -= a.r.c[3];
		r.c[4] -= a.r.c[4];
		r.c[5] -= a.r.c[5];
		r.c[6] -= a.r.c[6];
		r.c[7] -= a.r.c[7];
	}

	AML_FUNCTION void operator-=(const AML_PREFIX(Complex32) a) {
		i.c[0] -= a.c.c[1];
		i.c[1] -= a.c.c[1];
		i.c[2] -= a.c.c[1];
		i.c[3] -= a.c.c[1];
		i.c[4] -= a.c.c[1];
		i.c[5] -= a.c.c[1];
		i.c[6] -= a.c.c[1];
		i.c[7] -= a.c.c[1];
		r.c[0] -= a.c.c[0];
		r.c[1] -= a.c.c[0];
		r.c[2] -= a.c.c[0];
		r.c[3] -= a.c.c[0];
		r.c[4] -= a.c.c[0];
		r.c[5] -= a.c.c[0];
		r.c[6] -= a.c.c[0];
		r.c[7] -= a.c.c[0];
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) operator-(const AML_PREFIX(Array8Complex32) a) const {
		AML_PREFIX(Array8Complex32) ret;
		ret.i.c[0] = i.c[0] - a.i.c[0];
		ret.i.c[1] = i.c[1] - a.i.c[1];
		ret.i.c[2] = i.c[2] - a.i.c[2];
		ret.i.c[3] = i.c[3] - a.i.c[3];
		ret.i.c[4] = i.c[4] - a.i.c[4];
		ret.i.c[5] = i.c[5] - a.i.c[5];
		ret.i.c[6] = i.c[6] - a.i.c[6];
		ret.i.c[7] = i.c[7] - a.i.c[7];
		ret.r.c[0] = r.c[0] - a.r.c[0];
		ret.r.c[1] = r.c[1] - a.r.c[1];
		ret.r.c[2] = r.c[2] - a.r.c[2];
		ret.r.c[3] = r.c[3] - a.r.c[3];
		ret.r.c[4] = r.c[4] - a.r.c[4];
		ret.r.c[5] = r.c[5] - a.r.c[5];
		ret.r.c[6] = r.c[6] - a.r.c[6];
		ret.r.c[7] = r.c[7] - a.r.c[7];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) operator-(const AML_PREFIX(Complex32) a) const {
		AML_PREFIX(Array8Complex32) ret;
		ret.i.c[0] = i.c[0] - a.c.c[1];
		ret.i.c[1] = i.c[1] - a.c.c[1];
		ret.i.c[2] = i.c[2] - a.c.c[1];
		ret.i.c[3] = i.c[3] - a.c.c[1];
		ret.i.c[4] = i.c[4] - a.c.c[1];
		ret.i.c[5] = i.c[5] - a.c.c[1];
		ret.i.c[6] = i.c[6] - a.c.c[1];
		ret.i.c[7] = i.c[7] - a.c.c[1];
		ret.r.c[0] = r.c[0] - a.c.c[0];
		ret.r.c[1] = r.c[1] - a.c.c[0];
		ret.r.c[2] = r.c[2] - a.c.c[0];
		ret.r.c[3] = r.c[3] - a.c.c[0];
		ret.r.c[4] = r.c[4] - a.c.c[0];
		ret.r.c[5] = r.c[5] - a.c.c[0];
		ret.r.c[6] = r.c[6] - a.c.c[0];
		ret.r.c[7] = r.c[7] - a.c.c[0];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *
	subtract(const AML_PREFIX(Array8Complex32) &a, AML_PREFIX(VectorU8_8D) mask) {
		if (mask.v.c[0]) {
			i.c[0] -= a.i.c[0];
			r.c[0] -= a.r.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] -= a.i.c[1];
			r.c[1] -= a.r.c[1];
		}
		if (mask.v.c[2]) {
			i.c[2] -= a.i.c[2];
			r.c[2] -= a.r.c[2];
		}
		if (mask.v.c[3]) {
			i.c[3] -= a.i.c[3];
			r.c[3] -= a.r.c[3];
		}
		if (mask.v.c[4]) {
			i.c[4] -= a.i.c[4];
			r.c[4] -= a.r.c[4];
		}
		if (mask.v.c[5]) {
			i.c[5] -= a.i.c[5];
			r.c[5] -= a.r.c[5];
		}
		if (mask.v.c[6]) {
			i.c[6] -= a.i.c[6];
			r.c[6] -= a.r.c[6];
		}
		if (mask.v.c[7]) {
			i.c[7] -= a.i.c[7];
			r.c[7] -= a.r.c[7];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *subtract(const AML_PREFIX(Complex32) a, AML_PREFIX(VectorU8_8D) mask) {
		if (mask.v.c[0]) {
			i.c[0] -= a.c.c[1];
			r.c[0] -= a.c.c[0];
		}
		if (mask.v.c[1]) {
			i.c[1] -= a.c.c[1];
			r.c[1] -= a.c.c[0];
		}
		if (mask.v.c[2]) {
			i.c[2] -= a.c.c[1];
			r.c[2] -= a.c.c[0];
		}
		if (mask.v.c[3]) {
			i.c[3] -= a.c.c[1];
			r.c[3] -= a.c.c[0];
		}
		if (mask.v.c[4]) {
			i.c[4] -= a.c.c[1];
			r.c[4] -= a.c.c[0];
		}
		if (mask.v.c[5]) {
			i.c[5] -= a.c.c[1];
			r.c[5] -= a.c.c[0];
		}
		if (mask.v.c[6]) {
			i.c[6] -= a.c.c[1];
			r.c[6] -= a.c.c[0];
		}
		if (mask.v.c[7]) {
			i.c[7] -= a.c.c[1];
			r.c[7] -= a.c.c[0];
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array8Complex32) operator*(const AML_PREFIX(Array8Complex32) &a) const {
		AML_PREFIX(Array8Complex32) ret;
		ret.r.c[0] = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		ret.i.c[0] = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		ret.r.c[1] = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		ret.i.c[1] = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
		ret.r.c[2] = r.c[2] * a.r.c[2] - i.c[2] * a.i.c[2];
		ret.i.c[2] = r.c[2] * a.i.c[2] + i.c[2] * a.r.c[2];
		ret.r.c[3] = r.c[3] * a.r.c[3] - i.c[3] * a.i.c[3];
		ret.i.c[3] = r.c[3] * a.i.c[3] + i.c[3] * a.r.c[3];
		ret.r.c[4] = r.c[4] * a.r.c[4] - i.c[4] * a.i.c[4];
		ret.i.c[4] = r.c[4] * a.i.c[4] + i.c[4] * a.r.c[4];
		ret.r.c[5] = r.c[5] * a.r.c[5] - i.c[5] * a.i.c[5];
		ret.i.c[5] = r.c[5] * a.i.c[5] + i.c[5] * a.r.c[5];
		ret.r.c[6] = r.c[6] * a.r.c[6] - i.c[6] * a.i.c[6];
		ret.i.c[6] = r.c[6] * a.i.c[6] + i.c[6] * a.r.c[6];
		ret.r.c[7] = r.c[7] * a.r.c[7] - i.c[7] * a.i.c[7];
		ret.i.c[7] = r.c[7] * a.i.c[7] + i.c[7] * a.r.c[7];


		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *multiply(const AML_PREFIX(Array8Complex32) &a) {
		float d1;
		float d2;
		d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;

		d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * a.r.c[2] - i.c[2] * a.i.c[2];
		d2 = r.c[2] * a.i.c[2] + i.c[2] * a.r.c[2];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * a.r.c[3] - i.c[3] * a.i.c[3];
		d2 = r.c[3] * a.i.c[3] + i.c[3] * a.r.c[3];
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = r.c[4] * a.r.c[4] - i.c[4] * a.i.c[4];
		d2 = r.c[4] * a.i.c[4] + i.c[4] * a.r.c[4];
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = r.c[5] * a.r.c[5] - i.c[5] * a.i.c[5];
		d2 = r.c[5] * a.i.c[5] + i.c[5] * a.r.c[5];
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = r.c[6] * a.r.c[6] - i.c[6] * a.i.c[6];
		d2 = r.c[6] * a.i.c[6] + i.c[6] * a.r.c[6];
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = r.c[7] * a.r.c[7] - i.c[7] * a.i.c[7];
		d2 = r.c[7] * a.i.c[7] + i.c[7] * a.r.c[7];
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION void operator*=(const AML_PREFIX(Array8Complex32) &a) {
		float d1;
		float d2;
		d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
		d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;

		d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
		d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * a.r.c[2] - i.c[2] * a.i.c[2];
		d2 = r.c[2] * a.i.c[2] + i.c[2] * a.r.c[2];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * a.r.c[3] - i.c[3] * a.i.c[3];
		d2 = r.c[3] * a.i.c[3] + i.c[3] * a.r.c[3];
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = r.c[4] * a.r.c[4] - i.c[4] * a.i.c[4];
		d2 = r.c[4] * a.i.c[4] + i.c[4] * a.r.c[4];
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = r.c[5] * a.r.c[5] - i.c[5] * a.i.c[5];
		d2 = r.c[5] * a.i.c[5] + i.c[5] * a.r.c[5];
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = r.c[6] * a.r.c[6] - i.c[6] * a.i.c[6];
		d2 = r.c[6] * a.i.c[6] + i.c[6] * a.r.c[6];
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = r.c[7] * a.r.c[7] - i.c[7] * a.i.c[7];
		d2 = r.c[7] * a.i.c[7] + i.c[7] * a.r.c[7];
		r.c[7] = d1;
		i.c[7] = d2;
	}

	AML_FUNCTION void operator*=(const AML_PREFIX(Complex32) &a) {
		float d1;
		float d2;
		d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * a.c.c[0] - i.c[2] * a.c.c[1];
		d2 = r.c[2] * a.c.c[1] + i.c[2] * a.c.c[0];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * a.c.c[0] - i.c[3] * a.c.c[1];
		d2 = r.c[3] * a.c.c[1] + i.c[3] * a.c.c[0];
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = r.c[4] * a.c.c[0] - i.c[4] * a.c.c[1];
		d2 = r.c[4] * a.c.c[1] + i.c[4] * a.c.c[0];
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = r.c[5] * a.c.c[0] - i.c[5] * a.c.c[1];
		d2 = r.c[5] * a.c.c[1] + i.c[5] * a.c.c[0];
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = r.c[6] * a.c.c[0] - i.c[6] * a.c.c[1];
		d2 = r.c[6] * a.c.c[1] + i.c[6] * a.c.c[0];
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = r.c[7] * a.c.c[0] - i.c[7] * a.c.c[1];
		d2 = r.c[7] * a.c.c[1] + i.c[7] * a.c.c[0];
		r.c[7] = d1;
		i.c[7] = d2;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) operator*(const AML_PREFIX(Complex32) &a) const {
		AML_PREFIX(Array8Complex32) ret;
		ret.r.c[0] = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		ret.i.c[0] = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		ret.r.c[1] = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		ret.i.c[1] = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
		ret.r.c[2] = r.c[2] * a.c.c[0] - i.c[2] * a.c.c[1];
		ret.i.c[2] = r.c[2] * a.c.c[1] + i.c[2] * a.c.c[0];
		ret.r.c[3] = r.c[3] * a.c.c[0] - i.c[3] * a.c.c[1];
		ret.i.c[3] = r.c[3] * a.c.c[1] + i.c[3] * a.c.c[0];
		ret.r.c[4] = r.c[4] * a.c.c[0] - i.c[4] * a.c.c[1];
		ret.i.c[4] = r.c[4] * a.c.c[1] + i.c[4] * a.c.c[0];
		ret.r.c[5] = r.c[5] * a.c.c[0] - i.c[5] * a.c.c[1];
		ret.i.c[5] = r.c[5] * a.c.c[1] + i.c[5] * a.c.c[0];
		ret.r.c[6] = r.c[6] * a.c.c[0] - i.c[6] * a.c.c[1];
		ret.i.c[6] = r.c[6] * a.c.c[1] + i.c[6] * a.c.c[0];
		ret.r.c[7] = r.c[7] * a.c.c[0] - i.c[7] * a.c.c[1];
		ret.i.c[7] = r.c[7] * a.c.c[1] + i.c[7] * a.c.c[0];


		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *multiply(const AML_PREFIX(Complex32) &a) {
		float d1;
		float d2;
		d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
		d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
		d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * a.c.c[0] - i.c[2] * a.c.c[1];
		d2 = r.c[2] * a.c.c[1] + i.c[2] * a.c.c[0];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * a.c.c[0] - i.c[3] * a.c.c[1];
		d2 = r.c[3] * a.c.c[1] + i.c[3] * a.c.c[0];
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = r.c[4] * a.c.c[0] - i.c[4] * a.c.c[1];
		d2 = r.c[4] * a.c.c[1] + i.c[4] * a.c.c[0];
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = r.c[5] * a.c.c[0] - i.c[5] * a.c.c[1];
		d2 = r.c[5] * a.c.c[1] + i.c[5] * a.c.c[0];
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = r.c[6] * a.c.c[0] - i.c[6] * a.c.c[1];
		d2 = r.c[6] * a.c.c[1] + i.c[6] * a.c.c[0];
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = r.c[7] * a.c.c[0] - i.c[7] * a.c.c[1];
		d2 = r.c[7] * a.c.c[1] + i.c[7] * a.c.c[0];
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *
	multiply(const AML_PREFIX(Complex32) &a, const AML_PREFIX(VectorU8_8D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * a.c.c[0] - i.c[0] * a.c.c[1];
			d2 = r.c[0] * a.c.c[1] + i.c[0] * a.c.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * a.c.c[0] - i.c[1] * a.c.c[1];
			d2 = r.c[1] * a.c.c[1] + i.c[1] * a.c.c[0];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = r.c[2] * a.c.c[0] - i.c[2] * a.c.c[1];
			d2 = r.c[2] * a.c.c[1] + i.c[2] * a.c.c[0];
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = r.c[3] * a.c.c[0] - i.c[3] * a.c.c[1];
			d2 = r.c[3] * a.c.c[1] + i.c[3] * a.c.c[0];
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = r.c[4] * a.c.c[0] - i.c[4] * a.c.c[1];
			d2 = r.c[4] * a.c.c[1] + i.c[4] * a.c.c[0];
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = r.c[5] * a.c.c[0] - i.c[5] * a.c.c[1];
			d2 = r.c[5] * a.c.c[1] + i.c[5] * a.c.c[0];
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = r.c[6] * a.c.c[0] - i.c[6] * a.c.c[1];
			d2 = r.c[6] * a.c.c[1] + i.c[6] * a.c.c[0];
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = r.c[7] * a.c.c[0] - i.c[7] * a.c.c[1];
			d2 = r.c[7] * a.c.c[1] + i.c[7] * a.c.c[0];
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;


	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *
	multiply(const AML_PREFIX(Array8Complex32) &a, const AML_PREFIX(VectorU8_8D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * a.r.c[0] - i.c[0] * a.i.c[0];
			d2 = r.c[0] * a.i.c[0] + i.c[0] * a.r.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * a.r.c[1] - i.c[1] * a.i.c[1];
			d2 = r.c[1] * a.i.c[1] + i.c[1] * a.r.c[1];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = r.c[2] * a.r.c[2] - i.c[2] * a.i.c[2];
			d2 = r.c[2] * a.i.c[2] + i.c[2] * a.r.c[2];
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = r.c[3] * a.r.c[3] - i.c[3] * a.i.c[3];
			d2 = r.c[3] * a.i.c[3] + i.c[3] * a.r.c[3];
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = r.c[4] * a.r.c[4] - i.c[4] * a.i.c[4];
			d2 = r.c[4] * a.i.c[4] + i.c[4] * a.r.c[4];
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = r.c[5] * a.r.c[5] - i.c[5] * a.i.c[5];
			d2 = r.c[5] * a.i.c[5] + i.c[5] * a.r.c[5];
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = r.c[6] * a.r.c[6] - i.c[6] * a.i.c[6];
			d2 = r.c[6] * a.i.c[6] + i.c[6] * a.r.c[6];
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = r.c[7] * a.r.c[7] - i.c[7] * a.i.c[7];
			d2 = r.c[7] * a.i.c[7] + i.c[7] * a.r.c[7];
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *square() {
		float d1;
		float d2;
		d1 = r.c[0] * r.c[0] - i.c[0] * i.c[0];
		d2 = r.c[0] * i.c[0] + i.c[0] * r.c[0];
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = r.c[1] * r.c[1] - i.c[1] * i.c[1];
		d2 = r.c[1] * i.c[1] + i.c[1] * r.c[1];
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = r.c[2] * r.c[2] - i.c[2] * i.c[2];
		d2 = r.c[2] * i.c[2] + i.c[2] * r.c[2];
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = r.c[3] * r.c[3] - i.c[3] * i.c[3];
		d2 = r.c[3] * i.c[3] + i.c[3] * r.c[3];
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = r.c[4] * r.c[4] - i.c[4] * i.c[4];
		d2 = r.c[4] * i.c[4] + i.c[4] * r.c[4];
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = r.c[5] * r.c[5] - i.c[5] * i.c[5];
		d2 = r.c[5] * i.c[5] + i.c[5] * r.c[5];
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = r.c[6] * r.c[6] - i.c[6] * i.c[6];
		d2 = r.c[6] * i.c[6] + i.c[6] * r.c[6];
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = r.c[7] * r.c[7] - i.c[7] * i.c[7];
		d2 = r.c[7] * i.c[7] + i.c[7] * r.c[7];
		r.c[7] = d1;
		i.c[7] =
				d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *square(const AML_PREFIX(VectorU8_8D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = r.c[0] * r.c[0] - i.c[0] * i.c[0];
			d2 = r.c[0] * i.c[0] + i.c[0] * r.c[0];
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = r.c[1] * r.c[1] - i.c[1] * i.c[1];
			d2 = r.c[1] * i.c[1] + i.c[1] * r.c[1];
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = r.c[2] * r.c[2] - i.c[2] * i.c[2];
			d2 = r.c[2] * i.c[2] + i.c[2] * r.c[2];
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = r.c[3] * r.c[3] - i.c[3] * i.c[3];
			d2 = r.c[3] * i.c[3] + i.c[3] * r.c[3];
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = r.c[4] * r.c[4] - i.c[4] * i.c[4];
			d2 = r.c[4] * i.c[4] + i.c[4] * r.c[4];
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = r.c[5] * r.c[5] - i.c[5] * i.c[5];
			d2 = r.c[5] * i.c[5] + i.c[5] * r.c[5];
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = r.c[6] * r.c[6] - i.c[6] * i.c[6];
			d2 = r.c[6] * i.c[6] + i.c[6] * r.c[6];
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = r.c[7] * r.c[7] - i.c[7] * i.c[7];
			d2 = r.c[7] * i.c[7] + i.c[7] * r.c[7];
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *divide(const AML_PREFIX(Complex32) a) {
		float d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		float d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = (r.c[2] * a.c.c[0] + i.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[2] * a.c.c[0] - r.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = (r.c[3] * a.c.c[0] + i.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[3] * a.c.c[0] - r.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = (r.c[4] * a.c.c[0] + i.c[4] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[4] * a.c.c[0] - r.c[4] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = (r.c[5] * a.c.c[0] + i.c[5] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[5] * a.c.c[0] - r.c[5] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = (r.c[6] * a.c.c[0] + i.c[6] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[6] * a.c.c[0] - r.c[6] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = (r.c[7] * a.c.c[0] + i.c[7] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[7] * a.c.c[0] - r.c[7] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *
	divide(const AML_PREFIX(Complex32) a, const AML_PREFIX(VectorU8_8D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = (r.c[2] * a.c.c[0] + i.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[2] * a.c.c[0] - r.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = (r.c[3] * a.c.c[0] + i.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[3] * a.c.c[0] - r.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = (r.c[4] * a.c.c[0] + i.c[4] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[4] * a.c.c[0] - r.c[4] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = (r.c[5] * a.c.c[0] + i.c[5] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[5] * a.c.c[0] - r.c[5] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = (r.c[6] * a.c.c[0] + i.c[6] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[6] * a.c.c[0] - r.c[6] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = (r.c[7] * a.c.c[0] + i.c[7] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			d2 = (i.c[7] * a.c.c[0] - r.c[7] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *divide(const AML_PREFIX(Array8Complex32) &a) {
		float d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		float d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = (r.c[2] * a.r.c[2] + i.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		d2 = (i.c[2] * a.r.c[2] - r.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = (r.c[3] * a.r.c[3] + i.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		d2 = (i.c[3] * a.r.c[3] - r.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = (r.c[4] * a.r.c[4] + i.c[4] * a.i.c[4]) / (a.r.c[4] * a.r.c[4] + a.i.c[4] * a.i.c[4]);
		d2 = (i.c[4] * a.r.c[4] - r.c[4] * a.i.c[4]) / (a.r.c[4] * a.r.c[4] + a.i.c[4] * a.i.c[4]);
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = (r.c[5] * a.r.c[5] + i.c[5] * a.i.c[5]) / (a.r.c[5] * a.r.c[5] + a.i.c[5] * a.i.c[5]);
		d2 = (i.c[5] * a.r.c[5] - r.c[5] * a.i.c[5]) / (a.r.c[5] * a.r.c[5] + a.i.c[5] * a.i.c[5]);
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = (r.c[6] * a.r.c[6] + i.c[6] * a.i.c[6]) / (a.r.c[6] * a.r.c[6] + a.i.c[6] * a.i.c[6]);
		d2 = (i.c[6] * a.r.c[6] - r.c[6] * a.i.c[6]) / (a.r.c[6] * a.r.c[6] + a.i.c[6] * a.i.c[6]);
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = (r.c[7] * a.r.c[7] + i.c[7] * a.i.c[7]) / (a.r.c[7] * a.r.c[7] + a.i.c[7] * a.i.c[7]);
		d2 = (i.c[7] * a.r.c[7] - r.c[7] * a.i.c[7]) / (a.r.c[7] * a.r.c[7] + a.i.c[7] * a.i.c[7]);
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *
	divide(const AML_PREFIX(Array8Complex32) &a, const AML_PREFIX(VectorU8_8D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = (r.c[0] * a.i.c[0] + i.c[0] * a.i.c[1]) / (a.i.c[0] * a.i.c[0] + a.i.c[1] * a.i.c[1]);
			d2 = (i.c[0] * a.i.c[0] - r.c[0] * a.i.c[1]) / (a.i.c[0] * a.i.c[0] + a.i.c[1] * a.i.c[1]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
			d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = (r.c[2] * a.r.c[2] + i.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
			d2 = (i.c[2] * a.r.c[2] - r.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = (r.c[3] * a.r.c[3] + i.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
			d2 = (i.c[3] * a.r.c[3] - r.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = (r.c[4] * a.r.c[4] + i.c[4] * a.i.c[4]) / (a.r.c[4] * a.r.c[4] + a.i.c[4] * a.i.c[4]);
			d2 = (i.c[4] * a.r.c[4] - r.c[4] * a.i.c[4]) / (a.r.c[4] * a.r.c[4] + a.i.c[4] * a.i.c[4]);
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = (r.c[5] * a.r.c[5] + i.c[5] * a.i.c[5]) / (a.r.c[5] * a.r.c[5] + a.i.c[5] * a.i.c[5]);
			d2 = (i.c[5] * a.r.c[5] - r.c[5] * a.i.c[5]) / (a.r.c[5] * a.r.c[5] + a.i.c[5] * a.i.c[5]);
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = (r.c[6] * a.r.c[6] + i.c[6] * a.i.c[6]) / (a.r.c[6] * a.r.c[6] + a.i.c[6] * a.i.c[6]);
			d2 = (i.c[6] * a.r.c[6] - r.c[6] * a.i.c[6]) / (a.r.c[6] * a.r.c[6] + a.i.c[6] * a.i.c[6]);
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = (r.c[7] * a.r.c[7] + i.c[7] * a.i.c[7]) / (a.r.c[7] * a.r.c[7] + a.i.c[7] * a.i.c[7]);
			d2 = (i.c[7] * a.r.c[7] - r.c[7] * a.i.c[7]) / (a.r.c[7] * a.r.c[7] + a.i.c[7] * a.i.c[7]);
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) operator/(const AML_PREFIX(Complex32) &a) const {
		AML_PREFIX(Array8Complex32) ret;
		float d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		float d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[0] = d1;
		ret.i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[1] = d1;
		ret.i.c[1] = d2;
		d1 = (r.c[2] * a.c.c[0] + i.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[2] * a.c.c[0] - r.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[2] = d1;
		ret.i.c[2] = d2;
		d1 = (r.c[3] * a.c.c[0] + i.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[3] * a.c.c[0] - r.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[3] = d1;
		ret.i.c[3] = d2;
		d1 = (r.c[4] * a.c.c[0] + i.c[4] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[4] * a.c.c[0] - r.c[4] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[4] = d1;
		ret.i.c[4] = d2;
		d1 = (r.c[5] * a.c.c[0] + i.c[5] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[5] * a.c.c[0] - r.c[5] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[5] = d1;
		ret.i.c[5] = d2;
		d1 = (r.c[6] * a.c.c[0] + i.c[6] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[6] * a.c.c[0] - r.c[6] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[6] = d1;
		ret.i.c[6] = d2;
		d1 = (r.c[7] * a.c.c[0] + i.c[7] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[7] * a.c.c[0] - r.c[7] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.r.c[7] = d1;
		ret.i.c[7] = d2;
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) operator/(const AML_PREFIX(Array8Complex32) &a) const {
		AML_PREFIX(Array8Complex32) ret;
		float d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		float d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		ret.r.c[0] = d1;
		ret.i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		ret.r.c[1] = d1;
		ret.i.c[1] = d2;
		d1 = (r.c[2] * a.r.c[2] + i.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		d2 = (i.c[2] * a.r.c[2] - r.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		ret.r.c[2] = d1;
		ret.i.c[2] = d2;
		d1 = (r.c[3] * a.r.c[3] + i.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		d2 = (i.c[3] * a.r.c[3] - r.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		ret.r.c[3] = d1;
		ret.i.c[3] = d2;
		d1 = (r.c[4] * a.r.c[4] + i.c[4] * a.i.c[4]) / (a.r.c[4] * a.r.c[4] + a.i.c[4] * a.i.c[4]);
		d2 = (i.c[4] * a.r.c[4] - r.c[4] * a.i.c[4]) / (a.r.c[4] * a.r.c[4] + a.i.c[4] * a.i.c[4]);
		ret.r.c[4] = d1;
		ret.i.c[4] = d2;
		d1 = (r.c[5] * a.r.c[5] + i.c[5] * a.i.c[5]) / (a.r.c[5] * a.r.c[5] + a.i.c[5] * a.i.c[5]);
		d2 = (i.c[5] * a.r.c[5] - r.c[5] * a.i.c[5]) / (a.r.c[5] * a.r.c[5] + a.i.c[5] * a.i.c[5]);
		ret.r.c[5] = d1;
		ret.i.c[5] = d2;
		d1 = (r.c[6] * a.r.c[6] + i.c[6] * a.i.c[6]) / (a.r.c[6] * a.r.c[6] + a.i.c[6] * a.i.c[6]);
		d2 = (i.c[6] * a.r.c[6] - r.c[6] * a.i.c[6]) / (a.r.c[6] * a.r.c[6] + a.i.c[6] * a.i.c[6]);
		ret.r.c[6] = d1;
		ret.i.c[6] = d2;
		d1 = (r.c[7] * a.r.c[7] + i.c[7] * a.i.c[7]) / (a.r.c[7] * a.r.c[7] + a.i.c[7] * a.i.c[7]);
		d2 = (i.c[7] * a.r.c[7] - r.c[7] * a.i.c[7]) / (a.r.c[7] * a.r.c[7] + a.i.c[7] * a.i.c[7]);
		ret.r.c[7] = d1;
		ret.i.c[7] = d2;
		return ret;
	}

	AML_FUNCTION void operator/=(const AML_PREFIX(Complex32) &a) {
		float d1 = (r.c[0] * a.c.c[0] + i.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		float d2 = (i.c[0] * a.c.c[0] - r.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.c.c[0] + i.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[1] * a.c.c[0] - r.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = (r.c[2] * a.c.c[0] + i.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[2] * a.c.c[0] - r.c[2] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = (r.c[3] * a.c.c[0] + i.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[3] * a.c.c[0] - r.c[3] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = (r.c[4] * a.c.c[0] + i.c[4] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[4] * a.c.c[0] - r.c[4] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = (r.c[5] * a.c.c[0] + i.c[5] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[5] * a.c.c[0] - r.c[5] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = (r.c[6] * a.c.c[0] + i.c[6] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[6] * a.c.c[0] - r.c[6] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = (r.c[7] * a.c.c[0] + i.c[7] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		d2 = (i.c[7] * a.c.c[0] - r.c[7] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		r.c[7] = d1;
		i.c[7] = d2;
	}

	AML_FUNCTION void operator/=(const AML_PREFIX(Array8Complex32) &a) {
		float d1 = (r.c[0] * a.r.c[0] + i.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		float d2 = (i.c[0] * a.r.c[0] - r.c[0] * a.i.c[0]) / (a.r.c[0] * a.r.c[0] + a.i.c[0] * a.i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = (r.c[1] * a.r.c[1] + i.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		d2 = (i.c[1] * a.r.c[1] - r.c[1] * a.i.c[1]) / (a.r.c[1] * a.r.c[1] + a.i.c[1] * a.i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = (r.c[2] * a.r.c[2] + i.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		d2 = (i.c[2] * a.r.c[2] - r.c[2] * a.i.c[2]) / (a.r.c[2] * a.r.c[2] + a.i.c[2] * a.i.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = (r.c[3] * a.r.c[3] + i.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		d2 = (i.c[3] * a.r.c[3] - r.c[3] * a.i.c[3]) / (a.r.c[3] * a.r.c[3] + a.i.c[3] * a.i.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = (r.c[4] * a.r.c[4] + i.c[4] * a.i.c[4]) / (a.r.c[4] * a.r.c[4] + a.i.c[4] * a.i.c[4]);
		d2 = (i.c[4] * a.r.c[4] - r.c[4] * a.i.c[4]) / (a.r.c[4] * a.r.c[4] + a.i.c[4] * a.i.c[4]);
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = (r.c[5] * a.r.c[5] + i.c[5] * a.i.c[5]) / (a.r.c[5] * a.r.c[5] + a.i.c[5] * a.i.c[5]);
		d2 = (i.c[5] * a.r.c[5] - r.c[5] * a.i.c[5]) / (a.r.c[5] * a.r.c[5] + a.i.c[5] * a.i.c[5]);
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = (r.c[6] * a.r.c[6] + i.c[6] * a.i.c[6]) / (a.r.c[6] * a.r.c[6] + a.i.c[6] * a.i.c[6]);
		d2 = (i.c[6] * a.r.c[6] - r.c[6] * a.i.c[6]) / (a.r.c[6] * a.r.c[6] + a.i.c[6] * a.i.c[6]);
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = (r.c[7] * a.r.c[7] + i.c[7] * a.i.c[7]) / (a.r.c[7] * a.r.c[7] + a.i.c[7] * a.i.c[7]);
		d2 = (i.c[7] * a.r.c[7] - r.c[7] * a.i.c[7]) / (a.r.c[7] * a.r.c[7] + a.i.c[7] * a.i.c[7]);
		r.c[7] = d1;
		i.c[7] = d2;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *sqrt() {
		float d1;
		float d2;
		d2 = ::sqrt((-r.c[0] + ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[0]);
		} else LIKELY {
			d1 = i.c[0] / (2 * d2);
		}
		r.c[0] = d1;
		i.c[0] = d2;
		d2 = ::sqrt((-r.c[1] + ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[1]);
		} else LIKELY {
			d1 = i.c[1] / (2 * d2);
		}
		r.c[1] = d1;
		i.c[1] = d2;
		d2 = ::sqrt((-r.c[2] + ::sqrt(r.c[2] * r.c[2] + i.c[2] * i.c[2])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[2]);
		} else LIKELY {
			d1 = i.c[2] / (2 * d2);
		}
		r.c[2] = d1;
		i.c[2] = d2;
		d2 = ::sqrt((-r.c[3] + ::sqrt(r.c[3] * r.c[3] + i.c[3] * i.c[3])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[3]);
		} else LIKELY {
			d1 = i.c[3] / (2 * d2);
		}
		r.c[3] = d1;
		i.c[3] = d2;
		d2 = ::sqrt((-r.c[4] + ::sqrt(r.c[4] * r.c[4] + i.c[4] * i.c[4])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[4]);
		} else LIKELY {
			d1 = i.c[4] / (2 * d2);
		}
		r.c[4] = d1;
		i.c[4] = d2;
		d2 = ::sqrt((-r.c[5] + ::sqrt(r.c[5] * r.c[5] + i.c[5] * i.c[5])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[5]);
		} else LIKELY {
			d1 = i.c[5] / (2 * d2);
		}
		r.c[5] = d1;
		i.c[5] = d2;
		d2 = ::sqrt((-r.c[6] + ::sqrt(r.c[6] * r.c[6] + i.c[6] * i.c[6])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[6]);
		} else LIKELY {
			d1 = i.c[6] / (2 * d2);
		}
		r.c[6] = d1;
		i.c[6] = d2;
		d2 = ::sqrt((-r.c[7] + ::sqrt(r.c[7] * r.c[7] + i.c[7] * i.c[7])) / (2));
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(r.c[7]);
		} else LIKELY {
			d1 = i.c[7] / (2 * d2);
		}
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *sqrt(const AML_PREFIX(VectorU8_8D) mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d2 = ::sqrt((-r.c[0] + ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[0]);
			} else LIKELY {
				d1 = i.c[0] / (2 * d2);
			}
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d2 = ::sqrt((-r.c[1] + ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[1]);
			} else LIKELY {
				d1 = i.c[1] / (2 * d2);
			}
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d2 = ::sqrt((-r.c[2] + ::sqrt(r.c[2] * r.c[2] + i.c[2] * i.c[2])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[2]);
			} else LIKELY {
				d1 = i.c[2] / (2 * d2);
			}
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d2 = ::sqrt((-r.c[3] + ::sqrt(r.c[3] * r.c[3] + i.c[3] * i.c[3])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[3]);
			} else LIKELY {
				d1 = i.c[3] / (2 * d2);
			}
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d2 = ::sqrt((-r.c[4] + ::sqrt(r.c[4] * r.c[4] + i.c[4] * i.c[4])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[4]);
			} else LIKELY {
				d1 = i.c[4] / (2 * d2);
			}
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d2 = ::sqrt((-r.c[5] + ::sqrt(r.c[5] * r.c[5] + i.c[5] * i.c[5])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[5]);
			} else LIKELY {
				d1 = i.c[5] / (2 * d2);
			}
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d2 = ::sqrt((-r.c[6] + ::sqrt(r.c[6] * r.c[6] + i.c[6] * i.c[6])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[6]);
			} else LIKELY {
				d1 = i.c[6] / (2 * d2);
			}
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d2 = ::sqrt((-r.c[7] + ::sqrt(r.c[7] * r.c[7] + i.c[7] * i.c[7])) / (2));
			if (d2 == 0) UNLIKELY {
				d1 = ::sqrt(r.c[7]);
			} else LIKELY {
				d1 = i.c[7] / (2 * d2);
			}
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *sin() {
		float d1;
		float d2;
		d1 = ::sin(r.c[0]) * ::cosh(i.c[0]);
		d2 = ::cos(i.c[0]) * ::sinh(r.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::sin(r.c[1]) * ::cosh(i.c[1]);
		d2 = ::cos(i.c[1]) * ::sinh(r.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::sin(r.c[2]) * ::cosh(i.c[2]);
		d2 = ::cos(i.c[2]) * ::sinh(r.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::sin(r.c[3]) * ::cosh(i.c[3]);
		d2 = ::cos(i.c[3]) * ::sinh(r.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = ::sin(r.c[4]) * ::cosh(i.c[4]);
		d2 = ::cos(i.c[4]) * ::sinh(r.c[4]);
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = ::sin(r.c[5]) * ::cosh(i.c[5]);
		d2 = ::cos(i.c[5]) * ::sinh(r.c[5]);
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = ::sin(r.c[6]) * ::cosh(i.c[6]);
		d2 = ::cos(i.c[6]) * ::sinh(r.c[6]);
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = ::sin(r.c[7]) * ::cosh(i.c[7]);
		d2 = ::cos(i.c[7]) * ::sinh(r.c[7]);
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *cos() {
		float d1;
		float d2;
		d1 = ::cos(r.c[0]) * ::cosh(i.c[0]);
		d2 = -::sin(i.c[0]) * ::sinh(r.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::cos(r.c[1]) * ::cosh(i.c[1]);
		d2 = -::sin(i.c[1]) * ::sinh(r.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::cos(r.c[2]) * ::cosh(i.c[2]);
		d2 = -::sin(i.c[2]) * ::sinh(r.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::cos(r.c[3]) * ::cosh(i.c[3]);
		d2 = -::sin(i.c[3]) * ::sinh(r.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = ::cos(r.c[4]) * ::cosh(i.c[4]);
		d2 = -::sin(i.c[4]) * ::sinh(r.c[4]);
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = ::cos(r.c[5]) * ::cosh(i.c[5]);
		d2 = -::sin(i.c[5]) * ::sinh(r.c[5]);
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = ::cos(r.c[6]) * ::cosh(i.c[6]);
		d2 = -::sin(i.c[6]) * ::sinh(r.c[6]);
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = ::cos(r.c[7]) * ::cosh(i.c[7]);
		d2 = -::sin(i.c[7]) * ::sinh(r.c[7]);
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *tan() {
		float d1;
		float d2;
		d1 = ::sin(r.c[0] + r.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
		d2 = ::sinh(i.c[0] + i.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::sin(r.c[1] + r.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
		d2 = ::sinh(i.c[1] + i.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::sin(r.c[2] + r.c[2]) / (::cos(r.c[2] + r.c[2]) * ::cosh(i.c[2] + i.c[2]));
		d2 = ::sinh(i.c[2] + i.c[2]) / (::cos(r.c[2] + r.c[2]) * ::cosh(i.c[2] + i.c[2]));
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::sin(r.c[3] + r.c[3]) / (::cos(r.c[3] + r.c[3]) * ::cosh(i.c[3] + i.c[3]));
		d2 = ::sinh(i.c[3] + i.c[3]) / (::cos(r.c[3] + r.c[3]) * ::cosh(i.c[3] + i.c[3]));
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = ::sin(r.c[4] + r.c[4]) / (::cos(r.c[4] + r.c[4]) * ::cosh(i.c[4] + i.c[4]));
		d2 = ::sinh(i.c[4] + i.c[4]) / (::cos(r.c[4] + r.c[4]) * ::cosh(i.c[4] + i.c[4]));
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = ::sin(r.c[5] + r.c[5]) / (::cos(r.c[5] + r.c[5]) * ::cosh(i.c[5] + i.c[5]));
		d2 = ::sinh(i.c[5] + i.c[5]) / (::cos(r.c[5] + r.c[5]) * ::cosh(i.c[5] + i.c[5]));
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = ::sin(r.c[6] + r.c[6]) / (::cos(r.c[6] + r.c[6]) * ::cosh(i.c[6] + i.c[6]));
		d2 = ::sinh(i.c[6] + i.c[6]) / (::cos(r.c[6] + r.c[6]) * ::cosh(i.c[6] + i.c[6]));
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = ::sin(r.c[7] + r.c[7]) / (::cos(r.c[7] + r.c[7]) * ::cosh(i.c[7] + i.c[7]));
		d2 = ::sinh(i.c[7] + i.c[7]) / (::cos(r.c[7] + r.c[7]) * ::cosh(i.c[7] + i.c[7]));
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *sin(const AML_PREFIX(VectorU8_8D) mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::sin(r.c[0]) * ::cosh(i.c[0]);
			d2 = ::cos(i.c[0]) * ::sinh(r.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::sin(r.c[1]) * ::cosh(i.c[1]);
			d2 = ::cos(i.c[1]) * ::sinh(r.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::sin(r.c[2]) * ::cosh(i.c[2]);
			d2 = ::cos(i.c[2]) * ::sinh(r.c[2]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::sin(r.c[3]) * ::cosh(i.c[3]);
			d2 = ::cos(i.c[3]) * ::sinh(r.c[3]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::sin(r.c[4]) * ::cosh(i.c[4]);
			d2 = ::cos(i.c[4]) * ::sinh(r.c[4]);
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::sin(r.c[5]) * ::cosh(i.c[5]);
			d2 = ::cos(i.c[5]) * ::sinh(r.c[5]);
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::sin(r.c[6]) * ::cosh(i.c[6]);
			d2 = ::cos(i.c[6]) * ::sinh(r.c[6]);
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::sin(r.c[7]) * ::cosh(i.c[7]);
			d2 = ::cos(i.c[7]) * ::sinh(r.c[7]);
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *cos(const AML_PREFIX(VectorU8_8D) mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::cos(r.c[0]) * ::cosh(i.c[0]);
			d2 = -::sin(i.c[0]) * ::sinh(r.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::cos(r.c[1]) * ::cosh(i.c[1]);
			d2 = -::sin(i.c[1]) * ::sinh(r.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::cos(r.c[2]) * ::cosh(i.c[2]);
			d2 = -::sin(i.c[2]) * ::sinh(r.c[2]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::cos(r.c[3]) * ::cosh(i.c[3]);
			d2 = -::sin(i.c[3]) * ::sinh(r.c[3]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::cos(r.c[4]) * ::cosh(i.c[4]);
			d2 = -::sin(i.c[4]) * ::sinh(r.c[4]);
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::cos(r.c[5]) * ::cosh(i.c[5]);
			d2 = -::sin(i.c[5]) * ::sinh(r.c[5]);
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::cos(r.c[6]) * ::cosh(i.c[6]);
			d2 = -::sin(i.c[6]) * ::sinh(r.c[6]);
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::cos(r.c[7]) * ::cosh(i.c[7]);
			d2 = -::sin(i.c[7]) * ::sinh(r.c[7]);
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *tan(const AML_PREFIX(VectorU8_8D) mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::sin(r.c[0] + r.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
			d2 = ::sinh(i.c[0] + i.c[0]) / (::cos(r.c[0] + r.c[0]) * ::cosh(i.c[0] + i.c[0]));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::sin(r.c[1] + r.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
			d2 = ::sinh(i.c[1] + i.c[1]) / (::cos(r.c[1] + r.c[1]) * ::cosh(i.c[1] + i.c[1]));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::sin(r.c[2] + r.c[2]) / (::cos(r.c[2] + r.c[2]) * ::cosh(i.c[2] + i.c[2]));
			d2 = ::sinh(i.c[2] + i.c[2]) / (::cos(r.c[2] + r.c[2]) * ::cosh(i.c[2] + i.c[2]));
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::sin(r.c[3] + r.c[3]) / (::cos(r.c[3] + r.c[3]) * ::cosh(i.c[3] + i.c[3]));
			d2 = ::sinh(i.c[3] + i.c[3]) / (::cos(r.c[3] + r.c[3]) * ::cosh(i.c[3] + i.c[3]));
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::sin(r.c[4] + r.c[4]) / (::cos(r.c[4] + r.c[4]) * ::cosh(i.c[4] + i.c[4]));
			d2 = ::sinh(i.c[4] + i.c[4]) / (::cos(r.c[4] + r.c[4]) * ::cosh(i.c[4] + i.c[4]));
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::sin(r.c[5] + r.c[5]) / (::cos(r.c[5] + r.c[5]) * ::cosh(i.c[5] + i.c[5]));
			d2 = ::sinh(i.c[5] + i.c[5]) / (::cos(r.c[5] + r.c[5]) * ::cosh(i.c[5] + i.c[5]));
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::sin(r.c[6] + r.c[6]) / (::cos(r.c[6] + r.c[6]) * ::cosh(i.c[6] + i.c[6]));
			d2 = ::sinh(i.c[6] + i.c[6]) / (::cos(r.c[6] + r.c[6]) * ::cosh(i.c[6] + i.c[6]));
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::sin(r.c[7] + r.c[7]) / (::cos(r.c[7] + r.c[7]) * ::cosh(i.c[7] + i.c[7]));
			d2 = ::sinh(i.c[7] + i.c[7]) / (::cos(r.c[7] + r.c[7]) * ::cosh(i.c[7] + i.c[7]));
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array8Complex32) *exp() {
		float d1 = ::exp(r.c[0]) * ::cos(i.c[0]);
		float d2 = ::exp(r.c[0]) * ::sin(i.c[0]);
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::exp(r.c[1]) * ::cos(i.c[1]);
		d2 = ::exp(r.c[1]) * ::sin(i.c[1]);
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::exp(r.c[2]) * ::cos(i.c[2]);
		d2 = ::exp(r.c[2]) * ::sin(i.c[2]);
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::exp(r.c[3]) * ::cos(i.c[3]);
		d2 = ::exp(r.c[3]) * ::sin(i.c[3]);
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = ::exp(r.c[4]) * ::cos(i.c[4]);
		d2 = ::exp(r.c[4]) * ::sin(i.c[4]);
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = ::exp(r.c[5]) * ::cos(i.c[5]);
		d2 = ::exp(r.c[5]) * ::sin(i.c[5]);
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = ::exp(r.c[6]) * ::cos(i.c[6]);
		d2 = ::exp(r.c[6]) * ::sin(i.c[6]);
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = ::exp(r.c[7]) * ::cos(i.c[7]);
		d2 = ::exp(r.c[7]) * ::sin(i.c[7]);
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *exp(float n) {
		float d1 = ::pow(n, r.c[0]) * ::cos(i.c[0] * ::log(n));
		float d2 = ::pow(n, r.c[0]) * ::sin(i.c[0] * ::log(n));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::pow(n, r.c[1]) * ::cos(i.c[1] * ::log(n));
		d2 = ::pow(n, r.c[1]) * ::sin(i.c[1] * ::log(n));
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::pow(n, r.c[2]) * ::cos(i.c[2] * ::log(n));
		d2 = ::pow(n, r.c[2]) * ::sin(i.c[2] * ::log(n));
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::pow(n, r.c[3]) * ::cos(i.c[3] * ::log(n));
		d2 = ::pow(n, r.c[3]) * ::sin(i.c[3] * ::log(n));
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = ::pow(n, r.c[4]) * ::cos(i.c[4] * ::log(n));
		d2 = ::pow(n, r.c[4]) * ::sin(i.c[4] * ::log(n));
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = ::pow(n, r.c[5]) * ::cos(i.c[5] * ::log(n));
		d2 = ::pow(n, r.c[5]) * ::sin(i.c[5] * ::log(n));
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = ::pow(n, r.c[6]) * ::cos(i.c[6] * ::log(n));
		d2 = ::pow(n, r.c[6]) * ::sin(i.c[6] * ::log(n));
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = ::pow(n, r.c[7]) * ::cos(i.c[7] * ::log(n));
		d2 = ::pow(n, r.c[7]) * ::sin(i.c[7] * ::log(n));
		r.c[7] = d1;
		i.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *pow(const AML_PREFIX(Array8Complex32) &n) {
		float d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		float d2 = ::atan2(r.c[0], i.c[0]);
		float d3 = ::exp(d1 * n.i.c[0] - d2 * n.r.c[0]);
		float d4 = d1 * n.r.c[0] + d2 * n.i.c[0];
		float d5 = d3 * ::cos(d4);
		float d6 = d3 * ::sin(d4);
		i.c[0] = d5;
		r.c[0] = d6;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		d3 = ::exp(d1 * n.i.c[1] - d2 * n.r.c[1]);
		d4 = d1 * n.r.c[1] + d2 * n.i.c[1];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[1] = d5;
		r.c[1] = d6;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
		d2 = ::atan2(r.c[2], i.c[2]);
		d3 = ::exp(d1 * n.i.c[2] - d2 * n.r.c[2]);
		d4 = d1 * n.r.c[2] + d2 * n.i.c[2];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[2] = d5;
		r.c[2] = d6;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
		d2 = ::atan2(r.c[3], i.c[3]);
		d3 = ::exp(d1 * n.i.c[3] - d2 * n.r.c[3]);
		d4 = d1 * n.r.c[3] + d2 * n.i.c[3];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[3] = d5;
		r.c[3] = d6;
		d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / 2;
		d2 = ::atan2(r.c[4], i.c[4]);
		d3 = ::exp(d1 * n.i.c[4] - d2 * n.r.c[4]);
		d4 = d1 * n.r.c[4] + d2 * n.i.c[4];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[4] = d5;
		r.c[4] = d6;
		d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / 2;
		d2 = ::atan2(r.c[5], i.c[5]);
		d3 = ::exp(d1 * n.i.c[5] - d2 * n.r.c[5]);
		d4 = d1 * n.r.c[5] + d2 * n.i.c[5];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[5] = d5;
		r.c[5] = d6;
		d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / 2;
		d2 = ::atan2(r.c[6], i.c[6]);
		d3 = ::exp(d1 * n.i.c[6] - d2 * n.r.c[6]);
		d4 = d1 * n.r.c[6] + d2 * n.i.c[6];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[6] = d5;
		r.c[6] = d6;
		d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / 2;
		d2 = ::atan2(r.c[7], i.c[7]);
		d3 = ::exp(d1 * n.i.c[7] - d2 * n.r.c[7]);
		d4 = d1 * n.r.c[7] + d2 * n.i.c[7];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[7] = d5;
		r.c[7] = d6;
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array8Complex32) *pow(const AML_PREFIX(Complex32) n) {
		float d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		float d2 = ::atan2(r.c[0], i.c[0]);
		float d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		float d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		float d5 = d3 * ::cos(d4);
		float d6 = d3 * ::sin(d4);
		i.c[0] = d5;
		r.c[0] = d6;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[1] = d5;
		r.c[1] = d6;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
		d2 = ::atan2(r.c[2], i.c[2]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[2] = d5;
		r.c[2] = d6;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
		d2 = ::atan2(r.c[3], i.c[3]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[3] = d5;
		r.c[3] = d6;
		d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / 2;
		d2 = ::atan2(r.c[4], i.c[4]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[4] = d5;
		r.c[4] = d6;
		d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / 2;
		d2 = ::atan2(r.c[5], i.c[5]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[5] = d5;
		r.c[5] = d6;
		d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / 2;
		d2 = ::atan2(r.c[6], i.c[6]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[6] = d5;
		r.c[6] = d6;
		d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / 2;
		d2 = ::atan2(r.c[7], i.c[7]);
		d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		d5 = d3 * ::cos(d4);
		d6 = d3 * ::sin(d4);
		i.c[7] = d5;
		r.c[7] = d6;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *
	pow(const AML_PREFIX(Array8Complex32) &n, const AML_PREFIX(VectorU8_8D) &mask) {
		float d1;
		float d2;
		float d3;
		float d4;
		float d5;
		float d6;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			d3 = ::exp(d1 * n.i.c[0] - d2 * n.r.c[0]);
			d4 = d1 * n.r.c[0] + d2 * n.i.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[0] = d5;
			r.c[0] = d6;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			d3 = ::exp(d1 * n.i.c[1] - d2 * n.r.c[1]);
			d4 = d1 * n.r.c[1] + d2 * n.i.c[1];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[1] = d5;
			r.c[1] = d6;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
			d2 = ::atan2(r.c[2], i.c[2]);
			d3 = ::exp(d1 * n.i.c[2] - d2 * n.r.c[2]);
			d4 = d1 * n.r.c[2] + d2 * n.i.c[2];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[2] = d5;
			r.c[2] = d6;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
			d2 = ::atan2(r.c[3], i.c[3]);
			d3 = ::exp(d1 * n.i.c[3] - d2 * n.r.c[3]);
			d4 = d1 * n.r.c[3] + d2 * n.i.c[3];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[3] = d5;
			r.c[3] = d6;
		}
		if (mask.v.c[4]) {
			d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / 2;
			d2 = ::atan2(r.c[4], i.c[4]);
			d3 = ::exp(d1 * n.i.c[4] - d2 * n.r.c[4]);
			d4 = d1 * n.r.c[4] + d2 * n.i.c[4];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[4] = d5;
			r.c[4] = d6;
		}
		if (mask.v.c[5]) {
			d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / 2;
			d2 = ::atan2(r.c[5], i.c[5]);
			d3 = ::exp(d1 * n.i.c[5] - d2 * n.r.c[5]);
			d4 = d1 * n.r.c[5] + d2 * n.i.c[5];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[5] = d5;
			r.c[5] = d6;
		}
		if (mask.v.c[6]) {
			d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / 2;
			d2 = ::atan2(r.c[6], i.c[6]);
			d3 = ::exp(d1 * n.i.c[6] - d2 * n.r.c[6]);
			d4 = d1 * n.r.c[6] + d2 * n.i.c[6];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[6] = d5;
			r.c[6] = d6;
		}
		if (mask.v.c[7]) {
			d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / 2;
			d2 = ::atan2(r.c[7], i.c[7]);
			d3 = ::exp(d1 * n.i.c[7] - d2 * n.r.c[7]);
			d4 = d1 * n.r.c[7] + d2 * n.i.c[7];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[7] = d5;
			r.c[7] = d6;
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(Array8Complex32) *pow(const AML_PREFIX(Complex32) n, const AML_PREFIX(VectorU8_8D) &mask) {
		float d1;
		float d2;
		float d3;
		float d4;
		float d5;
		float d6;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[0] = d5;
			r.c[0] = d6;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[1] = d5;
			r.c[1] = d6;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
			d2 = ::atan2(r.c[2], i.c[2]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[2] = d5;
			r.c[2] = d6;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
			d2 = ::atan2(r.c[3], i.c[3]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[3] = d5;
			r.c[3] = d6;
		}
		if (mask.v.c[4]) {
			d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / 2;
			d2 = ::atan2(r.c[4], i.c[4]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[4] = d5;
			r.c[4] = d6;
		}
		if (mask.v.c[5]) {
			d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / 2;
			d2 = ::atan2(r.c[5], i.c[5]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[5] = d5;
			r.c[5] = d6;
		}
		if (mask.v.c[6]) {
			d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / 2;
			d2 = ::atan2(r.c[6], i.c[6]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[6] = d5;
			r.c[6] = d6;
		}
		if (mask.v.c[7]) {
			d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / 2;
			d2 = ::atan2(r.c[7], i.c[7]);
			d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			d5 = d3 * ::cos(d4);
			d6 = d3 * ::sin(d4);
			i.c[7] = d5;
			r.c[7] = d6;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *exp(const AML_PREFIX(VectorU8_8D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::exp(r.c[0]) * ::cos(i.c[0]);
			d2 = ::exp(r.c[0]) * ::sin(i.c[0]);
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::exp(r.c[1]) * ::cos(i.c[1]);
			d2 = ::exp(r.c[1]) * ::sin(i.c[1]);
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::exp(r.c[2]) * ::cos(i.c[2]);
			d2 = ::exp(r.c[2]) * ::sin(i.c[2]);
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::exp(r.c[3]) * ::cos(i.c[3]);
			d2 = ::exp(r.c[3]) * ::sin(i.c[3]);
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::exp(r.c[4]) * ::cos(i.c[4]);
			d2 = ::exp(r.c[4]) * ::sin(i.c[4]);
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::exp(r.c[5]) * ::cos(i.c[5]);
			d2 = ::exp(r.c[5]) * ::sin(i.c[5]);
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::exp(r.c[6]) * ::cos(i.c[6]);
			d2 = ::exp(r.c[6]) * ::sin(i.c[6]);
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::exp(r.c[7]) * ::cos(i.c[7]);
			d2 = ::exp(r.c[7]) * ::sin(i.c[7]);
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *exp(float n, const AML_PREFIX(VectorU8_8D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::pow(n, r.c[0]) * ::cos(i.c[0] * ::log(n));
			d2 = ::pow(n, r.c[0]) * ::sin(i.c[0] * ::log(n));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::pow(n, r.c[1]) * ::cos(i.c[1] * ::log(n));
			d2 = ::pow(n, r.c[1]) * ::sin(i.c[1] * ::log(n));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::pow(n, r.c[2]) * ::cos(i.c[2] * ::log(n));
			d2 = ::pow(n, r.c[2]) * ::sin(i.c[2] * ::log(n));
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::pow(n, r.c[3]) * ::cos(i.c[3] * ::log(n));
			d2 = ::pow(n, r.c[3]) * ::sin(i.c[3] * ::log(n));
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::pow(n, r.c[4]) * ::cos(i.c[4] * ::log(n));
			d2 = ::pow(n, r.c[4]) * ::sin(i.c[4] * ::log(n));
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::pow(n, r.c[5]) * ::cos(i.c[5] * ::log(n));
			d2 = ::pow(n, r.c[5]) * ::sin(i.c[5] * ::log(n));
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::pow(n, r.c[6]) * ::cos(i.c[6] * ::log(n));
			d2 = ::pow(n, r.c[6]) * ::sin(i.c[6] * ::log(n));
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::pow(n, r.c[7]) * ::cos(i.c[7] * ::log(n));
			d2 = ::pow(n, r.c[7]) * ::sin(i.c[7] * ::log(n));
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *pow(float n) {
		float d1;
		float d2;
		d1 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::cos(n * atan2(i.c[0], r.c[0]));
		d2 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::sin(n * atan2(i.c[0], r.c[0]));
		r.c[0] = d1;
		i.c[0] = d2;
		d1 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::cos(n * atan2(i.c[1], r.c[1]));
		d2 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::sin(n * atan2(i.c[1], r.c[1]));
		r.c[1] = d1;
		i.c[1] = d2;
		d1 = ::pow(r.c[2] * r.c[2] + i.c[2] * i.c[2], n / 2) * ::cos(n * atan2(i.c[2], r.c[2]));
		d2 = ::pow(r.c[2] * r.c[2] + i.c[2] * i.c[2], n / 2) * ::sin(n * atan2(i.c[2], r.c[2]));
		r.c[2] = d1;
		i.c[2] = d2;
		d1 = ::pow(r.c[3] * r.c[3] + i.c[3] * i.c[3], n / 2) * ::cos(n * atan2(i.c[3], r.c[3]));
		d2 = ::pow(r.c[3] * r.c[3] + i.c[3] * i.c[3], n / 2) * ::sin(n * atan2(i.c[3], r.c[3]));
		r.c[3] = d1;
		i.c[3] = d2;
		d1 = ::pow(r.c[4] * r.c[4] + i.c[4] * i.c[4], n / 2) * ::cos(n * atan2(i.c[4], r.c[4]));
		d2 = ::pow(r.c[4] * r.c[4] + i.c[4] * i.c[4], n / 2) * ::sin(n * atan2(i.c[4], r.c[4]));
		r.c[4] = d1;
		i.c[4] = d2;
		d1 = ::pow(r.c[5] * r.c[5] + i.c[5] * i.c[5], n / 2) * ::cos(n * atan2(i.c[5], r.c[5]));
		d2 = ::pow(r.c[5] * r.c[5] + i.c[5] * i.c[5], n / 2) * ::sin(n * atan2(i.c[5], r.c[5]));
		r.c[5] = d1;
		i.c[5] = d2;
		d1 = ::pow(r.c[6] * r.c[6] + i.c[6] * i.c[6], n / 2) * ::cos(n * atan2(i.c[6], r.c[6]));
		d2 = ::pow(r.c[6] * r.c[6] + i.c[6] * i.c[6], n / 2) * ::sin(n * atan2(i.c[6], r.c[6]));
		r.c[6] = d1;
		i.c[6] = d2;
		d1 = ::pow(r.c[7] * r.c[7] + i.c[7] * i.c[7], n / 2) * ::cos(n * atan2(i.c[7], r.c[7]));
		d2 = ::pow(r.c[7] * r.c[7] + i.c[7] * i.c[7], n / 2) * ::sin(n * atan2(i.c[7], r.c[7]));
		r.c[7] = d1;
		i.c[7] = d2;

		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *pow(float n, const AML_PREFIX(VectorU8_8D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::cos(n * atan2(i.c[0], r.c[0]));
			d2 = ::pow(r.c[0] * r.c[0] + i.c[0] * i.c[0], n / 2) * ::sin(n * atan2(i.c[0], r.c[0]));
			r.c[0] = d1;
			i.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::cos(n * atan2(i.c[1], r.c[1]));
			d2 = ::pow(r.c[1] * r.c[1] + i.c[1] * i.c[1], n / 2) * ::sin(n * atan2(i.c[1], r.c[1]));
			r.c[1] = d1;
			i.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::pow(r.c[2] * r.c[2] + i.c[2] * i.c[2], n / 2) * ::cos(n * atan2(i.c[2], r.c[2]));
			d2 = ::pow(r.c[2] * r.c[2] + i.c[2] * i.c[2], n / 2) * ::sin(n * atan2(i.c[2], r.c[2]));
			r.c[2] = d1;
			i.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::pow(r.c[3] * r.c[3] + i.c[3] * i.c[3], n / 2) * ::cos(n * atan2(i.c[3], r.c[3]));
			d2 = ::pow(r.c[3] * r.c[3] + i.c[3] * i.c[3], n / 2) * ::sin(n * atan2(i.c[3], r.c[3]));
			r.c[3] = d1;
			i.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::pow(r.c[4] * r.c[4] + i.c[4] * i.c[4], n / 2) * ::cos(n * atan2(i.c[4], r.c[4]));
			d2 = ::pow(r.c[4] * r.c[4] + i.c[4] * i.c[4], n / 2) * ::sin(n * atan2(i.c[4], r.c[4]));
			r.c[4] = d1;
			i.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::pow(r.c[5] * r.c[5] + i.c[5] * i.c[5], n / 2) * ::cos(n * atan2(i.c[5], r.c[5]));
			d2 = ::pow(r.c[5] * r.c[5] + i.c[5] * i.c[5], n / 2) * ::sin(n * atan2(i.c[5], r.c[5]));
			r.c[5] = d1;
			i.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::pow(r.c[6] * r.c[6] + i.c[6] * i.c[6], n / 2) * ::cos(n * atan2(i.c[6], r.c[6]));
			d2 = ::pow(r.c[6] * r.c[6] + i.c[6] * i.c[6], n / 2) * ::sin(n * atan2(i.c[6], r.c[6]));
			r.c[6] = d1;
			i.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::pow(r.c[7] * r.c[7] + i.c[7] * i.c[7], n / 2) * ::cos(n * atan2(i.c[7], r.c[7]));
			d2 = ::pow(r.c[7] * r.c[7] + i.c[7] * i.c[7], n / 2) * ::sin(n * atan2(i.c[7], r.c[7]));
			r.c[7] = d1;
			i.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(VectorFloat8D) abs() {
		AML_PREFIX(VectorFloat8D) ret;
		ret.v.c[0] = ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0]);
		ret.v.c[1] = ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1]);
		ret.v.c[2] = ::sqrt(r.c[2] * r.c[2] + i.c[2] * i.c[2]);
		ret.v.c[3] = ::sqrt(r.c[3] * r.c[3] + i.c[3] * i.c[3]);
		ret.v.c[4] = ::sqrt(r.c[4] * r.c[4] + i.c[4] * i.c[4]);
		ret.v.c[5] = ::sqrt(r.c[5] * r.c[5] + i.c[5] * i.c[5]);
		ret.v.c[6] = ::sqrt(r.c[6] * r.c[6] + i.c[6] * i.c[6]);
		ret.v.c[7] = ::sqrt(r.c[7] * r.c[7] + i.c[7] * i.c[7]);
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_8D) abs_gt(float a) {
		AML_PREFIX(VectorU8_8D) ret;
		ret.v.c[0] = a * a < r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a < r.c[1] * r.c[1] + i.c[1] * i.c[1];
		ret.v.c[2] = a * a < r.c[2] * r.c[2] + i.c[2] * i.c[2];
		ret.v.c[3] = a * a < r.c[3] * r.c[3] + i.c[3] * i.c[3];
		ret.v.c[4] = a * a < r.c[4] * r.c[4] + i.c[4] * i.c[4];
		ret.v.c[5] = a * a < r.c[5] * r.c[5] + i.c[5] * i.c[5];
		ret.v.c[6] = a * a < r.c[6] * r.c[6] + i.c[6] * i.c[6];
		ret.v.c[7] = a * a < r.c[7] * r.c[7] + i.c[7] * i.c[7];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_8D) abs_lt(float a) {
		AML_PREFIX(VectorU8_8D) ret;
		ret.v.c[0] = a * a > r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a > r.c[1] * r.c[1] + i.c[1] * i.c[1];
		ret.v.c[2] = a * a > r.c[2] * r.c[2] + i.c[2] * i.c[2];
		ret.v.c[3] = a * a > r.c[3] * r.c[3] + i.c[3] * i.c[3];
		ret.v.c[4] = a * a > r.c[4] * r.c[4] + i.c[4] * i.c[4];
		ret.v.c[5] = a * a > r.c[5] * r.c[5] + i.c[5] * i.c[5];
		ret.v.c[6] = a * a > r.c[6] * r.c[6] + i.c[6] * i.c[6];
		ret.v.c[7] = a * a > r.c[7] * r.c[7] + i.c[7] * i.c[7];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(VectorU8_8D) abs_eq(float a) {
		AML_PREFIX(VectorU8_8D) ret;
		ret.v.c[0] = a * a == r.c[0] * r.c[0] + i.c[0] * i.c[0];
		ret.v.c[1] = a * a == r.c[1] * r.c[1] + i.c[1] * i.c[1];
		ret.v.c[2] = a * a == r.c[2] * r.c[2] + i.c[2] * i.c[2];
		ret.v.c[3] = a * a == r.c[3] * r.c[3] + i.c[3] * i.c[3];
		ret.v.c[4] = a * a == r.c[4] * r.c[4] + i.c[4] * i.c[4];
		ret.v.c[5] = a * a == r.c[5] * r.c[5] + i.c[5] * i.c[5];
		ret.v.c[6] = a * a == r.c[6] * r.c[6] + i.c[6] * i.c[6];
		ret.v.c[7] = a * a == r.c[7] * r.c[7] + i.c[7] * i.c[7];
		return ret;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *ln(const AML_PREFIX(VectorU8_8D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			i.c[1] = d1;
			r.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
			d2 = ::atan2(r.c[2], i.c[2]);
			i.c[2] = d1;
			r.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
			d2 = ::atan2(r.c[3], i.c[3]);
			i.c[3] = d1;
			r.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / 2;
			d2 = ::atan2(r.c[4], i.c[4]);
			i.c[4] = d1;
			r.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / 2;
			d2 = ::atan2(r.c[5], i.c[5]);
			i.c[5] = d1;
			r.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / 2;
			d2 = ::atan2(r.c[6], i.c[6]);
			i.c[6] = d1;
			r.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / 2;
			d2 = ::atan2(r.c[7], i.c[7]);
			i.c[7] = d1;
			r.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *log(const AML_PREFIX(VectorU8_8D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
			d2 = ::atan2(r.c[0], i.c[0]);
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
			d2 = ::atan2(r.c[1], i.c[1]);
			i.c[1] = d1;
			r.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
			d2 = ::atan2(r.c[2], i.c[2]);
			i.c[2] = d1;
			r.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
			d2 = ::atan2(r.c[3], i.c[3]);
			i.c[3] = d1;
			r.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / 2;
			d2 = ::atan2(r.c[4], i.c[4]);
			i.c[4] = d1;
			r.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / 2;
			d2 = ::atan2(r.c[5], i.c[5]);
			i.c[5] = d1;
			r.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / 2;
			d2 = ::atan2(r.c[6], i.c[6]);
			i.c[6] = d1;
			r.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / 2;
			d2 = ::atan2(r.c[7], i.c[7]);
			i.c[7] = d1;
			r.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *log10(const AML_PREFIX(VectorU8_8D) &mask) {
		float d1;
		float d2;
		if (mask.v.c[0]) {
			d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[0], i.c[0]) / AML_LN10;
			i.c[0] = d1;
			r.c[0] = d2;
		}
		if (mask.v.c[1]) {
			d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[1], i.c[1]) / AML_LN10;
			i.c[1] = d1;
			r.c[1] = d2;
		}
		if (mask.v.c[2]) {
			d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[2], i.c[2]) / AML_LN10;
			i.c[2] = d1;
			r.c[2] = d2;
		}
		if (mask.v.c[3]) {
			d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[3], i.c[3]) / AML_LN10;
			i.c[3] = d1;
			r.c[3] = d2;
		}
		if (mask.v.c[4]) {
			d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[4], i.c[4]) / AML_LN10;
			i.c[4] = d1;
			r.c[4] = d2;
		}
		if (mask.v.c[5]) {
			d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[5], i.c[5]) / AML_LN10;
			i.c[5] = d1;
			r.c[5] = d2;
		}
		if (mask.v.c[6]) {
			d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[6], i.c[6]) / AML_LN10;
			i.c[6] = d1;
			r.c[6] = d2;
		}
		if (mask.v.c[7]) {
			d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / (2 * AML_LN10);
			d2 = ::atan2(r.c[7], i.c[7]) / AML_LN10;
			i.c[7] = d1;
			r.c[7] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *ln() {
		float d1;
		float d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		d2 = ::atan2(r.c[0], i.c[0]);
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		i.c[1] = d1;
		r.c[1] = d2;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
		d2 = ::atan2(r.c[2], i.c[2]);
		i.c[2] = d1;
		r.c[2] = d2;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
		d2 = ::atan2(r.c[3], i.c[3]);
		i.c[3] = d1;
		r.c[3] = d2;
		d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / 2;
		d2 = ::atan2(r.c[4], i.c[4]);
		i.c[4] = d1;
		r.c[4] = d2;
		d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / 2;
		d2 = ::atan2(r.c[5], i.c[5]);
		i.c[5] = d1;
		r.c[5] = d2;
		d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / 2;
		d2 = ::atan2(r.c[6], i.c[6]);
		i.c[6] = d1;
		r.c[6] = d2;
		d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / 2;
		d2 = ::atan2(r.c[7], i.c[7]);
		i.c[7] = d1;
		r.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *log() {
		float d1;
		float d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / 2;
		d2 = ::atan2(r.c[0], i.c[0]);
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / 2;
		d2 = ::atan2(r.c[1], i.c[1]);
		i.c[1] = d1;
		r.c[1] = d2;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / 2;
		d2 = ::atan2(r.c[2], i.c[2]);
		i.c[2] = d1;
		r.c[2] = d2;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / 2;
		d2 = ::atan2(r.c[3], i.c[3]);
		i.c[3] = d1;
		r.c[3] = d2;
		d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / 2;
		d2 = ::atan2(r.c[4], i.c[4]);
		i.c[4] = d1;
		r.c[4] = d2;
		d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / 2;
		d2 = ::atan2(r.c[5], i.c[5]);
		i.c[5] = d1;
		r.c[5] = d2;
		d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / 2;
		d2 = ::atan2(r.c[6], i.c[6]);
		i.c[6] = d1;
		r.c[6] = d2;
		d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / 2;
		d2 = ::atan2(r.c[7], i.c[7]);
		i.c[7] = d1;
		r.c[7] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(Array8Complex32) *log10() {
		float d1;
		float d2;
		d1 = ::log(i.c[0] * i.c[0] + r.c[0] * r.c[0]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[0], i.c[0]) / AML_LN10;
		i.c[0] = d1;
		r.c[0] = d2;
		d1 = ::log(i.c[1] * i.c[1] + r.c[1] * r.c[1]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[1], i.c[1]) / AML_LN10;
		i.c[1] = d1;
		r.c[1] = d2;
		d1 = ::log(i.c[2] * i.c[2] + r.c[2] * r.c[2]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[2], i.c[2]) / AML_LN10;
		i.c[2] = d1;
		r.c[2] = d2;
		d1 = ::log(i.c[3] * i.c[3] + r.c[3] * r.c[3]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[3], i.c[3]) / AML_LN10;
		i.c[3] = d1;
		r.c[3] = d2;
		d1 = ::log(i.c[4] * i.c[4] + r.c[4] * r.c[4]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[4], i.c[4]) / AML_LN10;
		i.c[4] = d1;
		r.c[4] = d2;
		d1 = ::log(i.c[5] * i.c[5] + r.c[5] * r.c[5]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[5], i.c[5]) / AML_LN10;
		i.c[5] = d1;
		r.c[5] = d2;
		d1 = ::log(i.c[6] * i.c[6] + r.c[6] * r.c[6]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[6], i.c[6]) / AML_LN10;
		i.c[6] = d1;
		r.c[6] = d2;
		d1 = ::log(i.c[7] * i.c[7] + r.c[7] * r.c[7]) / (2 * AML_LN10);
		d2 = ::atan2(r.c[7], i.c[7]) / AML_LN10;
		i.c[7] = d1;
		r.c[7] = d2;
		return this;
	}

};


#if !defined(AML_NO_STRING)

AML_FUNCTION std::string operator<<(std::string &lhs, const AML_PREFIX(Array8Complex32) &rhs) {
	std::ostringstream string;
	string << lhs << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1]
		   << "i ,  " << rhs.r.c[2] << " + " << rhs.i.c[2] << "i ,  " << rhs.r.c[3] << " + " << rhs.i.c[3] << "i ,  "
		   << rhs.r.c[4] << " + " << rhs.i.c[4] << "i ,  " << rhs.r.c[5] << " + " << rhs.i.c[5] << "i ,  " << rhs.r.c[6]
		   << " + " << rhs.i.c[6] << "i ,  " << rhs.r.c[7] << " + " << rhs.i.c[7] << "i }";
	return string.str();
}

AML_FUNCTION std::string operator<<(const char *lhs, const AML_PREFIX(Array8Complex32) &rhs) {
	std::ostringstream string;
	string << lhs << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1]
		   << "i ,  " << rhs.r.c[2] << " + " << rhs.i.c[2] << "i ,  " << rhs.r.c[3] << " + " << rhs.i.c[3] << "i ,  "
		   << rhs.r.c[4] << " + " << rhs.i.c[4] << "i ,  " << rhs.r.c[5] << " + " << rhs.i.c[5] << "i ,  " << rhs.r.c[6]
		   << " + " << rhs.i.c[6] << "i ,  " << rhs.r.c[7] << " + " << rhs.i.c[7] << "i }";
	return string.str();
}

template<class charT, class traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &o, const AML_PREFIX(Array8Complex32) &rhs) {
	std::basic_ostringstream<charT, traits> s;
	s.flags(o.flags());
	s.imbue(o.getloc());
	s.precision(o.precision());
	s << "{ " << rhs.r.c[0] << " + " << rhs.i.c[0] << "i ,  " << rhs.r.c[1] << " + " << rhs.i.c[1]
	  << "i ,  " << rhs.r.c[2] << " + " << rhs.i.c[2] << "i ,  " << rhs.r.c[3] << " + " << rhs.i.c[3] << "i ,  "
	  << rhs.r.c[4] << " + " << rhs.i.c[4] << "i ,  " << rhs.r.c[5] << " + " << rhs.i.c[5] << "i ,  " << rhs.r.c[6]
	  << " + " << rhs.i.c[6] << "i ,  " << rhs.r.c[7] << " + " << rhs.i.c[7] << "i }";
	return o << s.str();
}

#endif

AML_FUNCTION AML_PREFIX(Array8Complex32)
operator+(const AML_PREFIX(Complex32) &lhs, const AML_PREFIX(Array8Complex32) &rhs) {
	return rhs + lhs;
}


AML_FUNCTION AML_PREFIX(Array8Complex32)
operator-(const AML_PREFIX(Complex32) &lhs, const AML_PREFIX(Array8Complex32) &rhs) {
	AML_PREFIX(Array8Complex32) ret;
	ret.i.c[0] = lhs.c.c[1] - rhs.i.c[0];
	ret.i.c[1] = lhs.c.c[1] - rhs.i.c[1];
	ret.i.c[2] = lhs.c.c[1] - rhs.i.c[2];
	ret.i.c[3] = lhs.c.c[1] - rhs.i.c[3];
	ret.i.c[4] = lhs.c.c[1] - rhs.i.c[4];
	ret.i.c[5] = lhs.c.c[1] - rhs.i.c[5];
	ret.i.c[6] = lhs.c.c[1] - rhs.i.c[6];
	ret.i.c[7] = lhs.c.c[1] - rhs.i.c[7];
	ret.r.c[0] = lhs.c.c[0] - rhs.r.c[0];
	ret.r.c[1] = lhs.c.c[0] - rhs.r.c[1];
	ret.r.c[2] = lhs.c.c[0] - rhs.r.c[2];
	ret.r.c[3] = lhs.c.c[0] - rhs.r.c[3];
	ret.r.c[4] = lhs.c.c[0] - rhs.r.c[4];
	ret.r.c[5] = lhs.c.c[0] - rhs.r.c[5];
	ret.r.c[6] = lhs.c.c[0] - rhs.r.c[6];
	ret.r.c[7] = lhs.c.c[0] - rhs.r.c[7];
	return ret;
}

AML_FUNCTION AML_PREFIX(Array8Complex32)
operator*(const AML_PREFIX(Complex32) &lhs, const AML_PREFIX(Array8Complex32) &rhs) {
	return rhs * lhs;
}

AML_FUNCTION AML_PREFIX(Array8Complex32)
operator/(const AML_PREFIX(Complex32) &lhs, const AML_PREFIX(Array8Complex32) &rhs) {
	AML_PREFIX(Array8Complex32) ret;
	float d1 =
			(lhs.c.c[0] * rhs.r.c[0] + lhs.c.c[1] * rhs.i.c[0]) / (rhs.r.c[0] * rhs.r.c[0] + rhs.i.c[0] * rhs.i.c[0]);
	float d2 =
			(lhs.c.c[1] * rhs.r.c[0] - lhs.c.c[0] * rhs.i.c[0]) / (rhs.r.c[0] * rhs.r.c[0] + rhs.i.c[0] * rhs.i.c[0]);
	ret.r.c[0] = d1;
	ret.i.c[0] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[1] + lhs.c.c[1] * rhs.i.c[1]) / (rhs.r.c[1] * rhs.r.c[1] + rhs.i.c[1] * rhs.i.c[1]);
	d2 = (lhs.c.c[1] * rhs.r.c[1] - lhs.c.c[0] * rhs.i.c[1]) / (rhs.r.c[1] * rhs.r.c[1] + rhs.i.c[1] * rhs.i.c[1]);
	ret.r.c[1] = d1;
	ret.i.c[1] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[2] + lhs.c.c[1] * rhs.i.c[2]) / (rhs.r.c[2] * rhs.r.c[2] + rhs.i.c[2] * rhs.i.c[2]);
	d2 = (lhs.c.c[1] * rhs.r.c[2] - lhs.c.c[0] * rhs.i.c[2]) / (rhs.r.c[2] * rhs.r.c[2] + rhs.i.c[2] * rhs.i.c[2]);
	ret.r.c[2] = d1;
	ret.i.c[2] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[3] + lhs.c.c[1] * rhs.i.c[3]) / (rhs.r.c[3] * rhs.r.c[3] + rhs.i.c[3] * rhs.i.c[3]);
	d2 = (lhs.c.c[1] * rhs.r.c[3] - lhs.c.c[0] * rhs.i.c[3]) / (rhs.r.c[3] * rhs.r.c[3] + rhs.i.c[3] * rhs.i.c[3]);
	ret.r.c[3] = d1;
	ret.i.c[3] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[4] + lhs.c.c[1] * rhs.i.c[4]) / (rhs.r.c[4] * rhs.r.c[4] + rhs.i.c[4] * rhs.i.c[4]);
	d2 = (lhs.c.c[1] * rhs.r.c[4] - lhs.c.c[0] * rhs.i.c[4]) / (rhs.r.c[4] * rhs.r.c[4] + rhs.i.c[4] * rhs.i.c[4]);
	ret.r.c[4] = d1;
	ret.i.c[4] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[5] + lhs.c.c[1] * rhs.i.c[5]) / (rhs.r.c[5] * rhs.r.c[5] + rhs.i.c[5] * rhs.i.c[5]);
	d2 = (lhs.c.c[1] * rhs.r.c[5] - lhs.c.c[0] * rhs.i.c[5]) / (rhs.r.c[5] * rhs.r.c[5] + rhs.i.c[5] * rhs.i.c[5]);
	ret.r.c[5] = d1;
	ret.i.c[5] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[6] + lhs.c.c[1] * rhs.i.c[6]) / (rhs.r.c[6] * rhs.r.c[6] + rhs.i.c[6] * rhs.i.c[6]);
	d2 = (lhs.c.c[1] * rhs.r.c[6] - lhs.c.c[0] * rhs.i.c[6]) / (rhs.r.c[6] * rhs.r.c[6] + rhs.i.c[6] * rhs.i.c[6]);
	ret.r.c[6] = d1;
	ret.i.c[6] = d2;
	d1 = (lhs.c.c[0] * rhs.r.c[7] + lhs.c.c[1] * rhs.i.c[7]) / (rhs.r.c[7] * rhs.r.c[7] + rhs.i.c[7] * rhs.i.c[7]);
	d2 = (lhs.c.c[1] * rhs.r.c[7] - lhs.c.c[0] * rhs.i.c[7]) / (rhs.r.c[7] * rhs.r.c[7] + rhs.i.c[7] * rhs.i.c[7]);
	ret.r.c[7] = d1;
	ret.i.c[7] = d2;
	return ret;
}

#if defined(AML_USE_STD_COMPLEX)

#if defined(USE_CONCEPTS)
template<class T>
concept ComplexNumber = requires(T a, T b){
	a + a;
	a * a;
	a / b;
	a.ln();
	a.sin();
};
#endif


#if defined(USE_CONCEPTS)

template<ComplexNumber C>
#else
template<class C>
#endif
AML_FUNCTION auto log(C c) { return *c.ln(); }

#if defined(USE_CONCEPTS)

template<ComplexNumber C>
#else
template<class C>
#endif
AML_FUNCTION auto sin(C c) { return *c.sin(); }

#if defined(USE_CONCEPTS)

template<ComplexNumber C>
#else
template<class C>
#endif
AML_FUNCTION auto log10(C c) { return *c.log10(); }

#if defined(USE_CONCEPTS)

template<ComplexNumber C>
#else
template<class C>
#endif
AML_FUNCTION auto tan(C c) { return *c.tan(); }

#if defined(USE_CONCEPTS)

template<ComplexNumber C>
#else
template<class C>
#endif
AML_FUNCTION auto cos(C c) { return *c.cos(); }


#endif //std::complex compatibility


#endif //MATH_LIB_A_MATH_LIB_H
