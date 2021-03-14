//
// Created by af on 14.03.21.
//

#include <stdint.h>

#if !defined(USE_CUDA) && __cpp_lib_is_constant_evaluated
#define AML_CONSTEXPR constexpr
#else
#define AML_CONSTEXPR
#endif

class AML_PREFIX(AML_TYPE_NAME(Vector1D)) {
public:
	AML_TYPE v{};

	AML_PREFIX(AML_TYPE_NAME(Vector1D)) *set(AML_TYPE value, AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) { v = value; }
		return this;
	}

	AML_CONSTEXPR AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector1D))(const AML_TYPE *const values) {
		v = values[0];
	}

	AML_CONSTEXPR AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector1D))() {
		v = 0.0f;
	}

	AML_CONSTEXPR AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector1D))(AML_TYPE value) {
		v = value;
	}

	AML_CONSTEXPR AML_FUNCTION AML_TYPE &operator[]([[maybe_unused]]uint32_t location) {
		return v;
	}

};

class AML_PREFIX(AML_TYPE_NAME(Vector2D)) {
public:
#if AML_TYPE_ID == AML_TYPE_DOUBLE
	AML_PREFIX(doublevec2) v{};
#elif AML_TYPE_ID == AML_TYPE_FLOAT
	AML_PREFIX(floatvec2) v{};
#endif

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector2D)) *set(AML_TYPE value, const AML_PREFIX(VectorU8_2D) mask) {
		if (mask.v.c[0]) { v.c[0] = value; }
		if (mask.v.c[1]) { v.c[1] = value; }
		return this;
	}

	AML_CONSTEXPR AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector2D))(const AML_TYPE *const values) {
		v.c[0] = values[0];
		v.c[1] = values[1];
	}

	AML_CONSTEXPR AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector2D))() {
		v.c[0] = 0.0;
		v.c[1] = 0.0;
	}

	AML_CONSTEXPR AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector2D))(AML_PREFIX(const AML_TYPE_NAME(Vector1D)) value) {
		v.c[0] = value.v;
		v.c[1] = 0.0;
	};

	AML_CONSTEXPR AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector2D))(AML_TYPE value) {
		v.c[0] = value;
		v.c[1] = value;
	}

	AML_CONSTEXPR AML_FUNCTION AML_TYPE &operator[](uint32_t location) {
		return (AML_TYPE &) v.c[location];
	}

#if defined(USE_SSE) && AML_TYPE_ID == AML_TYPE_DOUBLE

	AML_CONSTEXPR AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector2D))(const __m128d value) {
		v.sse = value;
	}

#endif

};

class AML_PREFIX(AML_TYPE_NAME(Vector4D)) {
private:

public:
#if AML_TYPE_ID == AML_TYPE_DOUBLE
	AML_PREFIX(doublevec4) v{};
#endif
#if AML_TYPE_ID == AML_TYPE_FLOAT
	AML_PREFIX(floatvec4) v{};
#endif

	AML_FUNCTION AML_TYPE &operator[](uint32_t position) {
		return v.c[position];
	}

	AML_FUNCTION void operator+=(AML_PREFIX(AML_TYPE_NAME(Vector4D)) vec2) {
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
		v.avx = _mm256_add_pd(v.avx, vec2.v.avx);
#elif defined(USE_SSE) && AML_TYPE_ID == AML_TYPE_DOUBLE // SSE2
		v.sse[0] = _mm_add_pd(v.sse[0], vec2.v.sse[0]);
	v.sse[1] = _mm_add_pd(v.sse[1], vec2.v.sse[1]);
#elif defined(USE_NEON) && AML_TYPE_ID == AML_TYPE_DOUBLE
		v.neon[0] = vaddq_f64(v.neon[0], vec2.v.neon[0]);
	v.neon[1] = vaddq_f64(v.neon[1], vec2.v.neon[1]);
#else
		v.c[0] += vec2[0];
		v.c[1] += vec2[1];
		v.c[2] += vec2[2];
		v.c[3] += vec2[3];
#endif


	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) operator+(const AML_PREFIX(AML_TYPE_NAME(Vector4D)) vec2) {
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
		AML_PREFIX(AML_TYPE_NAME(Vector4D)) ret;
		ret.v.avx = _mm256_add_pd(v.avx, vec2.v.avx);
		return ret;
#elif defined(USE_SSE2) && AML_TYPE_ID == AML_TYPE_DOUBLE
		AML_PREFIX(AML_TYPE_NAME(Vector4D)) ret;
	ret.v.sse[0] = _mm_add_pd(v.sse[0], vec2.v.sse[0]);
	ret.v.sse[1] = _mm_add_pd(v.sse[1], vec2.v.sse[1]);
	return ret;
#elif defined(USE_NEON) && AML_TYPE_ID == AML_TYPE_DOUBLE
		AML_PREFIX(AML_TYPE_NAME(Vector4D)) ret;
	ret.v.neon[0] = vaddq_f64(v.neon[0], vec2.v.neon[0]);
	ret.v.neon[1] = vaddq_f64(v.neon[1], vec2.v.neon[1]);
	return ret;
#else
		AML_PREFIX(AML_TYPE_NAME(Vector4D)) ret(v.c[0] + vec2.v.c[0], v.c[1] + vec2.v.c[1], v.c[2] + vec2.v.c[2],
												v.c[3] + vec2.v.c[3]);
		return ret;
#endif


	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) operator+(AML_TYPE a) {
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
		AML_PREFIX(AML_TYPE_NAME(Vector4D)) ret(a);
		ret.v.avx = _mm256_add_pd(v.avx, ret.v.avx);
		return ret;
#elif defined(USE_SSE2) && AML_TYPE_ID == AML_TYPE_DOUBLE
		AML_PREFIX(AML_TYPE_NAME(Vector4D)) ret(a);
	ret.v.sse[0] = _mm_add_pd(v.sse[0], ret.v.sse[0]);
	ret.v.sse[1] = _mm_add_pd(v.sse[1], ret.v.sse[1]);
	return ret;
#elif defined(USE_NEON) && AML_TYPE_ID == AML_TYPE_DOUBLE
		AML_PREFIX(AML_TYPE_NAME(Vector4D)) ret(a);
	ret.v.neon[0] = vaddq_f64(v.neon[0], ret.v.neon[0]);
	ret.v.neon[1] = vaddq_f64(v.neon[1], ret.v.neon[1]);
	return ret;
#else
		AML_PREFIX(AML_TYPE_NAME(Vector4D)) ret(v.c[0] + a, v.c[1] + a, v.c[2] + a, v.c[3] + a);
		return ret;
#endif


	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) *add(const AML_PREFIX(AML_TYPE_NAME(Vector4D)) a) {
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
		v.avx = _mm256_add_pd(v.avx, a.v.avx);
#elif defined(USE_SSE) && AML_TYPE_ID == AML_TYPE_DOUBLE // SSE2
		v.sse[0] = _mm_add_pd(v.sse[0], a.v.sse[0]);
	v.sse[1] = _mm_add_pd(v.sse[1], a.v.sse[1]);
#elif defined(USE_NEON) && AML_TYPE_ID == AML_TYPE_DOUBLE
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

#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
		AML_PREFIX(doublevec4) a = {0.0f, 0.0f, 0.0f, 0.0f};
		v.avx = _mm256_sub_pd(a.avx, v.avx);
#elif defined(USE_SSE) && AML_TYPE_ID == AML_TYPE_DOUBLE // SSE2
		AML_TYPE a[2] = {0.0f, 0.0f};
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

	AML_FUNCTION void operator-=(AML_PREFIX(AML_TYPE_NAME(Vector4D)) vec2) {
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
		v.avx = _mm256_sub_pd(v.avx, vec2.v.avx);
#else
		v.c[0] -= vec2[0];
		v.c[1] -= vec2[1];
		v.c[2] -= vec2[2];
		v.c[3] -= vec2[3];
#endif


	}


	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) operator-(const AML_PREFIX(AML_TYPE_NAME(Vector4D)) vec2) {
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
		AML_PREFIX(AML_TYPE_NAME(Vector4D)) ret;
		ret.v.avx = _mm256_sub_pd(v.avx, vec2.v.avx);
		return ret;
#else
		AML_PREFIX(AML_TYPE_NAME(Vector4D)) ret(v.c[0] - vec2.v.c[0], v.c[1] - vec2.v.c[1], v.c[2] - vec2.v.c[2],
												v.c[3] - vec2.v.c[3]);
		return ret;
#endif


	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) operator-(AML_TYPE a) {
		AML_PREFIX(AML_TYPE_NAME(Vector4D)) ret(a);
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
		ret.v.avx = _mm256_sub_pd(v.avx, ret.v.avx);
		return ret;
#else
		ret = AML_PREFIX(AML_TYPE_NAME(Vector4D))(v.c[0] - a, v.c[1] - a, v.c[2] - a, v.c[3] - a);
		return ret;
#endif


	}

	AML_FUNCTION AML_TYPE length() {
		return sqrt(v.c[0] * v.c[0] + v.c[1] * v.c[1] + v.c[2] * v.c[2] + v.c[3] * v.c[3]);
	}

	AML_FUNCTION void normalize() {
		AML_TYPE vecLength = 1 / length();
		v.c[0] *= vecLength;
		v.c[1] *= vecLength;
		v.c[2] *= vecLength;
		v.c[3] *= vecLength;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) *forEachSin() {
#if defined(USE_AVX) && defined(__INTEL_COMPILER) && AML_TYPE_ID == AML_TYPE_DOUBLE
		v.avx = _mm256_sin_pd(v.avx);
#else
		v.c[0] = sin(v.c[0]);
		v.c[1] = sin(v.c[1]);
		v.c[2] = sin(v.c[2]);
		v.c[3] = sin(v.c[3]);
#endif
		return this;
	}

	AML_FUNCTION void operator*=(AML_TYPE scalar) {
		v.c[0] *= scalar;
		v.c[1] *= scalar;
		v.c[2] *= scalar;
		v.c[3] *= scalar;
	}

	// for each multiply
	AML_FUNCTION void operator*=(AML_PREFIX(AML_TYPE_NAME(Vector4D)) vec2) {
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
		v.avx = _mm256_mul_pd(v.avx, vec2.v.avx);
#elif defined(USE_NEON) && AML_TYPE_ID == AML_TYPE_DOUBLE
		v.neon[0] = vmulq_f64(v.neon[0], vec2.v.neon[0]);
	v.neon[1] = vmulq_f64(v.neon[1], vec2.v.neon[1]);
#else
		v.c[0] *= vec2.v.c[0];
		v.c[1] *= vec2.v.c[1];
		v.c[2] *= vec2.v.c[2];
		v.c[3] *= vec2.v.c[3];
#endif
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) *forEachSqrt() {
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
		v.avx = _mm256_sqrt_pd(v.avx);
#elif defined(USE_SSE) && AML_TYPE_ID == AML_TYPE_DOUBLE
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

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) operator*(AML_TYPE a) {
		AML_PREFIX(AML_TYPE_NAME(Vector4D)) ret;
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
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

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) operator/(AML_TYPE a) {
		AML_PREFIX(AML_TYPE_NAME(Vector4D)) ret;
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
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


	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) *capBetween1_0() {
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
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

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) *
	capBetweenX_Y(const AML_TYPE upperBoundary, const AML_TYPE lowerBoundary) {
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
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

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) *
	map(const AML_TYPE lowerInput, const AML_TYPE upperInput, const AML_TYPE lowerOutput, const AML_TYPE upperOutput) {
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
		AML_PREFIX(doublevec4) a = {lowerInput, lowerInput, lowerInput, lowerInput};
		a.avx = _mm256_sub_pd(v.avx, a.avx);
		AML_TYPE factor = (upperOutput - lowerOutput) / (upperInput - lowerInput);
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

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) *mapNonLinear(const AML_TYPE lowerInput, const AML_TYPE upperInput,
																   const AML_TYPE lowerOutput,
																   const AML_TYPE upperOutput,
																   const AML_TYPE factor) {
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

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) *forEachInterpolate(const AML_TYPE min, const AML_TYPE max) {
		v.c[0] = (v.c[0] * (min - max) + max);
		v.c[1] = (v.c[1] * (min - max) + max);
		v.c[2] = (v.c[2] * (min - max) + max);
		v.c[3] = (v.c[3] * (min - max) + max);
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) *
	interpolate(const AML_PREFIX(AML_TYPE_NAME(Vector4D)) max, AML_TYPE ratio) {
		v.c[0] = (ratio * (v.c[0] - max.v.c[0]) + max.v.c[0]);
		v.c[1] = (ratio * (v.c[1] - max.v.c[1]) + max.v.c[1]);
		v.c[2] = (ratio * (v.c[2] - max.v.c[2]) + max.v.c[2]);
		v.c[3] = (ratio * (v.c[3] - max.v.c[3]) + max.v.c[3]);
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) *
	interpolate(const AML_PREFIX(AML_TYPE_NAME(Vector4D)) val2, const AML_PREFIX(AML_TYPE_NAME(Vector4D)) val3,
				AML_TYPE ratio) {
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

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) *set(AML_TYPE value, AML_PREFIX(VectorU8_4D) mask) {
		if (mask.v.c[0]) { v.c[0] = value; }
		if (mask.v.c[1]) { v.c[1] = value; }
		if (mask.v.c[2]) { v.c[2] = value; }
		if (mask.v.c[3]) { v.c[3] = value; }
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) *
	set(const AML_PREFIX(AML_TYPE_NAME(Vector4D)) value, AML_PREFIX(VectorU8_4D) mask) {
		if (mask.v.c[0]) { v.c[0] = value.v.c[0]; }
		if (mask.v.c[1]) { v.c[1] = value.v.c[1]; }
		if (mask.v.c[2]) { v.c[2] = value.v.c[2]; }
		if (mask.v.c[3]) { v.c[3] = value.v.c[3]; }
		return this;
	}

	template<const int a, const int b, const int c, const int d>
	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) *permutation() {
#if defined(USE_AVX2) && AML_TYPE_ID == AML_TYPE_DOUBLE
		v.avx = _mm256_permute4x64_pd(v.avx, a + (b << 2) + (c << 4) + (d << 6));
#else
		AML_TYPE a1 = v.c[a];
		AML_TYPE b1 = v.c[b];
		AML_TYPE c1 = v.c[c];
		AML_TYPE d1 = v.c[d];
		v.c[0] = a1;
		v.c[1] = b1;
		v.c[2] = c1;
		v.c[3] = d1;
#endif
		return this;
	}


	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D))(AML_TYPE a, AML_TYPE b, AML_TYPE c, AML_TYPE d) {
		v.c[0] = a;
		v.c[1] = b;
		v.c[2] = c;
		v.c[3] = d;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D))() {
		v.c[0] = 0;
		v.c[1] = 0;
		v.c[2] = 0;
		v.c[3] = 0;
	}

	AML_FUNCTION explicit AML_PREFIX(AML_TYPE_NAME(Vector4D))(const AML_TYPE a) {
		v.c[0] = a;
		v.c[1] = a;
		v.c[2] = a;
		v.c[3] = a;
	}

#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE

	AML_FUNCTION explicit AML_PREFIX(AML_TYPE_NAME(Vector4D))(const __m256d a) {
		v.avx = a;
	}

#endif

#if defined(USE_SSE) && AML_TYPE_ID == AML_TYPE_DOUBLE

	AML_FUNCTION explicit AML_PREFIX(AML_TYPE_NAME(Vector4D))(const __m128d *const values) {
		v.sse[0] = values[0];
		v.sse[1] = values[1];
	}

#endif

	AML_FUNCTION explicit AML_PREFIX(AML_TYPE_NAME(Vector4D))(const AML_TYPE *const values) {
		v.c[0] = values[0];
		v.c[1] = values[1];
		v.c[2] = values[2];
		v.c[3] = values[3];
	}

};

class AML_PREFIX(AML_TYPE_NAME(Matrix4x4)) {
public:

#if AML_TYPE_ID == AML_TYPE_DOUBLE
	AML_PREFIX(doublemat4x4) m{};
#elif AML_TYPE_ID == AML_TYPE_FLOAT
	AML_PREFIX(floatmat4x4) m{};
#endif

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) operator[](uint32_t column) {
		AML_PREFIX(AML_TYPE_NAME(Vector4D)) ret;
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
		ret.v.avx = m.avx[column];
#else
		ret.v.c[0] = m.c[column * 4];
		ret.v.c[1] = m.c[column * 4 + 1];
		ret.v.c[2] = m.c[column * 4 + 2];
		ret.v.c[3] = m.c[column * 4 + 3];
#endif
		return ret;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Matrix4x4)) *identity() {
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

	AML_CONSTEXPR AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Matrix4x4))
	operator*(const AML_PREFIX(AML_TYPE_NAME(Matrix4x4)) &b) const noexcept {
#if defined(__cpp_lib_is_constant_evaluated)
		if (std::is_constant_evaluated()) {
			AML_PREFIX(AML_TYPE_NAME(Matrix4x4)) ret;
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
			return ret;
		}
#endif
		AML_PREFIX(AML_TYPE_NAME(Matrix4x4)) ret;
#if defined(USE_AVX512F) && AML_TYPE_ID == AML_TYPE_DOUBLE
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
#elif defined(USE_FMA) && AML_TYPE_ID == AML_TYPE_DOUBLE
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

#elif defined(USE_SSE2) && AML_TYPE_ID == AML_TYPE_DOUBLE

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
#elif defined(USE_NEON) && AML_TYPE_ID == AML_TYPE_DOUBLE
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

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector4D)) operator*(const AML_PREFIX(AML_TYPE_NAME(Vector4D)) b) {
		AML_PREFIX(AML_TYPE_NAME(Vector4D)) ret;
		ret.v.c[0] = m.c[0] * b.v.c[0] + m.c[4] * b.v.c[1] + m.c[8] * b.v.c[2] + m.c[12] * b.v.c[3];
		ret.v.c[1] = m.c[1] * b.v.c[0] + m.c[5] * b.v.c[1] + m.c[9] * b.v.c[2] + m.c[13] * b.v.c[3];
		ret.v.c[2] = m.c[2] * b.v.c[0] + m.c[6] * b.v.c[1] + m.c[10] * b.v.c[2] + m.c[14] * b.v.c[3];
		ret.v.c[3] = m.c[3] * b.v.c[0] + m.c[7] * b.v.c[1] + m.c[11] * b.v.c[2] + m.c[15] * b.v.c[3];
		return ret;
	}

	AML_CONSTEXPR AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Matrix4x4))() {
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

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Matrix4x4))(const AML_PREFIX(AML_TYPE_NAME(Vector4D)) &a,
													  const AML_PREFIX(AML_TYPE_NAME(Vector4D)) &b,
													  const AML_PREFIX(AML_TYPE_NAME(Vector4D)) &c,
													  const AML_PREFIX(AML_TYPE_NAME(Vector4D)) &d) {
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
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

	AML_CONSTEXPR AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Matrix4x4))(const AML_PREFIX(AML_TYPE_NAME(Matrix4x4)) &b) {
		if (std::is_constant_evaluated()) {
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
			return;
		}
#if defined(USE_AVX512) && AML_TYPE_ID == AML_TYPE_DOUBLE
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

class AML_PREFIX(AML_TYPE_NAME(Vector8D)) {
public:
#if AML_TYPE_ID == AML_TYPE_DOUBLE
	AML_PREFIX(doublevec8) v{};
#elif AML_TYPE_ID == AML_TYPE_FLOAT
	AML_PREFIX(floatvec8) v{};
#endif

	AML_FUNCTION AML_TYPE operator[](uint32_t position) {
		return v.c[position];
	}

	AML_FUNCTION void operator+=(const AML_PREFIX(AML_TYPE_NAME(Vector8D)) &vec2) {
#if defined(USE_AVX512F) || defined(KNCNI) && AML_TYPE_ID == AML_TYPE_DOUBLE
		v.avx512 = _mm512_add_pd(v.avx512, vec2.v.avx512);
#elif defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
		v.avx[0] = _mm256_add_pd(v.avx[0], vec2.v.avx[0]);
		v.avx[1] = _mm256_add_pd(v.avx[1], vec2.v.avx[1]);
#elif defined(USE_SSE) && AML_TYPE_ID == AML_TYPE_DOUBLE // SSE2
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

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector8D))(AML_TYPE a, AML_TYPE b, AML_TYPE c, AML_TYPE d, AML_TYPE e,
													 AML_TYPE f, AML_TYPE g,
													 AML_TYPE h) {
		v.c[0] = a;
		v.c[1] = b;
		v.c[2] = c;
		v.c[3] = d;
		v.c[4] = e;
		v.c[5] = f;
		v.c[6] = g;
		v.c[7] = h;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector8D))(const AML_PREFIX(AML_TYPE_NAME(Vector4D)) a,
													 const AML_PREFIX(AML_TYPE_NAME(Vector4D)) b) {
#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE
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

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector8D))() {
		v.c[0] = 0.0f;
		v.c[1] = 0.0f;
		v.c[2] = 0.0f;
		v.c[3] = 0.0f;
		v.c[4] = 0.0f;
		v.c[5] = 0.0f;
		v.c[6] = 0.0f;
		v.c[7] = 0.0f;
	}

#if defined(USE_AVX) && AML_TYPE_ID == AML_TYPE_DOUBLE

	AML_FUNCTION explicit AML_PREFIX(AML_TYPE_NAME(Vector8D))(__m256d *values) {
		v.avx[0] = values[0];
		v.avx[1] = values[1];
	}

#endif
#if defined(USE_SSE) && AML_TYPE_ID == AML_TYPE_DOUBLE

	AML_FUNCTION explicit AML_PREFIX(AML_TYPE_NAME(Vector8D))(__m128d *values) {
		v.sse[0] = values[0];
		v.sse[1] = values[1];
		v.sse[2] = values[2];
		v.sse[3] = values[3];
	}

#endif

#if defined(USE_AVX512) && AML_TYPE_ID == AML_TYPE_DOUBLE

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector8D))(__m512d value) {
	v.avx512 = value;
}

#endif

	AML_FUNCTION explicit AML_PREFIX(AML_TYPE_NAME(Vector8D))(const AML_TYPE *const values) {
		v.c[0] = values[0];
		v.c[1] = values[1];
		v.c[2] = values[2];
		v.c[3] = values[3];
		v.c[4] = values[4];
		v.c[5] = values[5];
		v.c[6] = values[6];
		v.c[7] = values[7];
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector8D))(const AML_TYPE value) {
		v.c[0] = value;
		v.c[1] = value;
		v.c[2] = value;
		v.c[3] = value;
		v.c[4] = value;
		v.c[5] = value;
		v.c[6] = value;
		v.c[7] = value;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Vector8D)) *set(AML_TYPE value, AML_PREFIX(VectorU8_8D) mask) {
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
