//
// Created by af on 10.03.21.
//

class AML_PREFIX(AML_TYPE_NAME(Array1)) {
public:
	AML_TYPE a;
};

class AML_PREFIX(AML_TYPE_NAME(Array2)) {
public:
	AML_TYPE a[2];
};

class AML_PREFIX(AML_TYPE_NAME(Array3)) {
public:
	AML_TYPE a[3];
};

class AML_PREFIX(AML_TYPE_NAME(Array4)) {
public:
#if AML_TYPE == double
	AML_PREFIX(doublevec4) v{};
#elif AML_TYPE == float
	AML_PREFIX(floatvec4) v{};
#endif

	constexpr AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Array4)) *add(const AML_PREFIX(AML_TYPE_NAME(Array4)) a) {
		if (std::is_constant_evaluated()) {
			v.c[0] += a.v.c[0];
			v.c[1] += a.v.c[1];
			v.c[2] += a.v.c[2];
			v.c[3] += a.v.c[3];
			return this;
		}
#if defined(USE_AVX) && AML_TYPE == double
		v.avx = _mm256_add_pd(v.avx, a.v.avx);
#elif defined(USE_SSE) && AML_TYPE == double // SSE2
		v.sse[0] = _mm_add_pd(v.sse[0], a.v.sse[0]);
		v.sse[1] = _mm_add_pd(v.sse[1], a.v.sse[1]);
#elif defined(USE_NEON) && AML_TYPE == double
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

	constexpr AML_FUNCTION auto f(int a) -> int {
		return a;
	}
};

class AML_PREFIX(AML_TYPE_NAME(Array8)) {
public:
	AML_TYPE a;
};

class AML_PREFIX(AML_TYPE_NAME(Array16)) {
public:
	AML_TYPE a;
};

template<const int size>
class AML_PREFIX(AML_TYPE_NAME(ArrayN)) {
public:
	AML_TYPE a[size];

	AML_TYPE_NAME(ArrayN)() {
	}
};
