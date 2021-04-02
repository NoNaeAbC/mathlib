//
// Created by af on 10.03.21.
//
class AML_TYPE_NAME(Array1) {
public:
	AML_TYPE a;
};

class AML_TYPE_NAME(Array2) {
public:
	AML_TYPE a[2];
};


class AML_TYPE_NAME(Array3) {
public:
	AML_TYPE a[3];
};

class AML_TYPE_NAME(Array4) {
public:
#if AML_TYPE_ID == 1
	AML_PREFIX(doublevec4) v{};
#elif AML_TYPE_ID == 2
	AML_PREFIX(floatvec4) v{};
#endif

	AML_CONSTEXPR AML_FUNCTION AML_TYPE_NAME(Array4) *add(const AML_TYPE_NAME(Array4) a) {
#if defined(__cpp_lib_is_constant_evaluated)
		if (std::is_constant_evaluated()) {
			v.c[0] += a.v.c[0];
			v.c[1] += a.v.c[1];
			v.c[2] += a.v.c[2];
			v.c[3] += a.v.c[3];
			return this;
		}
#endif
#if defined(USE_AVX) && AML_TYPE_ID == 1 && !defined(USE_CUDA)
		v.avx = _mm256_add_pd(v.avx, a.v.avx);
#elif defined(USE_SSE) && AML_TYPE_ID == 1 && !defined(USE_CUDA) // SSE2
		v.sse[0] = _mm_add_pd(v.sse[0], a.v.sse[0]);
		v.sse[1] = _mm_add_pd(v.sse[1], a.v.sse[1]);
#elif defined(USE_NEON) && AML_TYPE_ID == 1 && !defined(USE_CUDA)
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

class AML_TYPE_NAME(Array8) {
public:
	AML_TYPE a;
};

class AML_TYPE_NAME(Array16) {
public:
	AML_TYPE a;
};

template<const int size>
class AML_TYPE_NAME(ArrayN) {
public:
	AML_TYPE a[size];

	AML_TYPE_NAME(ArrayN)() {
	}
};
