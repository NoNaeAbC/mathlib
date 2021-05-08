
class AML_TYPE_NAME(Complex) {
		public:
		AML_PREFIX(doublevec8) r{};
		AML_PREFIX(doublevec8) i{};
		AML_FUNCTION AML_TYPE_NAME(Complex)() {}
		AML_FUNCTION AML_TYPE_NAME(Complex)(const AML_PREFIX( X ## 32) value) {
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

		LAML_FUNCTION AML_PREFIX(Vector8D_64) real() {
			return AML_PREFIX(Vector8D_64)(r.c);
		}

		AML_FUNCTION AML_PREFIX(Vector8D_64) complex() {
			return AML_PREFIX(Vector8D_64)(i.c);
		}

		AML_FUNCTION AML_PREFIX( X ## 32) operator[](uint64_t location) {
			return AML_PREFIX(X ## 32)(r.c[location], i.c[location]);
		}

		AML_FUNCTION void set(uint64_t location, AML_PREFIX( X ## 32) value) {
			r.c[location] = value.c.c[0];
			i.c[location] = value.c.c[1];
		}

		AML_FUNCTION AML_TYPE_NAME(Complex) *add(const AML_TYPE_NAME(Complex) &a) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *add(const AML_PREFIX( X ## 32) &a) {
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

		AML_FUNCTION void operator+=(const AML_TYPE_NAME(Complex) &a) {
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

		AML_FUNCTION void operator+=(const AML_PREFIX( X ## 32) a) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) operator+(const AML_TYPE_NAME(Complex) &a) const {
			AML_TYPE_NAME(Complex)
			ret;
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

		AML_FUNCTION AML_TYPE_NAME(Complex) operator+(const AML_PREFIX( X ## 32) a) const {
			AML_TYPE_NAME(Complex)
			ret{};
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *add(const AML_TYPE_NAME(Complex) &a, AML_PREFIX(VectorU8_8D) mask) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *add(const AML_PREFIX( X ## 32) a, AML_PREFIX(VectorU8_8D) mask) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *subtract(const AML_TYPE_NAME(Complex) a) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *subtract(const AML_PREFIX( X ## 32) a) {
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

		AML_FUNCTION void operator-=(const AML_TYPE_NAME(Complex) a) {
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

		AML_FUNCTION void operator-=(const AML_PREFIX( X ## 32) a) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) operator-(const AML_TYPE_NAME(Complex) &a) const {
			AML_TYPE_NAME(Complex)
			ret;
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

		AML_FUNCTION AML_TYPE_NAME(Complex) operator-(const AML_PREFIX( X ## 32) a) const {
			AML_TYPE_NAME(Complex)
			ret;
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *
		subtract(const AML_TYPE_NAME(Complex) &a, AML_PREFIX(VectorU8_8D) mask) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *subtract(const AML_PREFIX( X ## 32) a, AML_PREFIX(VectorU8_8D) mask) {
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


		AML_FUNCTION AML_TYPE_NAME(Complex) operator*(const AML_TYPE_NAME(Complex) &a) const {
			AML_TYPE_NAME(Complex)
			ret;
#if defined(USE_FMA)
			__m256d c_0 = _mm256_mul_pd(i.avx[0], a.i.avx[0]);
		ret.r.avx[0] = _mm256_fmsub_pd(r.avx[0], a.r.avx[0], c_0);
		__m256d c_1 = _mm256_mul_pd(i.avx[1], a.i.avx[1]);
		ret.r.avx[1] = _mm256_fmsub_pd(r.avx[1], a.r.avx[1], c_1);
		__m256d c_2 = _mm256_mul_pd(i.avx[0], a.r.avx[0]);
		ret.i.avx[0] = _mm256_fmadd_pd(r.avx[0], a.i.avx[0], c_2);
		__m256d c_3 = _mm256_mul_pd(i.avx[1], a.r.avx[1]);
		ret.i.avx[1] = _mm256_fmadd_pd(r.avx[1], a.i.avx[1], c_3);
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *multiply(const AML_TYPE_NAME(Complex) &a) {
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

		AML_FUNCTION void operator*=(const AML_TYPE_NAME(Complex) &a) {
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

		AML_FUNCTION void operator*=(const AML_PREFIX( X ## 32) &a) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) operator*(const AML_PREFIX( X ## 32) &a) const {
			AML_TYPE_NAME(Complex)
			ret;
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *multiply(const AML_PREFIX( X ## 32) &a) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *
		multiply(const AML_PREFIX( X ## 32) &a, const AML_PREFIX(VectorU8_8D) &mask) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *
		multiply(const AML_TYPE_NAME(Complex) &a, const AML_PREFIX(VectorU8_8D) &mask) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *square() {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *square(const AML_PREFIX(VectorU8_8D) &mask) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *divide(const AML_PREFIX( X ## 32) a) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *
		divide(const AML_PREFIX( X ## 32) a, const AML_PREFIX(VectorU8_8D) &mask) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *divide(const AML_TYPE_NAME(Complex) &a) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *
		divide(const AML_TYPE_NAME(Complex) &a, const AML_PREFIX(VectorU8_8D) &mask) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) operator/(const AML_PREFIX( X ## 32) &a) const {
			AML_TYPE_NAME(Complex)
			ret;
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

		AML_FUNCTION AML_TYPE_NAME(Complex) operator/(const AML_TYPE_NAME(Complex) &a) const {
			AML_TYPE_NAME(Complex)
			ret;
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

		AML_FUNCTION void operator/=(const AML_PREFIX( X ## 32) &a) {
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

		AML_FUNCTION void operator/=(const AML_TYPE_NAME(Complex) &a) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *sqrt() {
			double d1;
			double d2;
			d2 = ::sqrt((-r.c[0] + ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0])) / (2));
			if (d2 == 0)
				UNLIKELY{
						d1 = ::sqrt(r.c[0]);
				}
			else
				LIKELY{
						d1 = i.c[0] / (2 * d2);
				}
			r.c[0] = d1;
			i.c[0] = d2;
			d2 = ::sqrt((-r.c[1] + ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1])) / (2));
			if (d2 == 0)
				UNLIKELY{
						d1 = ::sqrt(r.c[1]);
				}
			else
				LIKELY{
						d1 = i.c[1] / (2 * d2);
				}
			r.c[1] = d1;
			i.c[1] = d2;
			d2 = ::sqrt((-r.c[2] + ::sqrt(r.c[2] * r.c[2] + i.c[2] * i.c[2])) / (2));
			if (d2 == 0)
				UNLIKELY{
						d1 = ::sqrt(r.c[2]);
				}
			else
				LIKELY{
						d1 = i.c[2] / (2 * d2);
				}
			r.c[2] = d1;
			i.c[2] = d2;
			d2 = ::sqrt((-r.c[3] + ::sqrt(r.c[3] * r.c[3] + i.c[3] * i.c[3])) / (2));
			if (d2 == 0)
				UNLIKELY{
						d1 = ::sqrt(r.c[3]);
				}
			else
				LIKELY{
						d1 = i.c[3] / (2 * d2);
				}
			r.c[3] = d1;
			i.c[3] = d2;
			d2 = ::sqrt((-r.c[4] + ::sqrt(r.c[4] * r.c[4] + i.c[4] * i.c[4])) / (2));
			if (d2 == 0)
				UNLIKELY{
						d1 = ::sqrt(r.c[4]);
				}
			else
				LIKELY{
						d1 = i.c[4] / (2 * d2);
				}
			r.c[4] = d1;
			i.c[4] = d2;
			d2 = ::sqrt((-r.c[5] + ::sqrt(r.c[5] * r.c[5] + i.c[5] * i.c[5])) / (2));
			if (d2 == 0)
				UNLIKELY{
						d1 = ::sqrt(r.c[5]);
				}
			else
				LIKELY{
						d1 = i.c[5] / (2 * d2);
				}
			r.c[5] = d1;
			i.c[5] = d2;
			d2 = ::sqrt((-r.c[6] + ::sqrt(r.c[6] * r.c[6] + i.c[6] * i.c[6])) / (2));
			if (d2 == 0)
				UNLIKELY{
						d1 = ::sqrt(r.c[6]);
				}
			else
				LIKELY{
						d1 = i.c[6] / (2 * d2);
				}
			r.c[6] = d1;
			i.c[6] = d2;
			d2 = ::sqrt((-r.c[7] + ::sqrt(r.c[7] * r.c[7] + i.c[7] * i.c[7])) / (2));
			if (d2 == 0)
				UNLIKELY{
						d1 = ::sqrt(r.c[7]);
				}
			else
				LIKELY{
						d1 = i.c[7] / (2 * d2);
				}
			r.c[7] = d1;
			i.c[7] = d2;
			return this;
		}

		AML_FUNCTION AML_TYPE_NAME(Complex) *sqrt(const AML_PREFIX(VectorU8_8D) mask) {
			double d1;
			double d2;
			if (mask.v.c[0]) {
				d2 = ::sqrt((-r.c[0] + ::sqrt(r.c[0] * r.c[0] + i.c[0] * i.c[0])) / (2));
				if (d2 == 0)
					UNLIKELY{
							d1 = ::sqrt(r.c[0]);
					}
				else
					LIKELY{
							d1 = i.c[0] / (2 * d2);
					}
				r.c[0] = d1;
				i.c[0] = d2;
			}
			if (mask.v.c[1]) {
				d2 = ::sqrt((-r.c[1] + ::sqrt(r.c[1] * r.c[1] + i.c[1] * i.c[1])) / (2));
				if (d2 == 0)
					UNLIKELY{
							d1 = ::sqrt(r.c[1]);
					}
				else
					LIKELY{
							d1 = i.c[1] / (2 * d2);
					}
				r.c[1] = d1;
				i.c[1] = d2;
			}
			if (mask.v.c[2]) {
				d2 = ::sqrt((-r.c[2] + ::sqrt(r.c[2] * r.c[2] + i.c[2] * i.c[2])) / (2));
				if (d2 == 0)
					UNLIKELY{
							d1 = ::sqrt(r.c[2]);
					}
				else
					LIKELY{
							d1 = i.c[2] / (2 * d2);
					}
				r.c[2] = d1;
				i.c[2] = d2;
			}
			if (mask.v.c[3]) {
				d2 = ::sqrt((-r.c[3] + ::sqrt(r.c[3] * r.c[3] + i.c[3] * i.c[3])) / (2));
				if (d2 == 0)
					UNLIKELY{
							d1 = ::sqrt(r.c[3]);
					}
				else
					LIKELY{
							d1 = i.c[3] / (2 * d2);
					}
				r.c[3] = d1;
				i.c[3] = d2;
			}
			if (mask.v.c[4]) {
				d2 = ::sqrt((-r.c[4] + ::sqrt(r.c[4] * r.c[4] + i.c[4] * i.c[4])) / (2));
				if (d2 == 0)
					UNLIKELY{
							d1 = ::sqrt(r.c[4]);
					}
				else
					LIKELY{
							d1 = i.c[4] / (2 * d2);
					}
				r.c[4] = d1;
				i.c[4] = d2;
			}
			if (mask.v.c[5]) {
				d2 = ::sqrt((-r.c[5] + ::sqrt(r.c[5] * r.c[5] + i.c[5] * i.c[5])) / (2));
				if (d2 == 0)
					UNLIKELY{
							d1 = ::sqrt(r.c[5]);
					}
				else
					LIKELY{
							d1 = i.c[5] / (2 * d2);
					}
				r.c[5] = d1;
				i.c[5] = d2;
			}
			if (mask.v.c[6]) {
				d2 = ::sqrt((-r.c[6] + ::sqrt(r.c[6] * r.c[6] + i.c[6] * i.c[6])) / (2));
				if (d2 == 0)
					UNLIKELY{
							d1 = ::sqrt(r.c[6]);
					}
				else
					LIKELY{
							d1 = i.c[6] / (2 * d2);
					}
				r.c[6] = d1;
				i.c[6] = d2;
			}
			if (mask.v.c[7]) {
				d2 = ::sqrt((-r.c[7] + ::sqrt(r.c[7] * r.c[7] + i.c[7] * i.c[7])) / (2));
				if (d2 == 0)
					UNLIKELY{
							d1 = ::sqrt(r.c[7]);
					}
				else
					LIKELY{
							d1 = i.c[7] / (2 * d2);
					}
				r.c[7] = d1;
				i.c[7] = d2;
			}
			return this;
		}

		AML_FUNCTION AML_TYPE_NAME(Complex) *sin() {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *cos() {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *tan() {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *sin(const AML_PREFIX(VectorU8_8D) mask) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *cos(const AML_PREFIX(VectorU8_8D) mask) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *tan(const AML_PREFIX(VectorU8_8D) mask) {
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


		AML_FUNCTION AML_TYPE_NAME(Complex) *exp() {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *exp(double n) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *pow(const AML_TYPE_NAME(Complex) &n) {
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


		AML_FUNCTION AML_TYPE_NAME(Complex) *pow(const AML_PREFIX( X ## 32) n) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *
		pow(const AML_TYPE_NAME(Complex) &n, const AML_PREFIX(VectorU8_8D) &mask) {
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


		AML_FUNCTION AML_TYPE_NAME(Complex) *pow(const AML_PREFIX( X ## 32) n, const AML_PREFIX(VectorU8_8D) &mask) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *exp(const AML_PREFIX(VectorU8_8D) &mask) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *exp(double n, const AML_PREFIX(VectorU8_8D) &mask) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *pow(double n) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *pow(double n, const AML_PREFIX(VectorU8_8D) &mask) {
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

		AML_FUNCTION AML_PREFIX(Vector8D_64) abs() {
			AML_PREFIX(Vector8D_64)
			ret;
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
			AML_PREFIX(VectorU8_8D)
			ret;
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
			AML_PREFIX(VectorU8_8D)
			ret;
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
			AML_PREFIX(VectorU8_8D)
			ret;
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *ln(const AML_PREFIX(VectorU8_8D) &mask) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *log(const AML_PREFIX(VectorU8_8D) &mask) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *log10(const AML_PREFIX(VectorU8_8D) &mask) {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *ln() {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *log() {
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

		AML_FUNCTION AML_TYPE_NAME(Complex) *log10() {
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

#if !defined(USE_OPENCL) && !defined(USE_CUDA)

		class AML_PREFIX(Complex64_8_Itr) :
		public std::iterator<
		std::input_iterator_tag,   // iterator_category
		AML_PREFIX(Complex64Ptr),                      // value_type
		long,                      // difference_type
		const AML_PREFIX(Complex64Ptr) *,               // pointer
		AML_PREFIX(Complex64Ptr)                       // reference
		> {

			AML_TYPE_NAME(Complex) * a;
			int position;

			public:
			AML_FUNCTION
			explicit AML_PREFIX(Complex64_8_Itr)(AML_TYPE_NAME(Complex) * array, int
			length) : a(array),
					position(
							length)
			{

			}

			AML_FUNCTION
					AML_PREFIX(Complex64_8_Itr)
			&operator++()
			{
				position++;
				return *this;
			}

			AML_FUNCTION bool operator==(const AML_PREFIX(Complex64_8_Itr) other) const {
				return position == other.position;
			}

			AML_FUNCTION bool operator!=(const AML_PREFIX(Complex64_8_Itr) other) const { return !(*this == other); }

			AML_FUNCTION
			reference operator*() const {
				return AML_PREFIX(Complex64Ptr)(&a->r.c[position], &a->i.c[position], position);
			}


		};

		AML_FUNCTION AML_PREFIX(Complex64_8_Itr) begin() {
			return AML_PREFIX(Complex64_8_Itr)(this, 0);
		}

		AML_FUNCTION AML_PREFIX(Complex64_8_Itr) end() {
			return AML_PREFIX(Complex64_8_Itr)(this, 8);
		}

#endif
};


#if !defined(AML_NO_STRING)

