//
// Created by af on 07.03.21.
//


#if defined( AML_USE_GMP) && !defined(USE_CUDA)

#include <mpfr.h>

namespace AML_PREFIX(AML) {

	static thread_local unsigned int defaultPrecision = 0;

	inline unsigned int
	sySTeMDEfAulTutILs() { // hopefully it sounds boring enough that it wont be used. everything in Spongebob case is private
		return defaultPrecision;
	}
}

class AmlDefaultPrecision {
	unsigned int cache;
public:
	inline AmlDefaultPrecision(unsigned int defaultPrecision) {
		cache = AML::defaultPrecision;
		AML::defaultPrecision = defaultPrecision;
	}

	inline ~AmlDefaultPrecision() {
		AML::defaultPrecision = cache;
	}


};


// https://stackoverflow.com/questions/1082192/how-to-generate-random-variable-names-in-c-using-macros
#define AML__PP_CAT(a, b) AML__PP_CAT_I(a, b)
#define AML__PP_CAT_I(a, b) AML__PP_CAT_II(~, a ## b)
#define AML__PP_CAT_II(p, res) res

#define AML__UNIQUE_NAME(base) AML__PP_CAT(base, __LINE__)

#define AML_SET_DEFAULT_PRECISION AmlDefaultPrecision AML__UNIQUE_NAME(amldefaultpre)  =


#define AML_DEFAULT_PRECISION (AML::sySTeMDEfAulTutILs())

class AmlNumber {
	mpfr_t number;
public:

	inline explicit AmlNumber(int precision = 512) {
		if (AML::defaultPrecision) precision = AML::defaultPrecision;
		mpfr_init2(number, precision);
	}

	inline AmlNumber(const long value, int precision = 512) {
		if (AML::defaultPrecision) precision = AML::defaultPrecision;
		mpfr_init2(number, precision);

		mpfr_set_si(number, value, MPFR_RNDZ);
	}

	inline AmlNumber(const unsigned long value, int precision = 512) {
		if (AML::defaultPrecision) precision = AML::defaultPrecision;
		mpfr_init2(number, precision);

		mpfr_set_ui(number, value, MPFR_RNDZ);
	}

	inline AmlNumber(const double value, int precision = 512) {
		if (AML::defaultPrecision) precision = AML::defaultPrecision;
		mpfr_init2(number, precision);
		mpfr_set_d(number, value, MPFR_RNDZ);
	}

	inline AmlNumber(const AmlNumber &value, int precision = 512) {
		if (AML::defaultPrecision) precision = AML::defaultPrecision;
		mpfr_init2(number, precision);

		mpfr_set(number, value.number, MPFR_RNDZ);
	}

	inline AmlNumber(const char *value, int base = 10, int precision = 512) {
		if (AML::defaultPrecision) precision = AML::defaultPrecision;
		mpfr_init2(number, precision);

		mpfr_set_str(number, value, base, MPFR_RNDZ);
	}

	inline static void swap(AmlNumber &a, AmlNumber &b) {
		mpfr_swap(a.number, b.number);
	}

	inline ~AmlNumber() {
		mpfr_clear(number);
	}

	[[nodiscard]] inline int getPrecision() const {
		return mpfr_get_prec(number);
	}

	inline operator double() const {
		return mpfr_get_d(number, MPFR_RNDZ);
	}

	inline operator long() const {
		return mpfr_get_si(number, MPFR_RNDZ);
	}

	inline operator unsigned long() const {
		return mpfr_get_ui(number, MPFR_RNDZ);
	}

	inline std::string toString() {
		const char *string = "hi";// mpf_get_str(nullptr, &exponent, 10, 0, number);
		return std::string(string);
	}

	inline void operator=(const AmlNumber rhs) {
		mpfr_set(number, rhs.number, MPFR_RNDZ);
	}

	inline AmlNumber operator+(const AmlNumber rhs) const {
		AmlNumber ret(mpfr_get_prec(number));
		mpfr_add(ret.number, number, rhs.number, MPFR_RNDZ);
		return ret;
	}

	inline void operator+=(const AmlNumber rhs) {
		mpfr_add(number, number, rhs.number, MPFR_RNDZ);
	}

	inline AmlNumber operator-(const AmlNumber rhs) const {
		AmlNumber ret(mpfr_get_prec(number));
		mpfr_sub(ret.number, number, rhs.number, MPFR_RNDZ);
		return ret;
	}

	inline void operator-=(const AmlNumber rhs) {
		mpfr_sub(number, number, rhs.number, MPFR_RNDZ);
	}

	inline AmlNumber operator*(const AmlNumber rhs) const {
		AmlNumber ret(mpfr_get_prec(number));
		mpfr_mul(ret.number, number, rhs.number, MPFR_RNDZ);
		return ret;
	}

	inline void operator*=(const AmlNumber rhs) {
		mpfr_mul(number, number, rhs.number, MPFR_RNDZ);
	}

	inline AmlNumber operator/(const AmlNumber rhs) const {
		AmlNumber ret(mpfr_get_prec(number));
		mpfr_div(ret.number, number, rhs.number, MPFR_RNDZ);
		return ret;
	}

	inline void operator/=(const AmlNumber rhs) {
		mpfr_div(number, number, rhs.number, MPFR_RNDZ);
	}

	inline AmlNumber *sqrt() {
		mpfr_sqrt(number, number, MPFR_RNDZ);
		return this;
	}

	inline AmlNumber *pow(unsigned long exponent) {
		mpfr_pow_ui(number, number, exponent, MPFR_RNDZ);
		return this;
	}

	inline AmlNumber *neg() {
		mpfr_neg(number, number, MPFR_RNDZ);
		return this;
	}

	inline AmlNumber *abs() {
		mpfr_abs(number, number, MPFR_RNDZ);
		return this;
	}

	inline AmlNumber *ln() {
		mpfr_log(number, number, MPFR_RNDZ);
		return this;
	}

	inline AmlNumber *log() {
		mpfr_log(number, number, MPFR_RNDZ);
		return this;
	}

	inline AmlNumber *log10() {
		mpfr_log10(number, number, MPFR_RNDZ);
		return this;
	}

	inline AmlNumber *log2() {
		mpfr_log2(number, number, MPFR_RNDZ);
		return this;
	}

	inline AmlNumber *exp() {
		mpfr_exp(number, number, MPFR_RNDZ);
		return this;
	}

	inline AmlNumber *exp10() {
		mpfr_exp10(number, number, MPFR_RNDZ);
		return this;
	}

	inline AmlNumber *exp2() {
		mpfr_exp2(number, number, MPFR_RNDZ);
		return this;
	}

	inline AmlNumber *pow(AmlNumber other) {
		mpfr_pow(number, number, other.number, MPFR_RNDZ);
		return this;
	}

	inline AmlNumber *cos() {
		mpfr_cos(number, number, MPFR_RNDZ);
		return this;
	}

	inline AmlNumber *sin() {
		mpfr_sin(number, number, MPFR_RNDZ);
		return this;
	}

	inline AmlNumber *tan() {
		mpfr_tan(number, number, MPFR_RNDZ);
		return this;
	}

	inline std::strong_ordering operator<=>(const AmlNumber other) const {
		int a = mpfr_cmp(number, other.number);
		if (a > 0) {
			return std::strong_ordering::greater;
		}
		if (a < 0) {
			return std::strong_ordering::less;
		}
		return std::strong_ordering::equal;
	}

	inline bool operator==(const AmlNumber other) const {
		int a = mpfr_cmp(number, other.number);
		return a == 0;
	}

	inline bool operator!=(const AmlNumber other) const {
		int a = mpfr_cmp(number, other.number);
		return a != 0;
	}
};

namespace AML_PREFIX(AML) {

	inline AmlNumber sqrt(const AmlNumber a) {
		AmlNumber x = a;
		x.sqrt();
		return x;
	}

	inline AmlNumber log(const AmlNumber a) {
		AmlNumber x = a;
		x.ln();
		return x;
	}

	inline AmlNumber exp(const AmlNumber a) {
		AmlNumber x = a;
		x.exp();
		return x;
	}

	inline AmlNumber pow(const AmlNumber a, const AmlNumber b) {
		AmlNumber x1 = a;
		AmlNumber x2 = b;
		x1.pow(x2);
		return x1;
	}
}
#define defaultPrecision sySTeMDEfAulTutILs()

#endif //MATHLIB_MULTIPRECISION_H
