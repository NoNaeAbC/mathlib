//
// Created by af on 26.03.21.
//


class AML_PREFIX(AML_TYPE_NAME(Complex)) {
public:
	AML_PREFIX(doublevec2) c{};

	AML_FUNCTION constexpr AML_PREFIX(AML_TYPE_NAME(Complex))(const AML_TYPE real, const AML_TYPE img = 0.0) {
		c.c[0] = real;
		c.c[1] = img;
	}

	AML_FUNCTION explicit AML_PREFIX(AML_TYPE_NAME(Complex))(AML_TYPE *values) {
		c.c[0] = values[0];
		c.c[1] = values[1];
	}

#if defined(AML_USE_STD_COMPLEX)

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex))(std::complex<AML_TYPE> sc) {
		c.c[0] = sc.real();
		c.c[1] = sc.imag();
	}

#endif

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex))() = default;

	AML_FUNCTION void set([[maybe_unused]]uint64_t location, AML_PREFIX(AML_TYPE_NAME(Complex)) value) {
		c.c[0] = value.c.c[0];
		c.c[1] = value.c.c[1];
	}

//add sub
	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *add(const AML_PREFIX(AML_TYPE_NAME(Complex)) a) {
		c.c[0] += a.c.c[0];
		c.c[1] += a.c.c[1];
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *
	add(const AML_PREFIX(AML_TYPE_NAME(Complex)) a, AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			c.c[1] += a.c.c[1];
			c.c[0] += a.c.c[0];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) operator+(const AML_PREFIX(AML_TYPE_NAME(Complex)) a) const {
		AML_PREFIX(AML_TYPE_NAME(Complex)) ret(c.c[0] + a.c.c[0], c.c[1] + a.c.c[1]);
		return ret;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) operator-(const AML_PREFIX(AML_TYPE_NAME(Complex)) a) const {
		AML_PREFIX(AML_TYPE_NAME(Complex)) ret(c.c[0] - a.c.c[0], c.c[1] - a.c.c[1]);
		return ret;
	}


	AML_FUNCTION void operator+=(const AML_PREFIX(AML_TYPE_NAME(Complex)) a) {
		c.c[0] += a.c.c[0];
		c.c[1] += a.c.c[1];
	}

	AML_FUNCTION void operator-=(const AML_PREFIX(AML_TYPE_NAME(Complex)) a) {
		c.c[0] -= a.c.c[0];
		c.c[1] -= a.c.c[1];
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *subtract(const AML_PREFIX(AML_TYPE_NAME(Complex)) a) {
		c.c[0] -= a.c.c[0];
		c.c[1] -= a.c.c[1];
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *
	subtract(const AML_PREFIX(AML_TYPE_NAME(Complex)) a, AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			c.c[0] -= a.c.c[0];
			c.c[1] -= a.c.c[1];
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *conjugate() {
		c.c[1] = -c.c[1];
		return this;
	}

//mul
	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *multiply(const AML_PREFIX(AML_TYPE_NAME(Complex)) a) {
		AML_TYPE d1 = c.c[0] * a.c.c[0] - c.c[1] * a.c.c[1];
		AML_TYPE d2 = c.c[0] * a.c.c[1];// + c.c[1] * a.c.c[0];
		d2 += d2;
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}


	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *
	multiply(const AML_PREFIX(AML_TYPE_NAME(Complex)) &a, const AML_PREFIX(VectorU8_1D) &mask) {
		AML_TYPE d1;
		AML_TYPE d2;
		if (mask.v.c) LIKELY {
			d1 = c.c[0] * a.c.c[0] - c.c[1] * a.c.c[1];
			d2 = c.c[0] * a.c.c[1] + c.c[1] * a.c.c[0];
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;


	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) operator*(const AML_PREFIX(AML_TYPE_NAME(Complex)) a) const {
		AML_PREFIX(AML_TYPE_NAME(Complex)) ret(c.c[0] * a.c.c[0] - c.c[1] * a.c.c[1],
											   c.c[0] * a.c.c[1] + c.c[1] * a.c.c[0]);
		return ret;
	}

	AML_FUNCTION void operator*=(const AML_PREFIX(AML_TYPE_NAME(Complex)) a) {
		AML_TYPE d1 = c.c[0] * a.c.c[0] - c.c[1] * a.c.c[1];
		AML_TYPE d2 = c.c[0] * a.c.c[1] + c.c[1] * a.c.c[0];
		c.c[0] = d1;
		c.c[1] = d2;
	}

	AML_FUNCTION void operator*=(AML_TYPE a) {
		c.c[0] = c.c[0] * a;
		c.c[1] = c.c[1] * a;
	}


	constexpr AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *square() {
		AML_TYPE d1 = c.c[0] * c.c[0] - c.c[1] * c.c[1];
		AML_TYPE d2 = c.c[0] * c.c[1] + c.c[1] * c.c[0];
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *square(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			AML_TYPE d1 = c.c[0] * c.c[0] - c.c[1] * c.c[1];
			AML_TYPE d2 = c.c[0] * c.c[1] + c.c[1] * c.c[0];
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

//division
	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) operator/(const AML_PREFIX(AML_TYPE_NAME(Complex)) a) const {
		AML_PREFIX(AML_TYPE_NAME(Complex)) ret;
		ret.c.c[0] = (c.c[0] * a.c.c[0] + c.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		ret.c.c[1] = (c.c[1] * a.c.c[0] - c.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		return ret;
	}

	AML_FUNCTION void operator/=(const AML_PREFIX(AML_TYPE_NAME(Complex)) a) {
		AML_TYPE d1 = (c.c[0] * a.c.c[0] + c.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		AML_TYPE d2 = (c.c[1] * a.c.c[0] - c.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		c.c[0] = d1;
		c.c[1] = d2;
	}

	AML_FUNCTION void operator/=(AML_TYPE a) {
		AML_TYPE d1 = c.c[0] / a;
		AML_TYPE d2 = c.c[1] / a;
		c.c[0] = d1;
		c.c[1] = d2;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *divide(const AML_PREFIX(AML_TYPE_NAME(Complex)) a) {
		AML_TYPE d1 = (c.c[0] * a.c.c[0] + c.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		AML_TYPE d2 = (c.c[1] * a.c.c[0] - c.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) operator/(AML_TYPE a) {
		AML_PREFIX(AML_TYPE_NAME(Complex)) ret;
		ret.c.c[0] = c.c[0] / a;
		ret.c.c[1] = c.c[1] / a;
		return ret;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *divide(AML_TYPE a) {
		AML_TYPE d1 = c.c[0] / a;
		AML_TYPE d2 = c.c[1] / a;
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *divide(const AML_TYPE a, const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			AML_TYPE d1 = c.c[0] / a;
			AML_TYPE d2 = c.c[1] / a;

			c.c[0] = d1;
			c.c[1] = d2;

		}

		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *
	divide(const AML_PREFIX(AML_TYPE_NAME(Complex)) a, const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			AML_TYPE d1 = (c.c[0] * a.c.c[0] + c.c[1] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);
			AML_TYPE d2 = (c.c[1] * a.c.c[0] - c.c[0] * a.c.c[1]) / (a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1]);

			c.c[0] = d1;
			c.c[1] = d2;

		}

		return this;
	}

//sqrt
	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *sqrt() {
		AML_TYPE d2 = ::sqrt((-c.c[0] + ::sqrt(c.c[0] * c.c[0] + c.c[1] * c.c[1])) / (2));
		AML_TYPE d1;
		if (d2 == 0) UNLIKELY {
			d1 = ::sqrt(c.c[0]);
		} else LIKELY {
			d1 = c.c[1] / (2 * d2);
		}
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *sqrt(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			AML_TYPE d2 = ::sqrt((-c.c[0] + ::sqrt(c.c[0] * c.c[0] + c.c[1] * c.c[1])) / (2));
			AML_TYPE d1;
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

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *sin() {
		AML_TYPE d1 = ::sin(c.c[0]) * ::cosh(c.c[1]);
		AML_TYPE d2 = ::cos(c.c[1]) * ::sinh(c.c[0]);

		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *cos() {
		AML_TYPE d1 = ::cos(c.c[0]) * ::cosh(c.c[1]);
		AML_TYPE d2 = -::sin(c.c[1]) * ::sinh(c.c[0]);

		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *tan() {
		AML_TYPE d1 = ::sin(c.c[0] + c.c[0]) / (::cos(c.c[0] + c.c[0]) * ::cosh(c.c[1] + c.c[1]));
		AML_TYPE d2 = ::sinh(c.c[1] + c.c[1]) / (::cos(c.c[0] + c.c[0]) * ::cosh(c.c[1] + c.c[1]));

		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *sin(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) {
			AML_TYPE d1 = ::sin(c.c[0]) * ::cosh(c.c[1]);
			AML_TYPE d2 = ::cos(c.c[1]) * ::sinh(c.c[0]);
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *cos(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) {
			AML_TYPE d1 = ::cos(c.c[0]) * ::cosh(c.c[1]);
			AML_TYPE d2 = -::sin(c.c[1]) * ::sinh(c.c[0]);
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *tan(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) {
			AML_TYPE d1 = ::sin(c.c[0] + c.c[0]) / (::cos(c.c[0] + c.c[0]) * ::cosh(c.c[1] + c.c[1]));
			AML_TYPE d2 = ::sinh(c.c[1] + c.c[1]) / (::cos(c.c[0] + c.c[0]) * ::cosh(c.c[1] + c.c[1]));
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}


	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *exp() {
		AML_TYPE d1 = ::exp(c.c[0]) * ::cos(c.c[1]);
		AML_TYPE d2 = ::exp(c.c[0]) * ::sin(c.c[1]);


		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *exp(AML_TYPE n) {
		AML_TYPE d1 = ::pow(n, c.c[0]) * ::cos(c.c[1] * ::log(n));
		AML_TYPE d2 = ::pow(n, c.c[0]) * ::sin(c.c[1] * ::log(n));
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *pow(AML_TYPE n) {
		AML_TYPE d1 = ::pow(c.c[0] * c.c[0] + c.c[1] * c.c[1], n / 2) * ::cos(n * atan2(c.c[1], c.c[0]));
		AML_TYPE d2 = ::pow(c.c[0] * c.c[0] + c.c[1] * c.c[1], n / 2) * ::sin(n * atan2(c.c[1], c.c[0]));
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *pow(const AML_PREFIX(AML_TYPE_NAME(Complex)) n) {
		AML_TYPE d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
		AML_TYPE d2 = ::atan2(c.c[1], c.c[0]);
		AML_TYPE d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
		AML_TYPE d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
		AML_TYPE d5 = d3 * ::cos(d4);
		AML_TYPE d6 = d3 * ::sin(d4);
		c.c[0] = d5;
		c.c[1] = d6;
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *pow(AML_TYPE n, const AML_PREFIX(VectorU8_1D) &mask) {
		if (mask.v.c) LIKELY {
			AML_TYPE d1 = ::pow(c.c[0] * c.c[0] + c.c[1] * c.c[1], n / 2) * ::cos(n * atan2(c.c[1], c.c[0]));
			AML_TYPE d2 = ::pow(c.c[0] * c.c[0] + c.c[1] * c.c[1], n / 2) * ::sin(n * atan2(c.c[1], c.c[0]));
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *
	pow(const AML_PREFIX(AML_TYPE_NAME(Complex)) n, const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			AML_TYPE d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
			AML_TYPE d2 = ::atan2(c.c[1], c.c[0]);
			AML_TYPE d3 = ::exp(d1 * n.c.c[0] - d2 * n.c.c[1]);
			AML_TYPE d4 = d1 * n.c.c[1] + d2 * n.c.c[0];
			AML_TYPE d5 = d3 * ::cos(d4);
			AML_TYPE d6 = d3 * ::sin(d4);
			c.c[0] = d5;
			c.c[1] = d6;
		}
		return this;
	}

	AML_FUNCTION AML_TYPE abs() {
		return ::sqrt(c.c[0] * c.c[0] + c.c[1] * c.c[1]);
	}

	AML_FUNCTION bool abs_gt(AML_TYPE a) {
		return a * a < c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}

	AML_FUNCTION bool abs_lt(AML_TYPE a) {
		return a * a > c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}

	AML_FUNCTION bool abs_eq(AML_TYPE a) {
		return a * a == c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}

	AML_FUNCTION bool abs_gt(const AML_PREFIX(AML_TYPE_NAME(Complex)) a) {
		return a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1] < c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}

	AML_FUNCTION bool abs_lt(const AML_PREFIX(AML_TYPE_NAME(Complex)) a) {
		return a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1] > c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}

	AML_FUNCTION bool abs_eq(const AML_PREFIX(AML_TYPE_NAME(Complex)) a) {
		return a.c.c[0] * a.c.c[0] + a.c.c[1] * a.c.c[1] == c.c[0] * c.c[0] + c.c[1] * c.c[1];
	}


	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *ln() {
		AML_TYPE d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
		AML_TYPE d2 = ::atan2(c.c[1], c.c[0]);
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *log() {
		AML_TYPE d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
		AML_TYPE d2 = ::atan2(c.c[1], c.c[0]);
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *log10() {
		AML_TYPE d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / (2 * AML_LN10);
		AML_TYPE d2 = ::atan2(c.c[1], c.c[0]) / AML_LN10;
		c.c[0] = d1;
		c.c[1] = d2;
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *ln(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			AML_TYPE d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
			AML_TYPE d2 = ::atan2(c.c[1], c.c[0]);
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *log(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			AML_TYPE d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / 2;
			AML_TYPE d2 = ::atan2(c.c[1], c.c[0]);
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *log10(const AML_PREFIX(VectorU8_1D) mask) {
		if (mask.v.c) LIKELY {
			AML_TYPE d1 = ::log(c.c[0] * c.c[0] + c.c[1] * c.c[1]) / (2 * AML_LN10);
			AML_TYPE d2 = ::atan2(c.c[1], c.c[0]) / AML_LN10;
			c.c[0] = d1;
			c.c[1] = d2;
		}
		return this;
	}


	AML_FUNCTION AML_TYPE imaginary() {
		return c.c[1];
	}

	AML_FUNCTION AML_TYPE real() {
		return c.c[0];
	}

	AML_FUNCTION AML_TYPE angle() {
		return ::atan2(c.c[1], c.c[0]);
	}

	AML_FUNCTION AML_TYPE length() {
		return ::sqrt(c.c[0] * c.c[0] + c.c[1] * c.c[1]);
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) *polar(AML_TYPE length, AML_TYPE angle) {
		c.c[0] = length * ::cos(angle);
		c.c[1] = length * ::sin(angle);
		return this;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) operator[]([[maybe_unused]]uint64_t location) {
		return *this;
	}


};

class AML_PREFIX(AML_POINTER_NAME(Complex)) : public AML_PREFIX(AML_TYPE_NAME(Complex)) {
	AML_TYPE *r;
	AML_TYPE *i;
	uint32_t index = 0;

	AML_FUNCTION void update() {
		*r = c.c[0];
		*i = c.c[1];
	}

public:
	AML_FUNCTION AML_PREFIX(AML_POINTER_NAME(Complex))(AML_TYPE *real, AML_TYPE *imag) : AML_PREFIX(AML_TYPE_NAME(
			Complex))(*real, *imag) {
		r = real;
		i = imag;
	}

	AML_FUNCTION AML_PREFIX(AML_POINTER_NAME(Complex))(AML_TYPE *real, AML_TYPE *imag, uint32_t position) : AML_PREFIX(
																													AML_TYPE_NAME(
																															Complex))(
			*real,
			*imag) {
		r = real;
		i = imag;
		index = position;
	}

	AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) &operator*() {
		return *this;
	}

	AML_FUNCTION ~AML_PREFIX(AML_POINTER_NAME(Complex))() {
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

	AML_FUNCTION void operator=(const AML_PREFIX(AML_TYPE_NAME(Complex)) newVal) {
		c.c[0] = newVal.c.c[0];
		c.c[1] = newVal.c.c[1];
		update();
	}

};

#if !defined(AML_NO_STRING)

AML_FUNCTION std::string operator<<(std::string &lhs, const AML_PREFIX(AML_TYPE_NAME(Complex)) &rhs) {
	std::ostringstream string;
	string << lhs << rhs.c.c[0] << " + " << rhs.c.c[1] << "i";
	return string.str();
}

AML_FUNCTION std::string operator<<(const char *lhs, const AML_PREFIX(AML_TYPE_NAME(Complex)) &rhs) {
	std::ostringstream string;
	string << lhs << rhs.c.c[0] << " + " << rhs.c.c[1] << "i";
	return string.str();
}

template<class charT, class traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &o, const AML_PREFIX(AML_TYPE_NAME(Complex)) &x) {
	std::basic_ostringstream<charT, traits> s;
	s.flags(o.flags());
	s.imbue(o.getloc());
	s.precision(o.precision());
	s << x.c.c[0] << " + " << x.c.c[1] << "i";
	return o << s.str();
}

#endif

AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex))
operator+(const AML_TYPE &lhs, const AML_PREFIX(AML_TYPE_NAME(Complex)) &rhs) {
	AML_PREFIX(AML_TYPE_NAME(Complex)) ret(lhs + rhs.c.c[0], 0.0 + rhs.c.c[1]);
	return ret;
}

AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex))
operator-(const AML_TYPE &lhs, const AML_PREFIX(AML_TYPE_NAME(Complex)) &rhs) {
	AML_PREFIX(AML_TYPE_NAME(Complex)) ret(lhs - rhs.c.c[0], 0.0 - rhs.c.c[1]);
	return ret;
}

AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex))
operator*(const AML_TYPE &lhs, const AML_PREFIX(AML_TYPE_NAME(Complex)) &rhs) {
	AML_PREFIX(AML_TYPE_NAME(Complex)) ret(lhs * rhs.c.c[0], lhs * rhs.c.c[1]);
	return ret;
}

AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex))
operator/(const AML_TYPE &lhs, const AML_PREFIX(AML_TYPE_NAME(Complex)) &rhs) {
	AML_PREFIX(AML_TYPE_NAME(Complex)) ret;
	ret.c.c[0] = (lhs * rhs.c.c[0]) / (rhs.c.c[0] * rhs.c.c[0] + rhs.c.c[1] * rhs.c.c[1]);
	ret.c.c[1] = (-lhs * rhs.c.c[1]) / (rhs.c.c[0] * rhs.c.c[0] + rhs.c.c[1] * rhs.c.c[1]);
	return ret;
}

#if defined(AML_USE_STD_COMPLEX)

AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex))
operator+(const std::complex<AML_TYPE> &lhs, const AML_PREFIX(AML_TYPE_NAME(Complex)) &rhs) {
	AML_PREFIX(AML_TYPE_NAME(Complex)) ret = lhs;
	return ret + rhs;
}

AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex))
operator-(const std::complex<AML_TYPE> &lhs, const AML_PREFIX(AML_TYPE_NAME(Complex)) &rhs) {
	AML_PREFIX(AML_TYPE_NAME(Complex)) ret = lhs;
	return ret - rhs;
}

AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex))
operator*(const std::complex<AML_TYPE> &lhs, const AML_PREFIX(AML_TYPE_NAME(Complex)) &rhs) {
	AML_PREFIX(AML_TYPE_NAME(Complex)) ret = lhs;
	return ret * rhs;
}

AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex))
operator/(const std::complex<AML_TYPE> &lhs, const AML_PREFIX(AML_TYPE_NAME(Complex)) &rhs) {
	AML_PREFIX(AML_TYPE_NAME(Complex)) ret = lhs;
	return ret / rhs;
}

#if AML_TYPE_ID == 1

class AML_PREFIX(STD_COMPLEX64_CAST) : public std::complex<AML_TYPE> {
public:
	AML_FUNCTION AML_PREFIX(STD_COMPLEX64_CAST)(const AML_PREFIX(AML_TYPE_NAME(Complex)) &other) : std::complex<AML_TYPE>(other.c.c[0],
																										   other.c.c[1]) {}
};

#elif AML_TYPE_ID == 2

class AML_PREFIX(STD_COMPLEX32_CAST) : public std::complex<AML_TYPE> {
public:
	AML_FUNCTION AML_PREFIX(STD_COMPLEX32_CAST)(const AML_PREFIX(AML_TYPE_NAME(Complex)) &other)
			: std::complex<AML_TYPE>(other.c.c[0],
									 other.c.c[1]) {}
};

#endif

#endif

#if !defined(USE_CUDA)

#if AML_TYPE_ID == 1

constexpr AML_PREFIX(AML_TYPE_NAME(Complex)) operator ""_i(long double d) {
	return AML_PREFIX(AML_TYPE_NAME(Complex))(0.0f, (AML_TYPE) d);
}

constexpr AML_PREFIX(AML_TYPE_NAME(Complex)) operator ""_i(unsigned long long d) {
	return AML_PREFIX(AML_TYPE_NAME(Complex))(0.0f, (AML_TYPE) d);
}

#elif AML_TYPE_ID == 2

constexpr AML_PREFIX(AML_TYPE_NAME(Complex)) operator ""_if(long double d) {
	return AML_PREFIX(AML_TYPE_NAME(Complex))(0.0f, (AML_TYPE) d);
}

constexpr AML_PREFIX(AML_TYPE_NAME(Complex)) operator ""_if(unsigned long long d) {
	return AML_PREFIX(AML_TYPE_NAME(Complex))(0.0f, (AML_TYPE) d);
}


#endif

#endif

#if defined(AML_USE_STD_COMPLEX)

AML_FUNCTION std::complex<AML_TYPE> toStdComplex(const AML_PREFIX(AML_TYPE_NAME(Complex)) a) {
	std::complex<AML_TYPE> ret(a.c.c[0], a.c.c[1]);
	return ret;
}

#endif

#define FUNCTION(S) AML_FUNCTION AML_PREFIX(AML_TYPE_NAME(Complex)) S(AML_PREFIX(AML_TYPE_NAME(Complex)) a){\
    AML_PREFIX(AML_TYPE_NAME(Complex)) b = a;\
    b.S();\
    return b;\
}

FUNCTION(square)

FUNCTION(sqrt)

FUNCTION(log)

FUNCTION(exp)

FUNCTION(log10)

