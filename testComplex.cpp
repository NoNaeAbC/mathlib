//
// Created by af on 31.01.21.
//

#define AML_USE_STD_COMPLEX

#include "amathlib.h"
#include <iostream>
#include <iomanip>


#define ASSERT_AML(A, B) if(!checkEquality(A,B)){std::cerr<<"Test " << testCaseId << " failed in line " << __LINE__ << std::endl;return -1;}
#define ASSERT_AML_BOOL(A) if(!A){std::cerr<<"Test "<< testCaseId <<" failed in line" << __LINE__ << std::endl;return -1;}

using namespace std::complex_literals;

struct ieedouble {
	unsigned long frac: 52;
	unsigned int exp: 11;
	unsigned int sign: 1;
};

union double_seg {
	double value;
	ieedouble seg;

};

bool checkEquality(Complex64 a, Complex64 b) {
	Complex64 diff = (a - b);
	if (a.c.c[0] != b.c.c[0]) {
		return false;
	}
	if (a.c.c[1] != b.c.c[1]) {
		return false;
	}
	return true;
}


bool checkEquality(const Complex64 a, const std::complex<double> b) {
	Complex64 diff = (a - b);
	if (a.c.c[0] != b.real()) {
		std::cout << diff.abs() << std::endl;
		double_seg real1;
		real1.value = a.c.c[0];
		double_seg real2;
		real2.value = b.real();
		std::cout << real1.seg.frac << "\n" << real2.seg.frac << std::endl;

		std::cout << real1.seg.exp << "\n" << real2.seg.exp << std::endl;
		return false;
	}
	if (a.c.c[1] != b.imag()) {
		std::cout << diff.abs() << std::endl;
		double_seg img1;
		img1.value = a.c.c[0];
		double_seg img2;
		img2.value = b.imag();
		std::cout << img1.seg.frac << "\n" << img2.seg.frac << std::endl;
		std::cout << img1.seg.exp << "\n" << img2.seg.exp << std::endl;
		return false;
	}
	return true;
}

bool checkEquality(Complex32 a, Complex32 b) {
	if (a.c.c[0] != b.c.c[0]) {
		return false;
	}
	if (a.c.c[1] != b.c.c[1]) {
		return false;
	}
	return true;
}


bool checkEquality(Complex32 a, std::complex<float> b) {
	if (a.c.c[0] != b.real()) {
		return false;
	}
	if (a.c.c[1] != b.imag()) {
		return false;
	}
	return true;
}


const char *getTruthValue(bool value) {
	if (value) {
		return "true";
	}
	return "false";
}

int main() {
	uint32_t testCaseId = 1;

	std::complex<double> sc1(6.2, 3.0);
	auto c1 = (Complex64) sc1;

	std::cout << testCaseId << "\t:\t" << c1 << "\t\t\t\t\t:\t" << sc1 << std::endl;//1
	ASSERT_AML(c1, sc1);
	testCaseId++;

	std::complex<double> sc2(4.6, 2.9);
	auto c2 = (Complex64) sc2;

	c2 += c1;
	sc2 += sc1;

	std::cout << testCaseId << "\t:\t" << c2 << "\t\t\t\t\t:\t" << sc2 << std::endl;//2
	ASSERT_AML(c2, sc2);
	testCaseId++;

	std::complex<double> sc3 = 2.0 + 4.7i;
	Complex64 c3 = 2.0 + 4.7_i;

	std::cout << testCaseId << "\t:\t" << c3 << "\t\t\t\t\t:\t" << sc3 << std::endl;//3
	ASSERT_AML(c3, sc3);
	testCaseId++;

	std::complex<double> sc4 = 2.0 + 4i;
	Complex64 c4 = 2 + 4_i;

	std::cout << testCaseId << "\t:\t" << c4 << "\t\t\t\t\t\t:\t" << sc4 << std::endl;//4
	ASSERT_AML(c4, sc4);
	testCaseId++;

	Complex64 c5 = 3.4 - 0.9_i;
	std::complex<double> sc5 = toStdComplex(c5);

	std::cout << testCaseId << "\t:\t" << c5 << "\t\t\t\t\t:\t" << sc5 << std::endl;//5
	ASSERT_AML(c5, sc5);
	testCaseId++;

	std::complex<double> sc6 = 0.2;
	Complex64 c6 = 0.2;

	std::cout << testCaseId << "\t:\t" << c6 << "\t\t\t\t\t:\t" << sc6 << std::endl;//6
	ASSERT_AML(c6, sc6);
	testCaseId++;

	std::complex<double> sc7 = 0.2 + 0.1i;
	Complex64 c7 = 0.2 + 0.1_i;

	std::complex<double> sc8 = -0.7 + 9.3i;
	Complex64 c8 = -0.7 + 9.3_i;

	c7.add(c8);
	sc7 = sc7 + sc8;

	std::cout << testCaseId << "\t:\t" << c7 << "\t\t\t\t\t:\t" << sc7 << std::endl;//7
	ASSERT_AML(c7, sc7);
	testCaseId++;

	std::complex<double> sc9 = std::polar<double>(20, 1);
	Complex64 c9;
	c9.polar(20, 1);

	std::cout << testCaseId << "\t:\t" << c9 << "\t\t\t:\t" << sc9 << std::endl;//8
	ASSERT_AML(c9, sc9);
	testCaseId++;

	Complex64 c10;
	c10.polar(c9.length(), c9.angle());

	std::cout << testCaseId << "\t:\t" << c9 << "\t\t\t:\t" << c10 << std::endl;//9
	ASSERT_AML(c10, c9);
	testCaseId++;

	c10 = Complex64(c9.real(), c9.imaginary());

	std::cout << testCaseId << "\t:\t" << c9 << "\t\t\t:\t" << c10 << std::endl;//10
	ASSERT_AML(c10, c9);
	testCaseId++;

	std::cout << testCaseId << "\t:\t" << c10.abs() << " == |" << c10 << "| \t:\t"
			  << getTruthValue(c10.abs_eq(c10.abs()))
			  << std::endl;//11
	ASSERT_AML_BOOL(c10.abs_eq(c10.abs()));
	testCaseId++;

	std::cout << testCaseId << "\t:\t" << c10.abs() << " == |" << c10 << "| \t:\t" << getTruthValue(c10.abs_eq(c10))
			  << std::endl;//12
	ASSERT_AML_BOOL(c10.abs_eq(c10));
	testCaseId++;

	std::cout << testCaseId << "\t:\t" << c10.abs() << " == |" << (c10 + 1) << "| \t:\t"
			  << getTruthValue(c10.abs_eq(c10 + 1)) << std::endl;//13
	ASSERT_AML_BOOL((!c10.abs_eq(c10 + 1)));
	testCaseId++;

	Complex64 c11 = c10;
	c10 += 1_i;
	std::cout << testCaseId << "\t:\t" << c11.abs() << " == |" << c10 << "| \t:\t"
			  << getTruthValue(c11.abs_eq(c10.abs()))
			  << std::endl;//14
	ASSERT_AML_BOOL((!c11.abs_eq(c10.abs())));
	testCaseId++;

	std::complex<double> sc12 = 150.2 + 213.23i;
	Complex64 c12 = 150.2 + 213.23_i;

	std::complex<double> sc13 = 432.3 + 2.3i;
	Complex64 c13 = 432.3 + 2.3_i;

	c12.subtract(c13);
	sc12 = sc12 - sc13;

	std::cout << testCaseId << "\t:\t" << c12 << "\t\t\t:\t" << sc12 << std::endl;//15
	ASSERT_AML(c12, sc12);
	testCaseId++;


	c12.subtract(c13, false);

	std::cout << testCaseId << "\t:\t" << c12 << "\t\t\t:\t" << sc12 << std::endl;//16
	ASSERT_AML(c12, sc12);
	testCaseId++;

	c12.subtract(c13, true);
	sc12 = sc12 - sc13;

	std::cout << testCaseId << "\t:\t" << c12 << "\t\t\t:\t" << sc12 << std::endl;//17
	ASSERT_AML(c12, sc12);
	testCaseId++;

	std::complex<double> sc14 = 12.4 + 42.4i;
	Complex64 c14 = 12.4 + 42.4_i;

	c14 = 1.0 / c14;
	sc14 = 1.0 / sc14;

	std::cout << testCaseId << "\t:\t" << c14 << "\t:\t" << sc14 << std::endl;//18
	ASSERT_AML(c14, sc14);
	testCaseId++;

	std::complex<double> sc15 = 12.4 + 42.4i;
	std::complex<double> sc16 = 19.6 + 11.3i;
	Complex64 c15 = 12.4 + 42.4_i;

	c15 = sc16 + c15;
	sc15 = sc16 + sc15;

	std::cout << testCaseId << "\t:\t" << c15 << "\t\t\t\t\t:\t" << sc15 << std::endl;//19
	ASSERT_AML(c15, sc15);
	testCaseId++;

	c15 = sc16 - c15;
	sc15 = sc16 - sc15;

	std::cout << testCaseId << "\t:\t" << c15 << "\t\t\t\t:\t" << sc15 << std::endl;//20
	ASSERT_AML(c15, sc15);
	testCaseId++;

	c15 = sc16 * c15;
	sc15 = sc16 * sc15;

	std::cout << testCaseId << "\t:\t" << c15 << "\t\t\t:\t" << sc15 << std::endl;//21
	ASSERT_AML(c15, sc15);
	testCaseId++;

	c15 = sc16 / c15;
	sc15 = sc16 / sc15;

	std::cout << std::setprecision(100) << testCaseId << "\t:\t" << c15 << "\t:\t" << sc15 << std::endl;//22
	ASSERT_AML(c15, sc15);
	testCaseId++;

	Complex64 c17 = 1.0 + 5.0_i;

	std::complex<double> sc17 = (STD_COMPLEX64_CAST) c17;

	std::cout << testCaseId << "\t:\t" << c17 << "\t\t\t\t\t\t:\t" << sc17 << std::endl;//23
	ASSERT_AML(c17, sc17);
	testCaseId++;

	Complex64 c18 = 1.7 + 4.1_i;

	std::complex<double> sc18 = (STD_COMPLEX64_CAST) c18;

	c18 = log(c18);
	sc18 = log(sc18);

	std::cout << testCaseId << "\t:\t" << c18 << "\t\t\t:\t" << sc18 << std::endl;//24
	ASSERT_AML(c18, sc18);
	testCaseId++;

	// vector test

	Complex64 c19 = 8.7 - 3.2_i;
	Array2Complex64 a2c19(c19);
	Array4Complex64 a4c19(c19);
	Array8Complex64 a8c19(c19);

	std::cout << testCaseId << "\t:\t" << c19 << "\t\t\t\t\t:\t" << a2c19[0] << std::endl;//25
	ASSERT_AML(c19, a2c19[0]);
	ASSERT_AML(c19, a2c19[1]);
	ASSERT_AML(c19, a4c19[0]);
	ASSERT_AML(c19, a4c19[0]);
	ASSERT_AML(c19, a4c19[1]);
	ASSERT_AML(c19, a4c19[2]);
	ASSERT_AML(c19, a4c19[3]);
	ASSERT_AML(c19, a8c19[0]);
	ASSERT_AML(c19, a8c19[1]);
	ASSERT_AML(c19, a8c19[2]);
	ASSERT_AML(c19, a8c19[3]);
	ASSERT_AML(c19, a8c19[4]);
	ASSERT_AML(c19, a8c19[5]);
	ASSERT_AML(c19, a8c19[6]);
	ASSERT_AML(c19, a8c19[7]);
	testCaseId++;

	std::cout << "test complete" << std::endl;


	return 0;
}