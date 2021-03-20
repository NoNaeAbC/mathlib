
#define USE_FMA
#define USE_CONCEPTS
#define AML_USE_GMP

#define AML_USE_STD_COMPLEX

#include <iostream>
#include "amathlib.h"
#include "amlavg.h"
#include <algorithm>

int main() {
	Average<double> a;
	a += 0.0;
	a += 3.0;
	a += 3.0;
	std::cout << a << std::endl;
	Average<double, HARMONIC_MEAN> d;
	d += 0.0;
	d += 3.0;
	d += 3.0;
	std::cout << d << std::endl;
	Average<double, GEOMETRIC_MEAN> f;
	f += 0.0;
	f += 3.0;
	f += 3.0;
	std::cout << f << std::endl;


	CREATE_POW_FUNC(AR, 0.0);
	Average<double, 0> avr;
	avr.add(2.0, POW_FUNC(AR));
	avr.add(3.0, POW_FUNC(AR));
	avr.add(3.0, POW_FUNC(AR));
	std::cout << avr.getValue(POW_FUNCI(AR)) << std::endl;

	Average<double, MAX_MEAN> abc;
	abc += 3.2;
	abc += 9.2;
	abc += 2.2;
	std::cout << abc << std::endl;
	Average<double, LOGARITHMIC_MEAN> abc2;
	abc2 += 3.2;
	abc2 += 9.2;
	std::cout << abc2 << std::endl;
	return 0;

}
