#define USE_OPENCL

#include "amathlib.h"

template<class T>
T add(T x, T y) {
	return x + y;
}

__kernel void test(__global double *a, __global double *b) {
	auto index = get_global_id(0);
	a[index] = add(b[index], b[index + 1]);
}
