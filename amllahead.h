//
// Created by af on 14.03.21.
//

#define AML_TYPE_FLOAT 2
#define AML_TYPE_DOUBLE 1

#define AML_TYPE_ID 2
#define AML_TYPE float
#define AML_TYPE_NAME(X) AML_PREFIX(X ## _32)


#include "amlla.h"

#undef AML_TYPE
#define AML_TYPE double
#undef AML_TYPE_ID
#define AML_TYPE_ID 1
#undef AML_TYPE_NAME
#define AML_TYPE_NAME(X) AML_PREFIX(X ## _64)

#include "amlla.h"


#undef AML_TYPE
#undef AML_TYPE_NAME
#undef AML_TYPE_ID
#undef AML_TYPE_FLOAT
#undef AML_TYPE_DOUBLE

class AML_PREFIX(Array2UInt_32) {
public:
	uint32_t a[2];

	AML_FUNCTION AML_PREFIX(Array2UInt_32)() {};

	AML_FUNCTION AML_PREFIX(Array2UInt_32)(uint32_t *val) {
		a[0] = val[0];
		a[1] = val[1];
	};

	AML_FUNCTION AML_PREFIX(Array2UInt_32)(uint32_t v1, uint32_t v2) {
		a[0] = v1;
		a[1] = v2;
	};

	constexpr AML_FUNCTION uint32_t &operator[](uint32_t pos) {
		return a[pos];
	}
};

