//
// Created by af on 10.03.21.
//


#define AML_TYPE_FLOAT 2
#define AML_TYPE_DOUBLE 1

#define AML_TYPE_ID 2
#define AML_TYPE float
#define AML_TYPE_NAME(X) AML_PREFIX(X ## _32)

#include "amlarray.h"

#undef AML_TYPE
#define AML_TYPE double
#undef AML_TYPE_ID
#define AML_TYPE_ID 1
#undef AML_TYPE_NAME

#ifdef AML_TYPE_NAME
#error ERROR IN COMPILER
#endif

#define AML_TYPE_NAME(X) AML_PREFIX(X ## _64)

#include "amlarray.h"


#undef AML_TYPE
#undef AML_TYPE_NAME
#undef AML_TYPE_ID
#undef AML_TYPE_FLOAT
#undef AML_TYPE_DOUBLE
