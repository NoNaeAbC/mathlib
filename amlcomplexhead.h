//
// Created by af on 26.03.21.
//

#ifndef MATHLIB_AMLCOMPLEXHEAD_H
#define MATHLIB_AMLCOMPLEXHEAD_H


#define AML_TYPE_ID 2
#define AML_TYPE float
#define AML_TYPE_NAME(X) X ## 32
#define AML_POINTER_NAME(X) X ## 32Ptr

#include "amlcomplex.h"

#undef AML_TYPE
#define AML_TYPE double
#undef AML_TYPE_ID
#define AML_TYPE_ID 1
#undef AML_TYPE_NAME
#define AML_TYPE_NAME(X) X ## 64
#undef AML_POINTER_NAME
#define AML_POINTER_NAME(X) X ## 64Ptr

#include "amlcomplex.h"


#undef AML_TYPE
#undef AML_TYPE_NAME
#undef AML_TYPE_ID
#undef AML_TYPE_FLOAT
#undef AML_TYPE_DOUBLE
#undef AML_POINTER_NAME


#endif //MATHLIB_AMLCOMPLEXHEAD_H
