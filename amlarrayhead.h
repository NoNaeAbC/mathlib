//
// Created by af on 10.03.21.
//

#ifndef MATHLIB_AMLARRAYHEAD_H
#define MATHLIB_AMLARRAYHEAD_H

#define AML_TYPE float
#define AML_TYPE_NAME(X) X ## _32

#include "amlarray.h"

#undef AML_TYPE
#define AML_TYPE double
#undef AML_TYPE_NAME
#define AML_TYPE_NAME(X) X ## _64

#include "amlarray.h"


#undef AML_TYPE
#undef AML_TYPE_NAME


#endif //MATHLIB_AMLARRAYHEAD_H
