//
// Created by af on 14.03.21.
//

#define AML_TYPE_FLOAT 2
#define AML_TYPE_DOUBLE 1

#define AML_TYPE_ID 2
#define AML_TYPE float
#define AML_TYPE_NAME(X) X ## _32

#include "amlla.h"

#undef AML_TYPE
#define AML_TYPE double
#undef AML_TYPE_ID
#define AML_TYPE_ID 1
#undef AML_TYPE_NAME
#define AML_TYPE_NAME(X) X ## _64

#include "amlla.h"


#undef AML_TYPE
#undef AML_TYPE_NAME
#undef AML_TYPE_ID
#undef AML_TYPE_FLOAT
#undef AML_TYPE_DOUBLE
