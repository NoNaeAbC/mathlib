#ifndef USE_CUDA

#define AML_TYPE_ID 2
#define AML_TYPE float
#define AML_BASE_TYPE_NAME AML_PREFIX( X ## 32)
#define AML_TYPE_NAME(X) AML_PREFIX(Array8 ## X ## 32)
#define AML_POINTER_NAME(X) X ## 32Ptr
#define AML_ARRAY_LENGTH

#include "amlcomplexarray.h"

#undef AML_TYPE
#define AML_TYPE double
#undef AML_TYPE_ID
#define AML_TYPE_ID 1
#undef AML_TYPE_NAME
#define AML_TYPE_NAME(X) AML_PREFIX(X ## 64)
#undef AML_POINTER_NAME
#define AML_POINTER_NAME(X) X ## 64Ptr

#include "amlcomplexarray.h"


#undef AML_TYPE
#undef AML_TYPE_NAME
#undef AML_TYPE_ID
#undef AML_TYPE_FLOAT
#undef AML_TYPE_DOUBLE
#undef AML_POINTER_NAME

#endif