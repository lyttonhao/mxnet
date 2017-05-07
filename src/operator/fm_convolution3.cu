/*!
 * Copyright (c) 2015 by Contributors
 * \file fm_convolution3.cu
 * \brief
 * \author Yanghao Li
*/

#include "./fm_convolution3-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(FMConvolution3Param param) {
  return new FMConvolution3Op<gpu>(param);
}
}  // namespace op
}  // namespace mxnet

