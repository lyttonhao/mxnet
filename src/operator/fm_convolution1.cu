/*!
 * Copyright (c) 2015 by Contributors
 * \file fm_convolution1.cu
 * \brief
 * \author Yanghao Li
*/

#include "./fm_convolution1-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(FMConvolution1Param param) {
  return new FMConvolution1Op<gpu>(param);
}
}  // namespace op
}  // namespace mxnet

