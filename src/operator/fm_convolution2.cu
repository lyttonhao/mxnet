/*!
 * Copyright (c) 2015 by Contributors
 * \file fm_convolution2.cu
 * \brief
 * \author Yanghao Li
*/

#include "./fm_convolution2-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(FMConvolution2Param param) {
  return new FMConvolution2Op<gpu>(param);
}
}  // namespace op
}  // namespace mxnet

