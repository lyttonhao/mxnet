/*!
 * Copyright (c) 2015 by Contributors
 * \file fm_convolution2.cc
 * \brief
 * \author Yanghao Li
*/

#include "./fm_convolution2-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(FMConvolution2Param param) {
  return new FMConvolution2Op<cpu>(param);
}

Operator* FMConvolution2Prop::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(FMConvolution2Param);

MXNET_REGISTER_OP_PROPERTY(FMConvolution2, FMConvolution2Prop)
.add_argument("data", "Symbol", "Input data to the Convolution2Op.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(FMConvolution2Param::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet

