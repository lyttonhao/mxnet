/*!
 * Copyright (c) 2015 by Contributors
 * \file fm_convolution3.cc
 * \brief
 * \author Yanghao Li
*/

#include "./fm_convolution3-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(FMConvolution3Param param) {
  return new FMConvolution3Op<cpu>(param);
}

Operator* FMConvolution3Prop::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(FMConvolution3Param);

MXNET_REGISTER_OP_PROPERTY(FMConvolution3, FMConvolution3Prop)
.add_argument("data", "Symbol", "Input data to the Convolution2Op.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(FMConvolution3Param::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet

