/*!
 * Copyright (c) 2015 by Contributors
 * \file fm_convolution.cc
 * \brief
 * \author Yanghao Li
*/

#include "./fm_convolution1-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(FMConvolution1Param param) {
  return new FMConvolution1Op<cpu>(param);
}

Operator* FMConvolution1Prop::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(FMConvolution1Param);

MXNET_REGISTER_OP_PROPERTY(FMConvolution1, FMConvolution1Prop)
.add_argument("data", "Symbol", "Input data to the Convolution1Op.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(FMConvolution1Param::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet

