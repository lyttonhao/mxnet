/*!
 * Copyright (c) 2015 by Contributors
 * \file fm_convolution3-inl.h
 * \brief factorized bilinear layer (speed up version by batch_dot)
 * \author Yanhao Li
*/
#ifndef MXNET_OPERATOR_FM_CONVOLUTION3_INL_H_
#define MXNET_OPERATOR_FM_CONVOLUTION3_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <math.h>
#include "./operator_common.h"
 #include "./mshadow_op.h"


namespace mxnet {
namespace op {

namespace fmconv {
enum FMConvolution3OpInputs {kData, kWeight, kBias};
enum FMConvolution3OpOutputs {kOut, kMask};
enum FMConvolution3OpResource {kTempSpace, kRandom};
//enum FMConvolution2OpForwardResource {kRandom};
}

struct FMConvolution3Param : public dmlc::Parameter<FMConvolution3Param> {
  TShape kernel;
  TShape stride;
  TShape dilate;
  TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  uint64_t workspace;
  uint32_t num_factor;
  bool no_bias;
  float eps;
  float p;
  DMLC_DECLARE_PARAMETER(FMConvolution3Param) {
    int shape[] = {1, 1};
    DMLC_DECLARE_FIELD(kernel).describe("convolution kernel size: (y, x)");
    DMLC_DECLARE_FIELD(stride).set_default(TShape(shape, shape + 2))
    .describe("convolution stride: (y, x)");
    DMLC_DECLARE_FIELD(dilate).set_default(TShape(shape, shape + 2))
    .describe("convolution dilate: (y, x)");
    shape[0] = shape[1] = 0;
    DMLC_DECLARE_FIELD(pad).set_default(TShape(shape, shape + 2))
    .describe("pad for convolution: (y, x)");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
    .describe("convolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of groups partition. "
              "This option is not supported by CuDNN, you can use SliceChannel to num_group,"
              "apply convolution and concat instead to achieve the same need.");
    DMLC_DECLARE_FIELD(workspace).set_default(512).set_range(0, 4096)
    .describe("Tmp workspace for convolution (MB).");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(num_factor).set_default(1).set_range(1, 10000)
    .describe("Number of factor dimension.");
    DMLC_DECLARE_FIELD(eps).set_default(1e-10f)
    .describe("Epsilon to prevent div 0");   
    DMLC_DECLARE_FIELD(p).set_default(0.5)
    .describe("Fraction of the weight that gets dropped out at training time");   
  }
};

template<typename xpu>
class FMConvolution3Op : public Operator {
 public:
  explicit FMConvolution3Op(FMConvolution3Param param) {
    this->param_ = param;
    this->pkeep_ = 1.0f - param.p;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(real_t);
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[fmconv::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 2);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[fmconv::kData].get<xpu, 4, real_t>(s);
    Shape<4> wmat_shape =
        Shape4(param_.num_group,
               param_.num_filter / param_.num_group,
               param_.num_factor,
               data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1]);
    Tensor<xpu, 4> wmat = in_data[fmconv::kWeight].get_with_shape<xpu, 4, real_t>(wmat_shape, s);
    Tensor<xpu, 4> out = out_data[fmconv::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 1> mask = out_data[fmconv::kMask].get<xpu, 1, real_t>(s);
    //Get drop mask
    Random<xpu> *prnd = ctx.requested[fmconv::kRandom].get_random<xpu, real_t>(s);
    if (ctx.is_train) {
      mask = F<mshadow_op::threshold>(prnd->uniform(mask.shape_), pkeep_) * sqrt(1.0f / pkeep_);
    }else {
      mask = sqrt(pkeep_);
    }

#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    const index_t nbatch = data.size(0);
    const index_t required_size = this->InitTemp(data.shape_, out.shape_);
    const index_t gstride = shape_colunit_[0] / param_.num_group; //feature size
    const index_t _ssize = shape_colunit_[1] * nstep_;  //step size
    const int output_size = param_.num_filter / param_.num_group;  //output size
    float *dptr, *_dptr;
    
    Tensor<xpu, 1> workspace = ctx.requested[fmconv::kTempSpace].get_space<xpu>(
        Shape1((required_size 
                + output_size * (1 * wmat_shape[2] * wmat_shape[3]
                                 + 1 * gstride * _ssize
                                 + param_.num_factor * _ssize
                                 + 2 * gstride
                                 + _ssize)
                + _ssize
                + wmat_shape[2] * wmat_shape[3])), s);
    dptr = workspace.dptr_;
    Tensor<xpu, 3> mask_w = Tensor<xpu, 3>(dptr, Shape3(output_size, wmat_shape[2], wmat_shape[3]), s);
    dptr += mask_w.shape_.Size();
    Tensor<xpu, 2> mask1 = Tensor<xpu, 2>(dptr, Shape2(wmat_shape[2], wmat_shape[3]), s);
    dptr += mask1.shape_.Size();
    _dptr = dptr;
    Tensor<xpu, 1, real_t*> bgemm_space(NULL, Shape1(3 * output_size), s);

    mask1 = broadcast<0>(mask, mask1.shape_);
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      const index_t ssize = shape_colunit_[1] * step;

      dptr = _dptr;
      Tensor<xpu, 2> temp_col = Tensor<xpu, 2>(dptr, Shape2(shape_colunit_[0], ssize), s);
      dptr += temp_col.shape_.Size();
      Tensor<xpu, 3> temp_dst = Tensor<xpu, 3>(dptr, Shape3(shape_dstunit_[0],
                                                            shape_dstunit_[1],
                                                            shape_dstunit_[2] * step), s);
      dptr += temp_dst.shape_.Size();
      Tensor<xpu, 3> temp = Tensor<xpu, 3>(dptr, Shape3(output_size, gstride, ssize), s);
      dptr += temp.shape_.Size();
      Tensor<xpu, 3> temp1 = Tensor<xpu, 3>(dptr, Shape3(output_size, param_.num_factor, ssize), s);
      dptr += temp1.shape_.Size();
      Tensor<xpu, 1> col = Tensor<xpu, 1>(dptr, Shape1(ssize), ssize, s);
      dptr += col.shape_.Size();
      Tensor<xpu, 2> col1 = Tensor<xpu, 2>(dptr, Shape2(output_size, gstride), s);
      dptr += col1.shape_.Size();
      Tensor<xpu, 3> temp2 = Tensor<xpu, 3>(dptr, Shape3(output_size, 1, gstride), s);
      dptr += temp2.shape_.Size();
      Tensor<xpu, 3> temp3 = Tensor<xpu, 3>(dptr, Shape3(output_size, 1, ssize), s);
      dptr += temp3.shape_.Size();

      if (param_.pad[0] == 0 && param_.pad[1] == 0) {
        temp_col = unpack_patch2col(data.Slice(i, i + step),
                                    param_.kernel[0],
                                    param_.kernel[1],
                                    param_.stride[0],
                                    param_.stride[1],
                                    param_.dilate[0],
                                    param_.dilate[1]);
      } else {
        temp_col = unpack_patch2col(pad(data.Slice(i, i + step),
                                        param_.pad[0], param_.pad[1]),
                                    param_.kernel[0],
                                    param_.kernel[1],
                                    param_.stride[0],
                                    param_.stride[1],
                                    param_.dilate[0],
                                    param_.dilate[1]);
      }

      for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
        mshadow::Tensor<xpu, 2> tmpc = temp_col.Slice(gstride * gid,
                                gstride * (gid + 1));

        for (int j = 0;j < output_size;++j) {
          mask_w[j] = mask1 * wmat[gid][j];
          temp[j] = tmpc * 1.0f;
        }
        BatchGEMM<false, false>(temp1, mask_w, temp, 1.0f, 0.0f, bgemm_space);
        BatchGEMM<true, false>(temp, mask_w, temp1, 1.0f, 0.0f, bgemm_space);

        for (int j = 0; j < output_size; ++j) {
          temp[j] = temp[j] * tmpc;
          col = sum_rows(temp[j]);
          temp_dst[gid].Slice(j, j+1) = reshape(col, Shape2(1, ssize));

          col1[j] = sum_rows(mask_w[j] * mask_w[j]);
          temp2[j] = reshape(col1[j], Shape2(1, gstride));
          temp[j] = tmpc * tmpc;
        }
        BatchGEMM<false, false>(temp3, temp2, temp, 1.0f, 0.0f, bgemm_space);

        for (int j = 0;j < output_size;++j){
          temp_dst[gid].Slice(j, j+1) -= temp3[j];
        }

      }
      out.Slice(i, i + step) = swapaxis<1, 0>(reshape(temp_dst,
                                              mshadow::Shape4(param_.num_filter,
                                                  step,
                                                  out.size(2),
                                                  out.size(3))));
    }
    if (!param_.no_bias) {
      // add bias, broadcast bias to dim 1: channel
      Tensor<xpu, 1> bias = in_data[fmconv::kBias].get<xpu, 1, real_t>(s);
      out += broadcast<1>(bias, out.shape_);
    }

  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // TODO(bing): check the BLAS Handle, be careful
    CHECK_EQ(out_grad.size(), 1);
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[fmconv::kWeight].CheckContiguous(), true);
    // get data
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[fmconv::kData].get<xpu, 4, real_t>(s);
    Shape<4> wmat_shape =
        Shape4(param_.num_group,
               param_.num_filter / param_.num_group,
               param_.num_factor, 
               data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1]);
    Tensor<xpu, 4> wmat = in_data[fmconv::kWeight].get_with_shape<xpu, 4, real_t>(wmat_shape, s);
    Tensor<xpu, 4> grad = out_grad[fmconv::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gdata = in_grad[fmconv::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gwmat = in_grad[fmconv::kWeight].get_with_shape<xpu, 4, real_t>(wmat_shape, s);
    Tensor<xpu, 1> mask = out_data[fmconv::kMask].get<xpu, 1, real_t>(s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    const index_t nbatch = data.size(0);
    const index_t required_size = this->InitTemp(data.shape_, grad.shape_);
    const index_t gstride = shape_colunit_[0] / param_.num_group;
    const index_t _ssize = shape_colunit_[1] * nstep_;
    const int output_size = param_.num_filter / param_.num_group;  //output size
    float *dptr, *_dptr;

    Tensor<xpu, 1> workspace = ctx.requested[fmconv::kTempSpace].get_space<xpu>(
              Shape1((required_size 
                      + output_size * (2 * wmat_shape[2] * wmat_shape[3]
                                       + 1 * gstride * _ssize
                                       + param_.num_factor * _ssize
                                       + 1 * gstride
                                       + _ssize)
                      + wmat_shape[2] * wmat_shape[3])
                      + gstride 
                      + shape_colunit_[0] * _ssize), s);
    dptr = workspace.dptr_;
    Tensor<xpu, 3> mask_w = Tensor<xpu, 3>(dptr, Shape3(output_size, wmat_shape[2], wmat_shape[3]), s);
    dptr += mask_w.shape_.Size();
    Tensor<xpu, 2> mask1 = Tensor<xpu, 2>(dptr, Shape2(wmat_shape[2], wmat_shape[3]), s);
    dptr += mask1.shape_.Size();
    _dptr = dptr;
    Tensor<xpu, 1, real_t*> bgemm_space(NULL, Shape1(3 * output_size), s);

    mask1 = broadcast<0>(mask, mask1.shape_);
    gwmat = 0.0;
    gdata = 0.0;
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      const index_t ssize = shape_colunit_[1] * step;
      dptr = _dptr;

      Tensor<xpu, 2> temp_col = Tensor<xpu, 2>(dptr, Shape2(shape_colunit_[0], ssize), s);
      dptr += temp_col.shape_.Size();
      Tensor<xpu, 3> temp_dst = Tensor<xpu, 3>(dptr, Shape3(shape_dstunit_[0],
                                                            shape_dstunit_[1],
                                                            shape_dstunit_[2] * step), s);
      dptr += temp_dst.shape_.Size();
      Tensor<xpu, 3> temp = Tensor<xpu, 3>(dptr, Shape3(output_size, gstride, ssize), s);
      dptr += temp.shape_.Size();
      Tensor<xpu, 3> temp1 = Tensor<xpu, 3>(dptr, Shape3(output_size, param_.num_factor, ssize), s);
      dptr += temp1.shape_.Size();
      Tensor<xpu, 1> col1 = Tensor<xpu, 1>(dptr, Shape1(gstride), gstride, s);
      dptr += col1.shape_.Size();
      Tensor<xpu, 3> temp2 = Tensor<xpu, 3>(dptr, Shape3(output_size, 1, gstride), s);
      dptr += temp2.shape_.Size();
      Tensor<xpu, 3> temp3 = Tensor<xpu, 3>(dptr, Shape3(output_size, 1, ssize), s);
      dptr += temp3.shape_.Size();
      Tensor<xpu, 3> temp_w = Tensor<xpu, 3>(dptr, Shape3(output_size, wmat_shape[2], wmat_shape[3]), s);
      dptr += temp_w.shape_.Size();
      Tensor<xpu, 2> temp_col1 = Tensor<xpu, 2>(dptr, Shape2(shape_colunit_[0], ssize), s);
      dptr += temp_col1.shape_.Size();

      temp_dst = reshape(swapaxis<1, 0>(grad.Slice(i, i + step)), temp_dst.shape_);
      if (param_.pad[0] == 0 && param_.pad[1] == 0) {
        temp_col = unpack_patch2col(data.Slice(i, i + step),
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.stride[1],
                                     param_.dilate[0],
                                     param_.dilate[1]);
      } else {
        temp_col = unpack_patch2col(pad(data.Slice(i, i + step), param_.pad[0], param_.pad[1]),
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.stride[1],
                                     param_.dilate[0],
                                     param_.dilate[1]);
      }
      temp_col1 = 0.0;
      for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
        Tensor<xpu, 2> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));

        for (uint32_t j = 0;j < output_size;++j) {
          mask_w[j] = mask1 * wmat[gid][j];
          temp[j] = tmpc * 1.0f;
        }
        BatchGEMM<false, false>(temp1, mask_w, temp, 1.0f, 0.0f, bgemm_space);
        for (uint32_t j = 0;j < output_size;++j) {
          temp[j] = repmat(temp_dst[gid][j], gstride);
          temp[j] = temp[j] * tmpc;
        }
        BatchGEMM<false, true>(temp_w, temp1, temp, 1.0f, 0.0f, bgemm_space);

        for (uint32_t j = 0;j < output_size;++j) {
          gwmat[gid][j] += 2 * mask1 * temp_w[j];

          temp3[j] = reshape(temp_dst[gid][j], Shape2(1, ssize));
          temp[j] = tmpc * tmpc;
        }
        BatchGEMM<false, true>(temp2, temp3, temp, 1.0f, 0.0f, bgemm_space);

        for (uint32_t j = 0;j < output_size;++j) {
          col1 = reshape(temp2[j], Shape1(gstride));
          temp_w[j] = repmat(col1, temp_w[j].shape_[0]);
          temp_w[j] *= mask_w[j];
          gwmat[gid][j] -= 2 * temp_w[j] * mask1;
        }


        if (req[fmconv::kData] == kWriteTo || req[fmconv::kData] == kWriteInplace) {
          Tensor<xpu, 2> gtmpc = temp_col1.Slice(gstride * gid, gstride * (gid + 1));

          BatchGEMM<true, false>(temp, mask_w, temp1, 2.0f, 0.0f, bgemm_space);
          for (uint32_t j = 0;j < output_size;++j) {
            temp[j] *= repmat(temp_dst[gid][j], gstride);
            gtmpc += temp[j];

            col1 = sum_rows(mask_w[j] * mask_w[j]);
            temp2[j] = reshape(col1, Shape2(1, gstride));
            temp3[j] = reshape(temp_dst[gid][j], Shape2(1, ssize));
          }
          BatchGEMM<true, false>(temp, temp2, temp3, 2.0f, 0.0f, bgemm_space);
          for (uint32_t j = 0;j < output_size;++j)
            gtmpc -= temp[j];

        }
      }
      if (req[fmconv::kData] == kWriteTo || req[fmconv::kData] == kWriteInplace) {  
        if (param_.pad[0] == 0 && param_.pad[1] == 0) {
          gdata.Slice(i, i + step) = pack_col2patch(temp_col1,
                                     data.Slice(i, i + step).shape_,
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.dilate[0]);
        } else {
          Shape<4> pshape = data.Slice(i, i + step).shape_;
          pshape[2] += 2 * param_.pad[0];
          pshape[3] += 2 * param_.pad[1];
          gdata.Slice(i, i + step) = crop(pack_col2patch(temp_col1,
                                          pshape,
                                          param_.kernel[0],
                                          param_.kernel[1],
                                          param_.stride[0],
                                          param_.dilate[0]),
                                          gdata[i][0].shape_);
        }
      }
    }
    if (!param_.no_bias) {
      Tensor<xpu, 1> gbias = in_grad[fmconv::kBias].get<xpu, 1, real_t>(s);
      Assign(gbias, req[fmconv::kBias], sumall_except_dim<1>(grad));
    }
  }

 private:
  inline index_t InitTemp(const mshadow::Shape<4> &ishape,
                          const mshadow::Shape<4> &oshape) {
    const int ksize_y = param_.kernel[0];
    const int ksize_x = param_.kernel[1];
    shape_colunit_ = mshadow::Shape2(ishape[1] * ksize_y * ksize_x,
                                     oshape[2] * oshape[3]);
    shape_dstunit_ = mshadow::Shape3(param_.num_group,
                                     param_.num_filter / param_.num_group,
                                     oshape[2] * oshape[3]);
    // param_.workspace is in elements of sizeof(real_t)
    // if param_.workspace is set to zero the nstep_ equals ishape[0] (batch)
    nstep_ = std::max(
        std::min(
          static_cast<index_t>(param_.workspace / (shape_colunit_.Size() + shape_dstunit_.Size())),
          ishape[0]),
        1U);

    mshadow::Shape<2> scol = mshadow::Shape2(shape_colunit_[0],
                                             shape_colunit_[1] * nstep_);
    mshadow::Shape<3> sdst = mshadow::Shape3(shape_dstunit_[0],
                                             shape_dstunit_[1],
                                             shape_dstunit_[2] * nstep_);
    index_t required_size = scol.Size() + sdst.Size();
    CHECK_GE(param_.workspace, required_size)
      << "\nMinimum workspace size: " << required_size * sizeof(real_t) << " Bytes\n"
      << "Given: " << param_.workspace * sizeof(real_t) << " Bytes";
    return required_size;
  }

  FMConvolution3Param param_;
  mshadow::Shape<2> shape_colunit_;
  mshadow::Shape<3> shape_dstunit_;
  index_t nstep_;
  real_t pkeep_;
};  // class FMConvolutionOp

template<typename xpu>
Operator* CreateOp(FMConvolution3Param param);

#if DMLC_USE_CXX11
class FMConvolution3Prop : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return {"data", "weight", "bias"};
    } else {
      return {"data", "weight"};
    }
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mask"};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 2;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
    }
    const TShape &dshape = (*in_shape)[fmconv::kData];
    if (dshape.ndim() ==  0) return false;
    CHECK_EQ(dshape.ndim(), 4) \
        << "Input data should be 4D in batch-num_filter-y-x";
    SHAPE_ASSIGN_CHECK(*in_shape,
                       fmconv::kWeight,
                       Shape5(param_.num_filter, param_.num_factor, dshape[1] / param_.num_group,
                              param_.kernel[0], param_.kernel[1]));
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, fmconv::kBias, Shape1(param_.num_filter));
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    const index_t ksize_y = static_cast<index_t>(param_.kernel[0]);
    const index_t ksize_x = static_cast<index_t>(param_.kernel[1]);
    CHECK_EQ(dshape[1] % param_.num_group, 0) \
        << "input num_filter must divide group size";
    CHECK_EQ(param_.num_filter % param_.num_group, 0) \
        << "output num_filter must divide group size";
    CHECK_GE(param_.kernel.Size(), 0) \
        << "incorrect kernel size: " << param_.kernel;
    CHECK_GE(param_.stride.Size(), 0) \
        << "incorrect stride size: " << param_.stride;
    CHECK_GE(param_.dilate.Size(), 0) \
        << "incorrect dilate size: " << param_.dilate;
    CHECK(ksize_x <= dshape[3] && ksize_y <= dshape[2])
        << "kernel size exceed input";
    (*out_shape)[fmconv::kOut][1] = param_.num_filter;
    (*out_shape)[fmconv::kOut][2] = (dshape[2] + 2 * param_.pad[0] -
        (param_.dilate[0] == 1 ? ksize_y : ksize_y * param_.dilate[0] - 1)) / param_.stride[0] + 1;
    (*out_shape)[fmconv::kOut][3] = (dshape[3] + 2 * param_.pad[1] -
        (param_.dilate[1] == 1 ? ksize_x : ksize_x * param_.dilate[1] - 1)) / param_.stride[1] + 1;
    out_shape->push_back(Shape1(param_.num_factor));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new FMConvolution3Prop();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "FMConvolution3";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[fmconv::kOut], 
      in_data[fmconv::kData], 
      in_data[fmconv::kWeight],
      out_data[fmconv::kMask]};
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace, ResourceRequest::kRandom};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  FMConvolution3Param param_;
};  // class FMConvolutionProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_FM_CONVOLUTION3_INL_H_
