#include <vector>

#include "gtest/gtest.h"

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FuseInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //printf("FuseInnerProduce_Forward_gpu\n");
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data_1 = top[0]->mutable_gpu_data();
  Dtype* top_data_2 = top[1]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data_1);
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data_2);
    // top_data_2 = top_data_1
    //caffe_gpu_axpby<Dtype>(N_, (Dtype)1., top_data_2, (Dtype)0., top_data_1);
    if (bias_term_) {
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data_1);
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data_2);
      //caffe_copy<Dtype>(N_,top_data_1,top_data_2);
      //caffe_gpu_axpby<Dtype>(N_, (Dtype)1., top_data_2, (Dtype)0., top_data_1);
    }
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data_1);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data_2);
    //caffe_copy<Dtype>(N_,top_data_1,top_data_2);
    //caffe_gpu_axpby<Dtype>(N_, (Dtype)1., top_data_2, (Dtype)0., top_data_1);
    if (bias_term_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data_1);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data_2);
      //caffe_copy<Dtype>(N_,top_data_1,top_data_2);
      //caffe_gpu_axpby<Dtype>(N_, (Dtype)1., top_data_2, (Dtype)0., top_data_1);
    }
  }
}

template <typename Dtype>
void FuseInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  //printf("FuseInnerProduce_Backward_gpu\n");
  if (this->param_propagate_down_[0]) {
    // sum two top_diff
    const Dtype* top_diff_1 = top[0]->gpu_diff();
    const Dtype* top_diff_2 = top[1]->gpu_diff();
    //caffe_gpu_axpy<Dtype>(N_, (Dtype)1., top_diff_1, top_diff);

    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff_1, bottom_data, (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff_2, bottom_data, (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff_1 = top[0]->gpu_diff();
    const Dtype* top_diff_2 = top[1]->gpu_diff();
    //caffe_gpu_axpy<Dtype>(N_, (Dtype)1., top_diff_1, top_diff);
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff_1,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff_2,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff_1 = top[0]->gpu_diff();
    const Dtype* top_diff_2 = top[1]->gpu_diff();
    //caffe_gpu_axpy<Dtype>(N_, (Dtype)1., top_diff_1, top_diff);
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff_1, this->blobs_[0]->gpu_data(), (Dtype)0.,
        bottom[0]->mutable_gpu_diff());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff_2, this->blobs_[0]->gpu_data(), (Dtype)1.,
        bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FuseInnerProductLayer);

}  // namespace caffe
