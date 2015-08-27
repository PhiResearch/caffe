#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNUnpoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::LayerSetUp(top, bottom);
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createPoolingDesc<Dtype>(&pooling_desc_,
      this->layer_param_.pooling_param().pool(), &mode_,
      this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_,
      this->stride_h_, this->stride_w_);
  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNUnpoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::Reshape(top, bottom);
  cudnn::setTensor4dDesc<Dtype>(&top_desc_, top[0]->num(),
      this->channels_, this->height_, this->width_);
  cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, top[0]->num(),
      this->channels_, this->pooled_height_, this->pooled_width_);
}

template <typename Dtype>
CuDNNUnpoolingLayer<Dtype>::~CuDNNPoolingLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyPoolingDescriptor(pooling_desc_);
  cudnnDestroy(handle_);
}

INSTANTIATE_CLASS(CuDNNUnpoolingLayer);

}   // namespace caffe
#endif
