#ifndef CAFFE_MAX_ACTIVATIONS_GROUPING_HPP_
#define CAFFE_MAX_ACTIVATIONS_GROUPING_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


template <typename Dtype>
class MaxActivationsGroupingLayer : public Layer<Dtype> {
 public:
  explicit MaxActivationsGroupingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual inline const char* type() const { return "MaxActivationsGrouping"; }

    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int MinNumTopBlobs() const { return 1; }

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

 protected:


  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


    Blob<int> max_indices_;

};



}  // namespace caffe

#endif  // CAFFE_MAX_ACTIVATIONS_GROUPING_LAYER_HPP_
