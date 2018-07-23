#ifndef CAFFE_MEMORY_OPT_LAYER_HPP_
#define CAFFE_MEMORY_OPT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/memory_bank.hpp"

namespace caffe {


template <typename Dtype>
class MemoryOptLayer : public Layer<Dtype> {
 public:

  explicit MemoryOptLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {
    }

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MemoryOpt"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


private:

    int bank_id_;
    bool can_be_shared_;

    bool optmize_data_train_;
    int data_bank_id_;
    bool data_bank_can_be_shared_;

};

}  // namespace caffe

#endif  // CAFFE_MEMORY_OPT_LAYER_HPP_
