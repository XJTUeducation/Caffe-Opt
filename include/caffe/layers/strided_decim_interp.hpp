#ifndef CAFFE_STRIDED_DECIM_INTERP_HPP_
#define CAFFE_STRIDED_DECIM_INTERP_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/memory_bank.hpp"

namespace caffe {


template <typename Dtype>
class StridedDecimInterpLayer : public Layer<Dtype> {
 public:

  explicit StridedDecimInterpLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {
    }

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "StridedDecimInterp"; }

    virtual inline int ExactNumBottomBlobs() const
    {
        return 1;
    }

    virtual inline int ExactNumTopBlobs() const
    {
        return 1;
    }


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

    int dilation_;
    Blob<Dtype> temp_buffer_;
    int n_batch_per_example_;
    int strided_op_;

};

}  // namespace caffe

#endif  // CAFFE_STRIDED_DECIM_INTERP_HPP_
