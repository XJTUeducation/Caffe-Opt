#ifndef CAFFE_BOTTLENECK_LAYER_HPP_
#define CAFFE_BOTTLENECK_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class BottleNeckLayer : public Layer<Dtype> {
 public:

  explicit BottleNeckLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BottleNeck"; }
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


    shared_ptr<Layer<Dtype> > conv_reduce_layer_;
    shared_ptr<Layer<Dtype> > bn_reduce_layer_;
    shared_ptr<Layer<Dtype> > relu_reduce_layer_;

    shared_ptr<Layer<Dtype> > conv_grow_layer_;
    shared_ptr<Layer<Dtype> > bn_grow_layer_;
    shared_ptr<Layer<Dtype> > relu_grow_layer_;

    bool has_bn_reduce_param_;
    bool has_bn_grow_param_;

    bool has_reducer_;

    static std::map<int, Blob<Dtype>* > global_data_map_;
    static std::map<int, Blob<Dtype>* > global_diff_map_;

    Blob<Dtype> reduce_top_;
};

}  // namespace caffe

#endif  // CAFFE_BOTTLENECK_LAYER_HPP_
