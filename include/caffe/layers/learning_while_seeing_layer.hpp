#ifndef CAFFE_LEARNING_WHILE_SEEING_LAYER_HPP_
#define CAFFE_LEARNING_WHILE_SEEING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class LearningWhileSeeingLayer : public Layer<Dtype> {
 public:

  explicit LearningWhileSeeingLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LearningWhileSeeing"; }
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


    shared_ptr<Layer<Dtype> > conv_layer_;
    shared_ptr<Layer<Dtype> > bn_lws_layer_;
    shared_ptr<Layer<Dtype> > rbf_layer_;
    shared_ptr<Layer<Dtype> > lws_loss_layer_; // Learning while seeing loss layer


    bool has_bn_lws_param_;

    static std::map<int, Blob<Dtype>* > bottoms_gpu_map_;
    static std::map<int, Blob<Dtype>* > tops_gpu_map_;

    Blob<Dtype> lws_loss_;

    Blob<Dtype> lws_bottom_;
    Blob<Dtype> lws_top_;

};

}  // namespace caffe

#endif  // CAFFE_LEARNING_WHILE_SEEING_LAYER_HPP_
