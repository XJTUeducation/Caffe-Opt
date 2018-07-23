#ifndef CAFFE_BN_LAYER_HPP_
#define CAFFE_BN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/memory_bank.hpp"

namespace caffe {

#define BN_OPTMIZE_X_NORM

/**
 * @brief Batch normalization the input blob along the channel axis while
 *        averaging over the spatial axes.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BNLayer : public Layer<Dtype> {
 public:
  explicit BNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}


  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BN"; }
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

  void AverageAllExceptChannel(const Dtype* input, Dtype* output);
  void BroadcastChannel(const Dtype* input, Dtype* output);

  bool frozen_;
  Dtype bn_momentum_;
  Dtype bn_eps_;

  int num_;
  int channels_;
  int height_;
  int width_;

  Blob<Dtype> broadcast_buffer_;
  Blob<Dtype> spatial_statistic_;
  Blob<Dtype> batch_statistic_;

  Blob<Dtype> x_norm_cpu_;
  Blob<Dtype> x_norm_;
  Blob<Dtype> x_inv_std_;

  Blob<Dtype> spatial_sum_multiplier_;
  Blob<Dtype> batch_sum_multiplier_;

  static   std::vector<unsigned long int> broadcast_buffer_old_count_;
  static   std::map<int,Dtype*>* broadcast_buffer_pointer_map_; // FOR MULTI GPU


private:

  Blob<Dtype> batch_statistic_instances_;

  static pthread_mutex_t mutex_instances_finished_;
  static pthread_cond_t cond_gathered_;

  static std::vector<Dtype*> batch_statistics_ptrs_gpu_;

  void bn_gather_statistics_gpu(bool normalize = false);

};

}  // namespace caffe

#endif  // CAFFE_BN_LAYER_HPP_
