#include <cmath>
#include <vector>

#include "caffe/layers/rbf_layer.hpp"
#include "caffe/filler.hpp"

namespace caffe {

template <typename Dtype>
void RBFLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {

    const int n_channels = bottom[0]->shape(1);

    this->blobs_.resize(2);

    this->blobs_[0]->Reshape(1, n_channels, 1, 1);
    this->blobs_[1]->Reshape(1, n_channels, 1, 1);

    // set mean and var to default values
    caffe_set(this->blobs_[0]->count(), Dtype(0), this->blobs_[0]->mutable_cpu_data());
    caffe_set(this->blobs_[1]->count(), Dtype(1), this->blobs_[1]->mutable_cpu_data());

    if(this->layer_param_.param_size() > 2)
        LOG(FATAL) << "More than two param defined ! Only two lr_mult and decay_mult can be added";

    while(this->layer_param_.param_size() < 2)
    {
        ParamSpec* param_spec = this->layer_param_.mutable_param()->Add();

        param_spec->set_lr_mult(0);
        param_spec->set_decay_mult(0);
    }

    if(this->layer_param_.param(0).lr_mult() == 0.0)
        this->set_param_propagate_down(0, false);

    // no learning of variance (but can learn in future)
    this->set_param_propagate_down(0, false);


}

template <typename Dtype>
void RBFLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top) {

    const int num = bottom[0]->shape(0);
    const int n_channels = bottom[0]->shape(1);
    const int height = bottom[0]->shape(2);
    const int width = bottom[0]->shape(3);

    top[0]->Reshape(num, n_channels, height, width);

    spatial_sum_multiplier_.Reshape(1, 1, height, width);
    batch_sum_multiplier_.Reshape(1, n_channels, 1, 1);

    caffe_set( spatial_sum_multiplier_.count(), Dtype(1.0), spatial_sum_multiplier_.mutable_cpu_data());
    caffe_set( batch_sum_multiplier_.count(), Dtype(1.0), batch_sum_multiplier_.mutable_cpu_data());

    spatial_diff_.Reshape(num, n_channels, 1, 1);
}



template <typename Dtype>
void RBFLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void RBFLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(RBFLayer);
#endif

INSTANTIATE_CLASS(RBFLayer);


}  // namespace caffe
