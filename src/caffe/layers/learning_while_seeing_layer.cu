#include <vector>

#include "caffe/layers/learning_while_seeing_layer.hpp"

namespace caffe {



template <typename Dtype>
void LearningWhileSeeingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {

    cudaMemcpy(lws_bottom_.mutable_gpu_data(), bottom[0]->gpu_data(), bottom[0]->count() * sizeof(Dtype), cudaMemcpyDeviceToDevice);

    std::vector<Blob<Dtype>*> lws_bottoms;
    std::vector<Blob<Dtype>*> lws_tops;

    lws_bottoms.push_back(&lws_bottom_);
    lws_tops.push_back(&lws_top_);

    conv_layer_->Forward(lws_bottoms, lws_tops);

    if(has_bn_lws_param_)
        bn_lws_layer_->Forward(lws_tops, lws_tops);

    rbf_layer_->Forward(lws_tops, lws_tops);

    std::vector<Blob<Dtype>*> lws_loss_tops;
    lws_loss_tops.push_back(&lws_loss_);

    lws_loss_layer_->Forward(lws_tops, lws_loss_tops);

    // backward
    const std::vector<bool> conv_propagate_down(false);
    std::vector<bool> bn_propagate_down(true);
    std::vector<bool> rbf_propagate_down(true);
    std::vector<bool> loss_propagate_down(true);

    lws_loss_layer_->Backward(lws_loss_tops, loss_propagate_down, lws_tops);

    rbf_layer_->Backward( lws_tops, rbf_propagate_down , lws_tops);

    if(has_bn_lws_param_)
        bn_lws_layer_->Backward(lws_tops, bn_propagate_down, lws_tops);


    conv_layer_->Backward(lws_tops, conv_propagate_down, lws_bottoms );


}

template <typename Dtype>
void LearningWhileSeeingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down, const vector<Blob<Dtype>*>&
                                           bottom) {

 // Nothing to do

}

INSTANTIATE_LAYER_GPU_FUNCS(LearningWhileSeeingLayer);

}  // namespace caffe
