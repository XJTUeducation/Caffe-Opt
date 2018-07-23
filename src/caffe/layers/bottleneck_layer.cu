#include <vector>

#include "caffe/layers/bottleneck_layer.hpp"

namespace caffe {



template <typename Dtype>
void BottleNeckLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {

    int device;
    cudaGetDevice(&device);

//    if(this->phase_ == TEST)
//    reduce_top_.data()->set_gpu_data(global_data_map_[device]->mutable_gpu_data());

    if(this->phase_ == TRAIN)
        reduce_top_.diff()->set_gpu_data(global_diff_map_[device]->mutable_gpu_data());


    std::vector<Blob<Dtype>*> reduce_tops;

    if(has_reducer_)
        reduce_tops.push_back(&reduce_top_);
    else
    {
        for(int i=0; i<bottom.size();i++)
            reduce_tops.push_back(bottom[i]);
    }

    // Forward  pass reduction
    if(has_reducer_)
    {
        conv_reduce_layer_->Forward(bottom, reduce_tops);

        if(has_bn_reduce_param_)
            bn_reduce_layer_->Forward(reduce_tops, reduce_tops);

        relu_reduce_layer_->Forward(reduce_tops, reduce_tops);
    }

    // Forward  pass grow
    {
        conv_grow_layer_->Forward(reduce_tops, top);

        if(has_bn_grow_param_)
            bn_grow_layer_->Forward(top, top);

        relu_grow_layer_->Forward(top, top);
    }


}

template <typename Dtype>
void BottleNeckLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>&
                                          bottom) {


    int device;
    cudaGetDevice(&device);

    //    if(this->phase_ == TEST)
    //    reduce_top_.data()->set_gpu_data(global_data_map_[device]->mutable_gpu_data());

    if(this->phase_ == TRAIN)
        reduce_top_.diff()->set_gpu_data(global_diff_map_[device]->mutable_gpu_data());

    // backward
    const std::vector<bool> conv_propagate_down(true);
    std::vector<bool> bn_propagate_down(true);
    std::vector<bool> relu_propagate_down(true);

    std::vector<Blob<Dtype>*> reduce_tops;

    if(has_reducer_)
        reduce_tops.push_back(&reduce_top_);
    else
    {
        for(int i=0; i<bottom.size();i++)
            reduce_tops.push_back(bottom[i]);
    }

    // backward  pass grow
    {
        relu_grow_layer_->Backward(top, relu_propagate_down, top);

        if(has_bn_grow_param_)
            bn_grow_layer_->Backward(top, bn_propagate_down, top);

        conv_grow_layer_->Backward(top, conv_propagate_down, reduce_tops);
    }

    // backward  pass reduction
    if(has_reducer_)
    {
        relu_reduce_layer_->Backward(reduce_tops, propagate_down, reduce_tops);

        if(has_bn_reduce_param_)
            bn_reduce_layer_->Backward(reduce_tops, propagate_down, reduce_tops);

        conv_reduce_layer_->Backward(reduce_tops, propagate_down, bottom);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(BottleNeckLayer);

}  // namespace caffe
