#include <cmath>
#include <vector>

#include "caffe/layers/max_activations_grouping_layer.hpp"

namespace caffe {


template <typename Dtype>
void MaxActivationsGroupingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {


    const int num = bottom[0]->shape(0);
    const int n_channels = bottom[0]->shape(1);
    const int height = bottom[0]->shape(2);
    const int width = bottom[0]->shape(3);

    top[0]->Reshape(num, 1, height, width);

    if(top.size() > 1)
        top[1]->Reshape(num, 1, height, width);

    max_indices_.Reshape(num, 1, height, width);
}


template <typename Dtype>
void MaxActivationsGroupingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void MaxActivationsGroupingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(MaxActivationsGroupingLayer);
#endif

INSTANTIATE_CLASS(MaxActivationsGroupingLayer);
REGISTER_LAYER_CLASS(MaxActivationsGrouping);


}  // namespace caffe
