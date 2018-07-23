#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/layers/label_transform_layer.hpp"

namespace caffe {

template <typename Dtype>
void LabelTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
    InterpParameter interp_param = this->layer_param_.interp_param();
    pad_beg_ = interp_param.pad_beg();
    pad_end_ = interp_param.pad_end();
    CHECK_LE(pad_beg_, 0) << "Only supports non-pos padding (cropping) for now";
    CHECK_LE(pad_end_, 0) << "Only supports non-pos padding (cropping) for now";
}

template <typename Dtype>
void LabelTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {

    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_in_ = bottom[0]->height();
    width_in_ = bottom[0]->width();
    height_out_ = bottom[0]->shape(2);
    width_out_ = bottom[0]->shape(3);

    top[0]->Reshape(num_, channels_, height_out_, width_out_);
}

template <typename Dtype>
void LabelTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {


    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    for(int row = 0; row < height_in_; row++)
        for(int col = 0; col < width_in_;  col++)
        {
            // CONVERT ALL LABELS GREATER THEN ZERO TO ONE IN ORDER TO ACHIEVE CLASS AGNOSTIC LABELS
            if(bottom_data[row * width_in_ + col] > 0 && bottom_data[row * width_in_ + col] < 255 )
                top_data[row * width_out_ + col] = 1;
        }
}


template <typename Dtype>
void LabelTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;

}

#ifndef CPU_ONLY
template <typename Dtype>
void LabelTransformLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    for(int row = 0; row < height_in_; row++)
        for(int col = 0; col < width_in_;  col++)
        {
            // CONVERT ALL LABELS GREATER THEN ZERO TO ONE IN ORDER TO ACHIEVE CLASS AGNOSTIC LABELS
            if(bottom_data[row * width_in_ + col] > 0 && bottom_data[row * width_in_ + col] < 255 )
                top_data[row * width_out_ + col] = 1;
        }
}

template <typename Dtype>
void LabelTransformLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;

}
#endif

#ifdef CPU_ONLY
STUB_GPU(LabelTransformLayer);
#endif

INSTANTIATE_CLASS(LabelTransformLayer);
REGISTER_LAYER_CLASS(LabelTransform);

}  // namespace caffe
