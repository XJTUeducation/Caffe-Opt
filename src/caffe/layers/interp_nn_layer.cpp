#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/layers/interp_nn_layer.hpp"

namespace caffe {

template <typename Dtype>
void InterpNNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
    InterpParameter interp_param = this->layer_param_.interp_param();
    pad_beg_ = interp_param.pad_beg();
    pad_end_ = interp_param.pad_end();
    CHECK_LE(pad_beg_, 0) << "Only supports non-pos padding (cropping) for now";
    CHECK_LE(pad_end_, 0) << "Only supports non-pos padding (cropping) for now";
}

template <typename Dtype>
void InterpNNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {



    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_in_ = bottom[0]->height();
    width_in_ = bottom[0]->width();
    height_in_eff_ = height_in_ + pad_beg_ + pad_end_;
    width_in_eff_ = width_in_ + pad_beg_ + pad_end_;



    InterpParameter interp_param = this->layer_param_.interp_param();

    // TO RESIZE A BLOB EQUAL TO ANOTHER
    if(bottom.size() > 1)
    {
        height_out_ = bottom[1]->shape(2);
        width_out_ = bottom[1]->shape(3);
    } else if (interp_param.has_shrink_factor() &&
               !interp_param.has_zoom_factor()) {
        const int shrink_factor = interp_param.shrink_factor();
        CHECK_GE(shrink_factor, 1) << "Shrink factor must be positive";
        height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
        width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
    } else if (interp_param.has_zoom_factor() &&
               !interp_param.has_shrink_factor()) {
        const int zoom_factor = interp_param.zoom_factor();
        CHECK_GE(zoom_factor, 1) << "Zoom factor must be positive";
        //        height_out_ = height_in_eff_ + (height_in_eff_ - 1) * (zoom_factor - 1);
        //        width_out_ = width_in_eff_ + (width_in_eff_ - 1) * (zoom_factor - 1);
                height_out_ = height_in_eff_  * (zoom_factor );
                width_out_ = width_in_eff_  * (zoom_factor);
    } else if (interp_param.has_height() && interp_param.has_width()) {
        height_out_  = interp_param.height();
        width_out_  = interp_param.width();
    } else if (interp_param.has_shrink_factor() &&
               interp_param.has_zoom_factor()) {
        const int shrink_factor = interp_param.shrink_factor();
        const int zoom_factor = interp_param.zoom_factor();
        CHECK_GE(shrink_factor, 1) << "Shrink factor must be positive";
        CHECK_GE(zoom_factor, 1) << "Zoom factor must be positive";
        height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
        width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
        height_out_ = height_out_ + (height_out_ - 1) * (zoom_factor - 1);
        width_out_ = width_out_ + (width_out_ - 1) * (zoom_factor - 1);
    } else {
        LOG(FATAL);
    }
    CHECK_GT(height_in_eff_, 0) << "height should be positive";
    CHECK_GT(width_in_eff_, 0) << "width should be positive";
    CHECK_GT(height_out_, 0) << "height should be positive";
    CHECK_GT(width_out_, 0) << "width should be positive";
    top[0]->Reshape(num_, channels_, height_out_, width_out_);
}

template <typename Dtype>
void InterpNNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {


    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();


    float interp_factor_h = (float)height_in_ / height_out_ ;
    float interp_factor_w = (float)width_in_ / width_out_ ;

    int c = num_ * channels_; // due to contigous memory location

    int input_offset = height_in_ * width_in_;
    int output_offset = height_out_ * width_out_;

    for(int row = 0; row < height_out_; row++)
        for(int col = 0; col < width_out_;  col++)
        {
            int input_row = floor( row * interp_factor_h );
            int input_col = floor( col * interp_factor_w );

            const Dtype* input_data = bottom_data;
            Dtype* output_data = top_data;

            for(int channels = 0; channels < c; channels++)
            {
                output_data[row * width_out_ + col] = input_data[input_row * width_in_ + input_col];

                input_data += input_offset;
                output_data += output_offset;
            }
        }


}

template <typename Dtype>
void InterpNNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) { return; }
    caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* top_diff = top[0]->cpu_diff();

    float interp_factor_h = (float)height_in_ / height_out_ ;
    float interp_factor_w = (float)width_in_ / width_out_ ;

    int c = num_ * channels_; // due to contigous memory location

    int input_offset = height_in_ * width_in_;
    int output_offset = height_out_ * width_out_;

    for(int row = 0; row < height_out_; row++)
        for(int col = 0; col < width_out_;  col++)
        {
            int input_row = floor( row * interp_factor_h );
            int input_col = floor( col * interp_factor_w );

            Dtype* input_diff = bottom_diff;
            const Dtype* output_diff = top_diff;

            for(int channels = 0; channels < c; channels++)
            {
                input_diff[input_row * width_in_ + input_col] += output_diff[row * width_out_ + col];

                input_diff += input_offset;
                output_diff += output_offset;
            }
        }
}


#ifdef CPU_ONLY
STUB_GPU(InterpNNLayer);
#endif

INSTANTIATE_CLASS(InterpNNLayer);
REGISTER_LAYER_CLASS(InterpNN);

}  // namespace caffe
