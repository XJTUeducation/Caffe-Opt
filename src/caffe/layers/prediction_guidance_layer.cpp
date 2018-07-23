#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/layers/predition_guidance_layer.hpp"

namespace caffe {

template <typename Dtype>
void PredictionGuidanceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
    InterpParameter interp_param = this->layer_param_.interp_param();
    pad_beg_ = interp_param.pad_beg();
    pad_end_ = interp_param.pad_end();
    CHECK_LE(pad_beg_, 0) << "Only supports non-pos padding (cropping) for now";
    CHECK_LE(pad_end_, 0) << "Only supports non-pos padding (cropping) for now";
}

template <typename Dtype>
void PredictionGuidanceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
void PredictionGuidanceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {

    const Dtype* predictions = bottom[0]->cpu_data();
    const Dtype* gnd_truth = bottom[1]->cpu_data();

    Dtype* guided_gnd_truth = top[0]->mutable_cpu_data();

    for(int n_ = 0; n_ < num_; n_++)
    {

       const Dtype*  predictions_data = predictions + n_ * channels_ * height_in_ * height_out_;
       const Dtype*  gnd_truth_data = gnd_truth + n_ * 1 * height_in_ * height_out_;
        Dtype*  guided_gnd_truth_data = guided_gnd_truth + n_ * 1 * height_in_ * height_out_;

        for(int row = 0; row < height_in_; row++)
            for(int col = 0; col < width_in_;  col++)
            {
                const Dtype* predition_pixel = predictions_data +  row * width_in_ + col;

                int index = 0;
                float max = (float)INT_MIN;

                for(int c_ = 0; c_ < channels_; c_++)
                {
                    const Dtype* predition_pixel_c = predition_pixel +  c_ * height_in_ * width_in_;

                    if(predition_pixel_c[0] > max)
                    {
                        max = predition_pixel_c[0];
                        index = c_;
                    }
                }

                const Dtype*  gnd_truth_pixel = gnd_truth_data + row * width_in_ + col;
                Dtype*  guided_gnd_truth_pixel = guided_gnd_truth_data + row * width_in_ + col;

                if(gnd_truth_pixel[0] == index)
                    guided_gnd_truth_pixel[0] = 255;
                else
                    guided_gnd_truth_pixel[0] = gnd_truth_pixel[0];

            }
    }

}


template <typename Dtype>
void PredictionGuidanceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;

}

#ifndef CPU_ONLY
template <typename Dtype>
void PredictionGuidanceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {

    const Dtype* predictions = bottom[0]->cpu_data();
    const Dtype* gnd_truth = bottom[1]->cpu_data();

    Dtype* guided_gnd_truth = top[0]->mutable_cpu_data();

    for(int n_ = 0; n_ < num_; n_++)
    {

       const Dtype*  predictions_data = predictions + n_ * channels_ * height_in_ * height_out_;
       const Dtype*  gnd_truth_data = gnd_truth + n_ * 1 * height_in_ * height_out_;
        Dtype*  guided_gnd_truth_data = guided_gnd_truth + n_ * 1 * height_in_ * height_out_;

        for(int row = 0; row < height_in_; row++)
            for(int col = 0; col < width_in_;  col++)
            {
                const Dtype* predition_pixel = predictions_data +  row * width_in_ + col;

                int index = 0;
                float max = (float)INT_MIN;

                for(int c_ = 0; c_ < channels_; c_++)
                {
                    const Dtype* predition_pixel_c = predition_pixel +  c_ * height_in_ * width_in_;

                    if(predition_pixel_c[0] > max)
                    {
                        max = predition_pixel_c[0];
                        index = c_;
                    }
                }

                const Dtype*  gnd_truth_pixel = gnd_truth_data + row * width_in_ + col;
                Dtype*  guided_gnd_truth_pixel = guided_gnd_truth_data + row * width_in_ + col;

                if(gnd_truth_pixel[0] == index)
                    guided_gnd_truth_pixel[0] = 255;
                else
                    guided_gnd_truth_pixel[0] = gnd_truth_pixel[0];

            }
    }

    // A CALL TO COPY THE DATA FROM CPU TO GPU
    top[0]->gpu_data();
}

template <typename Dtype>
void PredictionGuidanceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;

}
#endif

#ifdef CPU_ONLY
STUB_GPU(PredictionGuidanceLayer);
#endif

INSTANTIATE_CLASS(PredictionGuidanceLayer);
REGISTER_LAYER_CLASS(PredictionGuidance);

}  // namespace caffe
