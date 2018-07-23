#include <algorithm>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/common.cuh"
#include "caffe/layers/interp_nn_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void InterpNN_forward_gpu(int n_threads, int c, int height_in_, int width_in_,
                                     int height_out_, int width_out_,
                                     const Dtype* bottom_data, Dtype* top_data)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < n_threads)
    {
        float interp_factor_h = (float)height_in_ / height_out_ ;
        float interp_factor_w = (float)width_in_ / width_out_ ;

        int input_offset = height_in_ * width_in_;
        int output_offset = height_out_ * width_out_;

        int row = index / width_out_;
        int col =  index - row * width_out_;

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
__global__ void InterpNN_backward_gpu(int n_threads, int c, int height_in_, int width_in_,
                                      int height_out_, int width_out_,
                                      Dtype* bottom_diff, const Dtype* top_diff)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < n_threads)
    {
        float interp_factor_h = (float)height_in_ / height_out_ ;
        float interp_factor_w = (float)width_in_ / width_out_ ;

        int input_offset = height_in_ * width_in_;
        int output_offset = height_out_ * width_out_;

        int row = index / width_out_;
        int col =  index - row * width_out_;

        int input_row = floor( row * interp_factor_h );
        int input_col = floor( col * interp_factor_w );

        Dtype* input_diff = bottom_diff;
        const Dtype* output_diff = top_diff;

        for(int channels = 0; channels < c; channels++)
        {
            atomicAdd(&input_diff[input_row * width_in_ + input_col], output_diff[row * width_out_ + col]);

            input_diff += input_offset;
            output_diff += output_offset;
        }
    }
}


template <typename Dtype>
void InterpNNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    int c = num_ * channels_; // due to contigous memory location
    int n_threads = height_out_ * width_out_;

    InterpNN_forward_gpu<Dtype><<< CAFFE_GET_BLOCKS(n_threads),CAFFE_CUDA_NUM_THREADS >>>(n_threads, c,
                                  height_in_, width_in_, height_out_, width_out_, bottom_data, top_data);
    CUDA_POST_KERNEL_CHECK;



}

template <typename Dtype>
void InterpNNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) { return; }
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());

    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* top_diff = top[0]->gpu_diff();

    int c = num_ * channels_; // due to contigous memory location
    int n_threads = height_out_ * width_out_;

    InterpNN_backward_gpu<Dtype><<< CAFFE_GET_BLOCKS(n_threads),CAFFE_CUDA_NUM_THREADS >>>(n_threads, c,
                                  height_in_, width_in_, height_out_, width_out_, bottom_diff, top_diff);
    CUDA_POST_KERNEL_CHECK;

}


INSTANTIATE_LAYER_GPU_FUNCS(InterpNNLayer);


}  // namespace caffe
