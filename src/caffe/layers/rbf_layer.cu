#include <cmath>
#include <vector>

#include "caffe/layers/rbf_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RBFForward(const int n_kernels, const int n_channels,
                           const int height, const int width,
                           const Dtype* bottom_data, const Dtype* mu, const Dtype* var, Dtype* top_data)
{
#define RBF_EPSILON 0.0000001

    // run once per batch index
    const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_index < n_kernels)
    {
        const int c = thread_index / (height * width);
        const int h = (thread_index /  width) % height;
        const int w = thread_index %  width;

        const Dtype mu_= mu[c];
        const Dtype var_= var[c];

        const int spatial_offset = ( c * height + h ) * width + w;

        const Dtype bottom_data_ = bottom_data[spatial_offset];
        Dtype top_data_ = top_data[spatial_offset];

        const Dtype x_minus_mu = bottom_data_ - mu_;

        // y = exp(inf) = 0 --> in the backward pass log(y) = -inf
        // to prevent that y ->  y + epsilon and take max(y,1) which ensure that rbf output is >0 and <1

        top_data_ = min(exp(-((x_minus_mu * x_minus_mu) / (2* var_))) + RBF_EPSILON, 1.0);
    }


}

template <typename Dtype>
void RBFLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top)
{


    const Dtype* mu = this->blobs_[0]->gpu_data();
    const Dtype* var = this->blobs_[1]->gpu_data();

    const int num = bottom[0]->shape(0);
    const int n_channels = bottom[0]->shape(1);
    const int height = bottom[0]->shape(2);
    const int width = bottom[0]->shape(3);

    const int n_kernels = n_channels * height * width;

    for(int i=0; i< n_channels; i++)
    {
        const Dtype* bottom_data_ = bottom[0]->gpu_data() +  bottom[0]->offset(i,0,0,0);
        Dtype* top_data = top[0]->mutable_gpu_data() + top[0]->offset(i,0,0,0);

        RBFForward<<< CAFFE_GET_BLOCKS(n_kernels), CAFFE_CUDA_NUM_THREADS >>>(
                                                                                n_kernels,
                                                                                n_channels,
                                                                                height,
                                                                                width,
                                                                                bottom_data_,
                                                                                mu,
                                                                                var,
                                                                                top_data);

        CUDA_POST_KERNEL_CHECK;
    }



}

template <typename Dtype>
__global__ void RBFBackward(const int n_kernels, const int n_channels,
                            const int height, const int width,
                            const Dtype* top_data, const Dtype* top_diff,
                            const Dtype* var, Dtype* bottom_diff )
{
    // run once per batch index
    const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_index < n_kernels)
    {
        const int c = thread_index / (height * width);
        const int h = (thread_index /  width) % height;
        const int w = thread_index %  width;

        const Dtype var_= var[c];

        const int spatial_offset = ( c * height + h ) * width + w;

        const Dtype top_data_= top_data[spatial_offset];
        const Dtype top_diff_= top_diff[spatial_offset];

        Dtype bottom_diff_= bottom_diff[spatial_offset];

        bottom_diff_ =  top_diff_ * ( - top_data_ * sqrt(-((2 * log(top_data_)) / var_)) );
    }


}



template <typename Dtype>
void RBFLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<Dtype>*>& bottom)
{
    const int num = bottom[0]->shape(0);
    const int n_channels = bottom[0]->shape(1);
    const int height = bottom[0]->shape(2);
    const int width = bottom[0]->shape(3);

    const int n_kernels = n_channels * height * width;


    // Backward diff
    if(propagate_down[0])
    {
        const Dtype* var = this->blobs_[1]->gpu_data();


        for(int i=0; i< n_channels; i++)
        {
            Dtype* bottom_diff_ = bottom[0]->mutable_gpu_diff() +  bottom[0]->offset(i,0,0,0);
            const Dtype* top_data = top[0]->gpu_data() + top[0]->offset(i,0,0,0);
            const Dtype* top_diff = top[0]->gpu_diff() + top[0]->offset(i,0,0,0);

            RBFBackward<<< CAFFE_GET_BLOCKS(n_kernels), CAFFE_CUDA_NUM_THREADS >>>(
                                                                                     n_kernels,
                                                                                     n_channels,
                                                                                     height,
                                                                                     width,
                                                                                     top_data,
                                                                                     top_diff,
                                                                                     var,
                                                                                     bottom_diff_);

            CUDA_POST_KERNEL_CHECK;
        }
    }


    // Parameter gradient calculation
    if(this->param_propagate_down(0))
    {
        // using the memory of bottom diff  multipied with -1 will give gradient of mu
        // bcz of (x-mu) operation.

        Dtype* mu_diff = this->blobs_[0]->mutable_gpu_diff();
        const Dtype* bottom_diff = bottom[0]->gpu_diff();

        Dtype* spatial_diff = spatial_diff_.mutable_gpu_diff();

        // grad mu accumulation over spatial dims
        caffe_gpu_gemv(CblasNoTrans, num * n_channels, height * width, Dtype(1), bottom_diff, spatial_sum_multiplier_.gpu_data(), Dtype(0), spatial_diff);

        // the -1 is to get gradient w.r.t. mu
        // grad mu accumulation over batch
        caffe_gpu_gemv(CblasTrans, num, n_channels, Dtype(-1), spatial_diff, batch_sum_multiplier_.gpu_data(), Dtype(0), mu_diff);

    }




}

INSTANTIATE_LAYER_GPU_FUNCS(RBFLayer);


}  // namespace caffe
