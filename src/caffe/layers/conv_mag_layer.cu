#include <vector>

#include "caffe/layers/conv_mag_layer.hpp"

namespace caffe {



template <typename Dtype>
__global__ void set_max_value_in_the_bottom_data(const int n_kernels,
                                                 const int n_channels,
                                                 const int height, const int width,
                                                 const Dtype* max, const Dtype* arg_max,
                                                 Dtype* bottom_data)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_index < n_kernels)
    {
        const int n = n_kernels / (height * width);

        const int h = (thread_index / width) % height;
        const int w = thread_index % width;

        const int spatial_offset = h * width + w;

        const int channel_id = arg_max[ n * height * width + spatial_offset];

        bottom_data[ n * n_channels * height * width + channel_id * height * width + spatial_offset] =  max[n * height * width + spatial_offset];
    }

}


template <typename Dtype>
__global__ void set_max_value_in_the_bottom_diff(const int n_kernels,
                                                 const int num, const int n_channels,
                                                 const int height, const int width,
                                                 const Dtype* arg_max,
                                                 Dtype* bottom_diff)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_index < n_kernels)
    {
        const int n = n_kernels / (n_channels * height * width);
        const int c = (n_kernels / (height * width)) % n_channels;
        const int h = (thread_index / width) % height;
        const int w = thread_index % width;

        const int spatial_offset = h * width + w;
        const int max_channel_id = arg_max[ n * height * width + spatial_offset];

        if(max_channel_id != c)
            bottom_diff[ (n * n_channels + c) * height * width + spatial_offset] = 0;
    }
}

template <typename Dtype>
void ConvolutionMAGLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
#define MAG

#ifdef MAG

    // -  Forward pass the MAG Layer


//    std::cout << "\n\n bottom_data \n\n\n";

//    for(int i=0; i < bottom[0]->count();i++)
//    std::cout << i << "  " << bottom[0]->cpu_data()[i] << "    ";
//    std::cout << "\n\n";

    this->mag_layer_->Forward(bottom, this->mag_tops_);

    //        std::cout << "\n\nmax \n\n\n";

    //        for(int i=0; i < this->mag_tops_[0]->count();i++)
    //        std::cout << this->mag_tops_[0]->cpu_data()[i] << "   ";
    //        std::cout << "\n\n";

    std::cout << "\n\n argmax \n\n\n";



    // - Load the max indices weights into effective weights

    {
        // write the cuda kernel
        cudaMemset(bottom[0]->mutable_gpu_data(),0,bottom[0]->count()* sizeof(Dtype));

        const int num = bottom[0]->shape(0);
        const int n_channels = bottom[0]->shape(1);
        const int height = bottom[0]->shape(2);
        const int width = bottom[0]->shape(3);

        int n_kernels = num * height * width;
        const Dtype* max = this->mag_tops_[0]->gpu_data();
        const Dtype* arg_max = this->mag_tops_[1]->gpu_data();
        Dtype* bottom_data = bottom[0]->mutable_gpu_data();

        std::cout << "n kernels = " << n_kernels << "\n";

        set_max_value_in_the_bottom_data<<<CAFFE_GET_BLOCKS(n_kernels), CAFFE_CUDA_NUM_THREADS>>>(
                                                                                                    n_kernels,
                                                                                                    n_channels,
                                                                                                    height, width,
                                                                                                    max,
                                                                                                    arg_max,
                                                                                                    bottom_data);

        CUDA_POST_KERNEL_CHECK;

//        static bool once = false;

//        if(!once)
//        {
//            for(int i=0; i < this->mag_tops_[0]->count();i++)
//           if(this->mag_tops_[1]->cpu_data()[i] >= bottom[0]->shape(1))
//                std::cout << this->mag_tops_[1]->cpu_data()[i] << "   ";
//            std::cout << "\n\n";
//        once = true;
//        }



    }

#endif
    //----------------------------------------------------------------------

    const Dtype* weight = this->blobs_[0]->gpu_data();
    for (int i = 0; i < bottom.size(); ++i) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* top_data = top[i]->mutable_gpu_data();
        for (int n = 0; n < this->num_; ++n) {
            this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                                   top_data + n * this->top_dim_);
            if (this->bias_term_) {
                const Dtype* bias = this->blobs_[1]->gpu_data();
                this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
            }
        }
    }


}

template <typename Dtype>
void ConvolutionMAGLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>&
                                              bottom) {



    const Dtype* weight = this->blobs_[0]->gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    for (int i = 0; i < top.size(); ++i) {
        const Dtype* top_diff = top[i]->gpu_diff();
        // Bias gradient, if necessary.
        if (this->bias_term_ && this->param_propagate_down_[1]) {
            Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
            for (int n = 0; n < this->num_; ++n) {
                this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
            }
        }
        if (this->param_propagate_down_[0] || propagate_down[i]) {
            const Dtype* bottom_data = bottom[i]->gpu_data();
            for (int n = 0; n < this->num_; ++n) {
                // gradient w.r.t. weight. Note that we will accumulate diffs.
                if (this->param_propagate_down_[0]) {
                    this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
                                          top_diff + n * this->top_dim_, weight_diff);
                }
                // gradient w.r.t. bottom data, if necessary.
                if (propagate_down[i]) {
                    bool is_bottom_shared = bottom[i]->is_shared();

                    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();

                    this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
                                            bottom_diff + n * this->bottom_dim_, is_bottom_shared);



                }
            }
        }
    }

#ifdef MAG

    // NO NEED OF MAG BACKWARD PASS
    // BCZ WE KNOW THE MAXIMUM INDICES
    {
        const int num = bottom[0]->shape(0);
        const int n_channels = bottom[0]->shape(1);
        const int height = bottom[0]->shape(2);
        const int width = bottom[0]->shape(3);

        const Dtype* arg_max = this->mag_tops_[1]->gpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

        int n_kernels = num * n_channels * height * width;

        set_max_value_in_the_bottom_diff<<<CAFFE_GET_BLOCKS(n_kernels), CAFFE_CUDA_NUM_THREADS>>>(
                                                                                                    n_kernels,
                                                                                                    num, n_channels,
                                                                                                    height, width,
                                                                                                    arg_max,
                                                                                                    bottom_diff);

        CUDA_POST_KERNEL_CHECK;

    }

#endif


}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionMAGLayer);

}  // namespace caffe
