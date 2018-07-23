#include <cmath>
#include <vector>

#include "caffe/layers/max_activations_grouping_layer.hpp"

namespace caffe {

const int REDUCTION_MINI_ARRAY_SIZE = 16;  // aligned TO the power of 2

template <typename Dtype>
__global__ void MaxActivationsGroupingLayer_arg_max_forward(const int n_kernels, const int n_channels,
                                                            const int height, const int width,
                                                            Dtype* bottom_data, Dtype* output_max, Dtype* output_args_max,
                                                            int* max_indices) {

    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    if( thread_index < n_kernels)
    {
        // per array num threads required
        int n_threads_required = n_kernels / (height * width);

        const int linear_spatial_index = thread_index / n_threads_required;

        const int h = linear_spatial_index / width;
        const int w = linear_spatial_index % width;

        const int inc = height * width;

        // 0 based array index for each array
        int index = thread_index % n_threads_required;

        const int start_element = index * REDUCTION_MINI_ARRAY_SIZE;
        const int end_element = start_element + REDUCTION_MINI_ARRAY_SIZE - 1;

        int n_elements_next_cycle = n_channels;

        // array start and end data pointer
        Dtype* data_start_array = (Dtype*)bottom_data + h * width + w;
        Dtype* data_end_array = data_start_array + inc * (n_channels - 1);

        // mini array start data pointer
        //        const Dtype* data_start_mini_array = data_start_array + start_element * inc;

        Dtype max ;
        int arg_max_;

        // cz in first pass, no indices will be there.
        // Starting from the last location in array, Input is modified to accomodate
        // maximum index of max element index of mini array

        bool is_first_pass = true;

        while(1)
        {

            if(index < n_threads_required)
            {
                // Due to reduction, effective array size will be reduced by REDUCTION_MINI_ARRAY_SIZE
                // hence reduced array length is updated

                // check whether end element index excedes the new array length
                int end_element_ =  end_element > n_elements_next_cycle - 1 ? n_elements_next_cycle - 1 : end_element;

                max = data_start_array[start_element];
                arg_max_ = is_first_pass ? start_element : (data_end_array - start_element * inc)[0];

                for(int i=start_element+1; i <=end_element_; i++)
                {
                    const int data_start_ = data_start_array[i * inc];
                    if(max < data_start_)
                    {
                        max = data_start_;
                        arg_max_ = is_first_pass ? i : (data_end_array - i * inc)[0];
                    }
                }

                __syncthreads();

                data_start_array[index * inc] = max;
                (data_end_array - index * inc)[0] = arg_max_;

                __syncthreads();

                if(n_threads_required == 1)
                    break;
            }
            else
                break;

            n_elements_next_cycle = n_threads_required;
            n_threads_required = (n_threads_required - 1) / REDUCTION_MINI_ARRAY_SIZE + 1;

            if(is_first_pass)
                is_first_pass = false;
        }

        // only thread index zero have authority
        // to update the results in the output array

        if(index == 0)
        {
            output_max[h * width + w] = data_start_array[0];
//            output_args_max[h * width + w] = data_end_array[0];
            output_args_max[h * width + w] = (float)( data_end_array[0] + 1)  / n_channels;
            max_indices[h * width + w] = data_end_array[0];
        }
    }
}



template <typename Dtype>
__global__ void MaxActivationsGroupingLayer_arg_max_backward(const int n_kernels, const int n_channels,
                                                             const int height, const int width,
                                                             const Dtype* top_diff,
                                                             const Dtype* output_args_max,
                                                             const int* max_indices,
                                                             Dtype* bottom_diff) {

    int thread_index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;

    if( thread_index < n_kernels)
    {
        const int h = thread_index / width;
        const int w = thread_index % width;

        const int spatial_offset = h * width + w;

//        int max_activation_channel_index = (int)output_args_max[spatial_offset] * n_channels - 1;
        int max_activation_channel_index = max_indices[spatial_offset];

//        int max_activation_channel_index = (int)output_args_max[spatial_offset];

        //        if(max_activation_channel_index < 0)
        //            max_activation_channel_index = 0;

        //        if(max_activation_channel_index > 1)
        //            max_activation_channel_index = 1;

        bottom_diff[max_activation_channel_index * height * width + spatial_offset]   =  top_diff[spatial_offset];
    }
}

template <typename Dtype>
void MaxActivationsGroupingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                     const vector<Blob<Dtype>*>& top) {

    const int num = bottom[0]->shape(0);
    const int n_channels = bottom[0]->shape(1);
    const int height = bottom[0]->shape(2);
    const int width = bottom[0]->shape(3);

    const int n_threads_per_array = (n_channels - 1) / REDUCTION_MINI_ARRAY_SIZE + 1;

    const int n_kernels =  n_threads_per_array * height * width;

    int TPB = 0;

    if(n_threads_per_array <= CAFFE_CUDA_NUM_THREADS)
        TPB = ( CAFFE_CUDA_NUM_THREADS / n_threads_per_array ) * n_threads_per_array;
    else
        TPB = n_threads_per_array;

    //    TPB = CAFFE_CUDA_NUM_THREADS;

    dim3 grid_dim((n_kernels -1) / TPB + 1);
    dim3 block_dim(TPB);


    for(int i = 0; i < num; i++)
    {
        int offset = bottom[0]->offset(i,0,0,0);
        int top_offset = top[0]->offset(i,0,0,0);

        Dtype* bottom_data = bottom[0]->mutable_gpu_data() + offset;

        Dtype* output_max = top[0]->mutable_gpu_data() + top_offset;
        Dtype* output_arg_max = NULL;

        if(top.size() ==2)
        output_arg_max = top[1]->mutable_gpu_data() + top_offset;

        //not using to prevent memeory duplication
        int* max_indices = max_indices_.mutable_gpu_data() + top_offset;

        MaxActivationsGroupingLayer_arg_max_forward
                <<< grid_dim, block_dim >>>(
                                              n_kernels,
                                              n_channels,
                                              height, width,
                                              bottom_data,
                                              output_max,
                                              output_arg_max,
                                              max_indices);

        CUDA_POST_KERNEL_CHECK;
    }
}


template <typename Dtype>
void MaxActivationsGroupingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                      const vector<bool>& propagate_down,
                                                      const vector<Blob<Dtype>*>& bottom) {

    if(propagate_down[0])
    {
        const int num = bottom[0]->shape(0);
        const int n_channels = bottom[0]->shape(1);
        const int height = bottom[0]->shape(2);
        const int width = bottom[0]->shape(3);

        const int n_kernels = height * width;

        // SETTING ALL DIFF TO ZERO
        // HENCE WE WILL ONLY NEED TO LAUNCH HxW kernels, not CxHxW
        cudaMemset(bottom[0]->mutable_gpu_diff(), 0, bottom[0]->count()*sizeof(Dtype));

        for(int i = 0; i < num; i++)
        {
            int offset = bottom[0]->offset(i,0,0,0);
            int top_offset = top[0]->offset(i,0,0,0);

            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff() + offset;

            const Dtype* top_diff = top[0]->gpu_diff() + top_offset;

            Dtype* output_arg_max = NULL;

            if(top.size() ==2)
            output_arg_max = top[1]->mutable_gpu_data() + top_offset;

            //not using to prevent memeory duplication
            int* max_indices = max_indices_.mutable_gpu_data() + top_offset;

            MaxActivationsGroupingLayer_arg_max_backward
                    <<< CAFFE_GET_BLOCKS(n_kernels), CAFFE_CUDA_NUM_THREADS >>>(
                                                                                  n_kernels,
                                                                                  n_channels,
                                                                                  height, width,
                                                                                  top_diff,
                                                                                  output_arg_max,
                                                                                  max_indices,
                                                                                  bottom_diff);
            CUDA_POST_KERNEL_CHECK;
        }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(MaxActivationsGroupingLayer);


}  // namespace caffe
