#include <vector>

#include "caffe/layers/strided_decim_interp.hpp"

namespace caffe {


enum strided_decim_interp_op
{
    FWD_DECIM,
    BCKWD_DECIM,
    FWD_INTERP,
    BCKWD_INTERP,
    FWD_DECIM_CONCAT,
    BCKWD_DECIM_CONCAT,
    FWD_INTERP_CONCAT,
    BCKWD_INTERP_CONCAT,
};

template <typename Dtype>
__global__ void strided_decim_interp_fwd_bckwd_kernel(const int n_kernels,
                                                      const int n, const int n_channels, const int height, const int width,
                                                      const int top_height, const int top_width,
                                                      const int dilation,
                                                      const int op_type,
                                                      Dtype* bottom,
                                                      Dtype* top)
{

    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    if( thread_index < n_kernels )
    {
        int channel_id = thread_index / (height * width );
        int h_bottom =  (thread_index / width) % height;
        int w_bottom =  thread_index % width;

        int top_batch_index_h = h_bottom % dilation;
        int top_batch_index_w = w_bottom % dilation;

        int top_n_batch = top_batch_index_h * dilation + top_batch_index_w;

        int h_top = h_bottom / dilation;
        int w_top = w_bottom / dilation;

        //      const int bottom_offset = bottom_data + n_ * n_channels * height * width
        //                                            + channel_id * height * width +
        //                                            + h_bottom * width
        //                                            + w_bottom;

        switch(op_type)
        {
        case FWD_DECIM:
        {
            const Dtype* bottom_  = bottom + (channel_id * height + h_bottom ) * width + w_bottom;
            Dtype* top_  =    top + ((top_n_batch * n_channels + channel_id) * top_height + h_top ) * top_width + w_top;
            top_[0] = bottom_[0];
        }
            break;

        case BCKWD_DECIM:
        {
            Dtype* bottom_  = bottom + (channel_id * height + h_bottom ) * width + w_bottom;
            const Dtype* top_  =    top + ((top_n_batch * n_channels + channel_id) * top_height + h_top ) * top_width + w_top;
            bottom_[0] = top_[0];
        }
            break;

        case FWD_INTERP:
        {
            const Dtype* bottom_  =    bottom + ((top_n_batch * n_channels + channel_id) * top_height + h_top ) * top_width + w_top;
            Dtype* top_  = top + (channel_id * height + h_bottom ) * width + w_bottom;
            top_[0] = bottom_[0];
        }
            break;

        case BCKWD_INTERP:
        {
            Dtype* bottom_  =    bottom + ((top_n_batch * n_channels + channel_id) * top_height + h_top ) * top_width + w_top;
            const Dtype* top_  = top + (channel_id * height + h_bottom ) * width + w_bottom;
            bottom_[0] = top_[0];
        }
            break;


        }
    }
}



template <typename Dtype>
__global__ void strided_decim_interp_concat_fwd_bckwd_kernel(const int n_kernels,
                                                             const int n, const int n_channels, const int height, const int width,
                                                             const int top_height, const int top_width,
                                                             const int dilation,
                                                             const int op_type,
                                                             Dtype* bottom,
                                                             Dtype* top)
{

    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    if( thread_index < n_kernels )
    {
        int channel_id = thread_index / (height * width );
        int h_bottom =  (thread_index / width) % height;
        int w_bottom =  thread_index % width;

        int top_channel_index_h = h_bottom % dilation;
        int top_channel_index_w = w_bottom % dilation;

        int top_n_channel = top_channel_index_h * dilation + top_channel_index_w;

        int h_top = h_bottom / dilation;
        int w_top = w_bottom / dilation;

        //      const int bottom_offset = bottom_data + n_ * n_channels * height * width
        //                                            + channel_id * height * width +
        //                                            + h_bottom * width
        //                                            + w_bottom;

        switch(op_type)
        {
        case FWD_DECIM_CONCAT:
        {
            const Dtype* bottom_  = bottom + (channel_id * height + h_bottom ) * width + w_bottom;
            Dtype* top_  =    top + (top_n_channel * top_height + h_top ) * top_width + w_top;
            top_[0] = bottom_[0];
        }
            break;

        case BCKWD_DECIM_CONCAT:
        {
            Dtype* bottom_  = bottom + (channel_id * height + h_bottom ) * width + w_bottom;
            const Dtype* top_  =  top + (top_n_channel * top_height + h_top ) * top_width + w_top;
            bottom_[0] = top_[0];
        }
            break;

        case FWD_INTERP_CONCAT:
        {
            const Dtype* bottom_  =    bottom + (top_n_channel * top_height + h_top ) * top_width + w_top;;
            Dtype* top_  = top + (channel_id * height + h_bottom ) * width + w_bottom;
            top_[0] = bottom_[0];
        }
            break;

        case BCKWD_INTERP_CONCAT:
        {
            Dtype* bottom_  =    bottom + (top_n_channel * top_height + h_top ) * top_width + w_top;;
            const Dtype* top_  = top + (channel_id * height + h_bottom ) * width + w_bottom;
            bottom_[0] = top_[0];
        }
            break;


        }
    }
}


template <typename Dtype>
void StridedDecimInterpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {


    int n;
    int n_channels;
    int height;
    int width;

    int top_height;
    int top_width;

    int op_type;

    switch(strided_op_)
    {
    case StridedDecimInterpParameter_STRIDED_OP_DECIM :
    {
        op_type = FWD_DECIM;

        n = bottom[0]->shape(0);
        n_channels = bottom[0]->shape(1);
        height = bottom[0]->shape(2);
        width = bottom[0]->shape(3);

        top_height = top[0]->shape(2);
        top_width = top[0]->shape(3);

    }
        break;
    case StridedDecimInterpParameter_STRIDED_OP_INTERP:
    {
        op_type = FWD_INTERP;

        n = top[0]->shape(0);
        n_channels = top[0]->shape(1);
        height = top[0]->shape(2);
        width = top[0]->shape(3);

        top_height = bottom[0]->shape(2);
        top_width = bottom[0]->shape(3);

    }
        break;


    case StridedDecimInterpParameter_STRIDED_OP_DECIM_CONCAT :
    {
        op_type = FWD_DECIM_CONCAT;

        n = bottom[0]->shape(0);
        n_channels = bottom[0]->shape(1);
        height = bottom[0]->shape(2);
        width = bottom[0]->shape(3);

        top_height = top[0]->shape(2);
        top_width = top[0]->shape(3);

    }
        break;

    case StridedDecimInterpParameter_STRIDED_OP_INTERP_CONCAT:
    {
        op_type = FWD_INTERP_CONCAT;

        n = top[0]->shape(0);
        n_channels = top[0]->shape(1);
        height = top[0]->shape(2);
        width = top[0]->shape(3);

        top_height = bottom[0]->shape(2);
        top_width = bottom[0]->shape(3);
    }
        break;

    }


    for(int i=0; i< n;i++)
    {

        const int n_kernels = n_channels * height * width;

        Dtype* bottom_data_ = bottom[0]->mutable_gpu_data() + bottom[0]->offset(i,0,0,0);
        Dtype* top_data_ = temp_buffer_.mutable_gpu_data() + bottom[0]->offset(i,0,0,0);

        if( op_type == FWD_DECIM || op_type == FWD_INTERP)
        {
            strided_decim_interp_fwd_bckwd_kernel<<<CAFFE_GET_BLOCKS(n_kernels), CAFFE_CUDA_NUM_THREADS>>>
                                                                                                         (n_kernels,
                                                                                                          n, n_channels, height, width,
                                                                                                          top_height, top_width,
                                                                                                          dilation_,
                                                                                                          op_type,
                                                                                                          bottom_data_,
                                                                                                          top_data_);
        }
        else if(op_type == FWD_DECIM_CONCAT || op_type == FWD_INTERP_CONCAT)
        {
            strided_decim_interp_concat_fwd_bckwd_kernel<<<CAFFE_GET_BLOCKS(n_kernels), CAFFE_CUDA_NUM_THREADS>>>
                                                                                                         (n_kernels,
                                                                                                          n, n_channels, height, width,
                                                                                                          top_height, top_width,
                                                                                                          dilation_,
                                                                                                          op_type,
                                                                                                          bottom_data_,
                                                                                                          top_data_);
        }


        CUDA_POST_KERNEL_CHECK;

    }

    cudaMemcpy(bottom[0]->mutable_gpu_data(), temp_buffer_.mutable_gpu_data(), bottom[0]->count() * sizeof(Dtype), cudaMemcpyDeviceToDevice);


//    std::cout << (u_int64_t)bottom[0]->mutable_gpu_data() << "  " << (u_int64_t)top[0]->mutable_gpu_data() << "\n";


    //    std::cout << "****************** OUT PUT ******************************************\n\n";

    //    int count = 0;

    //    //    std::cout << "conv bottom pointer = " << (int64_t)(bottom[0]->gpu_data()) << "   " <<  (int64_t)(bottom[0]->cpu_data()) << "\n";
    //    //    std::cout << "conv top pointer = " << (int64_t)(top[0]->gpu_data()) << "   " <<  (int64_t)(top[0]->cpu_data()) << "\n";

    //    for(int l=0;l<top[0]->shape(0);l++)
    //    {
    //        for(int i=0;i<top[0]->shape(1);i++)
    //        {
    //            for(int j=0;j<top[0]->shape(2);j++)
    //            {
    //                for(int k=0;k<top[0]->shape(3);k++)
    //                {
    //                    std::cout << top[0]->cpu_data()[count++] << " ";
    //                    //                top[0]->mutable_cpu_data()[count++] = 2;

    //                }

    //                std::cout << "\n";
    //            }
    //            std::cout << "\n\n";
    //        }
    //        std::cout << "\n\nNEXT NUM \n\n";

    //    }

    //    std::cout << "****************** OUT PUT ENDS ******************************************\n\n\n\n\n";


}

template <typename Dtype>
void StridedDecimInterpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>&
                                                  bottom) {

    //    for(int i=0;i< top[0]->count();i++)
    //        top[0]->mutable_cpu_diff()[i] = i;

    if(propagate_down[0])
    {

        int n;
        int n_channels;
        int height;
        int width;

        int top_height;
        int top_width;

        int op_type;

        switch(strided_op_)
        {
        case StridedDecimInterpParameter_STRIDED_OP_DECIM :
        {
            op_type = BCKWD_DECIM;

            n = bottom[0]->shape(0);
            n_channels = bottom[0]->shape(1);
            height = bottom[0]->shape(2);
            width = bottom[0]->shape(3);

            top_height = top[0]->shape(2);
            top_width = top[0]->shape(3);

        }
            break;
        case StridedDecimInterpParameter_STRIDED_OP_INTERP:
        {
            op_type = BCKWD_INTERP;

            n = top[0]->shape(0);
            n_channels = top[0]->shape(1);
            height = top[0]->shape(2);
            width = top[0]->shape(3);

            top_height = bottom[0]->shape(2);
            top_width = bottom[0]->shape(3);

        }
            break;

        case StridedDecimInterpParameter_STRIDED_OP_DECIM_CONCAT :
        {
            op_type = BCKWD_DECIM_CONCAT;

            n = bottom[0]->shape(0);
            n_channels = bottom[0]->shape(1);
            height = bottom[0]->shape(2);
            width = bottom[0]->shape(3);

            top_height = top[0]->shape(2);
            top_width = top[0]->shape(3);

        }
            break;
        case StridedDecimInterpParameter_STRIDED_OP_INTERP_CONCAT:
        {
            op_type = BCKWD_INTERP_CONCAT;

            n = top[0]->shape(0);
            n_channels = top[0]->shape(1);
            height = top[0]->shape(2);
            width = top[0]->shape(3);

            top_height = bottom[0]->shape(2);
            top_width = bottom[0]->shape(3);

        }
            break;
        }


        for(int i=0; i< n;i++)
        {

            const int n_kernels = n_channels * height * width;

            Dtype* bottom_diff_ = temp_buffer_.mutable_gpu_data() + bottom[0]->offset(i,0,0,0);
            Dtype* top_diff_ = top[0]->mutable_gpu_diff() + bottom[0]->offset(i,0,0,0);

            if( op_type == BCKWD_DECIM || op_type == BCKWD_INTERP)
            {
                strided_decim_interp_fwd_bckwd_kernel<<<CAFFE_GET_BLOCKS(n_kernels), CAFFE_CUDA_NUM_THREADS>>>
                                                                                                         (n_kernels,
                                                                                                          n, n_channels, height, width,
                                                                                                          top_height, top_width,
                                                                                                          dilation_,
                                                                                                          op_type,
                                                                                                          bottom_diff_,
                                                                                                          top_diff_);
            }
            else if( op_type == BCKWD_DECIM_CONCAT || op_type == BCKWD_INTERP_CONCAT)
            {
                strided_decim_interp_concat_fwd_bckwd_kernel<<<CAFFE_GET_BLOCKS(n_kernels), CAFFE_CUDA_NUM_THREADS>>>
                                                                                                             (n_kernels,
                                                                                                              n, n_channels, height, width,
                                                                                                              top_height, top_width,
                                                                                                              dilation_,
                                                                                                              op_type,
                                                                                                              bottom_diff_,
                                                                                                              top_diff_);
            }

            CUDA_POST_KERNEL_CHECK;

        }

        cudaMemcpy(bottom[0]->mutable_gpu_diff(), temp_buffer_.gpu_data(), bottom[0]->count() * sizeof(Dtype), cudaMemcpyDeviceToDevice);

//        std::cout << "back " << (u_int64_t)bottom[0]->mutable_gpu_diff() << "  " << (u_int64_t)top[0]->mutable_gpu_diff() << "\n";

    }


    //    std::cout << "****************** BOTTOM DIFF ******************************************\n\n";

    //    int count = 0;

    //    //    std::cout << "conv bottom pointer = " << (int64_t)(bottom[0]->gpu_data()) << "   " <<  (int64_t)(bottom[0]->cpu_data()) << "\n";
    //    //    std::cout << "conv top pointer = " << (int64_t)(top[0]->gpu_data()) << "   " <<  (int64_t)(top[0]->cpu_data()) << "\n";

    //    for(int l=0;l<bottom[0]->shape(0);l++)
    //    {
    //        for(int i=0;i<bottom[0]->shape(1);i++)
    //        {
    //            for(int j=0;j<bottom[0]->shape(2);j++)
    //            {
    //                for(int k=0;k<bottom[0]->shape(3);k++)
    //                {
    //                    std::cout << bottom[0]->cpu_diff()[count++] << " ";
    //                    //                top[0]->mutable_cpu_data()[count++] = 2;

    //                }

    //                std::cout << "\n";
    //            }
    //            std::cout << "\n\n";
    //        }
    //        std::cout << "\n\nNEXT NUM \n\n";

    //    }

    //    std::cout << "****************** BOTTOM DIFF ENDS ******************************************\n\n\n\n\n";


}



INSTANTIATE_LAYER_GPU_FUNCS(StridedDecimInterpLayer);

}  // namespace caffe
