#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_mag_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
unsigned long int* BaseConvolutionMAGLayer<Dtype>::col_buffer_old_count_ = 0;

template <typename Dtype>
std::map<int, Dtype*>* BaseConvolutionMAGLayer<Dtype>::pointer_map_ = NULL;


template <typename Dtype>
void BaseConvolutionMAGLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
    // Configure the kernel size, padding, stride, and inputs.
    ConvolutionParameter conv_param = this->layer_param_.convolution_param();
    force_nd_im2col_ = conv_param.force_nd_im2col();
    channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
    const int first_spatial_axis = channel_axis_ + 1;
    const int num_axes = bottom[0]->num_axes();
    num_spatial_axes_ = num_axes - first_spatial_axis;
    CHECK_GE(num_spatial_axes_, 0);
    vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
    vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
    // Setup filter kernel dimensions (kernel_shape_).
    kernel_shape_.Reshape(spatial_dim_blob_shape);
    int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
    if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
        CHECK_EQ(num_spatial_axes_, 2)
                << "kernel_h & kernel_w can only be used for 2D convolution.";
        CHECK_EQ(0, conv_param.kernel_size_size())
                << "Either kernel_size or kernel_h/w should be specified; not both.";
        kernel_shape_data[0] = conv_param.kernel_h();
        kernel_shape_data[1] = conv_param.kernel_w();
    } else {
        const int num_kernel_dims = conv_param.kernel_size_size();
        CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
                << "kernel_size must be specified once, or once per spatial dimension "
                << "(kernel_size specified " << num_kernel_dims << " times; "
                << num_spatial_axes_ << " spatial dims).";
        for (int i = 0; i < num_spatial_axes_; ++i) {
            kernel_shape_data[i] =
                    conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
        }
    }
    for (int i = 0; i < num_spatial_axes_; ++i) {
        CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
    }
    // Setup stride dimensions (stride_).
    stride_.Reshape(spatial_dim_blob_shape);
    int* stride_data = stride_.mutable_cpu_data();
    if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
        CHECK_EQ(num_spatial_axes_, 2)
                << "stride_h & stride_w can only be used for 2D convolution.";
        CHECK_EQ(0, conv_param.stride_size())
                << "Either stride or stride_h/w should be specified; not both.";
        stride_data[0] = conv_param.stride_h();
        stride_data[1] = conv_param.stride_w();
    } else {
        const int num_stride_dims = conv_param.stride_size();
        CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
              num_stride_dims == num_spatial_axes_)
                << "stride must be specified once, or once per spatial dimension "
                << "(stride specified " << num_stride_dims << " times; "
                << num_spatial_axes_ << " spatial dims).";
        const int kDefaultStride = 1;
        for (int i = 0; i < num_spatial_axes_; ++i) {
            stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
                                                      conv_param.stride((num_stride_dims == 1) ? 0 : i);
            CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
        }
    }
    // Setup pad dimensions (pad_).
    pad_.Reshape(spatial_dim_blob_shape);
    int* pad_data = pad_.mutable_cpu_data();
    if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
        CHECK_EQ(num_spatial_axes_, 2)
                << "pad_h & pad_w can only be used for 2D convolution.";
        CHECK_EQ(0, conv_param.pad_size())
                << "Either pad or pad_h/w should be specified; not both.";
        pad_data[0] = conv_param.pad_h();
        pad_data[1] = conv_param.pad_w();
    } else {
        const int num_pad_dims = conv_param.pad_size();
        CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
              num_pad_dims == num_spatial_axes_)
                << "pad must be specified once, or once per spatial dimension "
                << "(pad specified " << num_pad_dims << " times; "
                << num_spatial_axes_ << " spatial dims).";
        const int kDefaultPad = 0;
        for (int i = 0; i < num_spatial_axes_; ++i) {
            pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
                                                conv_param.pad((num_pad_dims == 1) ? 0 : i);
        }
    }
    // Setup dilation dimensions (dilation_).
    dilation_.Reshape(spatial_dim_blob_shape);
    int* dilation_data = dilation_.mutable_cpu_data();
    const int num_dilation_dims = conv_param.dilation_size();
    CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
          num_dilation_dims == num_spatial_axes_)
            << "dilation must be specified once, or once per spatial dimension "
            << "(dilation specified " << num_dilation_dims << " times; "
            << num_spatial_axes_ << " spatial dims).";
    const int kDefaultDilation = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
        dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                                                      conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
    }
    // Special case: im2col is the identity for 1x1 convolution with stride 1
    // and no padding, so flag for skipping the buffer and transformation.
    is_1x1_ = true;
    for (int i = 0; i < num_spatial_axes_; ++i) {
        is_1x1_ &=
                kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
        if (!is_1x1_) { break; }
    }
    // Configure output channels and groups.
    channels_ = bottom[0]->shape(channel_axis_);
    num_output_ = this->layer_param_.convolution_param().num_output();
    CHECK_GT(num_output_, 0);
    group_ = this->layer_param_.convolution_param().group();
    CHECK_EQ(channels_ % group_, 0);
    CHECK_EQ(num_output_ % group_, 0)
            << "Number of output should be multiples of group.";
    if (reverse_dimensions()) {
        conv_out_channels_ = channels_;
        conv_in_channels_ = num_output_;
    } else {
        conv_out_channels_ = num_output_;
        conv_in_channels_ = channels_;
    }
    // Handle the parameters: weights and biases.
    // - blobs_[0] holds the filter weights
    // - blobs_[1] holds the biases (optional)
    vector<int> weight_shape(2);
    weight_shape[0] = conv_out_channels_;
    weight_shape[1] = conv_in_channels_ / group_;
    for (int i = 0; i < num_spatial_axes_; ++i) {
        weight_shape.push_back(kernel_shape_data[i]);
    }
    bias_term_ = this->layer_param_.convolution_param().bias_term();
    vector<int> bias_shape(bias_term_, num_output_);
    if (this->blobs_.size() > 0) {
        CHECK_EQ(1 + bias_term_, this->blobs_.size())
                << "Incorrect number of weight blobs.";
        if (weight_shape != this->blobs_[0]->shape()) {
            Blob<Dtype> weight_shaped_blob(weight_shape);
            LOG(FATAL) << "Incorrect weight shape: expected shape "
                       << weight_shaped_blob.shape_string() << "; instead, shape was "
                       << this->blobs_[0]->shape_string();
        }
        if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
            Blob<Dtype> bias_shaped_blob(bias_shape);
            LOG(FATAL) << "Incorrect bias shape: expected shape "
                       << bias_shaped_blob.shape_string() << "; instead, shape was "
                       << this->blobs_[1]->shape_string();
        }
        LOG(INFO) << "Skipping parameter initialization";
    } else {
        if (bias_term_) {
            this->blobs_.resize(2);
        } else {
            this->blobs_.resize(1);
        }
        // Initialize and fill the weights:
        // output channels x input channels per-group x kernel height x kernel width
        this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
        shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                                                     this->layer_param_.convolution_param().weight_filler()));
        weight_filler->Fill(this->blobs_[0].get());
        // If necessary, initialize and fill the biases.
        if (bias_term_) {
            this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
            shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                                                       this->layer_param_.convolution_param().bias_filler()));
            bias_filler->Fill(this->blobs_[1].get());
        }
    }
    kernel_dim_ = this->blobs_[0]->count(1);
    weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
    // Propagate gradients to the parameters (as directed by backward pass).
    this->param_propagate_down_.resize(this->blobs_.size(), true);


    {
        //------------------------------------------------------------------------------------------------------
        // - setup the mag layer

        std::string str_mag_layer = "mag_layer";

        LayerParameter mag_layer_param;
        mag_layer_param.set_name(str_mag_layer);
        mag_layer_param.set_type("MaxActivationsGrouping");

        mag_layer_param.add_bottom();
        mag_layer_param.set_bottom(0,"mag_bottom");

        mag_layer_param.add_top();
        mag_layer_param.set_top(0, "mag_max");

        mag_layer_param.add_top();
        mag_layer_param.set_top(1, "mag_arg_max");

        mag_layer_param.add_propagate_down(true);

        mag_tops_.push_back(new Blob<Dtype>);
        mag_tops_.push_back(new Blob<Dtype>);

        mag_layer_ = LayerRegistry<Dtype>::CreateLayer(mag_layer_param);
        mag_layer_->LayerSetUp(bottom, mag_tops_);
    }
}

template <typename Dtype>
void BaseConvolutionMAGLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {


    // call the reshape of the mag layer
    mag_layer_->Reshape(bottom, mag_tops_);

    //-------------------------------------------------------------------------------------------------------------------

    const int first_spatial_axis = channel_axis_ + 1;
    CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
            << "bottom num_axes may not change.";
    num_ = bottom[0]->count(0, channel_axis_);
    CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
            << "Input size incompatible with convolution kernel.";
    // TODO: generalize to handle inputs of different shapes.
    for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
        CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
                << "All inputs must have the same shape.";
    }
    // Shape the tops.
    bottom_shape_ = &bottom[0]->shape();
    compute_output_shape();
    vector<int> top_shape(bottom[0]->shape().begin(),
            bottom[0]->shape().begin() + channel_axis_);
    top_shape.push_back(num_output_);
    for (int i = 0; i < num_spatial_axes_; ++i) {
        top_shape.push_back(output_shape_[i]);
    }
    for (int top_id = 0; top_id < top.size(); ++top_id) {
        top[top_id]->Reshape(top_shape);
    }
    if (reverse_dimensions()) {
        conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
    } else {
        conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
    }
    col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
    output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
    // Setup input dimensions (conv_input_shape_).
    vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
    conv_input_shape_.Reshape(bottom_dim_blob_shape);
    int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
    for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
        if (reverse_dimensions()) {
            conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
        } else {
            conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
        }
    }
    //     The im2col result buffer will only hold one image at a time to avoid
    //     overly large memory usage. In the special case of 1x1 convolution
    //     it goes lazily unused to save memory.
    //    col_buffer_shape_.clear();
    //    col_buffer_shape_.push_back(kernel_dim_ * group_);
    //    for (int i = 0; i < num_spatial_axes_; ++i) {
    //        if (reverse_dimensions()) {
    //            col_buffer_shape_.push_back(input_shape(i + 1));
    //        } else {
    //            col_buffer_shape_.push_back(output_shape_[i]);
    //        }
    //    }
    //        col_buffer_.Reshape(col_buffer_shape_);

    // GLOBAL COL BUFFER
    {
        if(!pointer_map_)
            pointer_map_ = new std::map<int,Dtype*>();

        if(!col_buffer_old_count_)
        {
            int n_devices;
            cudaGetDeviceCount(&n_devices);
            col_buffer_old_count_ = new unsigned long int[n_devices];
            memset(col_buffer_old_count_,0,n_devices*sizeof(unsigned long int));
        }

        unsigned long int total_count = 1;

        total_count = total_count *kernel_dim_* group_;
        for (int i = 0; i < num_spatial_axes_; ++i) {
            if (reverse_dimensions()) {
                total_count *= input_shape(i + 1);
            } else {
                total_count *= output_shape_[i];
            }
        }

        unsigned long int max_buffer_limit_counts = 250000000;

        if(this->phase_ == TRAIN)
            max_buffer_limit_counts = 1000000000; // Bytes


        unsigned long int  per_loc_mem_required = kernel_dim_;

        unsigned long int  per_loc_backward_mem_required = (unsigned long int )conv_out_spatial_dim_ * kernel_shape_.cpu_data()[0] * kernel_shape_.cpu_data()[1];

        if(per_loc_mem_required < per_loc_backward_mem_required)
            per_loc_mem_required = per_loc_backward_mem_required;


        if(max_buffer_limit_counts < per_loc_mem_required * sizeof(Dtype))
            max_buffer_limit_counts = per_loc_mem_required * sizeof(Dtype);

        if(total_count * sizeof(Dtype) > max_buffer_limit_counts)
            total_count = max_buffer_limit_counts / sizeof(Dtype);



        int device;
        cudaGetDevice(&device);

        if(total_count > col_buffer_old_count_[device])
        {
            col_buffer_old_count_[device] = total_count;


            Dtype* gpu_col_buffer = NULL;
            if(pointer_map_->count(device) > 0)
            {
                gpu_col_buffer = pointer_map_->operator [](device);
                cudaFree(gpu_col_buffer);
            }
            cudaMalloc(&gpu_col_buffer,col_buffer_old_count_[device]*sizeof(Dtype));
            pointer_map_->operator [](device) = gpu_col_buffer;

            //                        std::cout << "\n\n\n\n\nLARGEST IM 2 COL MEM = " << total_count*4/1000000.0 << " MB\n\n\n\n\n";
        }

        //        for(int i=0;pointer_map_->count(i) > 0;i++)
        //        {
        //            int device;
        //            cudaGetDevice(&device);

        //            std::cout << "COUNTS = " << col_buffer_old_count_[i]*4 << " MB\n";
        //        }


    }

    bottom_dim_ = bottom[0]->count(channel_axis_);
    top_dim_ = top[0]->count(channel_axis_);
    num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
    num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
    // Set up the all ones "bias multiplier" for adding biases by BLAS
    out_spatial_dim_ = top[0]->count(first_spatial_axis);
    if (bias_term_) {
        vector<int> bias_multiplier_shape(1, out_spatial_dim_);
        bias_multiplier_.Reshape(bias_multiplier_shape);
        caffe_set(bias_multiplier_.count(), Dtype(1),
                  bias_multiplier_.mutable_cpu_data());
    }

}



template <typename Dtype>
void BaseConvolutionMAGLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
                                                      const Dtype* weights, Dtype* output, bool skip_im2col) {
    const Dtype* col_buff = input;
    if (!is_1x1_) {
        if (!skip_im2col) {
            conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
        }
        col_buff = col_buffer_.cpu_data();
    }
    for (int g = 0; g < group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
                              group_, conv_out_spatial_dim_, kernel_dim_,
                              (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
                              (Dtype)0., output + output_offset_ * g);
    }
}

template <typename Dtype>
void BaseConvolutionMAGLayer<Dtype>::forward_cpu_bias(Dtype* output,
                                                      const Dtype* bias) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                          out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
                          (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionMAGLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
                                                       const Dtype* weights, Dtype* input) {
    Dtype* col_buff = col_buffer_.mutable_cpu_data();
    if (is_1x1_) {
        col_buff = input;
    }
    for (int g = 0; g < group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
                              conv_out_spatial_dim_, conv_out_channels_ / group_,
                              (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
                              (Dtype)0., col_buff + col_offset_ * g);
    }
    if (!is_1x1_) {
        conv_col2im_cpu(col_buff, input);
    }
}

template <typename Dtype>
void BaseConvolutionMAGLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
                                                     const Dtype* output, Dtype* weights) {
    const Dtype* col_buff = input;
    if (!is_1x1_) {
        conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
        col_buff = col_buffer_.cpu_data();
    }
    for (int g = 0; g < group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
                              kernel_dim_, conv_out_spatial_dim_,
                              (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
                              (Dtype)1., weights + weight_offset_ * g);
    }
}

template <typename Dtype>
void BaseConvolutionMAGLayer<Dtype>::backward_cpu_bias(Dtype* bias,
                                                       const Dtype* input) {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
                          input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionMAGLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
                                                      const Dtype* weights, Dtype* output, bool skip_im2col) {
    //          const Dtype* col_buff = input;
    //          if (!is_1x1_) {
    //            if (!skip_im2col) {
    //            conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    //            }
    //            col_buff = col_buffer_.gpu_data();
    //          }
    //          for (int g = 0; g < group_; ++g) {
    //            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
    //                group_, conv_out_spatial_dim_, kernel_dim_,
    //                (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
    //                (Dtype)0., output + output_offset_ * g);
    //          }


    int device;
    cudaGetDevice(&device);

    unsigned long int total_buffer_required = (unsigned long int )kernel_dim_ * output_shape_[0] * output_shape_[1];
    unsigned long int avail_buffer = col_buffer_old_count_[device];

    if(is_1x1_ || avail_buffer >= total_buffer_required)
    {

        const Dtype* col_buff = input;
        if (!is_1x1_) {


            Dtype* gpu_col_buff = pointer_map_->operator [](device);

            conv_im2col_gpu(input, gpu_col_buff);
            col_buff = gpu_col_buff;
        }
        for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
                                  group_, conv_out_spatial_dim_, kernel_dim_,
                                  (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
                                  (Dtype)0., output + output_offset_ * g);
        }
    }
    else
    {
        // working
        Dtype* col_buff;
        Dtype* gpu_col_buff = pointer_map_->operator [](device);
        col_buff = gpu_col_buff;

        //        Dtype* temp_output_buff;
        //        gpu_col_buff = temp_output_pointer_map_->operator [](device);
        //        temp_output_buff = gpu_col_buff;

        int per_loc_mem_required = kernel_dim_;

        int n_loc_can_be_processed = floor((double)avail_buffer / per_loc_mem_required);

        if(n_loc_can_be_processed > (unsigned long int )output_shape_[0] * output_shape_[1])
            n_loc_can_be_processed = (unsigned long int )output_shape_[0] * output_shape_[1];


        int n_partition = ceil(((double)output_shape_[0] * output_shape_[1])/n_loc_can_be_processed);


        int n_loc_processed = 0;
        int n_loc_remaining = 0;

        //        std::cout << "N PARTITION = " << n_partition << "\n";

        for(int i=0; i < n_partition; i++)
        {
            n_loc_remaining = conv_out_spatial_dim_ - n_loc_processed;

            int to_process = n_loc_can_be_processed;

            if(n_loc_remaining < n_loc_can_be_processed)
                to_process = n_loc_remaining;


            im2col_gpu_selected(input, conv_in_channels_, to_process, n_loc_processed,
                                conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                    kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                    pad_.cpu_data()[0], pad_.cpu_data()[1],
                    stride_.cpu_data()[0], stride_.cpu_data()[1],
                    dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);




            //            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
            //                                  to_process, kernel_dim_,
            //                                  (Dtype)1., weights , col_buff ,
            //                                  (Dtype)0., temp_output_buff);

            //            copy_temp_output_2_output(temp_output_buff, num_output_,
            //                                      to_process, n_loc_processed,
            //                                      conv_out_spatial_dim_, output);

            int offset = n_loc_processed;

            caffe_gpu_gemm_ld<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
                                     to_process, kernel_dim_,
                                     (Dtype)1., weights , col_buff ,
                                     (Dtype)0., output + offset,-1,-1, conv_out_spatial_dim_);


            n_loc_processed += n_loc_can_be_processed;
        }



        //        Dtype* col_buff;
        //        Dtype* gpu_col_buff = pointer_map_->operator [](device);
        //        col_buff = gpu_col_buff;

        //        //        Dtype* temp_output_buff;
        //        //        gpu_col_buff = temp_output_pointer_map_->operator [](device);
        //        //        temp_output_buff = gpu_col_buff;


        //        int per_loc_mem_required = conv_out_spatial_dim_ * kernel_shape_.cpu_data()[0] * kernel_shape_.cpu_data()[1];

        //        int n_loc_can_be_processed = floor((double)avail_buffer / per_loc_mem_required);

        //        if(n_loc_can_be_processed > conv_in_channels_ )
        //            n_loc_can_be_processed = conv_in_channels_;

        //        int n_partition = ceil((double)conv_in_channels_/n_loc_can_be_processed);

        //        int n_loc_processed = 0;
        //        int n_loc_remaining = 0;

        ////        std::cout << "N PARTITION = " << n_partition << "\n";

        //        int kernel_extent = kernel_shape_.cpu_data()[0] * kernel_shape_.cpu_data()[1];

        //        for(int i=0; i < n_partition; i++)
        //        {
        //            n_loc_remaining = conv_in_channels_ - n_loc_processed;

        //            int to_process = n_loc_can_be_processed;

        //            if(n_loc_remaining < n_loc_can_be_processed)
        //                to_process = n_loc_remaining;


        //            int input_offset = n_loc_processed * conv_input_shape_.cpu_data()[1] * conv_input_shape_.cpu_data()[2];

        //            im2col_gpu(input + input_offset, to_process,
        //                       conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
        //                    kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
        //                    pad_.cpu_data()[0], pad_.cpu_data()[1],
        //                    stride_.cpu_data()[0], stride_.cpu_data()[1],
        //                    dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);

        //            Dtype beta = 0;

        //            if(i)
        //                beta = 1;

        //            int weight_offset = n_loc_processed * kernel_extent;

        //            caffe_gpu_gemm_ld<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
        //                                     conv_out_spatial_dim_, kernel_extent * to_process,
        //                                     (Dtype)1., weights + weight_offset, col_buff ,
        //                                      beta, output, kernel_dim_ , -1, -1);


        //            n_loc_processed += n_loc_can_be_processed;
        //        }

    }


    //        Dtype beta;

    //        if(!c)
    //            beta = 0;
    //        else
    //            beta = 1;

    ////        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_, conv_out_spatial_dim_, k_offset,
    ////                              (Dtype)1., temp_data_weights_gpu +weight_offset , col_buff ,
    ////                              (Dtype)beta, output);
    //    }






    //        const Dtype* col_buff = input;
    //        Dtype* gpu_col_buff = NULL;

    //        if (!is_1x1_) {

    //            int count = 1;
    //            for(int i=0;i<col_buffer_shape_.size();i++)
    //                count *= col_buffer_shape_[i];

    //            cudaMalloc(&gpu_col_buff,count*sizeof(Dtype));

    //            conv_im2col_gpu(input, gpu_col_buff);
    //            col_buff = gpu_col_buff;
    //        }
    //        for (int g = 0; g < group_; ++g) {
    //            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
    //                                  group_, conv_out_spatial_dim_, kernel_dim_,
    //                                  (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
    //                                  (Dtype)0., output + output_offset_ * g);
    //        }
    //        if(!is_1x1_)
    //            cudaFree(gpu_col_buff);
}

template <typename Dtype>
void BaseConvolutionMAGLayer<Dtype>::forward_gpu_bias(Dtype* output,
                                                      const Dtype* bias) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                          out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
                          (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionMAGLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
                                                       const Dtype* weights, Dtype* input, bool is_bottom_shared) {
    //        Dtype* col_buff = col_buffer_.mutable_gpu_data();
    //        if (is_1x1_) {
    //            col_buff = input;
    //        }
    //        for (int g = 0; g < group_; ++g) {
    //            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
    //                                  conv_out_spatial_dim_, conv_out_channels_ / group_,
    //                                  (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
    //                                  (Dtype)0., col_buff + col_offset_ * g);
    //        }
    //        if (!is_1x1_) {
    //                conv_col2im_gpu(col_buff, input);
    //        }

    //    std::cout << "is bottom shared = " << is_bottom_shared << "\n";

    int device;
    cudaGetDevice(&device);

    unsigned long int total_buffer_required = (unsigned long int )kernel_dim_ * output_shape_[0] * output_shape_[1];
    unsigned long int avail_buffer = col_buffer_old_count_[device];

    if(is_1x1_ || avail_buffer >= total_buffer_required)
    {

        Dtype* col_buff = input;
        if (!is_1x1_) {

            int device;
            cudaGetDevice(&device);
            Dtype* gpu_col_buff = pointer_map_->operator [](device);

            col_buff = gpu_col_buff;
        }

        Dtype beta = 0;
        if(is_1x1_ && is_bottom_shared)
        {
            //            std::cout << "beta = " << beta < "\n";
            beta = 1;
        }

        for (int g = 0; g < group_; ++g) {

            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
                                  conv_out_spatial_dim_, conv_out_channels_ / group_,
                                  (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
                                  beta, col_buff + col_offset_ * g);
        }
        if (!is_1x1_) {

            const int accumulate = is_bottom_shared;
            //            if(is_bottom_shared)
            //                std::cout << " 111 beta = " << beta < "\n";

            conv_col2im_gpu(col_buff, input, accumulate);
        }
    }
    else
    {

        Dtype* col_buff;
        Dtype* gpu_col_buff = pointer_map_->operator [](device);
        col_buff = gpu_col_buff;

        //        Dtype* temp_output_buff;
        //        gpu_col_buff = temp_output_pointer_map_->operator [](device);
        //        temp_output_buff = gpu_col_buff;


        int per_loc_mem_required = (unsigned long int )conv_out_spatial_dim_ * kernel_shape_.cpu_data()[0] * kernel_shape_.cpu_data()[1];

        int n_loc_can_be_processed = floor((double)avail_buffer / per_loc_mem_required);

        if(n_loc_can_be_processed > conv_in_channels_ )
            n_loc_can_be_processed = conv_in_channels_;

        int n_partition = ceil((double)conv_in_channels_/n_loc_can_be_processed);

        int n_loc_processed = 0;
        int n_loc_remaining = 0;

        //                std::cout << "N PARTITION = " << n_partition << "\n";

        int kernel_extent = (unsigned long int )kernel_shape_.cpu_data()[0] * kernel_shape_.cpu_data()[1];

        for(int i=0; i < n_partition; i++)
        {
            n_loc_remaining = conv_in_channels_ - n_loc_processed;

            int to_process = n_loc_can_be_processed;

            if(n_loc_remaining < n_loc_can_be_processed)
                to_process = n_loc_remaining;


            //            copy_output_2_temp_output(temp_output_buff, num_output_,
            //                                      to_process, n_loc_processed,
            //                                      conv_out_spatial_dim_, output);


            int weight_offset = (unsigned long int)n_loc_processed * kernel_extent;

            //            caffe_gpu_gemm_ld<Dtype>(CblasTrans, CblasNoTrans, kernel_extent * to_process,
            //                                     conv_out_spatial_dim_, conv_out_channels_ ,
            //                                     (Dtype)1., weights + offset , output,
            //                                     (Dtype)0., col_buff,-1, conv_out_spatial_dim_ , -1 );

            caffe_gpu_gemm_ld<Dtype>(CblasTrans, CblasNoTrans, kernel_extent * to_process,
                                     conv_out_spatial_dim_, conv_out_channels_ ,
                                     (Dtype)1., weights + weight_offset , output,
                                     (Dtype)0., col_buff, kernel_dim_, -1 , -1 );

            //            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
            //                                  conv_out_spatial_dim_, conv_out_channels_ / group_,
            //                                  (Dtype)1., weights + weight_offset, output ,
            //                                  (Dtype)0., col_buff );


            int input_offset = (unsigned long int )n_loc_processed * conv_input_shape_.cpu_data()[1] * conv_input_shape_.cpu_data()[2];

            const int accumulate = is_bottom_shared;

            col2im_gpu(col_buff, to_process,
                       conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                    kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                    pad_.cpu_data()[0], pad_.cpu_data()[1],
                    stride_.cpu_data()[0], stride_.cpu_data()[1],
                    dilation_.cpu_data()[0], dilation_.cpu_data()[1], input + input_offset, accumulate);

            n_loc_processed += n_loc_can_be_processed;

            //            std::cout << "N loc processed = " << n_loc_processed << "\n";
        }
    }



    //// WORKING FAST MEMORY OPTIMIZED
    //    if(0)
    //    {
    //        Dtype* col_buff;
    //        Dtype* gpu_col_buff = pointer_map_->operator [](device);
    //        col_buff = gpu_col_buff;

    //        //        Dtype* temp_output_buff;
    //        //        gpu_col_buff = temp_output_pointer_map_->operator [](device);
    //        //        temp_output_buff = gpu_col_buff;


    //        int per_loc_mem_required = kernel_dim_;

    //        int n_loc_can_be_processed = floor((double)avail_buffer / per_loc_mem_required);

    //        if(n_loc_can_be_processed > output_shape_[0] * output_shape_[1])
    //            n_loc_can_be_processed = output_shape_[0] * output_shape_[1];

    //        int n_partition = ceil((double)(output_shape_[0] * output_shape_[1])/n_loc_can_be_processed);

    //        int n_loc_processed = 0;
    //        int n_loc_remaining = 0;

    //        //        std::cout << "N PARTITION = " << n_partition << "\n";

    //        for(int i=0; i < n_partition; i++)
    //        {
    //            n_loc_remaining = conv_out_spatial_dim_ - n_loc_processed;

    //            int to_process = n_loc_can_be_processed;

    //            if(n_loc_remaining < n_loc_can_be_processed)
    //                to_process = n_loc_remaining;


    //            //            copy_output_2_temp_output(temp_output_buff, num_output_,
    //            //                                      to_process, n_loc_processed,
    //            //                                      conv_out_spatial_dim_, output);

    //            int offset = n_loc_processed;

    //            caffe_gpu_gemm_ld<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
    //                                     to_process, conv_out_channels_ ,
    //                                     (Dtype)1., weights  , output + offset,
    //                                     (Dtype)0., col_buff,-1, conv_out_spatial_dim_ , -1 );


    //            // COMPUTE INDEXS

    //            const int index_min = n_loc_processed;
    //            const int index_max = n_loc_processed + to_process -1;

    //            const int top_left_x_op = index_min % output_shape_[1];
    //            const int top_left_y_op = index_min / output_shape_[1];

    //            const int bottom_right_x_op = index_max % output_shape_[1];
    //            const int bottom_right_y_op = index_max / output_shape_[1];

    //            //            std::cout << "TOP left bottom right = " <<  top_left_x_op << "  "
    //            //                      << top_left_y_op << "  "
    //            //                      << bottom_right_x_op << "  "
    //            //                      << bottom_right_y_op << "\n\n";

    //            const int n_rows = abs(bottom_right_y_op - top_left_y_op);

    //            int min_x_op;
    //            int min_y_op;
    //            int max_x_op;
    //            int max_y_op;

    //            if(n_rows)
    //            {
    //                min_x_op = 0;
    //                max_x_op =  output_shape_[1] - 1;
    //            }
    //            else
    //            {
    //                min_x_op = top_left_x_op;
    //                max_x_op = bottom_right_x_op;
    //            }

    //            min_y_op = top_left_y_op;
    //            max_y_op = bottom_right_y_op;

    //            int crop_min_x;
    //            int crop_min_y;
    //            int crop_max_x;
    //            int crop_max_y;
    //            int crop_width;
    //            int crop_height;


    //            // TRANSFORM OUTPUT COORDINATES TO INPUT

    //            // X_in = (k-1)/2 -P + x_op*S;
    //            // Y_in = (k-1)/2 -P + y_op*S;
    //            // where k = (kernel-1)*(dilation-1) + kernel,
    //            // after solving: k = (kernel-1)*dilation + 1;

    //            //            const int w_col_start =
    //            //                    (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    //            //            const int w_col_end = min(w_im / stride_w + 1, width_col);
    //            //            const int h_col_start =
    //            //                    (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    //            //            const int h_col_end = min(h_im / stride_h + 1, height_col);



    //            //             std::cout << " min max = " << min_x_op << "  " <<  min_y_op << "  " << max_x_op << "  " << max_y_op << "\n\n";

    //            const int kernel_extent_w = (kernel_shape_.cpu_data()[1] - 1) * dilation_.cpu_data()[1] + 1;
    //            const int kernel_extent_h = (kernel_shape_.cpu_data()[0] - 1) * dilation_.cpu_data()[0] + 1;


    //            int min_x_in = min_x_op * stride_.cpu_data()[1] - pad_.cpu_data()[1] + (kernel_extent_w - 1)/2;
    //            int min_y_in = min_y_op * stride_.cpu_data()[0] - pad_.cpu_data()[0] + (kernel_extent_h - 1)/2;

    //            int max_x_in = max_x_op * stride_.cpu_data()[1] - pad_.cpu_data()[1] + (kernel_extent_w - 1)/2;
    //            int max_y_in = max_y_op * stride_.cpu_data()[0] - pad_.cpu_data()[0] + (kernel_extent_h - 1)/2;

    //            //             std::cout <<  "(" << min_y_in << "," << min_x_in <<  ")  (" << max_y_in << "," << max_x_in << ")\n";

    //            crop_min_x = min_x_in - (kernel_extent_w -1)/2 ;
    //            crop_max_x = max_x_in + (kernel_extent_w -1)/2 ;

    //            crop_min_y = min_y_in - (kernel_extent_h -1)/2 ;
    //            crop_max_y = max_y_in + (kernel_extent_h -1)/2 ;

    //            //            crop_min_y = min_y_op * stride_.cpu_data()[0] - pad_.cpu_data()[0];
    //            //            crop_max_y = max_y_op * stride_.cpu_data()[0] - pad_.cpu_data()[0] + (kernel_extent_h - 1);

    //            //            crop_max_x = (max_x_op-1) * stride_.cpu_data()[1] - pad_.cpu_data()[1] ;
    //            //            crop_min_x = (min_x_op-1) * stride_.cpu_data()[1] - pad_.cpu_data()[1] - kernel_extent_w;

    //            //            crop_max_y = (max_y_op-1) * stride_.cpu_data()[0] - pad_.cpu_data()[0];
    //            //            crop_min_y = (min_y_op-1) * stride_.cpu_data()[0] - pad_.cpu_data()[0] - kernel_extent_h;


    //            if(crop_min_x < 0)
    //                crop_min_x = 0;

    //            if(crop_max_x >= conv_input_shape_.cpu_data()[2])
    //                crop_max_x = conv_input_shape_.cpu_data()[2] - 1;

    //            if(crop_min_y < 0)
    //                crop_min_y = 0;

    //            if(crop_max_y >= conv_input_shape_.cpu_data()[1])
    //                crop_max_y = conv_input_shape_.cpu_data()[1] - 1;

    //            //            crop_width = conv_input_shape_.cpu_data()[2]; //crop_max_x - crop_min_x + 1;
    //            //            crop_height = conv_input_shape_.cpu_data()[1];//crop_max_y - crop_min_y + 1;

    //            crop_width = crop_max_x - crop_min_x + 1;
    //            crop_height = crop_max_y - crop_min_y + 1;

    //            //            std::cout << "indexs = " << index_min << "  " << index_max << "\n\n";

    //            //            std::cout //<< crop_min_x << "  "
    //            ////                      << crop_min_y << "  "
    //            ////                      << crop_max_x << "  "
    //            ////                      << crop_max_y << "  "
    //            //                      << crop_width << "  "
    //            //                      << crop_height<< "\n\n";

    //            col2im_gpu_selected(col_buff, conv_in_channels_, to_process,
    //                                index_min, index_max,
    //                                crop_min_x, crop_min_y,
    //                                crop_width, crop_height,
    //                                conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
    //                    kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
    //                    pad_.cpu_data()[0], pad_.cpu_data()[1],
    //                    stride_.cpu_data()[0], stride_.cpu_data()[1],
    //                    dilation_.cpu_data()[0], dilation_.cpu_data()[1], input);

    //            n_loc_processed += n_loc_can_be_processed;
    //        }
    //    }


    //    Dtype* col_buff = input;
    //    Dtype* gpu_col_buff = NULL;

    //    if (!is_1x1_) {

    //        int count = 1;
    //        for(int i=0;i<col_buffer_shape_.size();i++)
    //            count *= col_buffer_shape_[i];

    //        cudaMalloc(&gpu_col_buff,count*sizeof(Dtype));

    //        conv_im2col_gpu(input, gpu_col_buff);
    //        col_buff = gpu_col_buff;
    //    }
    //    for (int g = 0; g < group_; ++g) {
    //        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
    //                              conv_out_spatial_dim_, conv_out_channels_ / group_,
    //                              (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
    //                              (Dtype)0., col_buff + col_offset_ * g);
    //    }
    //    if (!is_1x1_) {
    //        conv_col2im_gpu(col_buff, input);
    //        cudaFree(gpu_col_buff);
    //    }

}

template <typename Dtype>
void BaseConvolutionMAGLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
                                                     const Dtype* output, Dtype* weights) {
    //        const Dtype* col_buff = input;
    //        if (!is_1x1_) {
    //            conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    //            col_buff = col_buffer_.gpu_data();
    //        }
    //        for (int g = 0; g < group_; ++g) {
    //            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
    //                                  kernel_dim_, conv_out_spatial_dim_,
    //                                  (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
    //                                  (Dtype)1., weights + weight_offset_ * g);
    //        }




    int device;
    cudaGetDevice(&device);

    unsigned long int total_buffer_required = (unsigned long int )kernel_dim_ * output_shape_[0] * output_shape_[1];
    unsigned long int avail_buffer = col_buffer_old_count_[device];

    if(is_1x1_  || avail_buffer >= total_buffer_required)
    {
        const Dtype* col_buff = input;
        if (!is_1x1_) {

            int device;
            cudaGetDevice(&device);
            Dtype* gpu_col_buff = pointer_map_->operator [](device);

            conv_im2col_gpu(input, gpu_col_buff);
            col_buff = gpu_col_buff;
        }
        for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
                                  kernel_dim_, conv_out_spatial_dim_,
                                  (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
                                  (Dtype)1., weights + weight_offset_ * g);
        }

    }
    else
    {
        Dtype* col_buff;
        Dtype* gpu_col_buff = pointer_map_->operator [](device);
        col_buff = gpu_col_buff;

        //        Dtype* temp_output_buff;
        //        gpu_col_buff = temp_output_pointer_map_->operator [](device);
        //        temp_output_buff = gpu_col_buff;


        int per_loc_mem_required = kernel_dim_;

        int n_loc_can_be_processed = floor((double)avail_buffer / per_loc_mem_required);

        if(n_loc_can_be_processed > (unsigned long int)output_shape_[0] * output_shape_[1])
            n_loc_can_be_processed = (unsigned long int)output_shape_[0] * output_shape_[1];

        int n_partition = ceil(((double)output_shape_[0] * output_shape_[1])/n_loc_can_be_processed);

        int n_loc_processed = 0;
        int n_loc_remaining = 0;

        //                std::cout << "N PARTITION = " << n_partition << "\n";

        for(int i=0; i < n_partition; i++)
        {
            n_loc_remaining = conv_out_spatial_dim_ - n_loc_processed;

            int to_process = n_loc_can_be_processed;

            if(n_loc_remaining < n_loc_can_be_processed)
                to_process = n_loc_remaining;


            im2col_gpu_selected(input, conv_in_channels_, to_process, n_loc_processed,
                                conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                    kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                    pad_.cpu_data()[0], pad_.cpu_data()[1],
                    stride_.cpu_data()[0], stride_.cpu_data()[1],
                    dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);


            //            copy_output_2_temp_output(temp_output_buff, num_output_,
            //                                      to_process, n_loc_processed,
            //                                      conv_out_spatial_dim_, output);


            int offset = n_loc_processed;

            caffe_gpu_gemm_ld<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_,
                                     kernel_dim_, to_process,
                                     (Dtype)1., output + offset , col_buff ,
                                     (Dtype)1., weights, conv_out_spatial_dim_, -1,-1 );

            n_loc_processed += n_loc_can_be_processed;
        }
    }







    //    const Dtype* col_buff = input;
    //    Dtype* gpu_col_buff = NULL;

    //    if (!is_1x1_) {

    //        int count = 1;
    //        for(int i=0;i<col_buffer_shape_.size();i++)
    //            count *= col_buffer_shape_[i];

    //        cudaMalloc(&gpu_col_buff,count*sizeof(Dtype));

    //        conv_im2col_gpu(input, gpu_col_buff);
    //        col_buff = gpu_col_buff;
    //    }
    //    for (int g = 0; g < group_; ++g) {
    //        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
    //                              kernel_dim_, conv_out_spatial_dim_,
    //                              (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
    //                              (Dtype)1., weights + weight_offset_ * g);
    //    }
    //    if(!is_1x1_)
    //        cudaFree(gpu_col_buff);
}

template <typename Dtype>
void BaseConvolutionMAGLayer<Dtype>::backward_gpu_bias(Dtype* bias,
                                                       const Dtype* input) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
                          input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionMAGLayer);

}  // namespace caffe

