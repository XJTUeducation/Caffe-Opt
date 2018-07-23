#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_constellation_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



template <typename Dtype>
void BaseConvolutionConstellationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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

    ///-------*****************************
    ///
    // Setup dilations, kernel extent, incremental dilations
    // and starting index for each kernel location to make the col2_im faster
    {

        int n_dilations_h = kernel_shape_.cpu_data()[0];
        int n_dilations_w = kernel_shape_.cpu_data()[1];

        if((n_dilations_h != conv_param.dilation_size() ) && (n_dilations_h * 2 != conv_param.dilation_size()))
            LOG(FATAL) << "NUM DILATIONS H MUST BE EQUAL TO K or 2K";

        if((n_dilations_w != conv_param.dilation_size() ) && (n_dilations_h * 2 != conv_param.dilation_size()))
            LOG(FATAL) << "NUM DILATIONS W MUST BE EQUAL TO K or 2K";

        // DILATION CONSTELLATION
        dilation_constellation_h_.Reshape(1,1,n_dilations_h,1);
        dilation_constellation_w_.Reshape(1,1,n_dilations_h,1);

        int* data_dilation_constellation_h_ = dilation_constellation_h_.mutable_cpu_data();
        int* data_dilation_constellation_w_ = dilation_constellation_w_.mutable_cpu_data();

        if(n_dilations_h * 2 == conv_param.dilation_size())
        {
            for(int i=0;i<conv_param.dilation_size()/2;i++)
                data_dilation_constellation_h_[i] = conv_param.dilation(i);

            for(int i=conv_param.dilation_size()/2;i<conv_param.dilation_size();i++)
                data_dilation_constellation_w_[i] = conv_param.dilation(i);

        }
        else
        {
            for(int i=0;i<conv_param.dilation_size();i++)
            {
                data_dilation_constellation_h_[i] = conv_param.dilation(i);
                data_dilation_constellation_w_[i] = conv_param.dilation(i);
            }

        }

        // INCREASING DILATION
        dilation_constellation_inc_h_.Reshape(1,1,n_dilations_h,1);
        dilation_constellation_inc_w_.Reshape(1,1,n_dilations_w,1);

        int* data_dilation_constellation_inc_h_ = dilation_constellation_inc_h_.mutable_cpu_data();
        int* data_dilation_constellation_inc_w_ = dilation_constellation_inc_w_.mutable_cpu_data();

        data_dilation_constellation_inc_h_[0] = 0;
        data_dilation_constellation_inc_w_[0] = 0;

        for(int i=1;i<dilation_constellation_h_.count();i++)
            data_dilation_constellation_inc_h_[i] = data_dilation_constellation_h_[i] + data_dilation_constellation_inc_h_[i-1];

        for(int i=1;i<dilation_constellation_w_.count();i++)
            data_dilation_constellation_inc_w_[i] = data_dilation_constellation_w_[i] + data_dilation_constellation_inc_w_[i-1];


        int kernel_extent_h_ = data_dilation_constellation_inc_h_[dilation_constellation_inc_h_.count()-1] + 1;
        int kernel_extent_w_ = data_dilation_constellation_inc_w_[dilation_constellation_inc_w_.count()-1] + 1;

        kernel_extent_.Reshape(1,1,num_spatial_axes_,1);

        kernel_extent_.mutable_cpu_data()[0] = kernel_extent_h_;
        kernel_extent_.mutable_cpu_data()[1] = kernel_extent_w_;



        // DILATION increment in col2im
        dilation_constellation_shift_h_.Reshape(1,1,n_dilations_h,1);
        dilation_constellation_shift_w_.Reshape(1,1,n_dilations_w,1);

        // DILATION index of next dilation for increment in col2im
        dilation_constellation_shift_index_h_.Reshape(1,1,n_dilations_h,1);
        dilation_constellation_shift_index_w_.Reshape(1,1,n_dilations_w,1);

        int* data_dilation_constellation_shift_h_ = dilation_constellation_shift_h_.mutable_cpu_data();
        int* data_dilation_constellation_shift_w_ = dilation_constellation_shift_w_.mutable_cpu_data();

        int* data_dilation_constellation_shift_index_h_ = dilation_constellation_shift_index_h_.mutable_cpu_data();
        int* data_dilation_constellation_shift_index_w_ = dilation_constellation_shift_index_w_.mutable_cpu_data();

        for(int i=0;i<dilation_constellation_inc_h_.count();i++)
        {
            int nbr_index = i-1;

            // wrap by num_dilations for negative index
            //for e.g. nbr_index = -1 --> nbr_index += num_dilations_h;
            // and shift the location in the kernel. so data_dilation_constellation_inc_w[i] + k * kernel_extent_h_
            // where k is number of times a negative has been encountered

            int n_negatives = 0;
            if(nbr_index < 0)
            {
                nbr_index += dilation_constellation_inc_h_.count();
                n_negatives++;
            }

            while((data_dilation_constellation_inc_h_[i] + n_negatives * kernel_extent_h_
                   - data_dilation_constellation_inc_h_[nbr_index] ) % stride_.cpu_data()[0])
            {
                nbr_index--;
                if(nbr_index < 0)
                {
                    nbr_index += dilation_constellation_inc_h_.count();
                    n_negatives++;
                }
            }

            data_dilation_constellation_shift_h_[i] =
                    abs(data_dilation_constellation_inc_h_[i]  + n_negatives * kernel_extent_h_
                        - data_dilation_constellation_inc_h_[nbr_index])/ stride_.cpu_data()[0];

            data_dilation_constellation_shift_index_h_[i] = nbr_index;
        }


        for(int i=0;i<dilation_constellation_inc_w_.count();i++)
        {
            int nbr_index = i-1;

            // wrap by num_dilations for negative index
            //for e.g. nbr_index = -1 --> nbr_index += num_dilations_h;
            // and shift the location in the kernel. so data_dilation_constellation_inc_w[i] + k * kernel_extent_h_
            // where k is number of times a negative has been encountered

            int n_negatives = 0;
            if(nbr_index < 0)
            {
                nbr_index += dilation_constellation_inc_w_.count();
                n_negatives++;
            }

            while((data_dilation_constellation_inc_w_[i] + n_negatives * kernel_extent_w_
                   - data_dilation_constellation_inc_w_[nbr_index] ) % stride_.cpu_data()[1])
            {
                nbr_index--;
                if(nbr_index < 0)
                {
                    nbr_index += dilation_constellation_inc_w_.count();
                    n_negatives++;
                }
            }

            data_dilation_constellation_shift_w_[i] =
                    abs(data_dilation_constellation_inc_w_[i]  + n_negatives * kernel_extent_w_
                        - data_dilation_constellation_inc_w_[nbr_index])/ stride_.cpu_data()[1];

            data_dilation_constellation_shift_index_w_[i] = nbr_index;
        }

        dilation_start_index_h_.Reshape(1,1,kernel_extent_h_,1);
        dilation_start_index_w_.Reshape(1,1,kernel_extent_w_,1);

        dilation_start_index_h_.mutable_cpu_data()[0] = 0;
        dilation_start_index_w_.mutable_cpu_data()[1] = 0;

        int* data_dilation_start_index_h_ = dilation_start_index_h_.mutable_cpu_data();
        int* data_dilation_start_index_w_ = dilation_start_index_w_.mutable_cpu_data();

        // start index for every kernel position
        // to avoid unncessory valid position check

        for(int i=1;i<kernel_extent_h_;i++)
        {
            bool done = false;

            int h_k = i;

            while(!done)
            {
                for(int j=1;j< dilation_constellation_inc_h_.count();j++)
                {

                    if(data_dilation_constellation_inc_h_[j] % h_k == 0 &&
                            abs(data_dilation_constellation_inc_h_[j] / h_k) == 1)
                    {
                        done = true;
                        data_dilation_start_index_h_[i] = j;

                        break;
                    }

                    if(data_dilation_constellation_inc_h_[j] > h_k)
                    {
                        h_k -= stride_.cpu_data()[0];

                        if(h_k < 0)
                            h_k += kernel_extent_h_;

                        if(h_k == 0)
                        {
                            data_dilation_start_index_h_[i] = 0;
                            done = true;
                        }

                        break;
                    }
                }
            }
        }

        for(int i=1;i<kernel_extent_w_;i++)
        {
            bool done = false;

            int w_k = i;

            while(!done)
            {
                for(int j=1;j< dilation_constellation_inc_w_.count();j++)
                {
                    if(data_dilation_constellation_inc_w_[j] % w_k == 0 &&
                            abs(data_dilation_constellation_inc_w_[j] / w_k) == 1)
                    {
                        done = true;
                        data_dilation_start_index_w_[i] = j;

                        break;
                    }

                    if(data_dilation_constellation_inc_w_[j] > w_k)
                    {
                        w_k -= stride_.cpu_data()[1];

                        if(w_k < 0)
                            w_k += kernel_extent_w_;

                        if(w_k == 0)
                        {
                            data_dilation_start_index_w_[i] = 0;
                            done = true;
                        }

                        break;
                    }
                }
            }
        }


        //        std::cout << "ke_h = " << kernel_extent_h_ << "\n";

        //        for(int i=0;i<dilation_constellation_h_.count();i++)
        //            std::cout << dilation_constellation_h_.cpu_data()[i] << "  ";
        //        std::cout << "\n\n";

        //        for(int i=0;i<dilation_constellation_inc_h_.count();i++)
        //            std::cout << dilation_constellation_inc_h_.cpu_data()[i] << "  ";
        //        std::cout << "\n\n";

        //        for(int i=0;i<dilation_constellation_shift_h_.count();i++)
        //            std::cout << dilation_constellation_shift_h_.cpu_data()[i] << "  ";
        //        std::cout << "\n\n";

        //        for(int i=0;i<dilation_constellation_shift_index_h_.count();i++)
        //            std::cout << dilation_constellation_shift_index_h_.cpu_data()[i] << "  ";
        //        std::cout << "\n\n";

        //        for(int i=0;i<dilation_start_index_h_.count();i++)
        //            std::cout << dilation_start_index_h_.cpu_data()[i] << "  ";
        //        std::cout << "\n\n";

        //        std::cout << "ke_w = " << kernel_extent_w_ << "\n";

        //        for(int i=0;i<dilation_constellation_w_.count();i++)
        //            std::cout << dilation_constellation_w_.cpu_data()[i] << "  ";
        //        std::cout << "\n\n";

        //        for(int i=0;i<dilation_constellation_inc_w_.count();i++)
        //            std::cout << dilation_constellation_inc_w_.cpu_data()[i] << "  ";
        //        std::cout << "\n\n";

        //        for(int i=0;i<dilation_constellation_shift_w_.count();i++)
        //            std::cout << dilation_constellation_shift_w_.cpu_data()[i] << "  ";
        //        std::cout << "\n\n";

        //        for(int i=0;i<dilation_constellation_shift_index_w_.count();i++)
        //            std::cout << dilation_constellation_shift_index_w_.cpu_data()[i] << "  ";
        //        std::cout << "\n\n";

        //        for(int i=0;i<dilation_start_index_w_.count();i++)
        //            std::cout << dilation_start_index_w_.cpu_data()[i] << "  ";
        //        std::cout << "\n\n";

    }
    ///-------*****************************


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
}

template <typename Dtype>
void BaseConvolutionConstellationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                       const vector<Blob<Dtype>*>& top) {
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


#ifdef BASE_CONV_NO_MEM_BANK

        if(!this->pointer_map_)
            this->pointer_map_ = new std::map<int,Dtype*>();

        int device;
        cudaGetDevice(&device);

        if(total_count > this->col_buffer_count_)
        {
            this->col_buffer_count_ = total_count;

            Dtype* gpu_col_buffer = NULL;
            if(this->pointer_map_->count(device) > 0)
            {
                gpu_col_buffer = this->pointer_map_->operator [](device);
                cudaFree(gpu_col_buffer);
            }
            cudaMalloc(&gpu_col_buffer, this->col_buffer_count_ * sizeof(Dtype));
            this->pointer_map_->operator [](device) = gpu_col_buffer;
        }
#else
        {
            const Blob<Dtype>& buffer = this->get_buffer(total_count);
            this->temp_buffer_.Reshape(total_count, 1, 1, 1);
            this->temp_buffer_.ShareData(buffer);
        }
#endif
        //  std::cout << "\n\nLARGEST IM 2 COL MEM = " << total_count*4/1000000.0 << " MB\n\n";
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
void BaseConvolutionConstellationLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
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
void BaseConvolutionConstellationLayer<Dtype>::forward_cpu_bias(Dtype* output,
                                                                const Dtype* bias) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                          out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
                          (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionConstellationLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
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
void BaseConvolutionConstellationLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
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
void BaseConvolutionConstellationLayer<Dtype>::backward_cpu_bias(Dtype* bias,
                                                                 const Dtype* input) {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
                          input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionConstellationLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
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
    unsigned long int total_buffer_required = (unsigned long int )kernel_dim_ * output_shape_[0] * output_shape_[1];

#ifdef BASE_CONV_NO_MEM_BANK

    int device;
    cudaGetDevice(&device);
    Dtype* gpu_col_buff = this->pointer_map_->operator [](device);
    unsigned long int avail_buffer = this->col_buffer_count_;


#else

    Dtype* gpu_col_buff = this->temp_buffer_.mutable_gpu_data();
    unsigned long int avail_buffer = this->temp_buffer_.data()->size() / sizeof(Dtype);

#endif


    if(is_1x1_ || avail_buffer >= total_buffer_required)
    {

        const Dtype* col_buff = input;
        if (!is_1x1_) {

            im2col_gpu_constellation(input, conv_in_channels_,
                                     conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                    kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                    pad_.cpu_data()[0], pad_.cpu_data()[1],
                    stride_.cpu_data()[0], stride_.cpu_data()[1],
                    kernel_extent_.cpu_data()[0], kernel_extent_.cpu_data()[1],
                    dilation_constellation_inc_h_.gpu_data(), dilation_constellation_inc_w_.gpu_data(),
                    gpu_col_buff);

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
        Dtype* col_buff = gpu_col_buff;

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


            im2col_gpu_constellation_selected(input, conv_in_channels_, to_process, n_loc_processed,
                                              conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                    kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                    pad_.cpu_data()[0], pad_.cpu_data()[1],
                    stride_.cpu_data()[0], stride_.cpu_data()[1],
                    kernel_extent_.cpu_data()[0], kernel_extent_.cpu_data()[1],
                    dilation_constellation_inc_h_.gpu_data(), dilation_constellation_inc_w_.gpu_data(),
                    col_buff);


            int offset = n_loc_processed;

            caffe_gpu_gemm_ld<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
                                     to_process, kernel_dim_,
                                     (Dtype)1., weights , col_buff ,
                                     (Dtype)0., output + offset,-1,-1, conv_out_spatial_dim_);


            n_loc_processed += n_loc_can_be_processed;
        }


    }
}

template <typename Dtype>
void BaseConvolutionConstellationLayer<Dtype>::forward_gpu_bias(Dtype* output,
                                                                const Dtype* bias) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                          out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
                          (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionConstellationLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
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


    unsigned long int total_buffer_required = (unsigned long int )kernel_dim_ * output_shape_[0] * output_shape_[1];

#ifdef BASE_CONV_NO_MEM_BANK

    int device;
    cudaGetDevice(&device);
    Dtype* gpu_col_buff = this->pointer_map_->operator [](device);
    unsigned long int avail_buffer = this->col_buffer_count_;


#else

    Dtype* gpu_col_buff = this->temp_buffer_.mutable_gpu_data();
    unsigned long int avail_buffer = this->temp_buffer_.data()->size() / sizeof(Dtype);

#endif

    if(is_1x1_ || avail_buffer >= total_buffer_required)
    {

        Dtype* col_buff = input;
        if (!is_1x1_) {
            col_buff = gpu_col_buff;
        }

        Dtype beta = is_1x1_ && is_bottom_shared ? 1 : 0;

        for (int g = 0; g < group_; ++g) {

            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
                                  conv_out_spatial_dim_, conv_out_channels_ / group_,
                                  (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
                                  beta, col_buff + col_offset_ * g);
        }
        if (!is_1x1_) {

            const int accumulate = is_bottom_shared;

            col2im_gpu_constellation(col_buff, conv_in_channels_,
                                     conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                    kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                    pad_.cpu_data()[0], pad_.cpu_data()[1],
                    stride_.cpu_data()[0], stride_.cpu_data()[1],
                    kernel_extent_.cpu_data()[0], kernel_extent_.cpu_data()[1],
                    dilation_constellation_shift_h_.gpu_data(), dilation_constellation_shift_w_.gpu_data(),
                    dilation_constellation_shift_index_h_.gpu_data(), dilation_constellation_shift_index_w_.gpu_data(),
                    dilation_constellation_inc_h_.gpu_data(), dilation_constellation_inc_w_.gpu_data(),
                    dilation_start_index_h_.gpu_data(), dilation_start_index_w_.gpu_data(),
                    input,  accumulate);
        }
    }
    else
    {

        Dtype* col_buff = gpu_col_buff;

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

            int weight_offset = (unsigned long int)n_loc_processed * kernel_extent;


            caffe_gpu_gemm_ld<Dtype>(CblasTrans, CblasNoTrans, kernel_extent * to_process,
                                     conv_out_spatial_dim_, conv_out_channels_ ,
                                     (Dtype)1., weights + weight_offset , output,
                                     (Dtype)0., col_buff, kernel_dim_, -1 , -1 );

            int input_offset = (unsigned long int )n_loc_processed * conv_input_shape_.cpu_data()[1] * conv_input_shape_.cpu_data()[2];

            const int accumulate = is_bottom_shared;

            col2im_gpu_constellation(col_buff, to_process,
                                     conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                    kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                    pad_.cpu_data()[0], pad_.cpu_data()[1],
                    stride_.cpu_data()[0], stride_.cpu_data()[1],
                    kernel_extent_.cpu_data()[0], kernel_extent_.cpu_data()[1],
                    dilation_constellation_shift_h_.gpu_data(), dilation_constellation_shift_w_.gpu_data(),
                    dilation_constellation_shift_index_h_.gpu_data(), dilation_constellation_shift_index_w_.gpu_data(),
                    dilation_constellation_inc_h_.gpu_data(), dilation_constellation_inc_w_.gpu_data(),
                    dilation_start_index_h_.gpu_data(), dilation_start_index_w_.gpu_data(),
                    input + input_offset, accumulate);

            n_loc_processed += n_loc_can_be_processed;
        }
    }
}

template <typename Dtype>
void BaseConvolutionConstellationLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
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



    unsigned long int total_buffer_required = (unsigned long int )kernel_dim_ * output_shape_[0] * output_shape_[1];

#ifdef BASE_CONV_NO_MEM_BANK

    int device;
    cudaGetDevice(&device);
    Dtype* gpu_col_buff = this->pointer_map_->operator [](device);
    unsigned long int avail_buffer = this->col_buffer_count_;


#else

    Dtype* gpu_col_buff = this->temp_buffer_.mutable_gpu_data();
    unsigned long int avail_buffer = this->temp_buffer_.data()->size() / sizeof(Dtype);

#endif

    if(is_1x1_  || avail_buffer >= total_buffer_required)
    {
        const Dtype* col_buff = input;
        if (!is_1x1_) {


            im2col_gpu_constellation(input, conv_in_channels_,
                                     conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                    kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                    pad_.cpu_data()[0], pad_.cpu_data()[1],
                    stride_.cpu_data()[0], stride_.cpu_data()[1],
                    kernel_extent_.cpu_data()[0], kernel_extent_.cpu_data()[1],
                    dilation_constellation_inc_h_.gpu_data(), dilation_constellation_inc_w_.gpu_data(),
                    gpu_col_buff);

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
        Dtype* col_buff = gpu_col_buff;

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

            im2col_gpu_constellation_selected(input, conv_in_channels_, to_process, n_loc_processed,
                                              conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                    kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                    pad_.cpu_data()[0], pad_.cpu_data()[1],
                    stride_.cpu_data()[0], stride_.cpu_data()[1],
                    kernel_extent_.cpu_data()[0], kernel_extent_.cpu_data()[1],
                    dilation_constellation_inc_h_.gpu_data(), dilation_constellation_inc_w_.gpu_data(),
                    col_buff);



            int offset = n_loc_processed;

            caffe_gpu_gemm_ld<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_,
                                     kernel_dim_, to_process,
                                     (Dtype)1., output + offset , col_buff ,
                                     (Dtype)1., weights, conv_out_spatial_dim_, -1,-1 );

            n_loc_processed += n_loc_can_be_processed;
        }
    }

}

template <typename Dtype>
void BaseConvolutionConstellationLayer<Dtype>::backward_gpu_bias(Dtype* bias,
                                                                 const Dtype* input) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
                          input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionConstellationLayer);

}  // namespace caffe

