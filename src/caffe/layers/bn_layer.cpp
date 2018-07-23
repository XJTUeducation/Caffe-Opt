#include <algorithm>
#include <vector>

#include "caffe/layers/bn_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
std::vector<unsigned long int> BNLayer<Dtype>::broadcast_buffer_old_count_ = std::vector<unsigned long int>(0);

template <typename Dtype>
std::map<int, Dtype*>* BNLayer<Dtype>::broadcast_buffer_pointer_map_ = NULL;


template <typename Dtype>
pthread_mutex_t BNLayer<Dtype>::mutex_instances_finished_;

template <typename Dtype>
std::vector<Dtype*> BNLayer<Dtype>::batch_statistics_ptrs_gpu_ = std::vector<Dtype*>(0);

template <typename Dtype>
pthread_cond_t BNLayer<Dtype>::cond_gathered_ ;




template <typename Dtype>
void BNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top) {
    frozen_ = this->layer_param_.bn_param().frozen();
    bn_momentum_ = this->layer_param_.bn_param().momentum();
    bn_eps_ = this->layer_param_.bn_param().eps();
    // Initialize parameters
    if (this->blobs_.size() > 0) {
        LOG(INFO) << "Skipping parameter initialization";
    } else {
        this->blobs_.resize(4);
        vector<int> shape;
        shape.push_back(1);
        shape.push_back(bottom[0]->channels());
        shape.push_back(1);
        shape.push_back(1);
        // slope
        this->blobs_[0].reset(new Blob<Dtype>(shape));
        shared_ptr<Filler<Dtype> > slope_filler(GetFiller<Dtype>(
                                                    this->layer_param_.bn_param().slope_filler()));
        slope_filler->Fill(this->blobs_[0].get());
        // bias
        this->blobs_[1].reset(new Blob<Dtype>(shape));
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                                                   this->layer_param_.bn_param().bias_filler()));
        bias_filler->Fill(this->blobs_[1].get());
        // moving average mean
        this->blobs_[2].reset(new Blob<Dtype>(shape));
        caffe_set(this->blobs_[2]->count(), Dtype(0),
                this->blobs_[2]->mutable_cpu_data());
        // moving average variance
        this->blobs_[3].reset(new Blob<Dtype>(shape));
        caffe_set(this->blobs_[3]->count(), frozen_ ? Dtype(1) : Dtype(0),
                this->blobs_[3]->mutable_cpu_data());
    }
    this->param_propagate_down_.resize(this->blobs_.size(), true);

    // runing average stats does not use weight decay and learning rate
    while (this->layer_param_.param_size() < 4){
        this->layer_param_.mutable_param()->Add();
    }
    this->layer_param_.mutable_param(2)->set_lr_mult(Dtype(0));
    this->layer_param_.mutable_param(2)->set_decay_mult(Dtype(0));

    this->layer_param_.mutable_param(3)->set_lr_mult(Dtype(0));
    this->layer_param_.mutable_param(3)->set_decay_mult(Dtype(0));

    // shutdown scale and bias update in frozen mode
    if (this->frozen_){
        // slope
        this->layer_param_.mutable_param(0)->set_lr_mult(Dtype(0));
        this->layer_param_.mutable_param(0)->set_decay_mult(Dtype(0));

        // bias
        this->layer_param_.mutable_param(1)->set_lr_mult(Dtype(0));
        this->layer_param_.mutable_param(1)->set_decay_mult(Dtype(0));
    }

#ifdef BN_NO_MEM_BANK_

    if(!broadcast_buffer_pointer_map_)
        broadcast_buffer_pointer_map_ = new std::map<int,Dtype*>();


    if(!broadcast_buffer_old_count_.size())
    {
        int n_devices;
        cudaGetDeviceCount(&n_devices);
        broadcast_buffer_old_count_.resize(n_devices,0);
        //memset(buffer_old_count_,0,n_devices*sizeof(unsigned long int));
    }

#endif

    if(this->phase_ == TRAIN)
    {
        static bool mutex_cond_initialized = false;

        if(!mutex_cond_initialized)
        {
            pthread_mutex_init(&mutex_instances_finished_, NULL);
            pthread_cond_init(&cond_gathered_, NULL);

            mutex_cond_initialized = true;
        }
    }

}

template <typename Dtype>
void BNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top) {
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();

    top[0]->ReshapeLike(*(bottom[0]));

    broadcast_buffer_.ReshapeLike(*(bottom[0]));
    spatial_statistic_.Reshape(num_, channels_, 1, 1);
    batch_statistic_.Reshape(1, channels_, 1, 1);

    x_norm_.ReshapeLike(*(bottom[0]));
    x_inv_std_.ReshapeLike(batch_statistic_);

    spatial_sum_multiplier_.Reshape(1, 1, height_, width_);
    caffe_set(spatial_sum_multiplier_.count(), Dtype(1),
              spatial_sum_multiplier_.mutable_cpu_data());
    batch_sum_multiplier_.Reshape(num_, 1, 1, 1);
    caffe_set(batch_sum_multiplier_.count(), Dtype(1),
              batch_sum_multiplier_.mutable_cpu_data());


    batch_statistic_instances_.Reshape(this->n_instances_ , channels_, 1, 1);

    // set broadcast buffer mem
    {
        int total_count = bottom[0]->count();

#ifdef BN_NO_MEM_BANK_

        int device;
        cudaGetDevice(&device);

        if(total_count > broadcast_buffer_old_count_[device])
        {
            broadcast_buffer_old_count_[device] = total_count;

            Dtype* gpu_buffer = NULL;
            if(broadcast_buffer_pointer_map_->count(device) > 0)
            {
                gpu_buffer = broadcast_buffer_pointer_map_->operator [](device);
                cudaFree(gpu_buffer);
            }
            cudaMalloc(&gpu_buffer,broadcast_buffer_old_count_[device]*sizeof(Dtype));
            broadcast_buffer_pointer_map_->operator [](device) = gpu_buffer;
        }

#else
        // set broadcast buffer memory from memory bank
        {
            const Blob<Dtype>& buffer = this->get_buffer(total_count);
            broadcast_buffer_.ShareData(buffer);
//            std::cout << "\n\nLARGEST Broad cast buff mem = " << total_count * 4/1000000.0 << " MB\n\n";
        }
#endif

#ifdef BN_OPTMIZE_X_NORM

        x_norm_cpu_.ReshapeLike(*bottom[0]);

#endif

    }

}

template <typename Dtype>
void BNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {

    const Dtype* const_bottom_data = bottom[0]->cpu_data();
    const Dtype* const_top_data = top[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();


    const Dtype* scale_data = this->blobs_[0]->cpu_data();
    const Dtype* shift_data = this->blobs_[1]->cpu_data();

    // Mean normalization
    if (frozen_ || this->phase_ == TEST) {
        // Use the moving average mean
        caffe_copy(batch_statistic_.count(), this->blobs_[2]->cpu_data(),
                batch_statistic_.mutable_cpu_data());
    } else {
        // Compute the mean by averaging over spatial and batch dimensions.
        caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1) / (height_ * width_), const_bottom_data,
                              spatial_sum_multiplier_.cpu_data(), Dtype(0),
                              spatial_statistic_.mutable_cpu_data());
        caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_,
                              Dtype(1) / num_, spatial_statistic_.cpu_data(),
                              batch_sum_multiplier_.cpu_data(), Dtype(0),
                              batch_statistic_.mutable_cpu_data());
        // Add to the moving average
        if (!frozen_) {
            caffe_cpu_axpby(batch_statistic_.count(),
                            Dtype(1) - bn_momentum_, batch_statistic_.cpu_data(),
                            bn_momentum_, this->blobs_[2]->mutable_cpu_data());
        }
    }
    // Broadcast the mean vector
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                          Dtype(1), batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(),
                          Dtype(0), spatial_statistic_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                          height_ * width_, 1, Dtype(-1),
                          spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
                          Dtype(0), broadcast_buffer_.mutable_cpu_data());
    // Subtract
    caffe_add(broadcast_buffer_.count(), const_bottom_data,
              broadcast_buffer_.cpu_data(), top_data);

    // Variance normalization
    if (frozen_ || this->phase_ == TEST) {
        // Use the moving average variance
        caffe_copy(batch_statistic_.count(), this->blobs_[3]->cpu_data(),
                batch_statistic_.mutable_cpu_data());
    } else {
        // calculate batch variance
        caffe_powx(broadcast_buffer_.count(), const_top_data, Dtype(2),
                   broadcast_buffer_.mutable_cpu_data());
        caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1) / (height_ * width_), broadcast_buffer_.cpu_data(),
                              spatial_sum_multiplier_.cpu_data(), Dtype(0),
                              spatial_statistic_.mutable_cpu_data());
        caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1) / num_,
                              spatial_statistic_.cpu_data(), batch_sum_multiplier_.cpu_data(),
                              Dtype(0), batch_statistic_.mutable_cpu_data());

        // Add to the moving average
        caffe_cpu_axpby(batch_statistic_.count(),
                        Dtype(1) - bn_momentum_, batch_statistic_.cpu_data(),
                        bn_momentum_, this->blobs_[3]->mutable_cpu_data());
    }

    // Add eps
    caffe_add_scalar(batch_statistic_.count(), bn_eps_,
                     batch_statistic_.mutable_cpu_data());
    // Inverse standard deviation
    caffe_powx(batch_statistic_.count(), batch_statistic_.cpu_data(),
               Dtype(-0.5), batch_statistic_.mutable_cpu_data());

    // Broadcast the inverse std
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                          Dtype(1), batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(),
                          Dtype(0), spatial_statistic_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                          height_ * width_, 1, Dtype(1),
                          spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
                          Dtype(0), broadcast_buffer_.mutable_cpu_data());
    // Multiply with the inverse std
    caffe_mul(broadcast_buffer_.count(), const_top_data,
              broadcast_buffer_.cpu_data(), top_data);

    // Save the normalized inputs and std for backprop
    if (!frozen_) {
        caffe_copy(broadcast_buffer_.count(), const_top_data,
                   x_norm_.mutable_cpu_data());
        caffe_copy(batch_statistic_.count(), batch_statistic_.cpu_data(),
                   x_inv_std_.mutable_cpu_data());
    }

    // Scale
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                          Dtype(1), batch_sum_multiplier_.cpu_data(), scale_data,
                          Dtype(0), spatial_statistic_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                          height_ * width_, 1, Dtype(1),
                          spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
                          Dtype(0), broadcast_buffer_.mutable_cpu_data());
    caffe_mul(broadcast_buffer_.count(), const_top_data,
              broadcast_buffer_.cpu_data(), top_data);

    // Shift
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                          Dtype(1), batch_sum_multiplier_.cpu_data(), shift_data,
                          Dtype(0), spatial_statistic_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                          height_ * width_, 1, Dtype(1),
                          spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
                          Dtype(0), broadcast_buffer_.mutable_cpu_data());
    caffe_add(broadcast_buffer_.count(), const_top_data,
              broadcast_buffer_.cpu_data(), top_data);
}

template <typename Dtype>
void BNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (frozen_) {
        if (propagate_down[0]) {
            const Dtype* const_top_diff = top[0]->cpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            // Use the moving average variance
            caffe_copy(batch_statistic_.count(), this->blobs_[3]->cpu_data(),
                    batch_statistic_.mutable_cpu_data());
            caffe_add_scalar(batch_statistic_.count(), bn_eps_,
                             batch_statistic_.mutable_cpu_data());
            caffe_powx(batch_statistic_.count(), batch_statistic_.cpu_data(),
                       Dtype(-0.5), batch_statistic_.mutable_cpu_data());
            // Divide slope with std
            caffe_mul(batch_statistic_.count(), this->blobs_[0]->cpu_data(),
                    batch_statistic_.cpu_data(), batch_statistic_.mutable_cpu_data());
            // Broadcast
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                                  Dtype(1), batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(),
                                  Dtype(0), spatial_statistic_.mutable_cpu_data());
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                                  height_ * width_, 1, Dtype(1),
                                  spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
                                  Dtype(0), broadcast_buffer_.mutable_cpu_data());
            // Elementwise multiply top grad with (slope / std)
            caffe_mul(broadcast_buffer_.count(), const_top_diff,
                      broadcast_buffer_.cpu_data(), bottom_diff);
        }
        return;
    }

    // gradient w.r.t. slope
    if (this->param_propagate_down_[0]) {
        const Dtype* const_top_diff = top[0]->cpu_diff();
        Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();
        caffe_mul(broadcast_buffer_.count(), x_norm_.cpu_data(), const_top_diff,
                  broadcast_buffer_.mutable_cpu_data());
        caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1), broadcast_buffer_.cpu_data(),
                              spatial_sum_multiplier_.cpu_data(), Dtype(0),
                              spatial_statistic_.mutable_cpu_data());
        caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
                              spatial_statistic_.cpu_data(), batch_sum_multiplier_.cpu_data(),
                              Dtype(1), scale_diff);
    }

    // gradient w.r.t. bias
    if (this->param_propagate_down_[1]) {
        const Dtype* const_top_diff = top[0]->cpu_diff();
        Dtype* shift_diff = this->blobs_[1]->mutable_cpu_diff();
        caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1), const_top_diff, spatial_sum_multiplier_.cpu_data(),
                              Dtype(0), spatial_statistic_.mutable_cpu_data());
        caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
                              spatial_statistic_.cpu_data(), batch_sum_multiplier_.cpu_data(),
                              Dtype(1), shift_diff);
    }

    // gradient w.r.t. normalized inputs
    if (propagate_down[0]) {
        const Dtype* const_top_diff = top[0]->cpu_diff();
        const Dtype* const_bottom_diff = bottom[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const Dtype* scale_data = this->blobs_[0]->cpu_data();

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                              Dtype(1), batch_sum_multiplier_.cpu_data(), scale_data,
                              Dtype(0), spatial_statistic_.mutable_cpu_data());
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                              height_ * width_, 1, Dtype(1), spatial_statistic_.cpu_data(),
                              spatial_sum_multiplier_.cpu_data(), Dtype(0),
                              broadcast_buffer_.mutable_cpu_data());
        caffe_mul(broadcast_buffer_.count(), const_top_diff,
                  broadcast_buffer_.cpu_data(), broadcast_buffer_.mutable_cpu_data());

        // sum of x_hat * (dl / dx_hat)
        caffe_mul(broadcast_buffer_.count(), x_norm_.cpu_data(),
                  broadcast_buffer_.cpu_data(), bottom_diff);
        caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1), const_bottom_diff, spatial_sum_multiplier_.cpu_data(),
                              Dtype(0), spatial_statistic_.mutable_cpu_data());
        caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
                              spatial_statistic_.cpu_data(), batch_sum_multiplier_.cpu_data(),
                              Dtype(0), batch_statistic_.mutable_cpu_data());

        // x_hat times the sum
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                              Dtype(1), batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(),
                              Dtype(0), spatial_statistic_.mutable_cpu_data());
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                              height_ * width_, 1, Dtype(1),
                              spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
                              Dtype(0), bottom_diff);
        caffe_mul(broadcast_buffer_.count(), x_norm_.cpu_data(),
                  const_bottom_diff, bottom_diff);

        // Subtract the average of x_hat times the sum
        caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1), broadcast_buffer_.cpu_data(),
                              spatial_sum_multiplier_.cpu_data(), Dtype(0),
                              spatial_statistic_.mutable_cpu_data());
        caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
                              spatial_statistic_.cpu_data(), batch_sum_multiplier_.cpu_data(),
                              Dtype(0), batch_statistic_.mutable_cpu_data());
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                              Dtype(1), batch_sum_multiplier_.cpu_data(), batch_statistic_.cpu_data(),
                              Dtype(0), spatial_statistic_.mutable_cpu_data());
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                              height_ * width_, 1, Dtype(1),
                              spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
                              Dtype(1), bottom_diff);
        caffe_cpu_axpby(broadcast_buffer_.count(), Dtype(1),
                        broadcast_buffer_.cpu_data(), Dtype(-1) / (num_ * height_ * width_),
                        bottom_diff);

        // Multiply with the inverse std
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                              Dtype(1), batch_sum_multiplier_.cpu_data(), x_inv_std_.cpu_data(),
                              Dtype(0), spatial_statistic_.mutable_cpu_data());
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                              height_ * width_, 1, Dtype(1),
                              spatial_statistic_.cpu_data(), spatial_sum_multiplier_.cpu_data(),
                              Dtype(0), broadcast_buffer_.mutable_cpu_data());
        caffe_mul(broadcast_buffer_.count(), const_bottom_diff,
                  broadcast_buffer_.cpu_data(), bottom_diff);
    }
}


#ifdef CPU_ONLY
STUB_GPU(BNLayer);
#endif

INSTANTIATE_CLASS(BNLayer);
REGISTER_LAYER_CLASS(BN);
}  // namespace caffe
