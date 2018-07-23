#define BN_MULTI_GPU_SYNC

#ifdef BN_MULTI_GPU_SYNC

#include <algorithm>
#include <vector>

#include "caffe/layers/bn_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void bn_gather_statistics_gpu_kernel(int n, int n_instances, bool normalize, Dtype* batch_statistics)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < n)
    {
        Dtype ch_wise_accumulation = 0;

        for(int i=0;i < n_instances; i++)
            ch_wise_accumulation += batch_statistics[i * n + index];

        if(normalize)
            ch_wise_accumulation /= n_instances;

        // ONLY FIRST LOCATION BECAUSE COPY FROM CPU TO GPU WILL BE PERFORMED
        // HENCE NO POINT OF MULTIPLE COMPUTATIONS

        batch_statistics[index] = ch_wise_accumulation;
    }
}


template <typename Dtype>
void BNLayer<Dtype>::bn_gather_statistics_gpu(bool normalize)
{
    if( this->phase_ == TRAIN)
    {
        // GET NUMBER OF PARRALLEL INSTANCES
        // FOR THIS I HAVE ADDED A VARIABLE 'n_instances' TO layer.hpp file
        // which is set during the net creation using layer->set_n_instances(Caffe::solver_count())

        if(this->n_instances_ > 1)
        {
            pthread_mutex_lock(&mutex_instances_finished_);
            {
                batch_statistics_ptrs_gpu_.push_back(batch_statistic_.mutable_cpu_data());

                if(batch_statistics_ptrs_gpu_.size() == this->n_instances_)
                {
                    int num_kernels = channels_;

                    for(int i=0;i<this->n_instances_;i++)
                    {
                        int offset = batch_statistic_instances_.offset(i);

                        memcpy(batch_statistic_instances_.mutable_cpu_data() + offset,
                               batch_statistics_ptrs_gpu_[i],
                               channels_ * sizeof(Dtype));
                    }


                    // LAUNCH KERNELS FOR CHANNEL WISE ACCUMULATION
                    bn_gather_statistics_gpu_kernel<<<CAFFE_GET_BLOCKS(num_kernels),
                            CAFFE_CUDA_NUM_THREADS>>>(num_kernels, this->n_instances_, normalize, batch_statistic_instances_.mutable_gpu_data());

                    CUDA_POST_KERNEL_CHECK;

                    for(int i=0;i<this->n_instances_;i++)
                    {
                        memcpy(batch_statistics_ptrs_gpu_[i],
                               batch_statistic_instances_.mutable_cpu_data(),
                               channels_ * sizeof(Dtype));
                    }


                    batch_statistics_ptrs_gpu_.clear();
                    pthread_cond_broadcast(&cond_gathered_);
                }
                else
                    pthread_cond_wait(&cond_gathered_, &mutex_instances_finished_);

            }
            pthread_mutex_unlock(&mutex_instances_finished_);

        }
    }
}


template <typename Dtype>
void BNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {


#ifdef BN_NO_MEM_BANK_

    int device;
    cudaGetDevice(&device);
    broadcast_buffer_.data()->set_gpu_data(broadcast_buffer_pointer_map_->operator [](device));

#endif

    const Dtype* const_bottom_data = bottom[0]->gpu_data();
    const Dtype* const_top_data = top[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    const Dtype* scale_data = this->blobs_[0]->gpu_data();
    const Dtype* shift_data = this->blobs_[1]->gpu_data();

    // Mean normalization
    if (frozen_ || this->phase_ == TEST) {
        // Use the moving average mean
        caffe_copy(batch_statistic_.count(), this->blobs_[2]->gpu_data(),
                batch_statistic_.mutable_gpu_data());
    } else {
        // Compute the mean by averaging over spatial and batch dimensions.
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1) / (height_ * width_), const_bottom_data,
                              spatial_sum_multiplier_.gpu_data(), Dtype(0),
                              spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_,
                              Dtype(1) / num_, spatial_statistic_.gpu_data(),
                              batch_sum_multiplier_.gpu_data(), Dtype(0),
                              batch_statistic_.mutable_gpu_data());

        // GATER MEAN STATISTICS ACROSS GPUS

#ifdef BN_MULTI_GPU_SYNC

        bn_gather_statistics_gpu(true);

#endif


        // Add to the moving average
        if (!frozen_) {
            caffe_gpu_axpby(batch_statistic_.count(),
                            Dtype(1) - bn_momentum_, batch_statistic_.gpu_data(),
                            bn_momentum_, this->blobs_[2]->mutable_gpu_data());
        }
    }
    // Broadcast the mean vector
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                          Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
                          Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                          height_ * width_, 1, Dtype(-1),
                          spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
                          Dtype(0), broadcast_buffer_.mutable_gpu_data());
    // Subtract
    caffe_gpu_add(broadcast_buffer_.count(), const_bottom_data,
                  broadcast_buffer_.gpu_data(), top_data);

    // Variance normalization
    if (frozen_ || this->phase_ == TEST) {
        // Use the moving average variance
        caffe_copy(batch_statistic_.count(), this->blobs_[3]->gpu_data(),
                batch_statistic_.mutable_gpu_data());
    } else {
        caffe_gpu_powx(broadcast_buffer_.count(), const_top_data, Dtype(2),
                       broadcast_buffer_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1) / (height_ * width_), broadcast_buffer_.gpu_data(),
                              spatial_sum_multiplier_.gpu_data(), Dtype(0),
                              spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1) / num_,
                              spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
                              Dtype(0), batch_statistic_.mutable_gpu_data());

        // GATER MEAN STATISTICS ACROSS GPUS

#ifdef BN_MULTI_GPU_SYNC

        bn_gather_statistics_gpu(true);

#endif

        // Add to the moving average
        caffe_gpu_axpby(batch_statistic_.count(),
                        Dtype(1) - bn_momentum_, batch_statistic_.gpu_data(),
                        bn_momentum_, this->blobs_[3]->mutable_gpu_data());
    }

    // Add eps
    caffe_gpu_add_scalar(batch_statistic_.count(), bn_eps_,
                         batch_statistic_.mutable_gpu_data());
    // Inverse standard deviation
    caffe_gpu_powx(batch_statistic_.count(), batch_statistic_.gpu_data(),
                   Dtype(-0.5), batch_statistic_.mutable_gpu_data());
    // Broadcast the inverse std
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                          Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
                          Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                          height_ * width_, 1, Dtype(1),
                          spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
                          Dtype(0), broadcast_buffer_.mutable_gpu_data());
    // Multiply with the inverse std
    caffe_gpu_mul(broadcast_buffer_.count(), const_top_data,
                  broadcast_buffer_.gpu_data(), top_data);

    // Save the normalized inputs and std for backprop
    if (!frozen_) {

#ifdef BN_OPTMIZE_X_NORM

        cudaMemcpy(x_norm_cpu_.mutable_cpu_data(), const_top_data, broadcast_buffer_.count()*sizeof(Dtype),cudaMemcpyDeviceToHost);

#else

                caffe_copy(broadcast_buffer_.count(), const_top_data,
                   x_norm_.mutable_gpu_data());
#endif

        caffe_copy(batch_statistic_.count(), batch_statistic_.gpu_data(),
                   x_inv_std_.mutable_gpu_data());

    }

    // Scale
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                          Dtype(1), batch_sum_multiplier_.gpu_data(), scale_data,
                          Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                          height_ * width_, 1, Dtype(1),
                          spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
                          Dtype(0), broadcast_buffer_.mutable_gpu_data());
    caffe_gpu_mul(broadcast_buffer_.count(), const_top_data,
                  broadcast_buffer_.gpu_data(), top_data);

    // Shift
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                          Dtype(1), batch_sum_multiplier_.gpu_data(), shift_data,
                          Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                          height_ * width_, 1, Dtype(1),
                          spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
                          Dtype(0), broadcast_buffer_.mutable_gpu_data());
    caffe_gpu_add(broadcast_buffer_.count(), const_top_data,
                  broadcast_buffer_.gpu_data(), top_data);
}

template <typename Dtype>
void BNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


#ifdef BN_NO_MEM_BANK_

    int device;
    cudaGetDevice(&device);
    broadcast_buffer_.data()->set_gpu_data(broadcast_buffer_pointer_map_->operator [](device));

#endif

    if (frozen_) {
        if (propagate_down[0]) {
            const Dtype* const_top_diff = top[0]->gpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            // Use the moving average variance
            caffe_copy(batch_statistic_.count(), this->blobs_[3]->gpu_data(),
                    batch_statistic_.mutable_gpu_data());
            caffe_gpu_add_scalar(batch_statistic_.count(), bn_eps_,
                                 batch_statistic_.mutable_gpu_data());
            caffe_gpu_powx(batch_statistic_.count(), batch_statistic_.gpu_data(),
                           Dtype(-0.5), batch_statistic_.mutable_gpu_data());
            // Multiple slope with inverse std
            caffe_gpu_mul(batch_statistic_.count(), this->blobs_[0]->gpu_data(),
                    batch_statistic_.gpu_data(), batch_statistic_.mutable_gpu_data());
            // Broadcast
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                                  Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
                                  Dtype(0), spatial_statistic_.mutable_gpu_data());
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                                  height_ * width_, 1, Dtype(1),
                                  spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
                                  Dtype(0), broadcast_buffer_.mutable_gpu_data());
            // Elementwise multiply top grad with (slope / std)
            caffe_gpu_mul(broadcast_buffer_.count(), const_top_diff,
                          broadcast_buffer_.gpu_data(), bottom_diff);
        }
        return;
    }


    // gradient w.r.t. slope
    if (this->param_propagate_down_[0]) {
        const Dtype* const_top_diff = top[0]->gpu_diff();
        Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();

#ifdef BN_OPTMIZE_X_NORM

    cudaMemcpy(broadcast_buffer_.mutable_gpu_data(), x_norm_cpu_.cpu_data(),x_norm_cpu_.count()*sizeof(Dtype),cudaMemcpyHostToDevice);

    caffe_gpu_mul(broadcast_buffer_.count(), broadcast_buffer_.gpu_data(), const_top_diff,
                  broadcast_buffer_.mutable_gpu_data());

#else
        caffe_gpu_mul(broadcast_buffer_.count(), x_norm_.gpu_data(), const_top_diff,
                      broadcast_buffer_.mutable_gpu_data());
#endif


        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1), broadcast_buffer_.gpu_data(),
                              spatial_sum_multiplier_.gpu_data(), Dtype(0),
                              spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
                              spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
                              Dtype(1), scale_diff);
    }

    // gradient w.r.t. bias
    if (this->param_propagate_down_[1]) {
        const Dtype* const_top_diff = top[0]->gpu_diff();
        Dtype* shift_diff = this->blobs_[1]->mutable_gpu_diff();
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1), const_top_diff, spatial_sum_multiplier_.gpu_data(),
                              Dtype(0), spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
                              spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
                              Dtype(1), shift_diff);
    }

    // gradient w.r.t. normalized inputs
    if (propagate_down[0]) {
        const Dtype* const_top_diff = top[0]->gpu_diff();
        const Dtype* const_bottom_diff = bottom[0]->gpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const Dtype* scale_data = this->blobs_[0]->gpu_data();
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                              Dtype(1), batch_sum_multiplier_.gpu_data(), scale_data,
                              Dtype(0), spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                              height_ * width_, 1, Dtype(1), spatial_statistic_.gpu_data(),
                              spatial_sum_multiplier_.gpu_data(), Dtype(0),
                              broadcast_buffer_.mutable_gpu_data());
        caffe_gpu_mul(broadcast_buffer_.count(), const_top_diff,
                      broadcast_buffer_.gpu_data(), broadcast_buffer_.mutable_gpu_data());

        // sum of x_hat * (dl / dx_hat)
#ifdef BN_OPTMIZE_X_NORM

    cudaMemcpy(bottom_diff, x_norm_cpu_.cpu_data(),x_norm_cpu_.count()*sizeof(Dtype),cudaMemcpyHostToDevice);

    caffe_gpu_mul(broadcast_buffer_.count(), bottom_diff,
                  broadcast_buffer_.gpu_data(), bottom_diff);


#else
        caffe_gpu_mul(broadcast_buffer_.count(), x_norm_.gpu_data(),
                      broadcast_buffer_.gpu_data(), bottom_diff);

#endif


        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1), const_bottom_diff, spatial_sum_multiplier_.gpu_data(),
                              Dtype(0), spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
                              spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
                              Dtype(0), batch_statistic_.mutable_gpu_data());


        // GATER MEAN STATISTICS ACROSS GPUS

#ifdef BN_MULTI_GPU_SYNC

        bn_gather_statistics_gpu();

#endif


        // x_hat times the sum
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                              Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
                              Dtype(0), spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                              height_ * width_, 1, Dtype(1),
                              spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
                              Dtype(0), bottom_diff);

#ifdef BN_OPTMIZE_X_NORM

        // top data is of no use at this point
    cudaMemcpy(top[0]->mutable_gpu_data(), x_norm_cpu_.cpu_data(),x_norm_cpu_.count()*sizeof(Dtype),cudaMemcpyHostToDevice);

    caffe_gpu_mul(broadcast_buffer_.count(), top[0]->mutable_gpu_data(),
                  const_bottom_diff, bottom_diff);


#else
        caffe_gpu_mul(broadcast_buffer_.count(), x_norm_.gpu_data(),
                      const_bottom_diff, bottom_diff);

#endif


        // Subtract the average of x_hat times the sum
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1), broadcast_buffer_.gpu_data(),
                              spatial_sum_multiplier_.gpu_data(), Dtype(0),
                              spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
                              spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
                              Dtype(0), batch_statistic_.mutable_gpu_data());


        // GATER the average of x_hat times the sum ACROSS GPUS

#ifdef BN_MULTI_GPU_SYNC

        bn_gather_statistics_gpu();

#endif

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                              Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
                              Dtype(0), spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                              height_ * width_, 1, Dtype(1),
                              spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
                              Dtype(1), bottom_diff);


        int n_elements = num_ * height_ * width_;

        // Total elements will be equal to N x H x W x num_instances

#ifdef BN_MULTI_GPU_SYNC

        n_elements *= this->n_instances_;

#endif

        caffe_gpu_axpby(broadcast_buffer_.count(), Dtype(1),
                        broadcast_buffer_.gpu_data(), Dtype(-1) / (n_elements),
                        bottom_diff);

        // Multiply with the inverse std
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                              Dtype(1), batch_sum_multiplier_.gpu_data(), x_inv_std_.gpu_data(),
                              Dtype(0), spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                              height_ * width_, 1, Dtype(1),
                              spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
                              Dtype(0), broadcast_buffer_.mutable_gpu_data());
        caffe_gpu_mul(broadcast_buffer_.count(), const_bottom_diff,
                      broadcast_buffer_.gpu_data(), bottom_diff);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(BNLayer);

}  // namespace caffe

#else

#include <algorithm>
#include <vector>

#include "caffe/layers/bn_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {

    int device;
    cudaGetDevice(&device);
    broadcast_buffer_.data()->set_gpu_data(broadcast_buffer_pointer_map_->operator [](device));


    const Dtype* const_bottom_data = bottom[0]->gpu_data();
    const Dtype* const_top_data = top[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();



    //    std::cout << "bn top bottom  =  " << (int64_t)top[0] << "  " << (int64_t)bottom[0] << "\n";
    //    std::cout << "bn top bottom data pointers =  " << (int64_t)const_top_data << "  " << (int64_t)const_bottom_data << "\n";



    const Dtype* scale_data = this->blobs_[0]->gpu_data();
    const Dtype* shift_data = this->blobs_[1]->gpu_data();

    // Mean normalization
    if (frozen_ || this->phase_ == TEST) {
        // Use the moving average mean
        caffe_copy(batch_statistic_.count(), this->blobs_[2]->gpu_data(),
                batch_statistic_.mutable_gpu_data());
    } else {
        // Compute the mean by averaging over spatial and batch dimensions.
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1) / (height_ * width_), const_bottom_data,
                              spatial_sum_multiplier_.gpu_data(), Dtype(0),
                              spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_,
                              Dtype(1) / num_, spatial_statistic_.gpu_data(),
                              batch_sum_multiplier_.gpu_data(), Dtype(0),
                              batch_statistic_.mutable_gpu_data());
        // Add to the moving average
        if (!frozen_) {
            caffe_gpu_axpby(batch_statistic_.count(),
                            Dtype(1) - bn_momentum_, batch_statistic_.gpu_data(),
                            bn_momentum_, this->blobs_[2]->mutable_gpu_data());
        }
    }
    // Broadcast the mean vector
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                          Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
                          Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                          height_ * width_, 1, Dtype(-1),
                          spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
                          Dtype(0), broadcast_buffer_.mutable_gpu_data());
    // Subtract
    caffe_gpu_add(broadcast_buffer_.count(), const_bottom_data,
                  broadcast_buffer_.gpu_data(), top_data);

    // Variance normalization
    if (frozen_ || this->phase_ == TEST) {
        // Use the moving average variance
        caffe_copy(batch_statistic_.count(), this->blobs_[3]->gpu_data(),
                batch_statistic_.mutable_gpu_data());
    } else {
        caffe_gpu_powx(broadcast_buffer_.count(), const_top_data, Dtype(2),
                       broadcast_buffer_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1) / (height_ * width_), broadcast_buffer_.gpu_data(),
                              spatial_sum_multiplier_.gpu_data(), Dtype(0),
                              spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1) / num_,
                              spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
                              Dtype(0), batch_statistic_.mutable_gpu_data());

        // Add to the moving average
        caffe_gpu_axpby(batch_statistic_.count(),
                        Dtype(1) - bn_momentum_, batch_statistic_.gpu_data(),
                        bn_momentum_, this->blobs_[3]->mutable_gpu_data());
    }

    // Add eps
    caffe_gpu_add_scalar(batch_statistic_.count(), bn_eps_,
                         batch_statistic_.mutable_gpu_data());
    // Inverse standard deviation
    caffe_gpu_powx(batch_statistic_.count(), batch_statistic_.gpu_data(),
                   Dtype(-0.5), batch_statistic_.mutable_gpu_data());
    // Broadcast the inverse std
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                          Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
                          Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                          height_ * width_, 1, Dtype(1),
                          spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
                          Dtype(0), broadcast_buffer_.mutable_gpu_data());
    // Multiply with the inverse std
    caffe_gpu_mul(broadcast_buffer_.count(), const_top_data,
                  broadcast_buffer_.gpu_data(), top_data);

    // Save the normalized inputs and std for backprop
    if (!frozen_) {
        caffe_copy(broadcast_buffer_.count(), const_top_data,
                   x_norm_.mutable_gpu_data());
        caffe_copy(batch_statistic_.count(), batch_statistic_.gpu_data(),
                   x_inv_std_.mutable_gpu_data());
    }

    // Scale
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                          Dtype(1), batch_sum_multiplier_.gpu_data(), scale_data,
                          Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                          height_ * width_, 1, Dtype(1),
                          spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
                          Dtype(0), broadcast_buffer_.mutable_gpu_data());
    caffe_gpu_mul(broadcast_buffer_.count(), const_top_data,
                  broadcast_buffer_.gpu_data(), top_data);

    // Shift
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                          Dtype(1), batch_sum_multiplier_.gpu_data(), shift_data,
                          Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                          height_ * width_, 1, Dtype(1),
                          spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
                          Dtype(0), broadcast_buffer_.mutable_gpu_data());
    caffe_gpu_add(broadcast_buffer_.count(), const_top_data,
                  broadcast_buffer_.gpu_data(), top_data);
}

template <typename Dtype>
void BNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    int device;
    cudaGetDevice(&device);
    broadcast_buffer_.data()->set_gpu_data(broadcast_buffer_pointer_map_->operator [](device));

    if (frozen_) {
        if (propagate_down[0]) {
            const Dtype* const_top_diff = top[0]->gpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            // Use the moving average variance
            caffe_copy(batch_statistic_.count(), this->blobs_[3]->gpu_data(),
                    batch_statistic_.mutable_gpu_data());
            caffe_gpu_add_scalar(batch_statistic_.count(), bn_eps_,
                                 batch_statistic_.mutable_gpu_data());
            caffe_gpu_powx(batch_statistic_.count(), batch_statistic_.gpu_data(),
                           Dtype(-0.5), batch_statistic_.mutable_gpu_data());
            // Multiple slope with inverse std
            caffe_gpu_mul(batch_statistic_.count(), this->blobs_[0]->gpu_data(),
                    batch_statistic_.gpu_data(), batch_statistic_.mutable_gpu_data());
            // Broadcast
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                                  Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
                                  Dtype(0), spatial_statistic_.mutable_gpu_data());
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                                  height_ * width_, 1, Dtype(1),
                                  spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
                                  Dtype(0), broadcast_buffer_.mutable_gpu_data());
            // Elementwise multiply top grad with (slope / std)
            caffe_gpu_mul(broadcast_buffer_.count(), const_top_diff,
                          broadcast_buffer_.gpu_data(), bottom_diff);
        }
        return;
    }

    // gradient w.r.t. slope
    if (this->param_propagate_down_[0]) {
        const Dtype* const_top_diff = top[0]->gpu_diff();
        Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();
        caffe_gpu_mul(broadcast_buffer_.count(), x_norm_.gpu_data(), const_top_diff,
                      broadcast_buffer_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1), broadcast_buffer_.gpu_data(),
                              spatial_sum_multiplier_.gpu_data(), Dtype(0),
                              spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
                              spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
                              Dtype(1), scale_diff);
    }

    // gradient w.r.t. bias
    if (this->param_propagate_down_[1]) {
        const Dtype* const_top_diff = top[0]->gpu_diff();
        Dtype* shift_diff = this->blobs_[1]->mutable_gpu_diff();
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1), const_top_diff, spatial_sum_multiplier_.gpu_data(),
                              Dtype(0), spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
                              spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
                              Dtype(1), shift_diff);
    }

    // gradient w.r.t. normalized inputs
    if (propagate_down[0]) {
        const Dtype* const_top_diff = top[0]->gpu_diff();
        const Dtype* const_bottom_diff = bottom[0]->gpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const Dtype* scale_data = this->blobs_[0]->gpu_data();
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                              Dtype(1), batch_sum_multiplier_.gpu_data(), scale_data,
                              Dtype(0), spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                              height_ * width_, 1, Dtype(1), spatial_statistic_.gpu_data(),
                              spatial_sum_multiplier_.gpu_data(), Dtype(0),
                              broadcast_buffer_.mutable_gpu_data());
        caffe_gpu_mul(broadcast_buffer_.count(), const_top_diff,
                      broadcast_buffer_.gpu_data(), broadcast_buffer_.mutable_gpu_data());

        // sum of x_hat * (dl / dx_hat)
        caffe_gpu_mul(broadcast_buffer_.count(), x_norm_.gpu_data(),
                      broadcast_buffer_.gpu_data(), bottom_diff);
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1), const_bottom_diff, spatial_sum_multiplier_.gpu_data(),
                              Dtype(0), spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
                              spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
                              Dtype(0), batch_statistic_.mutable_gpu_data());

        // x_hat times the sum
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                              Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
                              Dtype(0), spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                              height_ * width_, 1, Dtype(1),
                              spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
                              Dtype(0), bottom_diff);
        caffe_gpu_mul(broadcast_buffer_.count(), x_norm_.gpu_data(),
                      const_bottom_diff, bottom_diff);

        // Subtract the average of x_hat times the sum
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
                              Dtype(1), broadcast_buffer_.gpu_data(),
                              spatial_sum_multiplier_.gpu_data(), Dtype(0),
                              spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
                              spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
                              Dtype(0), batch_statistic_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                              Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
                              Dtype(0), spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                              height_ * width_, 1, Dtype(1),
                              spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
                              Dtype(1), bottom_diff);
        caffe_gpu_axpby(broadcast_buffer_.count(), Dtype(1),
                        broadcast_buffer_.gpu_data(), Dtype(-1) / (num_ * height_ * width_),
                        bottom_diff);

        // Multiply with the inverse std
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
                              Dtype(1), batch_sum_multiplier_.gpu_data(), x_inv_std_.gpu_data(),
                              Dtype(0), spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
                              height_ * width_, 1, Dtype(1),
                              spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
                              Dtype(0), broadcast_buffer_.mutable_gpu_data());
        caffe_gpu_mul(broadcast_buffer_.count(), const_bottom_diff,
                      broadcast_buffer_.gpu_data(), bottom_diff);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(BNLayer);

}  // namespace caffe

#endif


