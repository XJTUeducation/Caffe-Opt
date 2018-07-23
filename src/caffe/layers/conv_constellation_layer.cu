#include <vector>

#include "caffe/layers/conv_constellation_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionConstellationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                       const vector<Blob<Dtype>*>& top) {
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

    //    Dtype* weight1 = this->blobs_[0]->mutable_cpu_data();

    //    int count = this->blobs()[0]->count(0);

    //    for(int i=0;i<count;i++)
    //    {
    //        weight1[i] = 1;
    //    }

    //    const Dtype* weight = this->blobs_[0]->gpu_data();

    //    for (int i = 0; i < bottom.size(); ++i) {
    //        const Dtype* bottom_data = bottom[i]->gpu_data();
    //        Dtype* top_data = top[i]->mutable_gpu_data();
    //        for (int n = 0; n < this->num_; ++n) {
    //            this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
    //                                   top_data + n * this->top_dim_);
    //            if (this->bias_term_) {
    //                const Dtype* bias = this->blobs_[1]->gpu_data();
    //                this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
    //            }
    //        }
    //    }

    //    std::cout << "****************** OUT PUT ******************************************\n\n";
    //    count = 0;

    ////    std::cout << "conv bottom pointer = " << (int64_t)(bottom[0]->gpu_data()) << "   " <<  (int64_t)(bottom[0]->cpu_data()) << "\n";
    ////    std::cout << "conv top pointer = " << (int64_t)(top[0]->gpu_data()) << "   " <<  (int64_t)(top[0]->cpu_data()) << "\n";

    //    for(int l=0;l<top[0]->shape(0);l++)
    //    {
    //        for(int i=0;i<top[0]->shape(1);i++)
    //    {
    //        for(int j=0;j<top[0]->shape(2);j++)
    //        {
    //            for(int k=0;k<top[0]->shape(3);k++)
    //            {
    //                std::cout << top[0]->cpu_data()[count++] << " ";
    ////                top[0]->mutable_cpu_data()[count++] = 2;

    //            }

    //            std::cout << "\n";
    //        }
    //        std::cout << "\n\n";
    //    }
    //        std::cout << "\n\nNEXT NUM \n\n";

    //}

    //    std::cout << "****************** OUT PUT ENDS ******************************************\n\n\n\n\n";

    //    count = 0;
    //    for(int i=0;i<top[0]->shape(1);i++)
    //    {
    //        for(int j=0;j<top[0]->shape(2);j++)
    //        {
    //            for(int k=0;k<top[0]->shape(3);k++)
    //            {
    //                top[0]->mutable_cpu_diff()[count++] = 1;
    //            }

    //            //            std::cout << "\n";
    //        }
    //        //        std::cout << "\n\n";
    //    }

    //    const Dtype* top_diff = top[0]->gpu_diff();
    //    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    //    this->backward_gpu_gemm(top_diff + 0 * this->top_dim_, weight,
    //                            bottom_diff + 0 * this->bottom_dim_);

    //    std::cout << "\n";

    //    std::cout << "****************** DIFF ******************************************\n";
    //    count = 0;
    //    for(int l=0;l<bottom[0]->shape(0);l++)
    //    {
    //        for(int i=0;i<bottom[0]->shape(2);i++)
    //        {
    //            for(int j=0;j<bottom[0]->shape(3);j++)
    //            {
    //                  int sum = 0;
    //                for(int k=0;k<bottom[0]->shape(1);k++)
    //                {
    //                  sum +=  (bottom[0]->cpu_diff() + k *bottom[0]->shape(2) * bottom[0]->shape(3)
    //                          + i * bottom[0]->shape(3) + j)[0];
    //                }

    //                std::cout << sum << "  ";
    //            }
    //            std::cout << "\n";
    //        }
    //        std::cout << "NEXT KERNEL \n\n";
    //    }

    //        const Dtype* bottom_data = bottom[0]->gpu_data();
    //        Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    //        this->weight_gpu_gemm(bottom_data + 0 * this->bottom_dim_,
    //                              top_diff + 0 * this->top_dim_, weight_diff);


    //        std::cout << "****************** DIFF ******************************************\n";
    //        count = 0;
    //        for(int l=0;l<this->blobs_[0]->shape(0);l++)
    //        {
    //            for(int i=0;i<this->blobs_[0]->shape(1);i++)
    //        {
    //            for(int j=0;j<this->blobs_[0]->shape(2);j++)
    //            {
    //                for(int k=0;k<this->blobs_[0]->shape(3);k++)
    //                {
    //                    std::cout << this->blobs_[0]->cpu_diff()[count++] << " ";
    //                }

    //                std::cout << "\n";
    //            }
    //            std::cout << "\n\n";
    //        }
    //            std::cout << "NEXT KERNEL \n\n";
    //        }
}

template <typename Dtype>
void ConvolutionConstellationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>&
                                                        bottom) {

    ////    std::cout << "****************** WEIGHTS DIFF ******************************************\n";
    //    int count = 0;
    //    for(int l=0;l<this->blobs_[0]->shape(0);l++)
    //    {
    //        for(int i=0;i<this->blobs_[0]->shape(1);i++)
    //        {
    //            for(int j=0;j<this->blobs_[0]->shape(2);j++)
    //            {
    //                for(int k=0;k<this->blobs_[0]->shape(3);k++)
    //                {
    //                    this->blobs_[0]->mutable_cpu_diff()[count++] = 1.0f;
    ////                    std::cout << this->blobs_[0]->cpu_diff()[count++] << " ";
    //                }

    ////                std::cout << "\n";
    //            }
    ////            std::cout << "\n\n";
    //        }
    ////        std::cout << "NEXT KERNEL \n\n";
    //    }


    //    //    std::cout << "****************** TOP DIFF ******************************************\n";
    //       count = 0;
    //        for(int l=0;l<top[0]->shape(0);l++)
    //        {
    //            for(int i=0;i<top[0]->shape(1);i++)
    //            {
    //                for(int j=0;j<top[0]->shape(2);j++)
    //                {
    //                    for(int k=0;k<top[0]->shape(3);k++)
    //                    {
    //                        top[0]->mutable_cpu_diff()[count++] = 1.0;
    //    //                    std::cout << this->blobs_[0]->cpu_diff()[count++] << " ";
    //                    }

    //    //                std::cout << "\n";
    //                }
    //    //            std::cout << "\n\n";
    //            }
    //    //        std::cout << "NEXT KERNEL \n\n";
    //        }


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


    //    std::cout << "******************CONV OUT PUT DIFF******************************************\n\n";
    //    count = 0;
    //    for(int l=0;l<bottom[0]->shape(0);l++)
    //    {
    //        std::cout << "\n\n conv BOTTOM NUM = " << l << "  \n\n";

    //        for(int i=0;i<bottom[0]->shape(1);i++)
    //        {
    //            for(int j=0;j<bottom[0]->shape(2);j++)
    //            {
    //                for(int k=0;k<bottom[0]->shape(3);k++)
    //                {
    //                    std::cout << bottom[0]->cpu_diff()[count++] << " ";
    //                }

    //                std::cout << "\n";
    //            }
    //            std::cout << "\n\n";
    //        }
    //        std::cout << "\n\nNEXT NUM \n\n";
    //    }

    //    std::cout << "******************CONV OUT PUT ENDS DIFF ******************************************\n\n\n\n\n";


    //    std::cout << "****************** UPDATED WEIGHTS DIFF ******************************************\n";
    //    count = 0;
    //    for(int l=0;l<this->blobs_[0]->shape(0);l++)
    //    {
    //        for(int i=0;i<this->blobs_[0]->shape(1);i++)
    //        {
    //            for(int j=0;j<this->blobs_[0]->shape(2);j++)
    //            {
    //                for(int k=0;k<this->blobs_[0]->shape(3);k++)
    //                {
    //                    std::cout << this->blobs_[0]->cpu_diff()[count++] << " ";
    //                }

    //                std::cout << "\n";
    //            }
    //            std::cout << "\n\n";
    //        }
    //        std::cout << "NEXT KERNEL \n\n";
    //    }

}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionConstellationLayer);

}  // namespace caffe
