#include <vector>
#include "caffe/layers/concat_optimized_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {
template <typename Dtype>
void ConcatOptimizedLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {


    //    std::cout << "in concat forwd gpu\n";

    //    std::cout << "****************** CONCAT in PUT ******************************************\n\n";
    //    int count = 0;
    //    for(int l=0;l<bottom.size();l++)
    //    {
    //        count = 0;
    //        std::cout << "concate bottom pointer = " << (int64_t)(bottom[l]->gpu_data()) << "  " << (int64_t)(bottom[l]->cpu_data()) << "\n";
    //        for(int m=0;m<bottom[l]->shape(0);m++)
    //        {
    //            for(int i=0;i<bottom[l]->shape(1);i++)
    //            {
    //                for(int j=0;j<bottom[l]->shape(2);j++)
    //                {
    //                    for(int k=0;k<bottom[l]->shape(3);k++)
    //                    {
    //                        std::cout << bottom[l]->cpu_data()[count++] << " ";

    //                    }

    //                    std::cout << "\n";
    //                }
    //                std::cout << "\n\n";
    //            }
    //            std::cout << "\n\nNEXT NUM \n\n";
    //        }
    //        std::cout << "\n\nNEXT BOTTOM \n\n";
    //    }
    //    std::cout << "****************** CONCAT in PUT ENDS ******************************************\n\n";


    int num = bottom[0]->shape(0);

    if(num > 1)
    {
        //        std::cout << "in concat forwd gpu 111111\n";


#ifdef CONCAT_OPT_NO_MEM_BANK

        int device;
        cudaGetDevice(&device);
        Dtype* temp_gpu_buffer_ = temp_data_pointer_map_->operator [](device);

#else

        Dtype* temp_gpu_buffer_ = temp_buffer_.mutable_gpu_data();

#endif

//        std::cout << "concat temp pointer = " << (u_int64_t) temp_gpu_buffer_<< "\n";

        int offset = 0;

        for(int i=0;i<num;i++)
        {
            for(int j=0;j<bottom.size();j++)
            {
                int data_count = bottom[j]->shape(1) * bottom[j]->shape(2) * bottom[j]->shape(3);

                Dtype* temp_data = temp_gpu_buffer_ + offset;
                const Dtype* blob_data = bottom[j]->gpu_data() + bottom[j]->offset(i,0,0,0);

                //                Dtype* blob_data = data_pointer_map_->operator [](device) + offset;
                cudaMemcpy(temp_data, blob_data, data_count * sizeof(Dtype), cudaMemcpyDeviceToDevice);

                offset += data_count;
            }
        }

        int n_channels = 0;

        int h = bottom[0]->shape(2);
        int w = bottom[0]->shape(3);

        for(int i=0;i< bottom.size();i++)
            n_channels += bottom[i]->shape(1);

        int total_count = num * n_channels * h * w;

        Dtype* temp_data = temp_gpu_buffer_;
        Dtype* blob_data = top[0]->mutable_gpu_data();

        caffe_gpu_memcpy(total_count * sizeof(Dtype), temp_data, blob_data);
    }

    if(this->phase_ == TRAIN)
        cudaMemset(top[0]->mutable_gpu_diff(), 0, top[0]->count() * sizeof(Dtype));












    //    std::cout << "****************** CONCAT OUT PUT ******************************************\n\n";
    //   int count = 0;
    //    for(int l=0;l<top[0]->shape(0);l++)
    //    {
    //        std::cout << "concate top pointer = " << (int64_t)(top[0]->gpu_data()) << "  " << (int64_t)(top[0]->cpu_data()) << "\n";

    //        for(int i=0;i<top[0]->shape(1);i++)
    //        {
    //            for(int j=0;j<top[0]->shape(2);j++)
    //            {
    //                for(int k=0;k<top[0]->shape(3);k++)
    //                {
    //                    std::cout << top[0]->cpu_data()[count++] << " ";
    //                }

    //                std::cout << "\n";
    //            }
    //            std::cout << "\n\n";
    //        }
    //        std::cout << "\n\nNEXT NUM \n\n";
    //    }
    //    std::cout << "****************** CONCAT OUT PUT ENDS ******************************************\n\n";

}

template <typename Dtype>
void ConcatOptimizedLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


    //        static int diff_fill = 0;

    //    std::cout << "in concat backward gpu\n";


    int count = 0;
    //    if(diff_fill == 0)
    //    {
    ////        std::cout << "******************concat OUT PUT DIFF orgin******************************************\n\n";
    ////        std::cout << "concate diff top pointer = " << (int64_t)(top[0]->gpu_diff()) << "  " << (int64_t)(top[0]->cpu_diff()) << "\n";

    //        for(int l=0;l<top[0]->shape(0);l++)
    //        {

    //            for(int i=0;i<top[0]->shape(1);i++)
    //            {
    //                for(int j=0;j<top[0]->shape(2);j++)
    //                {
    //                    for(int k=0;k<top[0]->shape(3);k++)
    //                    {
    //                        top[0]->mutable_cpu_diff()[count++] = i+1;

    ////                        std::cout <<top[0]->cpu_diff()[count-1] << "  ";
    //                    }

    ////                    std::cout << "\n";
    //                }
    ////                std::cout << "\n\n";
    //            }
    ////            std::cout << "\n\nNEXT NUM \n\n";
    //        }
    //        diff_fill++;
    ////        std::cout << "******************concat OUT PUT ENDS DIFF orgin******************************************\n\n\n\n\n";

    //    top[0]->gpu_diff();

    //    }

    int num = bottom[0]->shape(0);

    if(num > 1)
    {
        //        std::cout << "in concat backward gpu  11111111111111\n";

#ifdef CONCAT_OPT_NO_MEM_BANK

        int device;
        cudaGetDevice(&device);
        Dtype* temp_gpu_buffer_ = temp_data_pointer_map_->operator [](device);

#else

        Dtype* temp_gpu_buffer_ = temp_buffer_.mutable_gpu_data();

#endif

        int n_channels = 0;

        int h = bottom[0]->shape(2);
        int w = bottom[0]->shape(3);

        for(int i=0;i< bottom.size();i++)
            n_channels += bottom[i]->shape(1);

        int top_offset = n_channels * h * w;


        // REARRANGE DATA
        {
            int offset = 0;
            int blob_offset = 0;

            for(int i=0;i<bottom.size();i++)
            {
                for(int j=0;j<num;j++)
                {
                    int data_count = bottom[i]->shape(1) * bottom[i]->shape(2) * bottom[i]->shape(3);

                    Dtype* temp_data = temp_gpu_buffer_ + offset;
                    const Dtype* blob_data = top[0]->gpu_data() + blob_offset + j * top_offset;

                    cudaMemcpy(temp_data, blob_data, data_count * sizeof(Dtype), cudaMemcpyDeviceToDevice);

                    offset += data_count;
                }
                blob_offset += bottom[i]->shape(1) * bottom[i]->shape(2) * bottom[i]->shape(3);
            }

            int total_count = num * n_channels * h * w;

            Dtype* temp_data = temp_gpu_buffer_;
            Dtype* blob_data = top[0]->mutable_gpu_data();

            cudaMemcpy(blob_data, temp_data, total_count * sizeof(Dtype), cudaMemcpyDeviceToDevice);
        }

        // REAARANGE DIFF
        {
            int offset = 0;
            int blob_offset = 0;

            for(int i=0;i<bottom.size();i++)
            {
                for(int j=0;j<num;j++)
                {
                    int data_count = bottom[i]->shape(1) * bottom[i]->shape(2) * bottom[i]->shape(3);

                    Dtype* temp_data = temp_gpu_buffer_ + offset;
                    const Dtype* blob_data = top[0]->gpu_diff() + blob_offset + j * top_offset;

                    cudaMemcpy(temp_data, blob_data, data_count * sizeof(Dtype), cudaMemcpyDeviceToDevice);

                    offset += data_count;
                }
                blob_offset += bottom[i]->shape(1) * bottom[i]->shape(2) * bottom[i]->shape(3);
            }

            int total_count = num * n_channels * h * w;

            Dtype* temp_data = temp_gpu_buffer_;
            Dtype* blob_data = top[0]->mutable_gpu_diff();

            cudaMemcpy(blob_data, temp_data, total_count * sizeof(Dtype), cudaMemcpyDeviceToDevice);
        }
    }

    for(int i=0;i<bottom.size();i++)
    {
        if(!propagate_down[i])
        {
            //            std::cout << "clearing diff concat gpu \n";
            cudaMemset(bottom[i]->mutable_gpu_diff(),0,bottom[i]->count() * sizeof(Dtype));
        }
    }

    //    std::cout << "****************** CONCAT OUT PUT BCWD ******************************************\n\n";


    //    for(int m=0;m<bottom.size();m++)
    //    {
    //        std::cout << "\n\n CONCAT BOTTOM NUM = " << m << "  \n\n";
    //        count = 0;

    //        for(int l=0;l<bottom[m]->shape(0);l++)
    //        {
    //            for(int i=0;i<bottom[m]->shape(1);i++)
    //            {
    //                for(int j=0;j<bottom[m]->shape(2);j++)
    //                {
    //                    for(int k=0;k<bottom[m]->shape(3);k++)
    //                    {
    //                        std::cout << bottom[m]->cpu_data()[count++] << " ";
    //                    }

    //                    std::cout << "\n";
    //                }
    //                std::cout << "\n\n";
    //            }
    //            std::cout << "\n\nNEXT NUM \n\n";
    //        }
    //    }
    //    std::cout << "****************** CONCAT OUT PUT ENDS BCWD******************************************\n\n";


    //    std::cout << "****************** CONCAT OUT PUT DIFF BCWD ******************************************\n\n";

    //    std::cout << "concate diff bottom pointer = " << (int64_t)(bottom[0]->gpu_diff())
    //            << "  " << (int64_t)(bottom[0]->cpu_diff()) << "\n";

    //      for(int m=0;m<bottom.size();m++)
    //    {
    //        std::cout << "\n\n CONCAT BOTTOM NUM = " << m << "  \n\n";
    //        count = 0;

    ////        for(int l=0;l<bottom[m]->shape(0);l++)
    //        {
    //            for(int i=0;i<10;i++)//bottom[m]->shape(1);i++)
    //            {
    //                for(int j=0;j<bottom[m]->shape(2);j++)
    //                {
    //                    for(int k=0;k<bottom[m]->shape(3);k++)
    //                    {
    //                        std::cout << bottom[m]->cpu_diff()[count++] << " ";
    //                    }

    //                    std::cout << "\n";
    //                }
    //                std::cout << "\n\n";
    //            }
    //            std::cout << "\n\nNEXT NUM \n\n";
    //        }
    //    }
    //    std::cout << "****************** CONCAT OUT PUT DIFF ENDS BCWD******************************************\n\n";



}

INSTANTIATE_LAYER_GPU_FUNCS(ConcatOptimizedLayer);

}  // namespace caffe
