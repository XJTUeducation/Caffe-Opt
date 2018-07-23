#include <vector>
#include "caffe/layers/concat_optimized_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {


template <typename Dtype>
std::vector<unsigned long int> ConcatOptimizedLayer<Dtype>::buffer_old_count_ = std::vector<unsigned long int>(0);

template <typename Dtype>
std::map<int, Dtype*>* ConcatOptimizedLayer<Dtype>::temp_data_pointer_map_ = NULL;

template <typename Dtype >
std::map<int, std::map<void*, typename ConcatOptimizedLayer<Dtype>::concat_child* >* > ConcatOptimizedLayer<Dtype>::concat_chain_ =
        std::map<int, std::map<void*, typename ConcatOptimizedLayer<Dtype>::concat_child* >* >();

template<typename Dtype>
ConcatOptimizedLayer<Dtype>::~ConcatOptimizedLayer()
{


#ifdef CONCAT_OPT_NO_MEM_BANK

    int device;
    cudaGetDevice(&device);

    if(concat_child_->own_data)
    {
        Dtype* gpu_buffer = NULL;
        data_pointer_map_->operator [](device) = gpu_buffer;
        cudaFree(gpu_buffer);

        if(this->phase_ == TRAIN)
        {
            Dtype* gpu_buffer = NULL;
            diff_pointer_map_->operator [](device) = gpu_buffer;
            cudaFree(gpu_buffer);
        }
    }
#endif

}


template <typename Dtype>
void ConcatOptimizedLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
    const ConcatParameter& concat_param = this->layer_param_.concat_param();
    CHECK(!(concat_param.has_axis() && concat_param.has_concat_dim()))
            << "Either axis or concat_dim should be specified; not both.";

#ifdef CONCAT_OPT_NO_MEM_BANK

    if(!data_pointer_map_)
        data_pointer_map_ = new std::map<int,Dtype*>();

    if(this->phase_ == TRAIN && !diff_pointer_map_)
        diff_pointer_map_ = new std::map<int,Dtype*>();

    if(!temp_data_pointer_map_)
        temp_data_pointer_map_ = new std::map<int,Dtype*>();

    if(!buffer_old_count_.size())
    {
        int n_devices;
        cudaGetDeviceCount(&n_devices);
        buffer_old_count_.resize(n_devices,0);
        //memset(buffer_old_count_,0,n_devices*sizeof(unsigned long int));
    }

#else
    cudaGetDevice(&device_id_);

    if(!concat_chain_[device_id_])
        concat_chain_[device_id_] = new std::map<void*, concat_child*>() ;

    concat_chain_map_ = concat_chain_[device_id_];

#endif

}


template< typename Dtype>
void ConcatOptimizedLayer<Dtype>::rellocate_the_map(std::map<void*, concat_child* >* map,
                                                    Dtype* data_pointer, Dtype* diff_pointer,
                                                    const std::vector<Blob<Dtype>*> &bottom
                                                    , const std::vector<Blob<Dtype>*> &top)
{
    for(int i=0;i<bottom.size();i++)
    {
        void* bottom_data_pointer = (void*)bottom[i];

        concat_child* concat_child_ = map->operator [](bottom_data_pointer);

        if(concat_child_)
        {
            std::vector<Blob<Dtype>*>& bottom_child = *(concat_child_->bottom);
            std::vector<Blob<Dtype>*>& top_child = *(concat_child_->top);

            if(concat_child_->own_data)
            {
#ifdef CONCAT_OPT_NO_MEM_BANK
                cudaFree(bottom[i]->mutable_gpu_data());
#endif
                std::cout << "Freeing the data memory w/o split capacity = " << ((float)top_child[0]->count()*sizeof(Dtype))/1000000.0  << " MB\n";

                if(this->phase_ == TRAIN)
                {
#ifdef CONCAT_OPT_NO_MEM_BANK
                    cudaFree(bottom[i]->mutable_gpu_diff());
#endif
                    std::cout << "Freeing the diff memory w/o split capacity = " << ((float)top_child[0]->count()*sizeof(Dtype))/1000000.0  << " MB\n";
                }

                concat_child_->own_data = false;
            }

            rellocate_the_map(map, data_pointer, diff_pointer, bottom_child, top_child);
        }


        // UPDATE BOTTOM MEMORY
        //                bottom[i]->data()->set_gpu_data(data_pointer);


        // TO HANDLE IF THE BLOB IS SPLITTED or SPLIT LAYER WAS INSERTED
        // IF YES THEN SEARCH FOR THE TOP MOST PARENT IN CASE OF MULTIPLE SPLITTS

        Blob<Dtype>* blob = bottom[i];

        if(blob->is_shared())
        {
            bool done = false;
            while(!done)
            {
                if(blob->is_shared())
                {
                    blob = blob->parent();
                }
                else
                {
                    concat_child* concat_child_ = map->operator [](blob);

                    if(concat_child_)
                    {
                        std::vector<Blob<Dtype>*>& bottom_child = *(concat_child_->bottom);
                        std::vector<Blob<Dtype>*>& top_child = *(concat_child_->top);

                        if(concat_child_->own_data)
                        {
#ifdef CONCAT_OPT_NO_MEM_BANK
                            cudaFree(blob->mutable_gpu_data());
#endif
                            std::cout << "Freeing the data memory capacity = " << blob->count()*sizeof(Dtype)/1000000.0  << " MB\n";

                            if(this->phase_ == TRAIN)
                            {
#ifdef CONCAT_OPT_NO_MEM_BANK
                                cudaFree(blob->mutable_gpu_diff());
#endif
                                std::cout << "Freeing the diff memory capacity = " << blob->count()*sizeof(Dtype)/1000000.0  << " MB\n";
                            }

                            concat_child_->own_data = false;
                        }

                        rellocate_the_map(map, data_pointer, diff_pointer, bottom_child, top_child);


                    }

                    blob->data()->set_gpu_data(data_pointer);
                    if(this->phase_ == TRAIN)
                        blob->diff()->set_gpu_data(diff_pointer);

                    done = true;
                }
            }
        }

        //        std::cout << " memory assigning to " << (int64_t)(bottom[i]->gpu_data()) <<                   "  to ";

        bottom[i]->data()->set_gpu_data(data_pointer);
        if(this->phase_ == TRAIN)
            bottom[i]->diff()->set_gpu_data(diff_pointer);

        //        std::cout << (int64_t)(bottom[i]->gpu_data()) << "\n";
        // ADVANCE THE MEMORY POINTERS

        int offset = bottom[i]->count();

        data_pointer += offset;

        if(this->phase_ == TRAIN)
            diff_pointer += offset;

        //        std::cout << "Setting the data diff mem\n";
    }

}



template< typename Dtype>
void ConcatOptimizedLayer<Dtype>::rellocate_the_map_mem_bank(std::map<void*, concat_child* >* map,
                                                             const Blob<Dtype>& layer_top, unsigned long int offset,
                                                             const std::vector<Blob<Dtype>*> &bottom,
                                                             const std::vector<Blob<Dtype>*> &top)
{
    for(int i=0;i<bottom.size();i++)
    {
        void* bottom_data_pointer = (void*)bottom[i];

        concat_child* concat_child_ = map->operator [](bottom_data_pointer);

        if(concat_child_)
        {
            std::vector<Blob<Dtype>*>& bottom_child = *(concat_child_->bottom);
            std::vector<Blob<Dtype>*>& top_child = *(concat_child_->top);

            if(concat_child_->own_data)
            {
                //                cudaFree(bottom[i]->mutable_gpu_data());
                std::cout << "Freeing the data memory w/o split capacity = " << ((float)top_child[0]->count()*sizeof(Dtype))/1000000.0  << " MB\n";

                if(this->phase_ == TRAIN)
                {
                    //                    cudaFree(bottom[i]->mutable_gpu_diff());
                    std::cout << "Freeing the diff memory w/o split capacity = " << ((float)top_child[0]->count()*sizeof(Dtype))/1000000.0  << " MB\n";
                }

                concat_child_->own_data = false;
            }

            rellocate_the_map_mem_bank(map, layer_top, offset, bottom_child, top_child);
        }


        // UPDATE BOTTOM MEMORY
        //                bottom[i]->data()->set_gpu_data(data_pointer);


        // TO HANDLE IF THE BLOB IS SPLITTED or SPLIT LAYER WAS INSERTED
        // IF YES THEN SEARCH FOR THE TOP MOST PARENT IN CASE OF MULTIPLE SPLITTS

        Blob<Dtype>* blob = bottom[i];

        if(blob->is_shared())
        {
            bool done = false;
            while(!done)
            {
                if(blob->is_shared())
                {
                    blob = blob->parent();
                }
                else
                {
                    concat_child* concat_child_ = map->operator [](blob);

                    if(concat_child_)
                    {
                        std::vector<Blob<Dtype>*>& bottom_child = *(concat_child_->bottom);
                        std::vector<Blob<Dtype>*>& top_child = *(concat_child_->top);

                        if(concat_child_->own_data)
                        {
                            //                            cudaFree(blob->mutable_gpu_data());
                            std::cout << "Freeing the data memory capacity = " << blob->count()*sizeof(Dtype)/1000000.0  << " MB\n";

                            if(this->phase_ == TRAIN)
                            {
                                //                                cudaFree(blob->mutable_gpu_diff());
                                std::cout << "Freeing the diff memory capacity = " << blob->count()*sizeof(Dtype)/1000000.0  << " MB\n";
                            }

                            concat_child_->own_data = false;
                        }

                        rellocate_the_map_mem_bank(map, layer_top, offset, bottom_child, top_child);


                    }

                    blob->ShareData(layer_top);
                    blob->set_ptr_offset(offset);
                    if(this->phase_ == TRAIN)
                        blob->ShareDiff(layer_top);

                    done = true;
                }
            }
        }

        bottom[i]->ShareData(layer_top);
        bottom[i]->set_ptr_offset(offset);
        if(this->phase_ == TRAIN)
            bottom[i]->ShareDiff(layer_top);
    // ADVANCE THE MEMORY POINTERS
        offset += bottom[i]->count();

    }
}


template <typename Dtype>
void ConcatOptimizedLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {

    int num = bottom[0]->shape(0);
    int h = bottom[0]->shape(2);
    int w = bottom[0]->shape(3);

    int n_channels = 0;
    for(int i=0;i< bottom.size();i++)
        n_channels += bottom[i]->shape(1);

    int total_count = num * n_channels * h * w;

    //UPDATE TOP MEMORY
    top[0]->Reshape(num,n_channels,h,w);

    if (bottom.size() == 1) {
        top[0]->ShareData(*bottom[0]);
        top[0]->ShareDiff(*bottom[0]);
    }
    else
    {
        //         GLOBAL DATA DIFF AND TEMP BUFFER only in GPU MODE
        if(caffe::Caffe::mode()== caffe::Caffe::GPU)
        {
            for (int i = 1; i < bottom.size(); ++i)
                for (int j = 0; j < 4; j++)
                    if(j != 1) // channels
                        CHECK_EQ(bottom[0]->shape(j), bottom[i]->shape(j));

            //            std::cout << "OPTIMIZING CONCATENATION OPERATION IN GPU MODE\n";


            //            if(!bottom_buffer_old_count_.size())
            //            {
            //                int n_devices;
            //                cudaGetDeviceCount(&n_devices);
            //                bottom_buffer_old_count_.resize(n_devices, std::vector<unsigned long int>(bottom.size(), 0));
            //                //memset(buffer_old_count_,0,n_devices*sizeof(unsigned long int));
            //            }

            //            for(int i=0;i<bottom.size();i++)
            //            std::cout << "botttom " << i << " gpu concat data = " << (int64_t)(bottom[i]->gpu_data()) << "\n";

            //            std::cout << "top 0 gpu concat data = " << (int64_t)(top[0]->gpu_data()) << "\n";


#ifdef CONCAT_OPT_NO_MEM_BANK

            int device;
            cudaGetDevice(&device);


            // FREE PREVIOUSLY ALLOCATED MEMORY
            Dtype* gpu_buffer = NULL;

            if( data_pointer_map_->operator [](device))
            {
                gpu_buffer = data_pointer_map_->operator [](device);
                cudaFree(gpu_buffer);
            }


            if(this->phase_ == TRAIN && diff_pointer_map_->operator [](device))
            {
                gpu_buffer = diff_pointer_map_->operator [](device);
                cudaFree(gpu_buffer);
            }

            // CREATE NEW MEMORY

            cudaMalloc(&gpu_buffer,total_count*sizeof(Dtype));
            data_pointer_map_->operator [](device) = gpu_buffer;

            //            std::cout << "Creating the data memory capacity = " << total_count*sizeof(Dtype)/1000000.0  << " MB\n";

            if(this->phase_ == TRAIN)
            {
                cudaMalloc(&gpu_buffer,total_count*sizeof(Dtype));
                diff_pointer_map_->operator [](device) = gpu_buffer;
                //                std::cout << "Creating the diff memory capacity = " << total_count*sizeof(Dtype)/1000000.0  << " MB\n";
            }

            std::map<void*, concat_child* > * map = concat_chain_.operator [](device);

            if(!map)
            {
                map = new std::map<void*, concat_child*>();
                concat_chain_.operator [](device) = map;
            }


            Dtype* data_pointer = NULL;
            gpu_buffer = data_pointer_map_->operator [](device);
            data_pointer = gpu_buffer;

            Dtype* diff_pointer = NULL;
            if( this->phase_ == TRAIN)
            {
                gpu_buffer = diff_pointer_map_->operator [](device);
                diff_pointer = gpu_buffer;
            }

            rellocate_the_map(map, data_pointer, diff_pointer, bottom, top);

            if(!concat_child_)
            {
                concat_child_ = new concat_child;

                concat_child_->bottom = (vector<Blob<Dtype>*>*)(&bottom);
                concat_child_->top = (vector<Blob<Dtype>*>*)(&top);
                concat_child_->own_data = true;

                map->operator []((void*)top[0]) = concat_child_ ;
            }

            gpu_buffer = data_pointer_map_->operator [](device);
            top[0]->data()->set_gpu_data(gpu_buffer);

            if(this->phase_ == TRAIN)
            {
                gpu_buffer = diff_pointer_map_->operator [](device);
                top[0]->diff()->set_gpu_data(gpu_buffer);
            }

            //            std::cout << "botttom 0 gpu concat data = " << (int64_t)(bottom[0]->gpu_data()) << "\n";

            //            std::cout << "top 0 gpu concat data = " << (int64_t)(top[0]->gpu_data()) << "\n";



            // UPDATE GLOBAL MEMORY
            if(num > 1)
            {
                if(total_count > buffer_old_count_[device])
                {
                    buffer_old_count_[device] = total_count;

                    gpu_buffer = NULL;
                    if(temp_data_pointer_map_->operator [](device))
                    {
                        gpu_buffer = temp_data_pointer_map_->operator [](device);
                        cudaFree(gpu_buffer);
                    }
                    cudaMalloc(&gpu_buffer,buffer_old_count_[device]*sizeof(Dtype));
                    temp_data_pointer_map_->operator [](device) = gpu_buffer;
                }
                std::cout << "\n\nLARGEST CONCATENET MEM = " << total_count*4/1000000.0 << " MB\n\n";

            }

#else

//            Dtype* data_pointer = NULL;
//            Dtype* diff_pointer = NULL;

//            data_pointer = (Dtype*)top[0]->gpu_data();

//            if( this->phase_ == TRAIN)
//                diff_pointer = (Dtype*)top[0]->gpu_diff();;

//            rellocate_the_map(concat_chain_map_, data_pointer, diff_pointer, bottom, top);

            rellocate_the_map_mem_bank(concat_chain_map_, *(top[0]), 0, bottom,top);

            if(!concat_child_)
            {
                concat_child_ = new concat_child;

                concat_child_->bottom = (vector<Blob<Dtype>*>*)(&bottom);
                concat_child_->top = (vector<Blob<Dtype>*>*)(&top);
                concat_child_->own_data = true;

                concat_chain_map_->operator []((void*)top[0]) = concat_child_ ;
            }


            // UPDATE GLOBAL MEMORY
            if(num > 1)
            {
                const Blob<Dtype>& buffer = this->get_buffer(total_count);
                temp_buffer_.Reshape(total_count, 1, 1, 1);
                temp_buffer_.ShareData(buffer);
//                std::cout << "\n\n\n\n\nLARGEST CONCATENET MEM = " << total_count*4/1000000.0 << " MB\n\n\n\n\n";

            }
#endif


        }
    }
}

template <typename Dtype>
void ConcatOptimizedLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
    if (bottom.size() == 1) { return; }
}

template <typename Dtype>
void ConcatOptimizedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (bottom.size() == 1) { return; }
}


#ifdef CPU_ONLY
STUB_GPU(ConcatOptimizedLayer);
#endif

INSTANTIATE_CLASS(ConcatOptimizedLayer);
REGISTER_LAYER_CLASS(ConcatOptimized);

}

