#include "caffe/memory_bank.hpp"

namespace caffe {

template<typename Dtype>
const Blob<Dtype>& MemoryBank<Dtype>::get_buffer(size_t count, bool can_be_shared, std::string str_buffer)
{
    std::string str_buffer_id = can_be_shared ? "0" : str_buffer;

    shared_ptr<Blob<Dtype> >& ptr_buffer = memory_bank_[str_buffer_id];

    if(!ptr_buffer)
     {
        ptr_buffer.reset(new Blob<Dtype>());
        ptr_buffer->Reshape(count,1,1,1);
    }

    if(ptr_buffer.get()->count() < count)
        ptr_buffer.get()->Reshape(count,1,1,1);

    return static_cast<Blob<Dtype>&>(*(ptr_buffer.get()));
}

//template<typename Dtype>
//boost::thread_specific_ptr<MemoryBank<Dtype> > MemoryBank<Dtype>::thread_specific_inst_memory_bank;


//template<typename Dtype>
//MemoryBank<Dtype>::~MemoryBank()
//{
//    if(thread_specific_inst_memory_bank.get())
//    {
//        MemoryBank<Dtype>& mem_bank_ = MemoryBank<Dtype>::Get();

//        for(typename blob_map::iterator it =  mem_bank_.memory_bank_.begin();
//             it != mem_bank_.memory_bank_.end();it++)
//        {
//            shared_ptr<Blob<Dtype> >& ptr_buffer = it->second;
//            if(ptr_buffer)
//                ptr_buffer.reset();
//        }

//        thread_specific_inst_memory_bank.reset();
//    }
//}

//template<typename Dtype>
//MemoryBank<Dtype>& MemoryBank<Dtype>::Get()
//{
//    if(!thread_specific_inst_memory_bank.get())
//        thread_specific_inst_memory_bank.reset(new MemoryBank());

//    return *(thread_specific_inst_memory_bank.get());
//}

//template<typename Dtype>
//const Blob<Dtype>& MemoryBank<Dtype>::get_buffer(size_t count, bool can_be_shared, std::string str_buffer)
//{

//    MemoryBank<Dtype>& mem_bank_ = MemoryBank<Dtype>::Get();

//    std::string str_buffer_id = can_be_shared ? "0" : str_buffer;

//    shared_ptr<Blob<Dtype> >& ptr_buffer = mem_bank_.memory_bank_[str_buffer_id];

//    if(!ptr_buffer)
//    {
//        ptr_buffer.reset(new Blob<Dtype>());
//        ptr_buffer->Reshape(count,1,1,1);
//    }

//    if(ptr_buffer.get()->count() < count)
//        ptr_buffer.get()->Reshape(count,1,1,1);

//    return static_cast<Blob<Dtype>&>(*(ptr_buffer.get()));
//}


INSTANTIATE_CLASS(MemoryBank);

}  // namespace caffe




