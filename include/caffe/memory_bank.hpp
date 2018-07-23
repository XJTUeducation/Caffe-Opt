#ifndef CAFFE_MEMORY_BANK_HPP_
#define CAFFE_MEMORY_BANK_HPP_

#include <algorithm>
#include <string>
#include <vector>
#include <iostream>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/blob.hpp"
#include "boost/thread.hpp"

namespace caffe {

//#define BN_NO_MEM_BANK_
//#define CONCAT_OPT_NO_MEM_BANK
//#define BASE_CONV_NO_MEM_BANK

template<typename Dtype>
class MemoryBank {

public:
    MemoryBank()
    {
    }

private:

    std::map<std::string, shared_ptr<Blob<Dtype> > > memory_bank_;
//    static boost::thread_specific_ptr<MemoryBank> thread_specific_inst_memory_bank;

public:

    const Blob<Dtype>& get_buffer(size_t count, bool can_be_shared = true, std::string str_buffer = "0");

    DISABLE_COPY_AND_ASSIGN(MemoryBank);
};

}  // namespace caffe

#endif  // CAFFE_MEMORY_BANK_HPP_

//#ifndef CAFFE_MEMORY_BANK_HPP_
//#define CAFFE_MEMORY_BANK_HPP_

//#include <algorithm>
//#include <string>
//#include <vector>
//#include <iostream>

//#include "caffe/common.hpp"
//#include "caffe/syncedmem.hpp"
//#include "caffe/blob.hpp"
//#include "boost/thread.hpp"

//namespace caffe {

////#define BN_NO_MEM_BANK_
////#define CONCAT_OPT_NO_MEM_BANK
////#define BASE_CONV_NO_MEM_BANK

//template<typename Dtype>
//class MemoryBank {

//public:
//    static MemoryBank<Dtype>& Get();
//   ~MemoryBank();

//private:
//    MemoryBank()
//    {
//    }

//    typedef std::map<std::string, shared_ptr<Blob<Dtype> > > blob_map;

//    std::map<std::string, shared_ptr<Blob<Dtype> > > memory_bank_;
//    static boost::thread_specific_ptr<MemoryBank> thread_specific_inst_memory_bank;

//public:

//    static const Blob<Dtype>& get_buffer(size_t count, bool can_be_shared = true, std::string str_buffer = "0");

//    DISABLE_COPY_AND_ASSIGN(MemoryBank);
//};

//}  // namespace caffe

//#endif  // CAFFE_MEMORY_BANK_HPP_

