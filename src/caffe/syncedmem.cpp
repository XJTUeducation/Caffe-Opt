//#include "caffe/common.hpp"
//#include "caffe/syncedmem.hpp"
//#include "caffe/util/math_functions.hpp"

//namespace caffe {

//SyncedMemory::~SyncedMemory() {
//  if (cpu_ptr_ && own_cpu_data_) {
//    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
//  }

//#ifndef CPU_ONLY
//  if (gpu_ptr_ && own_gpu_data_) {
//    int initial_device;
//    cudaGetDevice(&initial_device);
//    if (gpu_device_ != -1) {
//      CUDA_CHECK(cudaSetDevice(gpu_device_));
//    }
//    CUDA_CHECK(cudaFree(gpu_ptr_));
//    cudaSetDevice(initial_device);
//  }
//#endif  // CPU_ONLY
//}

//void SyncedMemory::resize(size_t size)
//{
//    if(size_ < size)
//    {
//        if (own_cpu_data_) {
//            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
//        }
//        cpu_ptr_ = NULL;

//#ifndef CPU_ONLY
//        if (own_gpu_data_) {
//            int initial_device;
//            cudaGetDevice(&initial_device);
//            if (gpu_device_ != -1) {
//                CUDA_CHECK(cudaSetDevice(gpu_device_));
//            }
//            CUDA_CHECK(cudaFree(gpu_ptr_));
//            cudaSetDevice(initial_device);
//        }
//        gpu_ptr_ = NULL;
//#endif  // CPU_ONLY

//        size_ = size;
//    }
//    own_cpu_data_ = false;
//    own_gpu_data_ = false;
//    cpu_malloc_use_cuda_ = false;
//    gpu_device_ = -1;
//    head_ = UNINITIALIZED;
//}


//inline void SyncedMemory::to_cpu() {
//  switch (head_) {
//  case UNINITIALIZED:
//    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
//    caffe_memset(size_, 0, cpu_ptr_);
//    head_ = HEAD_AT_CPU;
//    own_cpu_data_ = true;
//    break;
//  case HEAD_AT_GPU:
//#ifndef CPU_ONLY
//    if (cpu_ptr_ == NULL) {
//      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
//      own_cpu_data_ = true;
//    }
//    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
//    head_ = SYNCED;
//#else
//    NO_GPU;
//#endif
//    break;
//  case HEAD_AT_CPU:
//  case SYNCED:
//    break;
//  }
//}

//inline void SyncedMemory::to_gpu() {
//#ifndef CPU_ONLY
//  switch (head_) {
//  case UNINITIALIZED:
//    CUDA_CHECK(cudaGetDevice(&gpu_device_));
//    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
//    caffe_gpu_memset(size_, 0, gpu_ptr_);
//    head_ = HEAD_AT_GPU;
//    own_gpu_data_ = true;
//    break;
//  case HEAD_AT_CPU:
//    if (gpu_ptr_ == NULL) {
//      CUDA_CHECK(cudaGetDevice(&gpu_device_));
//      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
//      own_gpu_data_ = true;
//    }
//    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
//    head_ = SYNCED;
//    break;
//  case HEAD_AT_GPU:
//  case SYNCED:
//    break;
//  }
//#else
//  NO_GPU;
//#endif
//}

//const void* SyncedMemory::cpu_data() {
//  to_cpu();
//  return (const void*)cpu_ptr_;
//}

//void SyncedMemory::set_cpu_data(void* data) {
//  CHECK(data);
//  if (own_cpu_data_) {
//    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
//  }
//  cpu_ptr_ = data;
//  head_ = HEAD_AT_CPU;
//  own_cpu_data_ = false;
//}

//const void* SyncedMemory::gpu_data() {
//#ifndef CPU_ONLY
//  to_gpu();
//  return (const void*)gpu_ptr_;
//#else
//  NO_GPU;
//  return NULL;
//#endif
//}

//void SyncedMemory::set_gpu_data(void* data) {
//#ifndef CPU_ONLY
//  CHECK(data);
//  if (own_gpu_data_) {
//    int initial_device;
//    cudaGetDevice(&initial_device);
//    if (gpu_device_ != -1) {
//      CUDA_CHECK(cudaSetDevice(gpu_device_));
//    }
//    CUDA_CHECK(cudaFree(gpu_ptr_));
//    cudaSetDevice(initial_device);
//  }
//  gpu_ptr_ = data;
//  head_ = HEAD_AT_GPU;
//  own_gpu_data_ = false;
//#else
//  NO_GPU;
//#endif
//}

//void* SyncedMemory::mutable_cpu_data() {
//  to_cpu();
//  head_ = HEAD_AT_CPU;
//  return cpu_ptr_;
//}

//void* SyncedMemory::mutable_gpu_data() {
//#ifndef CPU_ONLY
//  to_gpu();
//  head_ = HEAD_AT_GPU;
//  return gpu_ptr_;
//#else
//  NO_GPU;
//  return NULL;
//#endif
//}

//#ifndef CPU_ONLY
//void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
//  CHECK(head_ == HEAD_AT_CPU);
//  if (gpu_ptr_ == NULL) {
//    CUDA_CHECK(cudaGetDevice(&gpu_device_));
//    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
//    own_gpu_data_ = true;
//  }
//  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
//  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
//  // Assume caller will synchronize on the stream before use
//  head_ = SYNCED;
//}
//#endif

//}  // namespace caffe




#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() {
    if (cpu_ptr_ && own_cpu_data_) {
        CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
    }

#ifndef CPU_ONLY
    if (gpu_ptr_ && own_gpu_data_) {
        int initial_device;
        cudaGetDevice(&initial_device);
        if (gpu_device_ != -1) {
            CUDA_CHECK(cudaSetDevice(gpu_device_));
        }
        CUDA_CHECK(cudaFree(gpu_ptr_));
        cudaSetDevice(initial_device);
    }
#endif  // CPU_ONLY
}

void SyncedMemory::resize(size_t size)
{
    if(parent_)
        LOG(FATAL) << "Parent is already set.. first set the parent to null using parent_ = NULL and then resize";

    if(size_ < size)
    {
        if (own_cpu_data_) {
            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
        }
        cpu_ptr_ = NULL;

#ifndef CPU_ONLY
        if (own_gpu_data_) {
            int initial_device;
            cudaGetDevice(&initial_device);
            if (gpu_device_ != -1) {
                CUDA_CHECK(cudaSetDevice(gpu_device_));
            }
            CUDA_CHECK(cudaFree(gpu_ptr_));
            cudaSetDevice(initial_device);
        }
        gpu_ptr_ = NULL;
#endif  // CPU_ONLY

        size_ = size;
    }
    own_cpu_data_ = false;
    own_gpu_data_ = false;
    cpu_malloc_use_cuda_ = false;
    gpu_device_ = -1;
    head_ = UNINITIALIZED;
}


inline void SyncedMemory::to_cpu() {
    switch (head_) {
    case UNINITIALIZED:
        CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
        caffe_memset(size_, 0, cpu_ptr_);
        head_ = HEAD_AT_CPU;
        own_cpu_data_ = true;
        break;
    case HEAD_AT_GPU:
#ifndef CPU_ONLY
        if (cpu_ptr_ == NULL) {
            CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
            own_cpu_data_ = true;
        }
        caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
        head_ = SYNCED;
#else
        NO_GPU;
#endif
        break;
    case HEAD_AT_CPU:
    case SYNCED:
        break;
    }
}

inline void SyncedMemory::to_gpu() {
#ifndef CPU_ONLY
    switch (head_) {
    case UNINITIALIZED:
        CUDA_CHECK(cudaGetDevice(&gpu_device_));
        CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
        caffe_gpu_memset(size_, 0, gpu_ptr_);
        head_ = HEAD_AT_GPU;
        own_gpu_data_ = true;
        break;
    case HEAD_AT_CPU:
        if (gpu_ptr_ == NULL) {
            CUDA_CHECK(cudaGetDevice(&gpu_device_));
            CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
            own_gpu_data_ = true;
        }
        caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
        head_ = SYNCED;
        break;
    case HEAD_AT_GPU:
    case SYNCED:
        break;
    }
#else
    NO_GPU;
#endif
}

const void* SyncedMemory::cpu_data() {

    if(parent_)
        return parent_->cpu_data();

    to_cpu();
    return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
    CHECK(data);
    if (own_cpu_data_) {
        CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
    }
    cpu_ptr_ = data;
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {

    if(parent_)
        return parent_->gpu_data();


#ifndef CPU_ONLY
    to_gpu();
    return (const void*)gpu_ptr_;
#else
    NO_GPU;
    return NULL;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
    CHECK(data);
    if (own_gpu_data_) {
        int initial_device;
        cudaGetDevice(&initial_device);
        if (gpu_device_ != -1) {
            CUDA_CHECK(cudaSetDevice(gpu_device_));
        }
        CUDA_CHECK(cudaFree(gpu_ptr_));
        cudaSetDevice(initial_device);
    }
    gpu_ptr_ = data;
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = false;
#else
    NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {

    if(parent_)
        return parent_->mutable_cpu_data();

    to_cpu();
    head_ = HEAD_AT_CPU;
    return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {

    if(parent_)
    return parent_->mutable_gpu_data();

#ifndef CPU_ONLY
    to_gpu();
    head_ = HEAD_AT_GPU;
    return gpu_ptr_;
#else
    NO_GPU;
    return NULL;
#endif
}

#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
    CHECK(head_ == HEAD_AT_CPU);
    if (gpu_ptr_ == NULL) {
        CUDA_CHECK(cudaGetDevice(&gpu_device_));
        CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
        own_gpu_data_ = true;
    }
    const cudaMemcpyKind put = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
    // Assume caller will synchronize on the stream before use
    head_ = SYNCED;
}
#endif

}  // namespace caffe

