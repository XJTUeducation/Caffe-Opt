#include <boost/thread.hpp>
#include "caffe/layer.hpp"

namespace caffe {

//template<typename Dtype>
//boost::thread_specific_ptr<MemoryBank<Dtype> > Layer<Dtype>::thread_instances_memory_bank_(0);

template<typename Dtype>
std::map<int, MemoryBank<Dtype>* > Layer<Dtype>::memory_banks_ = std::map<int, MemoryBank<Dtype>* >();



template <typename Dtype>
void Layer<Dtype>::InitMutex() {
  forward_mutex_.reset(new boost::mutex());
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
