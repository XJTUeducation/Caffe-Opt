#include <vector>

#include "caffe/layers/memory_opt_layer.hpp"

namespace caffe {



template <typename Dtype>
void MemoryOptLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void MemoryOptLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>&
                                          bottom) {


}

INSTANTIATE_LAYER_GPU_FUNCS(MemoryOptLayer);

}  // namespace caffe
