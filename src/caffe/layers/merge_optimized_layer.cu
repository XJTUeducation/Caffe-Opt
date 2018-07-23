#include <vector>

#include "caffe/layers/merge_optimized_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MergeOptimizedLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {

    return;
}

template <typename Dtype>
void MergeOptimizedLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}


INSTANTIATE_LAYER_GPU_FUNCS(MergeOptimizedLayer);

}  // namespace caffe
