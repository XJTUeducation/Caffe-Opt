#include <vector>

#include "caffe/layers/merge_optimized_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MergeOptimizedLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
    count_ = bottom[0]->count();

    top[0]->ReshapeLike(*bottom[0]);


    for (int i = 0; i < bottom.size(); ++i) {

        CHECK_EQ(bottom[i]->count(),bottom[0]->count()) <<  "All bottoms must have sames shape.";

        CHECK_NE(top[0], bottom[i]) << this->type() << " Layer does not allow in-place computation.";

        bottom[i]->ShareWith(top[0]);

        bottom[i]->ShareData(*top[0]);
        if(this->phase_ == TRAIN)
            bottom[i]->ShareDiff(*top[0]);
    }
}

template <typename Dtype>
void MergeOptimizedLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {

    return;
}

template <typename Dtype>
void MergeOptimizedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;
}


#ifdef CPU_ONLY
STUB_GPU(MergeOptimizedLayer);
#endif

INSTANTIATE_CLASS(MergeOptimizedLayer);
REGISTER_LAYER_CLASS(MergeOptimized);

}  // namespace caffe
