#include <algorithm>
#include <vector>

#include "caffe/layers/learning_while_seeing_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LearningWhileSeeingLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);


//  if (this->layer_param_.loss_param().has_normalization()) {
//    normalization_ = this->layer_param_.loss_param().normalization();
//  } else if (this->layer_param_.loss_param().has_normalize()) {
//    normalization_ = this->layer_param_.loss_param().normalize() ?
//                     LossParameter_NormalizationMode_VALID :
//                     LossParameter_NormalizationMode_BATCH_SIZE;
//  } else {
//    normalization_ = LossParameter_NormalizationMode_BATCH_SIZE;
//  }
}

template <typename Dtype>
void LearningWhileSeeingLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
//  outer_num_ = bottom[0]->shape(0);  // batch size
//  inner_num_ = bottom[0]->count(1);  // instance size: |output| == |target|

}

// TODO(shelhamer) loss normalization should be pulled up into LossLayer,
// instead of duplicated here and in SoftMaxWithLossLayer
template <typename Dtype>
Dtype LearningWhileSeeingLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
//  Dtype normalizer;
//  switch (normalization_mode) {
//    case LossParameter_NormalizationMode_FULL:
//      normalizer = Dtype(outer_num_ * inner_num_);
//      break;
//    case LossParameter_NormalizationMode_VALID:
//      if (valid_count == -1) {
//        normalizer = Dtype(outer_num_ * inner_num_);
//      } else {
//        normalizer = Dtype(valid_count);
//      }
//      break;
//    case LossParameter_NormalizationMode_BATCH_SIZE:
//      normalizer = Dtype(outer_num_);
//      break;
//    case LossParameter_NormalizationMode_NONE:
//      normalizer = Dtype(1);
//      break;
//    default:
//      LOG(FATAL) << "Unknown normalization mode: "
//          << LossParameter_NormalizationMode_Name(normalization_mode);
//  }
//  // Some users will have no labels for some examples in order to 'turn off' a
//  // particular loss in a multi-task setup. The max prevents NaNs in that case.
//  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void LearningWhileSeeingLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

//    const Dtype* input_data = bottom[0]->cpu_data();
//  const Dtype* target = bottom[1]->cpu_data();
//  int valid_count = 0;
//  Dtype loss = 0;
//  for (int i = 0; i < bottom[0]->count(); ++i) {
//    const int target_value = static_cast<int>(target[i]);
//    if (has_ignore_label_ && target_value == ignore_label_) {
//      continue;
//    }
//    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
//        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
//    ++valid_count;
//  }
//  normalizer_ = get_normalizer(normalization_, valid_count);
//  top[0]->mutable_cpu_data()[0] = loss / normalizer_;

}

template <typename Dtype>
void LearningWhileSeeingLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
//  if (propagate_down[1]) {
//    LOG(FATAL) << this->type()
//               << " Layer cannot backpropagate to label inputs.";
//  }
//  if (propagate_down[0]) {
//    // First, compute the diff
//    const int count = bottom[0]->count();
//    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
//    const Dtype* target = bottom[1]->cpu_data();
//    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
//    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
//    // Zero out gradient of ignored targets.
//    if (has_ignore_label_) {
//      for (int i = 0; i < count; ++i) {
//        const int target_value = static_cast<int>(target[i]);
//        if (target_value == ignore_label_) {
//          bottom_diff[i] = 0;
//        }
//      }
//    }
//    // Scale down gradient
//    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
//    caffe_scal(count, loss_weight, bottom_diff);
//  }
}

#ifdef CPU_ONLY
STUB_GPU(LearningWhileSeeingLossLayer);
#endif

INSTANTIATE_CLASS(LearningWhileSeeingLossLayer);
REGISTER_LAYER_CLASS(LearningWhileSeeingLoss);

}  // namespace caffe
