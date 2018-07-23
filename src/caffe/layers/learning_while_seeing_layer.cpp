#include <vector>

#include "caffe/layers/learning_while_seeing_layer.hpp"

namespace caffe {

template <typename Dtype>
std::map<int, Blob<Dtype>* > LearningWhileSeeingLayer<Dtype>::bottoms_gpu_map_ = std::map<int, Blob<Dtype>* >();

template <typename Dtype>
std::map<int, Blob<Dtype>* > LearningWhileSeeingLayer<Dtype>::tops_gpu_map_ = std::map<int, Blob<Dtype>* >();


template <typename Dtype>
void LearningWhileSeeingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{

    if(this->layer_param_.has_lws_param())
        LOG(FATAL) <<  "No parameters specified\n";


    // ADD conv
    LayerParameter conv_param = this->layer_param_.lws_param().conv_param();
    conv_layer_ = LayerRegistry<Dtype>::CreateLayer(conv_param);

    // ADD BN
    has_bn_lws_param_ = this->layer_param_.lws_param().has_bn_lws_param();
    if(has_bn_lws_param_)
    {
        LayerParameter bn_lws_param = this->layer_param_.lws_param().bn_lws_param();
        bn_lws_layer_ = LayerRegistry<Dtype>::CreateLayer(bn_lws_param);
    }

    // ADD RBF
    LayerParameter rbf_param = this->layer_param_.lws_param().rbf_param();
    rbf_layer_ = LayerRegistry<Dtype>::CreateLayer(rbf_param);

    // ADD LOSS LAYER
    LayerParameter lws_loss_param = this->layer_param_.lws_param().lws_loss_param();
    lws_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(lws_loss_param);

    // SET up conv layer
    conv_layer_->SetUp(bottom, top);

    // SET up bn layer
    if(has_bn_lws_param_)
        bn_lws_layer_->SetUp(top, top);

    // SET up rbf layer
    rbf_layer_->SetUp(top, top);

    std::vector<Blob<Dtype>*> lws_loss_tops;
    lws_loss_tops.push_back(&lws_loss_);

    // SET up lws loss layer
    lws_loss_layer_->SetUp(top, lws_loss_tops);
}

template <typename Dtype>
void LearningWhileSeeingLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{

    int device;
    cudaGetDevice(&device);

    Blob<Dtype>* bottom_global_ = bottoms_gpu_map_[device];
    Blob<Dtype>* top_global_ = tops_gpu_map_[device];

    if(!bottom_global_)
    {
        bottoms_gpu_map_[device] = new Blob<Dtype>();
        bottom_global_ = bottoms_gpu_map_[device];
    }

    if(!top_global_)
    {
        tops_gpu_map_[device] = new Blob<Dtype>();
        top_global_ = tops_gpu_map_[device];
    }

    const int global_buffer_size = std::max(bottom[0]->count(), top[0]->count());

    bottom_global_->Reshape(global_buffer_size, 1, 1, 1);
    top_global_->Reshape(global_buffer_size, 1, 1, 1);

    // Reshape all tops and bottoms
    conv_layer_->Reshape(top, top);

    if(has_bn_lws_param_)
        bn_lws_layer_->Reshape(top, top);

    rbf_layer_->Reshape(top, top);

    std::vector<Blob<Dtype>*> lws_loss_tops;
    lws_loss_tops.push_back(&lws_loss_);

    lws_loss_layer_->Reshape(top, lws_loss_tops);

    lws_bottom_.ReshapeLike(*bottom[0]);
    lws_top_.ReshapeLike(*top[0]);

    lws_bottom_.data()->set_gpu_data(bottom_global_->mutable_gpu_data());
    lws_top_.data()->set_gpu_data(top_global_->mutable_gpu_data());

    lws_bottom_.diff()->set_gpu_data(bottom_global_->mutable_gpu_diff());
    lws_top_.diff()->set_gpu_data(top_global_->mutable_gpu_diff());
}


template <typename Dtype>
void LearningWhileSeeingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {


}

template <typename Dtype>
void LearningWhileSeeingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(LearningWhileSeeingLayer);
#endif

INSTANTIATE_CLASS(LearningWhileSeeingLayer);
REGISTER_LAYER_CLASS(LearningWhileSeeing);

}  // namespace caffe
