#include <vector>

#include "caffe/layers/memory_opt_layer.hpp"
#include "caffe/memory_bank.hpp"

namespace caffe {


template <typename Dtype>
void MemoryOptLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{

    bank_id_ = this->layer_param_.memory_opt_param().bank_id();
    can_be_shared_ = this->layer_param_.memory_opt_param().can_be_shared();
    CHECK_GE(bank_id_, 0);


    data_bank_id_ = this->layer_param_.memory_opt_param().data_bank_id();
    data_bank_can_be_shared_ = this->layer_param_.memory_opt_param().data_bank_can_be_shared();
    optmize_data_train_ = this->layer_param_.memory_opt_param().optimize_data_train();

    CHECK_GE(data_bank_id_, 0);

}

template <typename Dtype>
void MemoryOptLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{

    std::stringstream str_buffer_id;
    str_buffer_id << bank_id_;

    top[0]->ReshapeLike(*bottom[0]);

    const Blob<Dtype>& buffer = this->get_buffer(bottom[0]->count(), can_be_shared_, str_buffer_id.str());

    if(this->phase_ == TEST)
    {
        bottom[0]->data()->parent_ = buffer.data();
        top[0]->data()->parent_ = buffer.data();
    }

    if(this->phase_ == TRAIN)
    {
        if(optmize_data_train_)
        {
            std::stringstream str_data_buffer_id;
            str_data_buffer_id << data_bank_id_;

            const Blob<Dtype>& data_buffer = this->get_buffer(bottom[0]->count(), data_bank_can_be_shared_, str_data_buffer_id.str());

            bottom[0]->data()->parent_ = data_buffer.data();
            top[0]->data()->parent_ = data_buffer.data();
        }
        else
            top[0]->ShareData(*bottom[0]);

        bottom[0]->diff()->parent_ = buffer.data();
        top[0]->diff()->parent_ = buffer.data();
    }
}


template <typename Dtype>
void MemoryOptLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {


}

template <typename Dtype>
void MemoryOptLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


    // do nothing
}

#ifdef CPU_ONLY
STUB_GPU(MemoryOptLayer);
#endif

INSTANTIATE_CLASS(MemoryOptLayer);
REGISTER_LAYER_CLASS(MemoryOpt);

}  // namespace caffe
