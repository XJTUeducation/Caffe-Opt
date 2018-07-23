#include <vector>

#include "caffe/layers/strided_decim_interp.hpp"
#include "caffe/memory_bank.hpp"

namespace caffe {


template <typename Dtype>
void StridedDecimInterpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{

    dilation_ = this->layer_param_.strided_decim_interp_param().dilation();
    CHECK_GE(dilation_, 1);

    strided_op_ = this->layer_param_.strided_decim_interp_param().strided_op();
    n_batch_per_example_ = dilation_ * dilation_;
}

template <typename Dtype>
void StridedDecimInterpLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{

    switch(strided_op_)
    {
    case StridedDecimInterpParameter_STRIDED_OP_DECIM :
    {
        int n = bottom[0]->shape(0);
        int n_channels = bottom[0]->shape(1);
        int height = bottom[0]->shape(2);
        int width = bottom[0]->shape(3);

        int top_height = height / dilation_;
        int top_width = width  / dilation_;

        CHECK_EQ( height % dilation_, 0);
        CHECK_EQ( width % dilation_, 0);

        int n_batch_all_examples = n_batch_per_example_ * n;
        top[0]->Reshape(n_batch_all_examples, n_channels, top_height, top_width);

    }
        break;

    case StridedDecimInterpParameter_STRIDED_OP_INTERP:
    {
        int n = bottom[0]->shape(0);
        int n_channels = bottom[0]->shape(1);
        int top_height = bottom[0]->shape(2);
        int top_width = bottom[0]->shape(3);

        int height = top_height * dilation_;
        int width = top_width  * dilation_;

        CHECK_EQ( height % dilation_, 0);
        CHECK_EQ( width % dilation_, 0);

        int n_batch_all_examples = n / n_batch_per_example_;

        top[0]->Reshape(n_batch_all_examples, n_channels, height, width);

    }
        break;

    case StridedDecimInterpParameter_STRIDED_OP_DECIM_CONCAT :
    {
        int n = bottom[0]->shape(0);
        int n_channels = bottom[0]->shape(1);
        int height = bottom[0]->shape(2);
        int width = bottom[0]->shape(3);

        int top_height = height / dilation_;
        int top_width = width  / dilation_;

        CHECK_EQ( height % dilation_, 0);
        CHECK_EQ( width % dilation_, 0);

        int top_n_channels = n_batch_per_example_ * n_channels;
        top[0]->Reshape(n, top_n_channels, top_height, top_width);

    }
        break;

    case StridedDecimInterpParameter_STRIDED_OP_INTERP_CONCAT:
    {
        int n = bottom[0]->shape(0);
        int n_channels = bottom[0]->shape(1);
        int top_height = bottom[0]->shape(2);
        int top_width = bottom[0]->shape(3);

        int height = top_height * dilation_;
        int width = top_width  * dilation_;

        CHECK_EQ( height % dilation_, 0);
        CHECK_EQ( width % dilation_, 0);

        int top_n_channels = n_channels / n_batch_per_example_;

        top[0]->Reshape(n, top_n_channels, height, width);

    }
        break;
    }

    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);

    const Blob<Dtype>& buffer = this->get_buffer(bottom[0]->count());
    temp_buffer_.Reshape(bottom[0]->count(), 1, 1, 1);
    temp_buffer_.ShareData(buffer);

//    static  int count;
//    if(count < temp_buffer_.count() )
//    {
//        count = temp_buffer_.count();
//    std::cout << "\n\nLARGEST TEMP strided buff = " << count * 4/1000000.0 << " MB\n\n";
//    }

}


template <typename Dtype>
void StridedDecimInterpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {


}

template <typename Dtype>
void StridedDecimInterpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


    // do nothing
}

#ifdef CPU_ONLY
STUB_GPU(StridedDecimInterpLayer);
#endif

INSTANTIATE_CLASS(StridedDecimInterpLayer);
REGISTER_LAYER_CLASS(StridedDecimInterp);

}  // namespace caffe
