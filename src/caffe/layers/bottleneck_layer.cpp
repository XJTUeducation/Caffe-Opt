#include <vector>

#include "caffe/layers/bottleneck_layer.hpp"

namespace caffe {

template <typename Dtype>
std::map<int, Blob<Dtype>* > BottleNeckLayer<Dtype>::global_data_map_ = std::map<int, Blob<Dtype>* >();

template <typename Dtype>
std::map<int, Blob<Dtype>* > BottleNeckLayer<Dtype>::global_diff_map_ = std::map<int, Blob<Dtype>* >();


template <typename Dtype>
void BottleNeckLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{

    if(!this->layer_param_.has_bottleneck_param())
        LOG(FATAL) <<  "No parameters specified\n";


    has_reducer_ = this->layer_param_.bottleneck_param().has_reducer();

    if(has_reducer_)
    {
        // ADD conv
        LayerParameter conv_reduce_param = this->layer_param_.bottleneck_param().conv_reduce_param();
        conv_reduce_layer_ = LayerRegistry<Dtype>::CreateLayer(conv_reduce_param);

        // ADD BN
        has_bn_reduce_param_ = this->layer_param_.bottleneck_param().has_bn_reduce_param();
        if(has_bn_reduce_param_)
        {
            LayerParameter bn_reduce_param = this->layer_param_.bottleneck_param().bn_reduce_param();
            bn_reduce_layer_ = LayerRegistry<Dtype>::CreateLayer(bn_reduce_param);
        }

        // ADD RELU
        LayerParameter relu_reduce_param = this->layer_param_.bottleneck_param().relu_reduce_param();
        relu_reduce_layer_ = LayerRegistry<Dtype>::CreateLayer(relu_reduce_param);
    }

    // ADD GROW
    {
        // ADD conv
        LayerParameter conv_grow_param = this->layer_param_.bottleneck_param().conv_grow_param();
        conv_grow_layer_ = LayerRegistry<Dtype>::CreateLayer(conv_grow_param);

        // ADD BN
        has_bn_grow_param_ = this->layer_param_.bottleneck_param().has_bn_grow_param();
        if(has_bn_grow_param_)
        {
            LayerParameter bn_grow_param = this->layer_param_.bottleneck_param().bn_grow_param();
            bn_grow_layer_ = LayerRegistry<Dtype>::CreateLayer(bn_grow_param);
        }

        // ADD RELU
        LayerParameter relu_grow_param = this->layer_param_.bottleneck_param().relu_grow_param();
        relu_grow_layer_ = LayerRegistry<Dtype>::CreateLayer(relu_grow_param);
    }

    std::vector<Blob<Dtype>*> reduce_tops;

    if(has_reducer_)
        reduce_tops.push_back(&reduce_top_);
    else
    {
        for(int i=0; i<bottom.size();i++)
            reduce_tops.push_back(bottom[i]);
    }

    if(has_reducer_)
    {
        // SET up conv layer
        conv_reduce_layer_->SetUp(bottom, reduce_tops);

        // SET up bn layer
        if(has_bn_reduce_param_)
         {
            bn_reduce_layer_->set_n_instances(Caffe::solver_count());
            bn_reduce_layer_->SetUp(reduce_tops, reduce_tops);
        }

        // SET up relu layer
        relu_reduce_layer_->SetUp(reduce_tops, reduce_tops);
    }

    {
        // SET up conv grow layer
        conv_grow_layer_->SetUp(reduce_tops, top);

          // SET up bn grow layer
        if(has_bn_grow_param_)
         {
            bn_grow_layer_->set_n_instances(Caffe::solver_count());
            bn_grow_layer_->SetUp(top, top);
        }

        // SET up relu grow layer
        relu_grow_layer_->SetUp(top, top);
    }


    this->blobs_.clear();

    for(int i=0;i<this->conv_reduce_layer_->blobs().size();i++)
        this->blobs_.push_back(conv_reduce_layer_->blobs()[i]);

    if(bn_reduce_layer_)
        for(int i=0;i<bn_reduce_layer_->blobs().size();i++)
            this->blobs_.push_back(bn_reduce_layer_->blobs()[i]);

    for(int i=0;i<conv_grow_layer_->blobs().size();i++)
        this->blobs_.push_back(conv_grow_layer_->blobs()[i]);

    if(bn_grow_layer_)
        for(int i=0;i<bn_grow_layer_->blobs().size();i++)
            this->blobs_.push_back(bn_grow_layer_->blobs()[i]);

    this->param_propagate_down_.clear();
    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BottleNeckLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{


    std::vector<Blob<Dtype>*> reduce_tops;

    if(has_reducer_)
        reduce_tops.push_back(&reduce_top_);
    else
    {
        for(int i=0; i<bottom.size();i++)
            reduce_tops.push_back(bottom[i]);
    }

    if(has_reducer_)
    {
        // SET up conv layer
        conv_reduce_layer_->Reshape(bottom, reduce_tops);

        // SET up bn layer
        if(has_bn_reduce_param_)
            bn_reduce_layer_->Reshape(reduce_tops, reduce_tops);

        // SET up relu layer
        relu_reduce_layer_->Reshape(reduce_tops, reduce_tops);
    }

    {
        // SET up conv grow layer
        conv_grow_layer_->Reshape(reduce_tops, top);

        // SET up bn grow layer
        if(has_bn_grow_param_)
            bn_grow_layer_->Reshape(top, top);

        // SET up relu grow layer
        relu_grow_layer_->Reshape(top, top);
    }

    int device;
    cudaGetDevice(&device);

    Blob<Dtype>* global_data_buffer_ = global_data_map_[device];
    Blob<Dtype>* global_diff_buffer_ = global_diff_map_[device];

    if(!global_data_buffer_)
    {
        global_data_map_[device] = new Blob<Dtype>(0,0,0,0);
        global_data_buffer_ = global_data_map_[device];
    }

    if(!global_diff_buffer_)
    {
        global_diff_map_[device] = new Blob<Dtype>(0,0,0,0);
        global_diff_buffer_ = global_diff_map_[device];
    }

    const int global_buffer_size = reduce_top_.count();

    if(global_data_buffer_->count() < global_buffer_size )
    {
        global_data_buffer_->Reshape(global_buffer_size, 1, 1, 1);
        global_diff_buffer_->Reshape(global_buffer_size, 1, 1, 1);
    }
}


template <typename Dtype>
void BottleNeckLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {


}

template <typename Dtype>
void BottleNeckLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(BottleNeckLayer);
#endif

INSTANTIATE_CLASS(BottleNeckLayer);
REGISTER_LAYER_CLASS(BottleNeck);

}  // namespace caffe
