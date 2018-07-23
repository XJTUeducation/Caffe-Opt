#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/layers/instance_interp_layer.hpp"

namespace caffe {

template <typename Dtype>
void InstanceInterpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
    InterpParameter interp_param = this->layer_param_.interp_param();
    pad_beg_ = interp_param.pad_beg();
    pad_end_ = interp_param.pad_end();
    CHECK_LE(pad_beg_, 0) << "Only supports non-pos padding (cropping) for now";
    CHECK_LE(pad_end_, 0) << "Only supports non-pos padding (cropping) for now";
}

template <typename Dtype>
void InstanceInterpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {


    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_in_ = bottom[0]->height();
    width_in_ = bottom[0]->width();
    height_in_eff_ = height_in_ + pad_beg_ + pad_end_;
    width_in_eff_ = width_in_ + pad_beg_ + pad_end_;

    InterpParameter interp_param = this->layer_param_.interp_param();

    height_out_  = interp_param.height();
    width_out_  = interp_param.width();

    CHECK_GT(height_out_, 0) << "height should be positive";
    CHECK_GT(width_out_, 0) << "width should be positive";

    //--fake nums of instances ..because n instances
    //--will depend on the batch and instances per image
    top[0]->Reshape(1, channels_, height_out_, width_out_);
}

template <typename Dtype>
void InstanceInterpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {

    int org_h = bottom[0]->height();
    int org_w = bottom[0]->width();

    int n_rois = bottom[1]->height();

    ((Blob<Dtype>*)top[0])->Reshape(n_rois,channels_,height_out_,width_out_);
    //    ((Blob<Dtype>*)top[1])->Reshape(1,1,n_rois,1);

    const Dtype* top_label = bottom[1]->cpu_data();

    int item_id;
    //    int label;
    int xmin;
    int ymin;
    int xmax;
    int ymax;
    int x1;
    int y1;

    int width;
    int height;

    int idx = 0;

    for(int i=0;i<n_rois;i++)
    {
        item_id = top_label[idx++];
        if(item_id > -1)
        {
            idx++; // label = top_label[idx++];
            idx++; // top_label[idx++] = anno.instance_id();
            xmin = org_w* top_label[idx++];
            ymin = org_h* top_label[idx++];
            xmax = org_w* top_label[idx++];
            ymax = org_h* top_label[idx++];
            idx++; // top_label[idx++] = bbox.difficult();

            if(xmin <0)xmin = 0;
            if(ymin <0)ymin = 0;

            if(xmin >= org_w)xmin = org_w - 1;
            if(ymin >= org_h)ymin = org_h - 1;

            if(xmax <0)xmax = 0;
            if(ymax <0)ymax = 0;

            if(xmax >= org_w)xmax = org_w - 1;
            if(ymax >= org_h)ymax = org_h - 1;

            x1 = xmin;
            y1 = ymin;

            width = xmax - xmin;
            height = ymax - ymin;

            if(!width) width = 1;
            if(!height) height = 1;
            const Dtype* data_in = bottom[0]->cpu_data() + item_id * channels_*height_in_*width_in_;
            Dtype* data_out = top[0]->mutable_cpu_data() + i * channels_*height_out_*width_out_;

            caffe_cpu_interp2<Dtype,false>(channels_,
                                           data_in, x1, y1, height, width, height_in_, width_in_,
                                           data_out, 0, 0, height_out_, width_out_, height_out_, width_out_);
            //        Dtype data_label = top[1]->mutable_cpu_data()[i];
            //        data_label = label;
        }
        else
            caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());

    }
}
template <typename Dtype>
void InstanceInterpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) { return; }
    caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());


    int org_h = bottom[0]->height();
    int org_w = bottom[0]->width();

    int n_rois = bottom[1]->height();

    const Dtype* top_label = bottom[1]->cpu_data();

    int item_id;
    //    int label;
    int xmin;
    int ymin;
    int xmax;
    int ymax;
    int x1;
    int y1;

    int width;
    int height;

    int idx = 0;
    for(int i=0;i<n_rois;i++)
    {
        item_id = top_label[idx++];
        if(item_id > -1)
        {
            idx++; //label = top_label[idx++];
            idx++; // top_label[idx++] = anno.instance_id();
            xmin = org_w* top_label[idx++];
            ymin = org_h* top_label[idx++];
            xmax = org_w* top_label[idx++];
            ymax = org_h* top_label[idx++];
            idx++; // top_label[idx++] = bbox.difficult();

            if(xmin <0)xmin = 0;
            if(ymin <0)ymin = 0;

            if(xmin >= org_w)xmin = org_w - 1;
            if(ymin >= org_h)ymin = org_h - 1;

            if(xmax <0)xmax = 0;
            if(ymax <0)ymax = 0;

            if(xmax >= org_w)xmax = org_w - 1;
            if(ymax >= org_h)ymax = org_h - 1;

            x1 = xmin;
            y1 = ymin;

            width = xmax - xmin;
            height = ymax - ymin;

            if(!width) width = 1;
            if(!height) height = 1;


            Dtype* data_bottom_diff = bottom[0]->mutable_cpu_diff() + item_id * channels_*height_in_*width_in_;
            const Dtype* data_top_diff = top[0]->cpu_diff() + i * channels_*height_out_*width_out_;

            caffe_cpu_interp2_backward<Dtype,false>(channels_,
                                                    data_bottom_diff, x1, y1, height, width, height_in_, width_in_,
                                                    data_top_diff, 0, 0, height_out_, width_out_, height_out_, width_out_);
        }
    }


}

#ifndef CPU_ONLY
template <typename Dtype>
void InstanceInterpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {

    int org_h = bottom[0]->height();
    int org_w = bottom[0]->width();

    int n_rois = bottom[1]->height();

    ((Blob<Dtype>*)top[0])->Reshape(n_rois,channels_,height_out_,width_out_);

    //    ((Blob<Dtype>*)top[1])->Reshape(1,1,n_rois,1);

    const Dtype* top_label = bottom[1]->cpu_data();

    int item_id;
    //    int label;
    int xmin;
    int ymin;
    int xmax;
    int ymax;
    int x1;
    int y1;

    int width;
    int height;

    int idx = 0;
    for(int i=0;i<n_rois;i++)
    {
        item_id = top_label[idx++];
        if(item_id > -1)
        {
            idx++; //label = top_label[idx++];
            idx++; // top_label[idx++] = anno.instance_id();
            xmin = org_w* top_label[idx++];
            ymin = org_h* top_label[idx++];
            xmax = org_w* top_label[idx++];
            ymax = org_h* top_label[idx++];
            idx++; // top_label[idx++] = bbox.difficult();

            if(xmin <0)xmin = 0;
            if(ymin <0)ymin = 0;

            if(xmin >= org_w)xmin = org_w - 1;
            if(ymin >= org_h)ymin = org_h - 1;

            if(xmax <0)xmax = 0;
            if(ymax <0)ymax = 0;

            if(xmax >= org_w)xmax = org_w - 1;
            if(ymax >= org_h)ymax = org_h - 1;

            x1 = xmin;
            y1 = ymin;

            width = xmax - xmin;
            height = ymax - ymin;

            if(!width) width = 1;
            if(!height) height = 1;

            //            std::cout << i << " " <<
            //                         x1 << "  " <<
            //                         y1 << "  " <<
            //                         width << "  " <<
            //                         org_w << "  " <<
            //                         height << "  " <<
            //                         org_h << "  " << "\n";


            const Dtype* data_in = bottom[0]->gpu_data() + item_id * channels_*height_in_*width_in_;
            Dtype* data_out = top[0]->mutable_gpu_data() + i * channels_*height_out_*width_out_;

            caffe_gpu_interp2<Dtype,false>(channels_,
                                           data_in, x1, y1, height, width, height_in_, width_in_,
                                           data_out, 0, 0, height_out_, width_out_, height_out_, width_out_);
            //        Dtype data_label = top[1]->mutable_cpu_data()[i];
            //        data_label = label;
        }
        else
            caffe_gpu_set(top[0]->count(), Dtype(0), top[0]->mutable_gpu_data());
    }
}

template <typename Dtype>
void InstanceInterpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) { return; }
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());

    int org_h = bottom[0]->height();
    int org_w = bottom[0]->width();

    int n_rois = bottom[1]->height();

    const Dtype* top_label = bottom[1]->cpu_data();

    int item_id;
    //    int label;
    int xmin;
    int ymin;
    int xmax;
    int ymax;
    int x1;
    int y1;

    int width;
    int height;

    int idx = 0;
    for(int i=0;i<n_rois;i++)
    {
        item_id = top_label[idx++];

        if(item_id > -1)
        {
            idx++; //label = top_label[idx++];
            idx++; // top_label[idx++] = anno.instance_id();
            xmin = org_w* top_label[idx++];
            ymin = org_h* top_label[idx++];
            xmax = org_w* top_label[idx++];
            ymax = org_h* top_label[idx++];
            idx++; // top_label[idx++] = bbox.difficult();

            if(xmin <0)xmin = 0;
            if(ymin <0)ymin = 0;

            if(xmin >= org_w)xmin = org_w - 1;
            if(ymin >= org_h)ymin = org_h - 1;

            if(xmax <0)xmax = 0;
            if(ymax <0)ymax = 0;

            if(xmax >= org_w)xmax = org_w - 1;
            if(ymax >= org_h)ymax = org_h - 1;

            x1 = xmin;
            y1 = ymin;

            width = xmax - xmin;
            height = ymax - ymin;

            if(!width) width = 1;
            if(!height) height = 1;

            Dtype* data_bottom_diff = bottom[0]->mutable_gpu_diff() + item_id * channels_*height_in_*width_in_;
            const Dtype* data_top_diff = top[0]->gpu_diff() + i * channels_*height_out_*width_out_;

            caffe_gpu_interp2_backward<Dtype,false>(channels_,
                                                    data_bottom_diff, x1, y1, height, width, height_in_, width_in_,
                                                    data_top_diff, 0, 0, height_out_, width_out_, height_out_, width_out_);

        }
    }

}
#endif

#ifdef CPU_ONLY
STUB_GPU(InstanceInterpLayer);
#endif

INSTANTIATE_CLASS(InstanceInterpLayer);
REGISTER_LAYER_CLASS(InstanceInterp);

}  // namespace caffe
