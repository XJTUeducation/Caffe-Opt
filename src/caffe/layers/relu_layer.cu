#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

#include<opencv2/opencv.hpp>

namespace caffe {

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

//  std::cout << "relu top bottom  =  " << (int64_t)top[0] << "  " << (int64_t)bottom[0] << "\n";
//  std::cout << "relu top bottom data pointers =  " << (int64_t)top_data << "  " << (int64_t)bottom_data << "\n";

  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, negative_slope);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;

//#define SAVE_FEATURES

#ifdef SAVE_FEATURES

  if(this->phase_ != TRAIN)
  {
      static int instance = 0;

      int f_map = 0;
      int out_count = 0;
      for(int l=0;l<top[0]->shape(0);l++)
      {
          for(int i=0;i<top[0]->shape(1);i++)
          {
              cv::Mat image = cv::Mat::zeros(top[0]->shape(2),top[0]->shape(3),CV_8UC1);

              int count  = 0 ;
              for(int j=0;j<top[0]->shape(2);j++)
              {
                  for(int k=0;k<top[0]->shape(3);k++)
                  {
//                      std::cout << top[0]->cpu_data()[count++] << " ";
                        image.data[count] =  255 * top[0]->mutable_cpu_data()[out_count++];

                        image.data[count] = image.data[count] > 255 ? 255 : image.data[count];
                        count++;
                  }

              }

              std::stringstream im_name;
              im_name << "/home/isl-server/ashish/op_feature_maps/" << instance << "_" << f_map << ".png";

              cv::imwrite(im_name.str(),image);

              f_map++;
          }
      }

      instance++;
  }

#endif


}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, negative_slope);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);


}  // namespace caffe
