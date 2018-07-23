#include <vector>

#include "caffe/layers/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Concat(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index = concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    if (forward) {
      out_data[top_index] = in_data[index];
    } else {
      out_data[index] = in_data[top_index];
    }
  }
}

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


//  std::cout << "in concat forwd gpu\n";
//  int num = bottom[0]->shape(0);

//  std::cout << "****************** CONCAT in PUT ******************************************\n\n";
//  int count = 0;
//  for(int l=0;l<bottom.size();l++)
//  {
//      count = 0;
//      std::cout << "concate bottom pointer = " << (int64_t)(bottom[l]->gpu_data()) << "  " << (int64_t)(bottom[l]->cpu_data()) << "\n";
//      for(int m=0;m<bottom[l]->shape(0);m++)
//      {
//          for(int i=0;i<bottom[l]->shape(1);i++)
//          {
//              for(int j=0;j<bottom[l]->shape(2);j++)
//              {
//                  for(int k=0;k<bottom[l]->shape(3);k++)
//                  {
//                      std::cout << bottom[l]->cpu_data()[count++] << " ";

//                  }

//                  std::cout << "\n";
//              }
//              std::cout << "\n\n";
//          }
//          std::cout << "\n\nNEXT NUM \n\n";
//      }
//      std::cout << "\n\nNEXT BOTTOM \n\n";
//  }
//  std::cout << "****************** CONCAT in PUT ENDS ******************************************\n\n";

    if (bottom.size() == 1) { return; }

  Dtype* top_data = top[0]->mutable_gpu_data();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = true;
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
    const int nthreads = bottom_concat_size * num_concats_;
    Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_data, kForward, num_concats_, concat_input_size_,
        top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data);
    offset_concat_axis += bottom_concat_axis;
  }

//  std::cout << "****************** CONCAT OUT PUT ******************************************\n\n";
//  count = 0;
//  for(int l=0;l<top[0]->shape(0);l++)
//  {
//      std::cout << "concate top pointer = " << (int64_t)(top[0]->gpu_data()) << "  " << (int64_t)(top[0]->cpu_data()) << "\n";

//      for(int i=0;i<top[0]->shape(1);i++)
//      {
//          for(int j=0;j<top[0]->shape(2);j++)
//          {
//              for(int k=0;k<top[0]->shape(3);k++)
//              {
//                  std::cout << top[0]->cpu_data()[count++] << " ";
//              }

//              std::cout << "\n";
//          }
//          std::cout << "\n\n";
//      }
//      std::cout << "\n\nNEXT NUM \n\n";
//  }
//  std::cout << "****************** CONCAT OUT PUT ENDS ******************************************\n\n";
}

template <typename Dtype>
void ConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

//  static int diff_fill = 0;

//  std::cout << "in concat backward gpu\n";

//  int num = bottom[0]->shape(0);

//  int count = 0;
//  if(diff_fill == 0)
//  {
////      std::cout << "******************concat OUT PUT DIFF orgin******************************************\n\n";
////      std::cout << "concate diff top pointer = " << (int64_t)(top[0]->gpu_diff()) << "  " << (int64_t)(top[0]->cpu_diff()) << "\n";

//      for(int l=0;l<top[0]->shape(0);l++)
//      {

//          for(int i=0;i<top[0]->shape(1);i++)
//          {
//              for(int j=0;j<top[0]->shape(2);j++)
//              {
//                  for(int k=0;k<top[0]->shape(3);k++)
//                  {
//                      top[0]->mutable_cpu_diff()[count++] = i+1;

////                      std::cout <<top[0]->cpu_diff()[count-1] << "  ";
//                  }

////                  std::cout << "\n";
//              }
////              std::cout << "\n\n";
//          }
////          std::cout << "\n\nNEXT NUM \n\n";
//      }
//      diff_fill++;
////      std::cout << "******************concat OUT PUT ENDS DIFF orgin******************************************\n\n\n\n\n";

//  top[0]->gpu_diff();

//  }

    if (bottom.size() == 1) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = false;
  for (int i = 0; i < bottom.size(); ++i) {
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
      const int nthreads = bottom_concat_size * num_concats_;
      Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, top_diff, kForward, num_concats_, concat_input_size_,
          top_concat_axis, bottom_concat_axis, offset_concat_axis, bottom_diff);
    }
    offset_concat_axis += bottom_concat_axis;
  }

//  std::cout << "****************** CONCAT OUT PUT BCWD ******************************************\n\n";


//  for(int m=0;m<bottom.size();m++)
//  {
//      std::cout << "\n\n CONCAT BOTTOM NUM = " << m << "  \n\n";
//      count = 0;

//      for(int l=0;l<bottom[m]->shape(0);l++)
//      {
//          for(int i=0;i<bottom[m]->shape(1);i++)
//          {
//              for(int j=0;j<bottom[m]->shape(2);j++)
//              {
//                  for(int k=0;k<bottom[m]->shape(3);k++)
//                  {
//                      std::cout << bottom[m]->cpu_data()[count++] << " ";
//                  }

//                  std::cout << "\n";
//              }
//              std::cout << "\n\n";
//          }
//          std::cout << "\n\nNEXT NUM \n\n";
//      }
//  }
//  std::cout << "****************** CONCAT OUT PUT ENDS BCWD******************************************\n\n";


//  std::cout << "****************** CONCAT OUT PUT DIFF BCWD ******************************************\n\n";

//  std::cout << "concate diff bottom pointer = " << (int64_t)(bottom[0]->gpu_diff())
//          << "  " << (int64_t)(bottom[0]->cpu_diff()) << "\n";

//    for(int m=0;m<bottom.size();m++)
//  {
//      std::cout << "\n\n CONCAT BOTTOM NUM = " << m << "  \n\n";
//      count = 0;

////      for(int l=0;l<bottom[m]->shape(0);l++)
//      {
//          for(int i=0;i<10;i++)//bottom[m]->shape(1);i++)
//          {
//              for(int j=0;j<bottom[m]->shape(2);j++)
//              {
//                  for(int k=0;k<bottom[m]->shape(3);k++)
//                  {
//                      std::cout << bottom[m]->cpu_diff()[count++] << " ";
//                  }

//                  std::cout << "\n";
//              }
//              std::cout << "\n\n";
//          }
//          std::cout << "\n\nNEXT NUM \n\n";
//      }
//  }
//  std::cout << "****************** CONCAT OUT PUT DIFF ENDS BCWD******************************************\n\n";


}

INSTANTIATE_LAYER_GPU_FUNCS(ConcatLayer);

}  // namespace caffe
