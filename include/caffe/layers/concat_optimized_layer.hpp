#ifndef CAFFE_CONCAT_OPTIMIZED_LAYER_HPP_
#define CAFFE_CONCAT_OPTIMIZED_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/memory_bank.hpp"

namespace caffe {



/**
 * @brief Takes at least two Blob%s and concatenates them along either the num
 *        or channel dimension, outputting the result.
 */
template <typename Dtype>
class ConcatOptimizedLayer : public Layer<Dtype> {
public:
    explicit ConcatOptimizedLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {

        data_pointer_map_ =0;
        diff_pointer_map_ = 0;
        concat_child_ = 0;

    }

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    ~ConcatOptimizedLayer();


    virtual inline const char* type() const { return "Concat"; }
    virtual inline int MinBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }




    typedef struct _concat_child_
    {
        std::vector<Blob<Dtype>*>* bottom;
        std::vector<Blob<Dtype>*>* top;
        bool own_data;
    } concat_child ;



protected:
    /**
   * @param bottom input Blob vector (length 2+)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x_1 @f$
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x_2 @f$
   *   -# ...
   *   - K @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x_K @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (KN \times C \times H \times W) @f$ if axis == 0, or
   *      @f$ (N \times KC \times H \times W) @f$ if axis == 1:
   *      the concatenated output @f$
   *        y = [\begin{array}{cccc} x_1 & x_2 & ... & x_K \end{array}]
   *      @f$
   */
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

    /**
   * @brief Computes the error gradient w.r.t. the concatenate inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *        respect to the outputs
   *   -# @f$ (KN \times C \times H \times W) @f$ if axis == 0, or
   *      @f$ (N \times KC \times H \times W) @f$ if axis == 1:
   *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
   *      with respect to concatenated outputs @f$ y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length K), into which the top gradient
   *        @f$ \frac{\partial E}{\partial y} @f$ is deconcatenated back to the
   *        inputs @f$
   *        \left[ \begin{array}{cccc}
   *          \frac{\partial E}{\partial x_1} &
   *          \frac{\partial E}{\partial x_2} &
   *          ... &
   *          \frac{\partial E}{\partial x_K}
   *        \end{array} \right] =
   *        \frac{\partial E}{\partial y}
   *        @f$
   */
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


private:

    void rellocate_the_map(std::map<void*, concat_child* >* map,
                           Dtype* data_pointer, Dtype* diff_pointer,
                           const std::vector<Blob<Dtype>*> & bottom,
                           const std::vector<Blob<Dtype>*> & top);

    void rellocate_the_map_mem_bank(std::map<void*, concat_child* >* map,
                                    const Blob<Dtype>& layer_top, unsigned long int offset,
                                    const std::vector<Blob<Dtype>*> &bottom,
                                    const std::vector<Blob<Dtype>*> &top);

    concat_child* concat_child_;

    static std::map<int, std::map<void*, concat_child* >* > concat_chain_;

    std::map<int,Dtype*>* data_pointer_map_; // FOR MULTI GPU
    std::map<int,Dtype*>* diff_pointer_map_; // FOR MULTI GPU

    static   std::vector<unsigned long int> buffer_old_count_;
    static   std::map<int,Dtype*>* temp_data_pointer_map_; // FOR MULTI GPU


    std::map<void*, concat_child* >* concat_chain_map_;
    int device_id_;
    Blob<Dtype> temp_buffer_;


    int count_;
    int num_concats_;
    int concat_input_size_;
    int concat_axis_;
};

}  // namespace caffe

#endif  // CAFFE_CONCAT_OPTMIZED_LAYER_HPP_

