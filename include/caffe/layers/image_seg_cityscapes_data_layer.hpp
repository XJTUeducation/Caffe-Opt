#ifndef CAFFE_IMAGE_SEG_DATA_LAYER_HPP_
#define CAFFE_IMAGE_SEG_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class ImageSegCityScapesDataLayer : public ImageDimPrefetchingDataLayer<Dtype> {
 public:
  explicit ImageSegCityScapesDataLayer(const LayerParameter& param)
    : ImageDimPrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageSegCityScapesDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageSegData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  virtual inline bool AutoTopBlobs() const { return true; }

 protected:
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  Blob<Dtype> transformed_label_;
  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, std::string> > lines_;
  int lines_id_;

  std::vector<cv::Mat> images_;
  std::vector<cv::Mat> labels_;

  std::vector<int> n_times_x_;
  std::vector<int> n_times_y_;

  int n_times_to_parse_full_resolution_x_;
  int n_times_to_parse_full_resolution_y_;

  int stride_x_;
  int stride_y_;

  bool flag_single_batch_finished_;
  bool flag_all_images_finished_;

  enum CropType
  {
      CROP_TYPE_ROW_MAJOR_FWD,
      CROP_TYPE_ROW_MAJOR_REV,
      CROP_TYPE_COL_MAJOR_FWD,
      CROP_TYPE_COL_MAJOR_REV
  };

  std::vector<CropType> crop_types_;

  int n_crops_processed_;

  float interpolation_factor_;

  int im_width_;
  int im_height_;

};

}  // namespace caffe

#endif  // CAFFE_IMAGE_SEG_DATA_LAYER_HPP_
