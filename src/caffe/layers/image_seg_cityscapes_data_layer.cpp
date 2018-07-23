#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_seg_cityscapes_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include<ctime>

namespace caffe {


//#define ALL_IMAGE_FIRST


template <typename Dtype>
ImageSegCityScapesDataLayer<Dtype>::~ImageSegCityScapesDataLayer<Dtype>() {
    this->StopInternalThread();
}

template <typename Dtype>
void ImageSegCityScapesDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                        const vector<Blob<Dtype>*>& top) {
    const int new_height = this->layer_param_.image_data_param().new_height();
    const int new_width  = this->layer_param_.image_data_param().new_width();
    const bool is_color  = this->layer_param_.image_data_param().is_color();
    const int label_type = this->layer_param_.image_data_param().label_type();
    string root_folder = this->layer_param_.image_data_param().root_folder();

    TransformationParameter transform_param = this->layer_param_.transform_param();
    CHECK(transform_param.has_mean_file() == false) <<
                                                       "ImageSegDataLayer does not support mean file";
    CHECK((new_height == 0 && new_width == 0) ||
          (new_height > 0 && new_width > 0)) << "Current implementation requires "
                                                "new_height and new_width to be set at the same time.";


    // Read the file with filenames and labels
    const string& source = this->layer_param_.image_data_param().source();
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());

    string linestr;
    while (std::getline(infile, linestr)) {
        std::istringstream iss(linestr);
        string imgfn;
        iss >> imgfn;
        string segfn = "";
        if (label_type != ImageDataParameter_LabelType_NONE) {
            iss >> segfn;
        }
        lines_.push_back(std::make_pair(imgfn, segfn));
    }

    if (this->layer_param_.image_data_param().shuffle()) {
        // randomly shuffle data
        LOG(INFO) << "Shuffling data";
        const unsigned int prefetch_rng_seed = caffe_rng_rand();
        prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
        ShuffleImages();
    }
    LOG(INFO) << "A total of " << lines_.size() << " images.";

    lines_id_ = 0;
    // Check if we would need to randomly skip a few data points
    if (this->layer_param_.image_data_param().rand_skip()) {
        unsigned int skip = caffe_rng_rand() %
                this->layer_param_.image_data_param().rand_skip();
        LOG(INFO) << "Skipping first " << skip << " data points.";
        CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
        lines_id_ = skip;
    }
    // Read an image, and use it to initialize the top blob.
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                      new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

    const int channels = cv_img.channels();
    const int height = cv_img.rows;
    const int width = cv_img.cols;

    // image
    //const int crop_size = this->layer_param_.transform_param().crop_size();
    int crop_width = 0;
    int crop_height = 0;
    CHECK((!transform_param.has_crop_size() && transform_param.has_crop_height() && transform_param.has_crop_width())
          || (!transform_param.has_crop_height() && !transform_param.has_crop_width()))
            << "Must either specify crop_size or both crop_height and crop_width.";
    if (transform_param.has_crop_size()) {
        crop_width = transform_param.crop_size();
        crop_height = transform_param.crop_size();
    }
    if (transform_param.has_crop_height() && transform_param.has_crop_width()) {
        crop_width = transform_param.crop_width();
        crop_height = transform_param.crop_height();
    }

    const int batch_size = this->layer_param_.image_data_param().batch_size();
    if (crop_width > 0 && crop_height > 0) {
        top[0]->Reshape(batch_size, channels, crop_height, crop_width);
        this->transformed_data_.Reshape(batch_size, channels, crop_height, crop_width);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].data_.Reshape(batch_size, channels, crop_height, crop_width);
        }

        //label
        top[1]->Reshape(batch_size, 1, crop_height, crop_width);
        this->transformed_label_.Reshape(batch_size, 1, crop_height, crop_width);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].label_.Reshape(batch_size, 1, crop_height, crop_width);
        }
    } else {
        top[0]->Reshape(batch_size, channels, height, width);
        this->transformed_data_.Reshape(batch_size, channels, height, width);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].data_.Reshape(batch_size, channels, height, width);
        }

        //label
        top[1]->Reshape(batch_size, 1, height, width);
        this->transformed_label_.Reshape(batch_size, 1, height, width);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].label_.Reshape(batch_size, 1, height, width);
        }
    }
    // image dimensions, for each image, stores (img_height, img_width)
    top[2]->Reshape(batch_size, 1, 1, 2);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].dim_.Reshape(batch_size, 1, 1, 2);
    }

    LOG(INFO) << "output data size: " << top[0]->num() << ","
              << top[0]->channels() << "," << top[0]->height() << ","
              << top[0]->width();
    // label
    LOG(INFO) << "output label size: " << top[1]->num() << ","
              << top[1]->channels() << "," << top[1]->height() << ","
              << top[1]->width();
    // image_dim
    LOG(INFO) << "output data_dim size: " << top[2]->num() << ","
              << top[2]->channels() << "," << top[2]->height() << ","
              << top[2]->width();

#ifdef ALL_IMAGE_FIRST
    flag_all_images_finished_ = true;

    n_times_x_.resize(lines_.size(),0);
    n_times_y_.resize(lines_.size(),0);

    crop_types_.resize(lines_.size(),(CropType)0);

    n_crops_processed_ = 0;

    im_width_ = width;
    im_height_ = height;
#else
    flag_single_batch_finished_ = true;

    n_times_x_.resize(batch_size);
    n_times_y_.resize(batch_size);

    crop_types_.resize(batch_size);

    n_crops_processed_ = 0;

    interpolation_factor_ = 1.0;

#endif
}

template <typename Dtype>
void ImageSegCityScapesDataLayer<Dtype>::ShuffleImages() {
    caffe::rng_t* prefetch_rng =
            static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageSegCityScapesDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());

    Dtype* top_data     = batch->data_.mutable_cpu_data();
    Dtype* top_label    = batch->label_.mutable_cpu_data();
    Dtype* top_data_dim = batch->dim_.mutable_cpu_data();

    const int max_height = batch->data_.height();
    const int max_width  = batch->data_.width();

    ImageDataParameter image_data_param = this->layer_param_.image_data_param();
    const int batch_size = image_data_param.batch_size();
    const int new_height = image_data_param.new_height();
    const int new_width  = image_data_param.new_width();
    const int label_type = this->layer_param_.image_data_param().label_type();
    const int ignore_label = image_data_param.ignore_label();
    const bool is_color  = image_data_param.is_color();
    string root_folder   = image_data_param.root_folder();

    const int lines_size = lines_.size();
    int top_data_dim_offset;

    const int crop_w = this->transformed_data_.shape(3);
    const int crop_h = this->transformed_data_.shape(2);

    // GET UNIX EPOCH
    uint64 random_seed = (uint64_t)std::time(0);

    static cv::RNG_MT19937 rng_(random_seed);


#ifdef ALL_IMAGE_FIRST

    if(flag_all_images_finished_)
    {
        for(int i=0;i<crop_types_.size();i++)
            crop_types_[i] = (CropType)rng_(4);

        // RANDOMLY SELECT THE STRIDES IN ORDER TO AVOID BIASING
        // TO KEEP LESSER CROPS PER IMAGE, RANDOMLY SELECT
        // STRIDE BETWEEN 50% to 80%
        // SET STRIDES AND N_TIME_TO_PARSE_FULL_RESOLUTION

        int min_factor = 5;
        int max_factor = 9;


        float factor = rng_.uniform(min_factor, max_factor + 1) / 10.0;

        stride_x_ = crop_w * factor;
        stride_y_ = crop_h * factor;


        n_times_to_parse_full_resolution_x_ = ceil(((float)im_width_ - crop_w) / stride_x_) + 1;
        n_times_to_parse_full_resolution_y_ = ceil(((float)im_height_ - crop_h) / stride_y_) + 1;

        for(int i=0;i<n_times_x_.size();i++)
        {
            n_times_x_[i] = 0;
            n_times_y_[i] = 0;
        }

        flag_all_images_finished_ = false;
    }


    for (int item_id = 0; item_id < batch_size; ++item_id) {
        top_data_dim_offset = batch->dim_.offset(item_id);

        std::vector<cv::Mat> cv_img_seg;

        // get a blob
        timer.Start();
        CHECK_GT(lines_size, lines_id_);

        int img_row, img_col;
        cv_img_seg.push_back(PSP_ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                                  new_height, new_width, is_color, &img_row, &img_col));

        // TODO(jay): implement resize in ReadImageToCVMat
        // NOTE data_dim may not work when min_scale and max_scale != 1
        top_data_dim[top_data_dim_offset]     = static_cast<Dtype>(std::min(max_height, img_row));
        top_data_dim[top_data_dim_offset + 1] = static_cast<Dtype>(std::min(max_width, img_col));

        if (!cv_img_seg[0].data) {
            DLOG(INFO) << "Fail to load img: " << root_folder + lines_[lines_id_].first;
        }
        if (label_type == ImageDataParameter_LabelType_PIXEL) {
            cv_img_seg.push_back(ReadImageToCVMat(root_folder + lines_[lines_id_].second,
                                                  new_height, new_width, false));
            //      std::cout << "BATCH NO = " << batch_no++ << lines_[lines_id_].second.c_str() << "\n";
            if (!cv_img_seg[1].data) {
                DLOG(INFO) << "Fail to load seg: " << root_folder + lines_[lines_id_].second;
            }
        }
        else if (label_type == ImageDataParameter_LabelType_IMAGE) {
            const int label = atoi(lines_[lines_id_].second.c_str());
            //std::cout << lines_[lines_id_].second.c_str() << "  LABEL = " << label << "\n";
            cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols,
                    CV_8UC1, cv::Scalar(label));
            cv_img_seg.push_back(seg);
        }
        else {
            cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols,
                    CV_8UC1, cv::Scalar(ignore_label));
            cv_img_seg.push_back(seg);
        }


        int x;
        int y;

        switch(crop_types_[lines_id_])
        {
        case CROP_TYPE_ROW_MAJOR_FWD:
        case CROP_TYPE_COL_MAJOR_FWD:

            x = n_times_x_[lines_id_] * stride_x_;
            y = n_times_y_[lines_id_] * stride_y_;

            if(crop_types_[lines_id_] == CROP_TYPE_ROW_MAJOR_FWD)
            {
                n_times_x_[lines_id_]++;

                if( !(n_times_x_[lines_id_] % n_times_to_parse_full_resolution_x_))
                {
                    n_times_x_[lines_id_] %= n_times_to_parse_full_resolution_x_;

                    n_times_y_[lines_id_]++;
                }
            }

            if(crop_types_[lines_id_] == CROP_TYPE_COL_MAJOR_FWD)
            {

                n_times_y_[lines_id_]++;

                if( !(n_times_y_[lines_id_] % n_times_to_parse_full_resolution_y_))
                {
                    n_times_y_[lines_id_] %= n_times_to_parse_full_resolution_y_;

                    n_times_x_[lines_id_]++;
                }
            }

            break;

        case CROP_TYPE_ROW_MAJOR_REV:
        case CROP_TYPE_COL_MAJOR_REV:

            x = (n_times_to_parse_full_resolution_x_ -1 -n_times_x_[lines_id_]) * stride_x_;
            y = (n_times_to_parse_full_resolution_y_ -1 - n_times_y_[lines_id_]) * stride_y_;

            if(crop_types_[lines_id_] == CROP_TYPE_ROW_MAJOR_REV)
            {
                n_times_x_[lines_id_]++;

                if( !(n_times_x_[lines_id_] % n_times_to_parse_full_resolution_x_))
                {
                    n_times_x_[lines_id_] %= n_times_to_parse_full_resolution_x_;

                    n_times_y_[lines_id_]++;
                }
            }

            if(crop_types_[lines_id_] == CROP_TYPE_COL_MAJOR_REV)
            {

                n_times_y_[lines_id_]++;

                if( !(n_times_y_[lines_id_] % n_times_to_parse_full_resolution_y_))
                {
                    n_times_y_[lines_id_] %= n_times_to_parse_full_resolution_y_;

                    n_times_x_[lines_id_]++;
                }
            }

            break;
        }


        const int im_width = cv_img_seg[0].cols;
        const int im_height = cv_img_seg[1].rows;

        if( x + crop_w > im_width )
            x -= (x + crop_w ) - im_width;

        if( y + crop_h > im_height)
            y -= (y + crop_h ) - im_height;

        cv::Rect roi(x,y,crop_w,crop_h);

        std::vector<cv::Mat> cv_img_seg_cropped;

        cv_img_seg_cropped.push_back(cv::Mat());
        cv_img_seg_cropped.push_back(cv::Mat());

        cv_img_seg[0](roi).copyTo(cv_img_seg_cropped[0]);
        cv_img_seg[1](roi).copyTo(cv_img_seg_cropped[1]);


        read_time += timer.MicroSeconds();
        timer.Start();
        // Apply transformations (mirror, crop...) to the image
        int offset;
        offset = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(top_data + offset);

        offset = batch->label_.offset(item_id);
        this->transformed_label_.set_cpu_data(top_label + offset);

        //        this->data_transformer_->TransformImgAndSeg(cv_img_seg,
        //                                                    &(this->transformed_data_), &(this->transformed_label_),
        //                                                    ignore_label);

        this->data_transformer_->TransformImgAndSegOnline(cv_img_seg_cropped,
                                                          &(this->transformed_data_), &(this->transformed_label_),
                                                          ignore_label);

        //        cv::imshow("live",cv_img_seg[0]);
        //        cv::imshow("seg",5*cv_img_seg[1]);
        //        cv::waitKey(1);


        trans_time += timer.MicroSeconds();

        // go to the next std::vector<int>::iterator iter;
        lines_id_++;
        if (lines_id_ >= lines_size) {
            // We have reached the end. Restart from the first.
            DLOG(INFO) << "Restarting data prefetching from start.";
            lines_id_ = 0;
            if (this->layer_param_.image_data_param().shuffle()) {
                ShuffleImages();
            }
        }

        n_crops_processed_++;
    }

    if(n_crops_processed_ == lines_.size() * n_times_to_parse_full_resolution_x_ * n_times_to_parse_full_resolution_y_)
    {
        n_crops_processed_ = 0;
        flag_all_images_finished_ = true;
    }


#else

    if( flag_single_batch_finished_)
    {

        images_.clear();
        labels_.clear();

        for (int item_id = 0; item_id < batch_size; ++item_id)
        {

            CHECK_GT(lines_size, lines_id_);

            int img_row, img_col;
            images_.push_back(PSP_ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                                   new_height, new_width, is_color, &img_row, &img_col));

            if (!images_[item_id].data) {
                DLOG(INFO) << "Fail to load img: " << root_folder + lines_[lines_id_].first;
            }

            if (label_type == ImageDataParameter_LabelType_PIXEL) {
//                labels_.push_back(ReadImageToCVMat(root_folder + lines_[lines_id_].second,
//                                                   new_height, new_width, false));

                labels_.push_back(cv::imread(root_folder + lines_[lines_id_].second, cv::IMREAD_ANYDEPTH  | cv::IMREAD_GRAYSCALE));

                if (!labels_[item_id].data) {
                    DLOG(INFO) << "Fail to load seg: " << root_folder + lines_[lines_id_].second;
                }
            }

            // go to the next std::vector<int>::iterator iter;
            lines_id_++;
            if (lines_id_ >= lines_size) {
                // We have reached the end. Restart from the first.
                DLOG(INFO) << "Restarting data prefetching from start.";
                lines_id_ = 0;
                if (this->layer_param_.image_data_param().shuffle()) {
                    ShuffleImages();
                }
            }

            n_times_x_[item_id] = 0;
            n_times_y_[item_id] = 0;

            crop_types_[item_id] = (CropType)rng_(4);
        }

//        // resize for 600 or 700 or 800
//        int min_resize_factor = crop_w  / 100;
//        int max_resize_factor = (crop_w + 200) / 100;  // if crop_h = 600 --> max factor = 600 +200 = 800

//        int n_resize_steps = max_resize_factor - min_resize_factor + 1 ;

//        interpolation_factor_ = crop_w;//  ( min_resize_factor + rng_.uniform(0, n_resize_steps) ) * 100.0;

//        for (int item_id = 0; item_id < batch_size; ++item_id)
//        {
//            cv::resize(images_[item_id], images_[item_id], cv::Size(interpolation_factor_ * 2.0 , interpolation_factor_ ));
//            cv::resize(labels_[item_id], labels_[item_id], cv::Size(interpolation_factor_ * 2.0 , interpolation_factor_),0, 0, cv::INTER_NEAREST);
//        }

        // RANDOMLY SELECT THE STRIDES IN ORDER TO AVOID BIASING
        // TO KEEP LESSER CROPS PER IMAGE, RANDOMLY SELECT
        // STRIDE BETWEEN 50% to 80%
        // SET STRIDES AND N_TIME_TO_PARSE_FULL_RESOLUTION

        int min_factor = 5;
        int max_factor = 8;

        float factor = rng_.uniform(min_factor, max_factor + 1) / 10.0;

        stride_x_ = crop_w * factor;
        stride_y_ = crop_h * factor;

        int im_width = images_[0].cols;
        int im_height = images_[0].rows;

        n_times_to_parse_full_resolution_x_ = ceil(((float)im_width - crop_w) / stride_x_) + 1;
        n_times_to_parse_full_resolution_y_ = ceil(((float)im_height - crop_h) / stride_y_) + 1;


        flag_single_batch_finished_ = false;
    }


    for (int item_id = 0; item_id < batch_size; ++item_id)
    {
        int x;
        int y;

        switch(crop_types_[item_id])
        {
        case CROP_TYPE_ROW_MAJOR_FWD:
        case CROP_TYPE_COL_MAJOR_FWD:

            x = n_times_x_[item_id] * stride_x_;
            y = n_times_y_[item_id] * stride_y_;

            if(crop_types_[item_id] == CROP_TYPE_ROW_MAJOR_FWD)
            {
                n_times_x_[item_id]++;

                if( !(n_times_x_[item_id] % n_times_to_parse_full_resolution_x_))
                {
                    n_times_x_[item_id] %= n_times_to_parse_full_resolution_x_;

                    n_times_y_[item_id]++;
                }
            }

            if(crop_types_[item_id] == CROP_TYPE_COL_MAJOR_FWD)
            {

                n_times_y_[item_id]++;

                if( !(n_times_y_[item_id] % n_times_to_parse_full_resolution_y_))
                {
                    n_times_y_[item_id] %= n_times_to_parse_full_resolution_y_;

                    n_times_x_[item_id]++;
                }
            }

            break;

        case CROP_TYPE_ROW_MAJOR_REV:
        case CROP_TYPE_COL_MAJOR_REV:

            x = (n_times_to_parse_full_resolution_x_ -1 -n_times_x_[item_id]) * stride_x_;
            y = (n_times_to_parse_full_resolution_y_ -1 - n_times_y_[item_id]) * stride_y_;

            if(crop_types_[item_id] == CROP_TYPE_ROW_MAJOR_REV)
            {
                n_times_x_[item_id]++;

                if( !(n_times_x_[item_id] % n_times_to_parse_full_resolution_x_))
                {
                    n_times_x_[item_id] %= n_times_to_parse_full_resolution_x_;

                    n_times_y_[item_id]++;
                }
            }

            if(crop_types_[item_id] == CROP_TYPE_COL_MAJOR_REV)
            {

                n_times_y_[item_id]++;

                if( !(n_times_y_[item_id] % n_times_to_parse_full_resolution_y_))
                {
                    n_times_y_[item_id] %= n_times_to_parse_full_resolution_y_;

                    n_times_x_[item_id]++;
                }
            }

            break;
        }


        const int im_width = images_[item_id].cols;
        const int im_height = images_[item_id].rows;

        if( x + crop_w > im_width )
            x -= (x + crop_w ) - im_width;

        if( y + crop_h > im_height)
            y -= (y + crop_h ) - im_height;

        cv::Rect roi(x,y,crop_w,crop_h);

        std::vector<cv::Mat> cv_img_seg;

        cv_img_seg.push_back(cv::Mat());
        cv_img_seg.push_back(cv::Mat());

        images_[item_id](roi).copyTo(cv_img_seg[0]);
        labels_[item_id](roi).copyTo(cv_img_seg[1]);

           // Apply transformations (mirror, crop...) to the image
        int offset;
        offset = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(top_data + offset);

        offset = batch->label_.offset(item_id);
        this->transformed_label_.set_cpu_data(top_label + offset);

        //        this->data_transformer_->TransformImgAndSeg(cv_img_seg,
        //                                                    &(this->transformed_data_), &(this->transformed_label_),
        //                                                    ignore_label);

        this->data_transformer_->TransformImgAndSegOnline(cv_img_seg,
                                                          &(this->transformed_data_), &(this->transformed_label_),
                                                          ignore_label);


//        cv::Mat rect_input_image;
//        images_[item_id].copyTo(rect_input_image);

//        cv::rectangle(rect_input_image,roi,cv::Scalar(0,255,255),1);

//        cv::imshow("rect", rect_input_image);
//        cv::imshow("croped im", cv_img_seg[0]);
//        cv::imshow("cropped label", 5 * cv_img_seg[1]);

//        cv::waitKey(400);


        //        cv::imshow("live",cv_img_seg[0]);
        //        cv::imshow("seg",5*cv_img_seg[1]);
        //        cv::waitKey(1);
    }

    n_crops_processed_++;

    //    std::cout << "CROPS AND TOTAL = " << n_crops_processed_ << "  " << n_times_to_parse_full_resolution_x_ * n_times_to_parse_full_resolution_y_ << "\n";

    if(n_crops_processed_ == n_times_to_parse_full_resolution_x_ * n_times_to_parse_full_resolution_y_)
    {
        n_crops_processed_ = 0;
        flag_single_batch_finished_ = true;
    }

#endif

    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageSegCityScapesDataLayer);
REGISTER_LAYER_CLASS(ImageSegCityScapesData);

}  // namespace caffe
