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
#include "caffe/layers/classification_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ClassificationDataLayer<Dtype>::~ClassificationDataLayer<Dtype>() {
    this->StopInternalThread();
}

template <typename Dtype>
void ClassificationDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
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

    int instance_h = this->layer_param_.image_data_param().instance_h();
    int instance_w = this->layer_param_.image_data_param().instance_w();

    const int batch_size = this->layer_param_.image_data_param().batch_size();

    {
        top[0]->Reshape(batch_size, channels, instance_h, instance_w);
        this->transformed_data_.Reshape(batch_size, channels, instance_h, instance_w);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].data_.Reshape(batch_size, channels, instance_h, instance_w);
        }

        // SOFTMAX
        //label
        top[1]->Reshape(batch_size, 1, 1, 1);
        this->transformed_label_.Reshape(1, 1, 1, 1);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].label_.Reshape(batch_size, 1, 1, 1);
        }
    }
    // image dimensions, for each image, stores (img_height, img_width)

    LOG(INFO) << "output data size: " << top[0]->num() << ","
              << top[0]->channels() << "," << top[0]->height() << ","
              << top[0]->width();
    // label
    LOG(INFO) << "output label size: " << top[1]->num() << ","
              << top[1]->channels() << "," << top[1]->height() << ","
              << top[1]->width();

}

template <typename Dtype>
void ClassificationDataLayer<Dtype>::ShuffleImages() {
    caffe::rng_t* prefetch_rng =
            static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ClassificationDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());

    ImageDataParameter image_data_param = this->layer_param_.image_data_param();
    const int batch_size = image_data_param.batch_size();
    const int new_height = image_data_param.new_height();
    const int new_width  = image_data_param.new_width();
    const bool is_color  = image_data_param.is_color();
    string root_folder   = image_data_param.root_folder();


    int instance_h = this->layer_param_.image_data_param().instance_h();
    int instance_w = this->layer_param_.image_data_param().instance_w();

    batch->data_.Reshape(batch_size, 3, instance_h, instance_w);
    batch->label_.Reshape(batch_size, 1, 1, 1);


    Dtype* top_data     = batch->data_.mutable_cpu_data();

    const int lines_size = lines_.size();

    //  static int batch_no;

    for (int item_id = 0; item_id < batch_size; ++item_id) {

        // get a blob
        timer.Start();
        CHECK_GT(lines_size, lines_id_);

        int img_row, img_col;
        cv::Mat cv_img = PSP_ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                              new_height, new_width, is_color, &img_row, &img_col);


        if (!cv_img.data) {
            DLOG(INFO) << "Fail to load img: " << root_folder + lines_[lines_id_].first;
        }

        int label = std::atoi(lines_[lines_id_].second.c_str()) - 1;

        cv::resize(cv_img,cv_img,cv::Size(instance_h,instance_w));


        read_time += timer.MicroSeconds();
        timer.Start();
        // Apply transformations (mirror, crop...) to the image
        int offset;
        offset = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(top_data + offset);

        this->data_transformer_->TransformImg(cv_img,
                                              &(this->transformed_data_));

        //         std::cout << "LABEL = " << label << "\n";

//         cv::imshow("live after",cv_img);
//         cv::waitKey(0);


        offset = batch->label_.offset(item_id);
        Dtype* label_data = batch->label_.mutable_cpu_data() + offset;

        label_data[0] = label;

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
    }
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ClassificationDataLayer);
REGISTER_LAYER_CLASS(ClassificationData);

}  // namespace caffe
