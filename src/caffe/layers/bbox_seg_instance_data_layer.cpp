#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/bbox_seg_instance_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"
#include<opencv2/opencv.hpp>
#include "caffe/util/rng.hpp"

using namespace std;

namespace caffe {

void image_seg_to_instance_annotated_datum(std::string& root_folder,
                                           vector<std::pair<std::string, std::string> >& lines_,
                                           int lines_id_,
                                           AnnotatedDatum& annotated_datum,
                                           int n_classes,int h,int w)
{

    //    std::cout << root_folder + lines_[lines_id_].first << "\n";

    //    // Read an image, and use it to initialize the top blob.
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,true);
    CHECK(cv_img.data) << "Fail to load img: " << root_folder + lines_[lines_id_].first;

    cv::Mat cv_seg = ReadImageToCVMat(root_folder + lines_[lines_id_].second,true);
    CHECK(cv_seg.data) << "Fail to load seg: " << root_folder + lines_[lines_id_].second;

    //    cv::Rect roi;
    //    //RANDOM CROP
    //    {
    //        int crop_width = w;
    //        int crop_height = h;

    //        int img_height = cv_img.rows;
    //        int img_width = cv_img.cols;

    //        int seg_height = cv_seg.rows;
    //        int seg_width = cv_seg.cols;

    //        int h_off = 0;
    //        int w_off = 0;

    //        // Check if we need to pad img to fit for crop_size
    //        // copymakeborder
    //        int pad_height = std::max(crop_height - img_height, 0);
    //        int pad_width  = std::max(crop_width - img_width, 0);
    //        if (pad_height > 0 || pad_width > 0) {
    //            cv::copyMakeBorder(cv_img, cv_img, 0, pad_height,
    //                               0, pad_width, cv::BORDER_CONSTANT,
    //                               cv::Scalar(0, 0, 0));
    //            cv::copyMakeBorder(cv_seg, cv_seg, 0, pad_height,
    //                               0, pad_width, cv::BORDER_CONSTANT,
    //                               cv::Scalar(0,0,0));
    //            // update height/width
    //            img_height   = cv_img.rows;
    //            img_width    = cv_img.cols;

    //            seg_height   = cv_seg.rows;
    //            seg_width    = cv_seg.cols;
    //        }

    //        static cv::RNG rng;

    //        // We only do random crop when we do training.
    //        h_off = rng(img_height - crop_height + 1);
    //        w_off = rng(img_width - crop_width + 1);

    //        roi.x = w_off;
    //        roi.y = h_off;
    //        roi.width = crop_width;
    //        roi.height = crop_height;
    //    }

    //    //    roi.x = (cv_img.cols-w)/2;
    //    //    roi.y = (cv_img.rows-h)/2;
    //    //    roi.width = w;
    //    //    roi.height = h;

    //    cv_img = cv_img(roi);
    //    cv_seg = cv_seg(roi);


    cv::resize(cv_img,cv_img,cv::Size(w,h));
    cv::resize(cv_seg,cv_seg,cv::Size(w,h),0,0,CV_INTER_NN);

    cv::Mat rect_img;
    cv_img.copyTo(rect_img);

    annotated_datum.clear_annotation_group();
    annotated_datum.clear_datum();
    annotated_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);

    //    for(int i=1;i<n_classes;i++)
    //    {
    //        cv::Mat object_mask = (cv_seg == i);

    //        std::vector<std::vector<cv::Point2i> > contours;
    //        cv::findContours(object_mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

    //        cv::Rect bounding_rect;

    //        if(contours.size())
    //        {
    //            std::vector<cv::Point2i> points;

    //            for(int contour = 0;contour < contours.size();contour++)
    //            {
    //                for(int n_points=0;n_points<contours[contour].size();n_points++)
    //                    points.push_back(contours[contour][n_points]);
    //            }

    //            bounding_rect =  cv::boundingRect(points);
    //            AnnotationGroup* anno_group =  annotated_datum.add_annotation_group();
    //            anno_group->set_group_label(i);

    //            Annotation* anno = anno_group->add_annotation();
    //            anno->set_instance_id(1);
    //            anno->mutable_bbox()->set_xmin(((float)bounding_rect.x) / cv_img.cols);
    //            anno->mutable_bbox()->set_ymin(((float)bounding_rect.y) / cv_img.rows);
    //            anno->mutable_bbox()->set_xmax(((float)bounding_rect.x + bounding_rect.width + 1.0f) / cv_img.cols);
    //            anno->mutable_bbox()->set_ymax(((float)bounding_rect.y + bounding_rect.height + 1.0f) / cv_img.rows);
    //            anno->mutable_bbox()->set_label(i);
    //            anno->mutable_bbox()->set_difficult(true);

    //            bounding_rect.x = 512*anno->bbox().xmin();
    //            bounding_rect.y = 512*anno->bbox().ymin();
    //            bounding_rect.width = 512*(anno->bbox().xmax() - anno->bbox().xmin());
    //            bounding_rect.height = 512*(anno->bbox().ymax() - anno->bbox().ymin());

    ////            std::cout << "ANNO BOX = " << anno->bbox().xmin() << "  " << anno->bbox().ymin() <<
    ////                         "  " << anno->bbox().xmax() << "  " << anno->bbox().ymax() << "\n";
    //            cv::rectangle(rect_img,bounding_rect,cv::Scalar(0,255,0),2);


    //        }

    //    }


    //----MIT ADEK DATASET

    cv::Mat mask;
    cv::Mat mask_instances;

    cv::Mat splitted_anno[3];
    cv::split(cv_seg,splitted_anno);

    mask = splitted_anno[2];
    mask_instances = splitted_anno[1];

    double max_val;
    cv::minMaxLoc(mask_instances,NULL,&max_val);

    int n_instances = max_val;
    n_classes = n_instances+1; // 1 base indexing

    cv::Mat cv_segs;
    cv_seg.copyTo(cv_segs);

    //    std::cout << cv_segs.rows << "  " << cv_segs.cols << "  " << cv_segs.channels() << "\n";
    cv_seg = mask_instances;
    //-----

    AnnotationGroup* anno_group =  annotated_datum.add_annotation_group();
    anno_group->set_group_label(1);

    int object_label = 1;

    int instance_id = 1;
    for(int i=1;i<n_classes;i++)
    {
        cv::Mat object_mask = (cv_seg == i);

        std::vector<std::vector<cv::Point2i> > contours;
        cv::findContours(object_mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

        cv::Rect bounding_rect;

        if(contours.size())
        {

            cv::Mat label_mask = (cv_seg == i)/255;

            cv::multiply(label_mask,mask,label_mask);

            cv::minMaxLoc(label_mask,NULL,&max_val);

            object_label = max_val;


            std::vector<cv::Point2i> points;

            for(int contour = 0;contour < contours.size();contour++)
            {
                for(int n_points=0;n_points<contours[contour].size();n_points++)
                    points.push_back(contours[contour][n_points]);
            }

            bounding_rect =  cv::boundingRect(points);

            Annotation* anno = anno_group->add_annotation();
            anno->set_instance_id(instance_id++);
            anno->mutable_bbox()->set_xmin(((float)bounding_rect.x) / cv_img.cols);
            anno->mutable_bbox()->set_ymin(((float)bounding_rect.y) / cv_img.rows);
            anno->mutable_bbox()->set_xmax(((float)bounding_rect.x + bounding_rect.width + 1.0f) / cv_img.cols);
            anno->mutable_bbox()->set_ymax(((float)bounding_rect.y + bounding_rect.height + 1.0f) / cv_img.rows);
            anno->mutable_bbox()->set_label(object_label);
            anno->mutable_bbox()->set_difficult(true);

            bounding_rect.x = w*anno->bbox().xmin();
            bounding_rect.y = h*anno->bbox().ymin();
            bounding_rect.width = w*(anno->bbox().xmax() - anno->bbox().xmin());
            bounding_rect.height = h*(anno->bbox().ymax() - anno->bbox().ymin());

            //            std::cout << "ANNO BOX = " << anno->bbox().xmin() << "  " << anno->bbox().ymin() <<
            //                         "  " << anno->bbox().xmax() << "  " << anno->bbox().ymax() << "\n";
            cv::rectangle(rect_img,bounding_rect,cv::Scalar(0,255,0),2);


        }

    }

    //        cv::imshow("cv_img",rect_img);
    //    //    cv::imshow("cv_seg",cv_seg);
    //        cv::waitKey(1);
    annotated_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);
    EncodeCVMatToDatum(cv_img,"jpg", annotated_datum.mutable_datum());
    EncodeCVMatToDatum(cv_segs,"png", annotated_datum.mutable_datum_seg());
}


template <typename Dtype>
BboxSegInstanceDataLayer<Dtype>::BboxSegInstanceDataLayer(const LayerParameter& param)
    : BoxSegInstancePrefetchingDataLayer<Dtype>(param)
{
}

template <typename Dtype>
BboxSegInstanceDataLayer<Dtype>::~BboxSegInstanceDataLayer() {
    this->StopInternalThread();
}

template <typename Dtype>
void BboxSegInstanceDataLayer<Dtype>::DataLayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    //    const int batch_size = this->layer_param_.data_param().batch_size();

    const int batch_size = this->layer_param_.image_data_param().batch_size();
    const AnnotatedDataParameter& anno_data_param =
            this->layer_param_.annotated_data_param();
    for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
        batch_samplers_.push_back(anno_data_param.batch_sampler(i));
    }
    label_map_file_ = anno_data_param.label_map_file();
    // Make sure dimension is consistent within batch.
    const TransformationParameter& transform_param =
            this->layer_param_.transform_param();
    if (transform_param.has_resize_param()) {
        if (transform_param.resize_param().resize_mode() ==
                ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
            CHECK_EQ(batch_size, 1)
                    << "Only support batch size of 1 for FIT_SMALL_SIZE.";
        }
    }

    ///--------------- INITIALIZING IMAGE AND SEG DATA READER

    string root_folder = this->layer_param_.image_data_param().root_folder();
    const int label_type = this->layer_param_.image_data_param().label_type();
    const int n_classes = this->layer_param_.image_data_param().n_classes();

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

    ///--------------------ENDS
    // Read a data point, and use it to initialize the top blob.
    int h = this->transform_param_.resize_param().height();
    int w = this->transform_param_.resize_param().width();

    AnnotatedDatum anno_datum;

    image_seg_to_instance_annotated_datum(root_folder,lines_,lines_id_,anno_datum,n_classes,h,w);

    // Use data_transformer to infer the expected blob shape from anno_datum.
    vector<int> top_shape =
            this->data_transformer_->InferBlobShape(anno_datum.datum());
    this->transformed_data_.Reshape(top_shape);
    this->transformed_seg_.Reshape(top_shape);

    // Reshape top[0] and prefetch_data according to the batch_size.
    top_shape[0] = batch_size;
    top[0]->Reshape(top_shape);

    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].data_.Reshape(top_shape);
    }

    top_shape[1] = 1;
    top[2]->Reshape(top_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i){
        this->prefetch_[i].seg_.Reshape(top_shape);
    }

    LOG(INFO) << "output data size: " << top[0]->num() << ","
              << top[0]->channels() << "," << top[0]->height() << ","
              << top[0]->width();


    // label
    if (this->output_labels_) {
        has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
        vector<int> label_shape(4, 1);
        if (has_anno_type_) {


            anno_type_ = anno_datum.type();
            if (anno_data_param.has_anno_type()) {
                // If anno_type is provided in AnnotatedDataParameter, replace
                // the type stored in each individual AnnotatedDatum.
                LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
                anno_type_ = anno_data_param.anno_type();
            }
            // Infer the label shape from anno_datum.AnnotationGroup().
            int num_bboxes = 0;
            if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
                // Since the number of bboxes can be different for each image,
                // we store the bbox information in a specific format. In specific:
                // All bboxes are stored in one spatial plane (num and channels are 1)
                // And each row contains one and only one box in the following format:
                // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
                // Note: Refer to caffe.proto for details about group_label and
                // instance_id.
                for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
                    num_bboxes += anno_datum.annotation_group(g).annotation_size();
                }
                label_shape[0] = 1;
                label_shape[1] = 1;
                // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
                // cpu_data and gpu_data for consistent prefetch thread. Thus we make
                // sure there is at least one bbox.
                label_shape[2] = std::max(num_bboxes, 1);
                label_shape[3] = 8;
            } else {
                LOG(FATAL) << "Unknown annotation type.";
            }
        } else {
            label_shape[0] = batch_size;
        }
        top[1]->Reshape(label_shape);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].label_.Reshape(label_shape);
        }

        vector<int> instance_seg_shape(4);

        int instance_c = 1;
        int instance_h = this->layer_param_.image_data_param().instance_h();
        int instance_w = this->layer_param_.image_data_param().instance_w();

        instance_seg_shape[0] = 1;
        instance_seg_shape[1] = instance_c;
        instance_seg_shape[2] = instance_h;
        instance_seg_shape[3] = instance_w;

        top[3]->Reshape(instance_seg_shape);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].label_.Reshape(instance_seg_shape);
        }
    }

    //    lines_id_ = 20000;
}



template <typename Dtype>
void BboxSegInstanceDataLayer<Dtype>::ShuffleImages() {
    caffe::rng_t* prefetch_rng =
            static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template<typename Dtype>
void BboxSegInstanceDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    //    std::cout << "LOADING BATCH\n";

    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());

    // Reshape according to the first anno_datum of each batch
    // on single input batches allows for inputs of varying dimension.
    //    const int batch_size = this->layer_param_.data_param().batch_size();
    const int batch_size = this->layer_param_.image_data_param().batch_size();

    const AnnotatedDataParameter& anno_data_param =
            this->layer_param_.annotated_data_param();
    const TransformationParameter& transform_param =
            this->layer_param_.transform_param();

    // IMAGE SEG STYLE
    string root_folder = this->layer_param_.image_data_param().root_folder();
    const int lines_size = lines_.size();
    const int n_classes = this->layer_param_.image_data_param().n_classes();

    int h = this->transform_param_.resize_param().height();
    int w = this->transform_param_.resize_param().width();

    AnnotatedDatum anno_datum;
    image_seg_to_instance_annotated_datum(root_folder,lines_,lines_id_,anno_datum,n_classes,h,w);


    // Use data_transformer to infer the expected blob shape from anno_datum.
    vector<int> top_shape =
            this->data_transformer_->InferBlobShape(anno_datum.datum());
    this->transformed_data_.Reshape(top_shape);
    // Reshape batch according to the batch_size.
    top_shape[0] = batch_size;
    batch->data_.Reshape(top_shape);


    vector<int> top_seg_shape(4);

    top_seg_shape[0] = 1;
    top_seg_shape[1] = 3;
    top_seg_shape[2] = top_shape[2];
    top_seg_shape[3] = top_shape[3];

    this->transformed_seg_.Reshape(top_seg_shape);

    top_seg_shape[0] = batch_size;
    top_seg_shape[1] = 1;
    batch->seg_.Reshape(top_seg_shape);

    Dtype* top_data = batch->data_.mutable_cpu_data();

    Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
    if (this->output_labels_ && !has_anno_type_) {
        top_label = batch->label_.mutable_cpu_data();
    }


    std::vector<cv::Mat> cv_transformed_img;
    std::vector<cv::Mat> cv_transformed_seg;

    // Store transformed annotation.
    map<int, vector<AnnotationGroup> > all_anno;
    int num_bboxes = 0;

    //    std::cout << "FILLING BATCH\n";
    //    std::cout << root_folder + lines_[lines_id_].first << "\n";

    for (int item_id = 0; item_id < batch_size; ++item_id) {
        timer.Start();
        // get a anno_datum

        AnnotatedDatum anno_datum;

        // IMAGE SEG STYLE
        image_seg_to_instance_annotated_datum(root_folder,lines_,lines_id_,anno_datum,n_classes,h,w);
        //        AnnotatedDatum& anno_datum = *(reader_.full().pop("Waiting for data"));

        read_time += timer.MicroSeconds();
        timer.Start();
        AnnotatedDatum distort_datum;
        AnnotatedDatum* expand_datum = NULL;

        if (transform_param.has_distort_param()) {
            distort_datum.CopyFrom(anno_datum);
            this->data_transformer_->DistortImage(anno_datum.datum(),
                                                  distort_datum.mutable_datum());
            if (transform_param.has_expand_param()) {
                expand_datum = new AnnotatedDatum();
                this->data_transformer_->ExpandImageSeg(distort_datum, expand_datum);
            } else {
                expand_datum = &distort_datum;
            }
        } else {
            if (transform_param.has_expand_param()) {
                expand_datum = new AnnotatedDatum();
                this->data_transformer_->ExpandImageSeg(anno_datum, expand_datum);
            } else {
                expand_datum = &anno_datum;
            }
        }

        AnnotatedDatum* sampled_datum = NULL;
        bool has_sampled = false;
        if (batch_samplers_.size() > 0) {
            // Generate sampled bboxes from expand_datum.
            vector<NormalizedBBox> sampled_bboxes;
            GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
            if (sampled_bboxes.size() > 0) {
                // Randomly pick a sampled bbox and crop the expand_datum.
                int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
                sampled_datum = new AnnotatedDatum();
                this->data_transformer_->CropImageSeg(*expand_datum,
                                                      sampled_bboxes[rand_idx],
                                                      sampled_datum);
                has_sampled = true;
            } else {
                sampled_datum = expand_datum;
            }
        } else {
            sampled_datum = expand_datum;
        }

        CHECK(sampled_datum != NULL);
        timer.Start();
        vector<int> shape =
                this->data_transformer_->InferBlobShape(sampled_datum->datum());
        if (transform_param.has_resize_param()) {
            if (transform_param.resize_param().resize_mode() ==
                    ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {

            } else {
                CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
                                 shape.begin() + 1));
            }
        } else {
            CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
                             shape.begin() + 1));
        }
        // Apply data transformations (mirror, scale, crop...)
        int offset = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(top_data + offset);

        vector<AnnotationGroup> transformed_anno_vec;
        if (this->output_labels_) {
            if (has_anno_type_) {
                // Make sure all data have same annotation type.
                CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
                if (anno_data_param.has_anno_type()) {
                    sampled_datum->set_type(anno_type_);
                } else {
                    CHECK_EQ(anno_type_, sampled_datum->type()) <<
                                                                   "Different AnnotationType.";
                }
                // Transform datum and annotation_group at the same time
                transformed_anno_vec.clear();
                this->data_transformer_->TransformSeg(*sampled_datum,
                                                      &(this->transformed_data_),
                                                      &(this->transformed_seg_),
                                                      &transformed_anno_vec);


                if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
                    // Count the number of bboxes.
                    for (int g = 0; g < transformed_anno_vec.size(); ++g) {
                        num_bboxes += transformed_anno_vec[g].annotation_size();
                    }

                } else {
                    LOG(FATAL) << "Unknown annotation type.";
                }
                all_anno[item_id] = transformed_anno_vec;
            }
        }
        // clear memory
        if (has_sampled) {
            delete sampled_datum;
        }
        if (transform_param.has_expand_param()) {
            delete expand_datum;
        }
        trans_time += timer.MicroSeconds();

        int im_width = this->transformed_data_.width();
        int im_height =  this->transformed_data_.height();

        cv::Mat rect_seg(im_height,im_width,CV_8UC3);

        offset = batch->seg_.offset(item_id);
        Dtype* top_seg = batch->seg_.mutable_cpu_data() + offset;

        int seg_index = 0;
        for(int py=0;py<im_height;py++)
            for(int px=0;px<im_width;px++)
            {
                unsigned char* seg_pixel = rect_seg.data + py* rect_seg.step[0] + px * rect_seg.step[1];
                seg_pixel[0] = (this->transformed_seg_.cpu_data() + py * im_width + px)[0];
                seg_pixel[1] = (this->transformed_seg_.cpu_data() + im_height * im_width + py * im_width + px)[0];
                seg_pixel[2] = (this->transformed_seg_.cpu_data() + 2*im_height * im_width + py * im_width + px)[0];

                top_seg[seg_index++] = seg_pixel[2]; // RED CHANNEL FOR CATEGORY
            }

        cv_transformed_seg.push_back(rect_seg);

        cv::Mat rect_img(im_height,im_width,CV_8UC3);

        for(int py=0;py<im_height;py++)
            for(int px=0;px<im_width;px++)
            {
                unsigned char* im_pixel = rect_img.data + py* rect_img.step[0] + px * rect_img.step[1];
                im_pixel[0] = (this->transformed_data_.cpu_data() + py * im_width + px)[0];
                im_pixel[1] = (this->transformed_data_.cpu_data() + im_height * im_width + py * im_width + px)[0];
                im_pixel[2] = (this->transformed_data_.cpu_data() + 2*im_height * im_width + py * im_width + px)[0];
            }

        cv_transformed_img.push_back(rect_img);

        //        cv::imshow("cv_img",rect_img);
        //        cv::imshow("cv_seg",5*rect_seg);
        //        cv::waitKey(0);

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
        //        reader_.free().push(const_cast<AnnotatedDatum*>(&anno_datum));
    }


    //    std::cout << "FILLING ANNO\n";
    //    std::cout << root_folder + lines_[lines_id_].first << "\n";

    // Store "rich" annotation if needed.
    if (this->output_labels_ && has_anno_type_) {
        vector<int> label_shape(4);
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
            label_shape[0] = 1;
            label_shape[1] = 1;
            label_shape[3] = 8;
            if (num_bboxes == 0) {
                // Store all -1 in the label.
                label_shape[2] = 1;
                batch->label_.Reshape(label_shape);
                caffe_set<Dtype>(8, -1, batch->label_.mutable_cpu_data());
            } else {
                // Reshape the label and store the annotation.
                label_shape[2] = num_bboxes;
                batch->label_.Reshape(label_shape);
                top_label = batch->label_.mutable_cpu_data();
                int idx = 0;
                for (int item_id = 0; item_id < batch_size; ++item_id) {
                    const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
                    for (int g = 0; g < anno_vec.size(); ++g) {
                        const AnnotationGroup& anno_group = anno_vec[g];
                        for (int a = 0; a < anno_group.annotation_size(); ++a) {
                            const Annotation& anno = anno_group.annotation(a);
                            const NormalizedBBox& bbox = anno.bbox();
                            top_label[idx++] = item_id;
                            top_label[idx++] = bbox.label();//  anno_group.group_label();
                            top_label[idx++] = anno.instance_id();
                            top_label[idx++] = bbox.xmin();
                            top_label[idx++] = bbox.ymin();
                            top_label[idx++] = bbox.xmax();
                            top_label[idx++] = bbox.ymax();
                            top_label[idx++] = bbox.difficult();
                        }
                    }
                }

                //                std::cout << "FILLING RECTANGLES\n";
            }
        } else {
            LOG(FATAL) << "Unknown annotation type.";
        }
    }


    //    std::cout << "FILLING INSTANCE SEG\n";
    //    std::cout << root_folder + lines_[lines_id_].first << "\n";

    int n_rois = batch->label_.height();

    vector<int> instance_seg_shape(4);

    int instance_c = 1;
    int instance_h = this->layer_param_.image_data_param().instance_h();
    int instance_w = this->layer_param_.image_data_param().instance_w();

    instance_seg_shape[0] = n_rois;
    instance_seg_shape[1] = instance_c;
    instance_seg_shape[2] = instance_h;
    instance_seg_shape[3] = instance_w;


    batch->instance_seg_.Reshape(instance_seg_shape);

    int im_height = batch->data_.height();
    int im_width = batch->data_.width();


    std::vector<cv::Mat> instance_segs;
    std::vector<cv::Mat> segs;


    //    std::cout << "FILLING SPLITTING\n";
    //    std::cout << root_folder + lines_[lines_id_].first << "\n";

    for(int i=0;i<batch_size;i++)
    {
        cv::Mat splitted_seg[3];
        cv::split(cv_transformed_seg[i],splitted_seg);
        instance_segs.push_back(splitted_seg[1]);
        segs.push_back(splitted_seg[2]);
    }

    double max;
    cv::minMaxLoc(segs[0],NULL,&max);

    //    std::cout << "MAX VAL = " << (int)max << "\n";

    //    std::cout << "FILLING INSTANCES MASK TO BATCH\n";
    //    std::cout << root_folder + lines_[lines_id_].first << "   "  << n_rois << "\n";

    //    std::cout << "TOP LABEL SIZE = " << batch->label_.shape_string() << "\n";

    top_label = batch->label_.mutable_cpu_data();

    int idx = 0;
    for (int i = 0; i< n_rois; i++)
    {
        int item_id = top_label[idx++];
        int label = top_label[idx++];
        int instance_id = top_label[idx++];
        int xmin = im_width * top_label[idx++];
        int ymin = im_height * top_label[idx++];
        int xmax = im_width * top_label[idx++];
        int ymax = im_height * top_label[idx++];
        idx++;

        cv::Mat instance_seg_ = cv::Mat::zeros(instance_h,instance_w,CV_8UC1);

        if(item_id > -1)
        {

            int w= xmax - xmin;
            int h= ymax - ymin;

            cv::Rect roi(xmin,ymin,w,h);
            //            std::cout << "ROI = " << roi <<  "  " << item_id << "\n";

            cv::Mat& instance_segs_ = instance_segs[item_id];

            instance_segs_(roi).copyTo(instance_seg_);

            instance_seg_ = (instance_seg_ == instance_id) / 255;
            instance_seg_ *= label;

            cv::resize(instance_seg_,instance_seg_,cv::Size(instance_w,instance_h),0,0,CV_INTER_NN);
        }

        //        // FILL LABEL FOR CLASS SPECIFIC TRAINING


        //        cv::Mat& img = cv_transformed_img[item_id];
        //        cv::Mat instance_img_;
        //        img(roi).copyTo(instance_img_);
        //        cv::resize(instance_img_,instance_img_,cv::Size(instance_w,instance_h),0,0,CV_INTER_LINEAR);

        //        cv::resize(instance_seg_,instance_seg_,cv::Size(256,256),0,0,CV_INTER_NN);
        //        cv::resize(instance_img_,instance_img_,cv::Size(256,256),0,0,CV_INTER_LINEAR);

        int offset = batch->instance_seg_.offset(i);
        Dtype* instance_seg_data = batch->instance_seg_.mutable_cpu_data() + offset;

        for(int j=0;j<instance_h*instance_w;j++)
            instance_seg_data[j] = static_cast<Dtype>(instance_seg_.data[j]);

        //        cv::rectangle(cv_transformed_img[item_id],roi,cv::Scalar(0,255,0));
        //        cv::imshow("img",cv_transformed_img[item_id]);
        //        cv::imshow("seg",20*segs[item_id]);
        //        cv::imshow("instance_img_",instance_img_);
        //        cv::imshow("instance_seg_",20*instance_seg_);
        //        cv::waitKey(1);
    }

    //    std::cout << "FILLED -----------------__****************\n";
    //    std::cout << root_folder + lines_[lines_id_].first << "\n";

    //    std::cout << "BATCH BLOB DATA = ";
    //    for(int i=3;i<7;i++)
    //    std::cout << batch->label_.cpu_data()[i] << " ";
    //    std::cout << "\n";

    timer.Stop();
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(BboxSegInstanceDataLayer);
REGISTER_LAYER_CLASS(BboxSegInstanceData);

}  // namespace caffe
