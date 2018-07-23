#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"

#include "boost/math/common_factor.hpp"

namespace caffe {



template <typename Dtype>
__global__ void im2col_gpu_kernel_constellation(const int n, const Dtype* data_im,
                                                const int height, const int width, const int kernel_h, const int kernel_w,
                                                const int pad_h, const int pad_w,
                                                const int stride_h, const int stride_w,
                                                const int* dilation_h, const int* dilation_w,
                                                const int height_col, const int width_col,
                                                Dtype* data_col) {


    CUDA_KERNEL_LOOP(index, n) {
        const int h_index = index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = index % width_col;
        const int c_im = h_index / height_col;
        const int c_col = c_im * kernel_h * kernel_w;
        const int h_offset = h_col * stride_h - pad_h;
        const int w_offset = w_col * stride_w - pad_w;
        Dtype* data_col_ptr = data_col;
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
        const Dtype* data_im_ptr = data_im;
        data_im_ptr += c_im * height * width;

        for (int i = 0; i < kernel_h; ++i) {

            const int dilation_h_ = dilation_h[i];
            const int h_im = h_offset + dilation_h_;

            for (int j = 0; j < kernel_w; ++j) {

                const int dilation_w_ = dilation_w[j];
                const int w_im = w_offset + dilation_w_;

                *data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                            data_im_ptr[ h_im * width + w_im] : 0;

                data_col_ptr += height_col * width_col;
            }
        }
    }
}




template <typename Dtype>
void im2col_gpu_constellation(const Dtype* data_im, const int channels,
                              const int height, const int width, const int kernel_h, const int kernel_w,
                              const int pad_h, const int pad_w,
                              const int stride_h, const int stride_w,
                              const int kernel_extent_h, const int kernel_extent_w,
                              const int* dilation_h, const int* dilation_w,
                              Dtype* data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.

    int height_col = (height + 2 * pad_h - kernel_extent_h ) / stride_h + 1;

    int width_col = (width + 2 * pad_w - kernel_extent_w) / stride_w + 1;

    int num_kernels = channels * height_col * width_col;

    // NOLINT_NEXT_LINE(whitespace/operators)
    im2col_gpu_kernel_constellation<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
            CAFFE_CUDA_NUM_THREADS>>>(
                                        num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
                                        pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
                                        width_col, data_col);
    CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_gpu_constellation<float>(const float* data_im, const int channels,
const int height, const int width, const int kernel_h, const int kernel_w,
const int pad_h, const int pad_w, const int stride_h, const int stride_w,
const int kernel_extent_h, const int kernel_extent_w,
const int* dilation_h, const int* dilation_w, float* data_col);

template void im2col_gpu_constellation<double>(const double* data_im, const int channels,
const int height, const int width, const int kernel_h, const int kernel_w,
const int pad_h, const int pad_w, const int stride_h, const int stride_w,
const int kernel_extent_h, const int kernel_extent_w,
const int* dilation_h, const int* dilation_w, double* data_col);



template <typename Dtype>
__global__ void im2col_gpu_constellation_selected_kernel(const int n, const int n_elements,
                                                         const int n_processed,
                                                         const Dtype* data_im,
                                                         const int height, const int width,
                                                         const int kernel_h, const int kernel_w,
                                                         const int pad_h, const int pad_w,
                                                         const int stride_h, const int stride_w,
                                                         const int* dilation_h, const int* dilation_w,
                                                         const int height_col, const int width_col,
                                                         Dtype* data_col) {
    CUDA_KERNEL_LOOP(index, n) {

        int ch_col_buffer = index / n_elements;
        int w_col_buffer = index % n_elements;

        Dtype* data_col_ptr = data_col;
        data_col_ptr += ch_col_buffer * kernel_h * kernel_w * n_elements + w_col_buffer;


        int temp_index = n_processed + w_col_buffer + ch_col_buffer * height_col * width_col;

        const int h_index = temp_index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = temp_index % width_col;
        const int c_im = h_index / height_col;
        const int h_offset = h_col * stride_h - pad_h;
        const int w_offset = w_col * stride_w - pad_w;


        const Dtype* data_im_ptr = data_im;
        data_im_ptr += c_im * height * width ;

        for (int i = 0; i < kernel_h; ++i) {

            const int dilation_h_ = dilation_h[i];
            const int h_im = h_offset + dilation_h_;

            for (int j = 0; j < kernel_w; ++j) {

                const int dilation_w_ = dilation_w[j];
                const int w_im = w_offset + dilation_w_;

                *data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                            data_im_ptr[ h_im * width + w_im] : 0;

                data_col_ptr += n_elements;
            }
        }
    }
}

template <typename Dtype>
void im2col_gpu_constellation_selected(const Dtype* data_im, const int channels,
                                       const int n_elements, const int n_processed,
                                       const int height, const int width,
                                       const int kernel_h, const int kernel_w,
                                       const int pad_h, const int pad_w,
                                       const int stride_h, const int stride_w,
                                       const int kernel_extent_h, const int kernel_extent_w,
                                       const int* dilation_h, const int* dilation_w,
                                       Dtype* data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h - kernel_extent_h ) / stride_h + 1;

    int width_col = (width + 2 * pad_w - kernel_extent_w) / stride_w + 1;

    int num_kernels = channels * n_elements;

    //    return;


    // NOLINT_NEXT_LINE(whitespace/operators)
    im2col_gpu_constellation_selected_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
            CAFFE_CUDA_NUM_THREADS>>>(
                                        num_kernels, n_elements, n_processed,
                                        data_im, height, width,
                                        kernel_h, kernel_w, pad_h,
                                        pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
                                        width_col, data_col);
    CUDA_POST_KERNEL_CHECK;
}


// EXPLICIT
template void im2col_gpu_constellation_selected<float>(const float* data_im,
const int channels,const int n_elements, const int n_processed,
const int height, const int width,
const int kernel_h, const int kernel_w,
const int pad_h, const int pad_w,
const int stride_h, const int stride_w,
const int kernel_extent_h, const int kernel_extent_w,
const int* dilation_h, const int* dilation_w,
float* data_col);

template void im2col_gpu_constellation_selected<double>(const double* data_im,
const int channels,const int n_elements, const int n_processed,
const int height, const int width,
const int kernel_h, const int kernel_w,
const int pad_h, const int pad_w,
const int stride_h, const int stride_w,
const int kernel_extent_h, const int kernel_extent_w,
const int* dilation_h, const int* dilation_w,
double* data_col);



///////COL2IM CONSTELLATION




template <typename Dtype>
__global__ void col2im_gpu_constellation_kernel(const int n, const Dtype* data_col,
                                                const int height, const int width, const int channels,
                                                const int kernel_h, const int kernel_w,
                                                const int pad_h, const int pad_w,
                                                const int stride_h, const int stride_w,
                                                const int kernel_extent_h, const int kernel_extent_w,
                                                const int* dilation_shift_h,const int* dilation_shift_w,
                                                const int* dilation_shift_index_h,const int* dilation_shift_index_w,
                                                const int* dilation_inc_h,const int* dilation_inc_w,
                                                const int* dilation_start_index_h, const int* dilation_start_index_w,
                                                const int height_col, const int width_col,
                                                Dtype* data_im,  const int accumulate) {
    CUDA_KERNEL_LOOP(index, n) {
        Dtype val = 0;
        const int w_im = index % width + pad_w;
        const int h_im = (index / width) % height + pad_h;
        const int c_im = index / (width * height);


        // compute the start and end of the output
        int w_col_start =
                (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
        const int w_col_end = min(w_im / stride_w + 1, width_col);

        int h_col_start =
                (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
        const int h_col_end = min(h_im / stride_h + 1, height_col);
        // TODO: use LCM of stride and dilation to avoid unnecessary loops

        //                for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
        //                    for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {

        const int h_k_start = (h_im - h_col_start * stride_h);
        const int w_k_start = (w_im - w_col_start * stride_w);

        const int start_index_h = dilation_start_index_h[h_k_start];
        const int start_index_w = dilation_start_index_w[w_k_start];

        const int h_k_desired = dilation_inc_h[start_index_h];
        const int w_k_desired = dilation_inc_w[start_index_w];

        const int start_shift_h = abs(h_k_desired - h_k_start) / stride_h;
        const int start_shift_w = abs(w_k_desired - w_k_start) / stride_w;

        h_col_start += start_shift_h;
        w_col_start += start_shift_w;

        int shift_index_h = start_index_h;

        for (int h_col = h_col_start; h_col < h_col_end;
             h_col += dilation_shift_h[shift_index_h],  shift_index_h = dilation_shift_index_h[shift_index_h] )
        {
            int shift_index_w = start_index_w;

            for (int w_col = w_col_start; w_col < w_col_end;
                 w_col += dilation_shift_w[shift_index_w],  shift_index_w = dilation_shift_index_w[shift_index_w] )
            {
                const int h_k = shift_index_h;
                const int w_k = shift_index_w;

                int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                      height_col + h_col) * width_col + w_col;
                val += data_col[data_col_index];
            }
        }
        data_im[index] = val + data_im[index] * accumulate;
    }
}

template <typename Dtype>
void col2im_gpu_constellation(const Dtype* data_col, const int channels,
                              const int height, const int width, const int kernel_h, const int kernel_w,
                              const int pad_h, const int pad_w,
                              const int stride_h, const int stride_w,
                              const int kernel_extent_h, const int kernel_extent_w,
                              const int* dilation_shift_h,const int* dilation_shift_w,
                              const int* dilation_shift_index_h,const int* dilation_shift_index_w,
                              const int* dilation_inc_h,const int* dilation_inc_w,
                              const int* dilation_start_index_h, const int* dilation_start_index_w,
                              Dtype* data_im, const int accumulate) {

    int height_col = (height + 2 * pad_h - kernel_extent_h ) / stride_h + 1;

    int width_col = (width + 2 * pad_w - kernel_extent_w) / stride_w + 1;

    int num_kernels = channels * height * width;
    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    // NOLINT_NEXT_LINE(whitespace/operators)

    //    if(accumulate)
    //    std::cout << "col 22 im accumulate = " << accumulate << "\n";

    // in the input space, diff will be back propagated by dilation_shift_index_h or w

    col2im_gpu_constellation_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
            CAFFE_CUDA_NUM_THREADS>>>(
                                        num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
                                        pad_h, pad_w,
                                        stride_h, stride_w,
                                        kernel_extent_h, kernel_extent_w,
                                        dilation_shift_h, dilation_shift_w,
                                        dilation_shift_index_h, dilation_shift_index_w,
                                        dilation_inc_h, dilation_inc_w,
                                        dilation_start_index_h, dilation_start_index_w,
                                        height_col, width_col, data_im, accumulate);
    CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_gpu_constellation<float>(const float* data_col, const int channels,
const int height, const int width, const int kernel_h, const int kernel_w,
const int pad_h, const int pad_w,
const int stride_h, const int stride_w,
const int kernel_extent_h, const int kernel_extent_w,
const int* dilation_shift_h,const int* dilation_shift_w,
const int* dilation_shift_index_h,const int* dilation_shift_index_w,
const int* dilation_inc_h,const int* dilation_inc_w,
const int* dilation_start_index_h, const int* dilation_start_index_w,
float* data_im, const int accumulate);

template void col2im_gpu_constellation<double>(const double* data_col, const int channels,
const int height, const int width, const int kernel_h, const int kernel_w,
const int pad_h, const int pad_w,
const int stride_h, const int stride_w,
const int kernel_extent_h, const int kernel_extent_w,
const int* dilation_shift_h,const int* dilation_shift_w,
const int* dilation_shift_index_h,const int* dilation_shift_index_w,
const int* dilation_inc_h,const int* dilation_inc_w,
const int* dilation_start_index_h, const int* dilation_start_index_w,
double* data_im, const int accumulate);



///------------************* CONSTELLATION CODE ENDS













template <typename Dtype>
__global__ void im2col_gpu_selected_kernel(const int n, const int n_elements,
                                           const int n_processed,
                                           const Dtype* data_im,
                                           const int height, const int width,
                                           const int kernel_h, const int kernel_w,
                                           const int pad_h, const int pad_w,
                                           const int stride_h, const int stride_w,
                                           const int dilation_h, const int dilation_w,
                                           const int height_col, const int width_col,
                                           Dtype* data_col) {
    CUDA_KERNEL_LOOP(index, n) {

        int ch_col_buffer = index / n_elements;
        int w_col_buffer = index % n_elements;

        Dtype* data_col_ptr = data_col;
        data_col_ptr += ch_col_buffer * kernel_h * kernel_w * n_elements + w_col_buffer;


        int temp_index = n_processed + w_col_buffer + ch_col_buffer * height_col * width_col;

        const int h_index = temp_index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = temp_index % width_col;
        const int c_im = h_index / height_col;
        const int h_offset = h_col * stride_h - pad_h;
        const int w_offset = w_col * stride_w - pad_w;


        const Dtype* data_im_ptr = data_im;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;

        for (int i = 0; i < kernel_h; ++i)
        {
            for (int j = 0; j < kernel_w; ++j)
            {
                int h_im = h_offset + i * dilation_h;
                int w_im = w_offset + j * dilation_w;
                *data_col_ptr =
                        (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                            data_im_ptr[i * dilation_h * width + j * dilation_w] :0;
                data_col_ptr += n_elements;
            }
        }
    }
}

template <typename Dtype>
void im2col_gpu_selected(const Dtype* data_im, const int channels,
                         const int n_elements, const int n_processed,
                         const int height, const int width,
                         const int kernel_h, const int kernel_w,
                         const int pad_h, const int pad_w,
                         const int stride_h, const int stride_w,
                         const int dilation_h, const int dilation_w,
                         Dtype* data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h -
                      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w -
                     (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * n_elements;

    //    return;

    // NOLINT_NEXT_LINE(whitespace/operators)
    im2col_gpu_selected_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
            CAFFE_CUDA_NUM_THREADS>>>(
                                        num_kernels, n_elements, n_processed,
                                        data_im, height, width,
                                        kernel_h, kernel_w, pad_h,
                                        pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
                                        width_col, data_col);
    CUDA_POST_KERNEL_CHECK;
}


// EXPLICIT
template void im2col_gpu_selected<float>(const float* data_im,
const int channels,const int n_elements, const int n_processed,
const int height, const int width,
const int kernel_h, const int kernel_w,
const int pad_h, const int pad_w,
const int stride_h, const int stride_w,
const int dilation_h, const int dilation_w,
float* data_col);

template void im2col_gpu_selected<double>(const double* data_im,
const int channels,const int n_elements, const int n_processed,
const int height, const int width,
const int kernel_h, const int kernel_w,
const int pad_h, const int pad_w,
const int stride_h, const int stride_w,
const int dilation_h, const int dilation_w,
double* data_col);


template <typename Dtype>
__global__ void col2im_gpu_selected_kernel(const int n, const Dtype* data_col,
                                           const int height, const int width,
                                           const int n_elements,
                                           const int index_min, const int index_max,
                                           const int crop_min_x, const int crop_min_y,
                                           const int crop_width, const int crop_height,
                                           const int kernel_h, const int kernel_w,
                                           const int pad_h, const int pad_w,
                                           const int stride_h, const int stride_w,
                                           const int dilation_h, const int dilation_w,
                                           const int height_col, const int width_col,
                                           Dtype* data_im) {
    CUDA_KERNEL_LOOP(index, n) {
        Dtype val = 0;

        const int crop_x = index % crop_width;
        const int crop_y = ( index / crop_width) % crop_height;


        //        const int w_im = crop_x + crop_min_x + pad_w;
        //        const int h_im = crop_y + crop_min_y + pad_h;
        const int c_im = index / (crop_width * crop_height);

        const int temp_index = (c_im * height + crop_y + crop_min_y) * width + crop_x + crop_min_x;

        const int w_im = temp_index % width + pad_w;
        const int h_im = (temp_index / width) % height + pad_h;
        //        const int c_im = temp_index / (width * height);

        int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
        int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
        // compute the start and end of the output
        const int w_col_start =
                (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
        const int w_col_end = min(w_im / stride_w + 1, width_col);
        const int h_col_start =
                (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
        const int h_col_end = min(h_im / stride_h + 1, height_col);
        // TODO: use LCM of stride and dilation to avoid unnecessary loops
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {

                int col_index = h_col * width_col + w_col;

                if(col_index >= index_min && col_index <= index_max)
                {
                    col_index -= index_min;

                    int h_k = (h_im - h_col * stride_h);
                    int w_k = (w_im - w_col * stride_w);
                    if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                        h_k /= dilation_h;
                        w_k /= dilation_w;

                        //                        int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                        //                                              height_col + h_col) * width_col + w_col;

                        //                        int data_col_index = c_im * kernel_h * kernel_w * n_elements +
                        //                                             h_k * kernel_w * n_elements +
                        //                                             w_k * n_elements +
                        //                                             col_index;

                        int data_col_index = ((c_im * kernel_h + h_k)* kernel_w + w_k) * n_elements + col_index;

                        val += data_col[data_col_index];
                    }
                }
            }
        }
        data_im[temp_index] += val;
    }
}

template <typename Dtype>
void col2im_gpu_selected(const Dtype* data_col, const int channels, const int n_elements,
                         const int index_min, const int index_max,
                         const int crop_min_x, const int crop_min_y,
                         const int crop_width, const int crop_height,
                         const int height, const int width, const int kernel_h, const int kernel_w,
                         const int pad_h, const int pad_w, const int stride_h,
                         const int stride_w, const int dilation_h, const int dilation_w,
                         Dtype* data_im) {
    int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
            stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
            stride_w + 1;
    int num_kernels = channels * (crop_width * crop_height);

    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    // NOLINT_NEXT_LINE(whitespace/operators)
    col2im_gpu_selected_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
            CAFFE_CUDA_NUM_THREADS>>>(
                                        num_kernels, data_col, height, width, n_elements,
                                        index_min, index_max,
                                        crop_min_x, crop_min_y, crop_width, crop_height,
                                        kernel_h, kernel_w,
                                        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                        height_col, width_col, data_im);

    CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_gpu_selected<float>(const float* data_col, const int channels, const int n_elements,
const int index_min, const int index_max,
const int crop_min_x, const int crop_min_y,
const int crop_width, const int crop_height,
const int height, const int width, const int kernel_h, const int kernel_w,
const int pad_h, const int pad_w, const int stride_h,
const int stride_w, const int dilation_h, const int dilation_w,
float* data_im);
template void col2im_gpu_selected<double>(const double* data_col, const int channels, const int n_elements,
const int index_min, const int index_max,
const int crop_min_x, const int crop_min_y,
const int crop_width, const int crop_height,
const int height, const int width, const int kernel_h, const int kernel_w,
const int pad_h, const int pad_w, const int stride_h,
const int stride_w, const int dilation_h, const int dilation_w,
double* data_im);




template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
                                  const int height, const int width, const int kernel_h, const int kernel_w,
                                  const int pad_h, const int pad_w,
                                  const int stride_h, const int stride_w,
                                  const int dilation_h, const int dilation_w,
                                  const int height_col, const int width_col,
                                  Dtype* data_col) {
    CUDA_KERNEL_LOOP(index, n) {
        const int h_index = index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = index % width_col;
        const int c_im = h_index / height_col;
        const int c_col = c_im * kernel_h * kernel_w;
        const int h_offset = h_col * stride_h - pad_h;
        const int w_offset = w_col * stride_w - pad_w;
        Dtype* data_col_ptr = data_col;
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
        const Dtype* data_im_ptr = data_im;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h_im = h_offset + i * dilation_h;
                int w_im = w_offset + j * dilation_w;
                *data_col_ptr =
                        (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                            data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}




template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w,
                const int stride_h, const int stride_w,
                const int dilation_h, const int dilation_w,
                Dtype* data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h -
                      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w -
                     (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;
    // NOLINT_NEXT_LINE(whitespace/operators)
    im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
            CAFFE_CUDA_NUM_THREADS>>>(
                                        num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
                                        pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
                                        width_col, data_col);
    CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
const int height, const int width, const int kernel_h, const int kernel_w,
const int pad_h, const int pad_w, const int stride_h, const int stride_w,
const int dilation_h, const int dilation_w, float* data_col);

template void im2col_gpu<double>(const double* data_im, const int channels,
const int height, const int width, const int kernel_h, const int kernel_w,
const int pad_h, const int pad_w, const int stride_h, const int stride_w,
const int dilation_h, const int dilation_w, double* data_col);




template <typename Dtype, int num_axes>
__global__ void im2col_nd_gpu_kernel(const int n, const Dtype* data_im,
                                     const int* im_shape, const int* col_shape,
                                     const int* kernel_shape, const int* pad, const int* stride,
                                     const int* dilation, Dtype* data_col) {
    int d_temp[num_axes];  // NOLINT(runtime/arrays)
    int d_iter[num_axes];  // NOLINT(runtime/arrays)

    __shared__ int shared_dilation[num_axes];
    __shared__ int shared_kernel_shape[num_axes];
    __shared__ int shared_pad[num_axes];
    __shared__ int shared_stride[num_axes];
    __shared__ int shared_col_shape[num_axes + 1];
    __shared__ int shared_im_shape[num_axes + 1];

    if (threadIdx.x < num_axes) {
        shared_dilation[threadIdx.x] = dilation[threadIdx.x];
        shared_kernel_shape[threadIdx.x] = kernel_shape[threadIdx.x];
        shared_pad[threadIdx.x] = pad[threadIdx.x];
        shared_stride[threadIdx.x] = stride[threadIdx.x];
    }
    if (threadIdx.x < num_axes + 1) {
        shared_col_shape[threadIdx.x] = col_shape[threadIdx.x];
        shared_im_shape[threadIdx.x] = im_shape[threadIdx.x];
    }
    __syncthreads();

    int i;
    CUDA_KERNEL_LOOP(index, n) {
        // Initialize channel_in, computed in the loop below, with intermediate
        // computations used to compute the spatial indices.
        int channel_in = index;
        int channel_out = 1;
        for (i = num_axes - 1; i >= 0; --i) {
            d_temp[i] = channel_in % shared_col_shape[i + 1];
            channel_in /= shared_col_shape[i + 1];
            channel_out *= shared_kernel_shape[i];
        }
        channel_out *= channel_in;
        int data_col_inc = 1;
        for (i = 0; i < num_axes; ++i) {
            channel_out *= shared_col_shape[i + 1];
            channel_out += d_temp[i];
            d_temp[i] = d_temp[i] * shared_stride[i] - shared_pad[i];
            channel_in *= shared_im_shape[i + 1];
            channel_in += d_temp[i];
            data_col_inc *= shared_col_shape[i + 1];
            d_iter[i] = 0;
        }
        Dtype* data_col_ptr = data_col + channel_out;
        const Dtype* data_im_ptr = data_im + channel_in;
        bool incremented;
        do {
            bool in_range = true;
            for (i = 0; i < num_axes; ++i) {
                const int d_iter_im = d_iter[i] * shared_dilation[i] + d_temp[i];
                in_range &= d_iter_im >= 0 && d_iter_im < shared_im_shape[i + 1];
                if (!in_range) { break; }
            }
            if (in_range) {
                int data_im_offset = d_iter[0] * shared_dilation[0];
                for (i = 1; i < num_axes; ++i) {
                    data_im_offset *= shared_im_shape[i + 1];
                    data_im_offset += d_iter[i] * shared_dilation[i];
                }
                *data_col_ptr = data_im_ptr[data_im_offset];
            } else {
                *data_col_ptr = 0;
            }
            data_col_ptr += data_col_inc;
            incremented = false;
            for (i = num_axes - 1; i >= 0; --i) {
                const int d_max = shared_kernel_shape[i];
                if (d_iter[i] == d_max - 1) {
                    d_iter[i] = 0;
                } else {  // d_iter[i] < d_max - 1
                    ++d_iter[i];
                    incremented = true;
                    break;
                }
            }  // for (int i = num_axes - 1; i >= 0; --i)
        } while (incremented);  // do
    }  // CUDA_KERNEL_LOOP(index, n)
}

template <typename Dtype>
void im2col_nd_gpu(const Dtype* data_im, const int num_spatial_axes,
                   const int num_kernels, const int* im_shape, const int* col_shape,
                   const int* kernel_shape, const int* pad, const int* stride,
                   const int* dilation, Dtype* data_col) {
    // num_axes should be smaller than block size
    DCHECK_LT(num_spatial_axes, CAFFE_CUDA_NUM_THREADS);
    switch (num_spatial_axes) {
    case 1:
        im2col_nd_gpu_kernel<Dtype, 1>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
                                                                              num_kernels, data_im, im_shape, col_shape,
                                                                              kernel_shape, pad, stride, dilation, data_col);
        break;
    case 2:
        im2col_nd_gpu_kernel<Dtype, 2>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
                                                                              num_kernels, data_im, im_shape, col_shape,
                                                                              kernel_shape, pad, stride, dilation, data_col);
        break;
    case 3:
        im2col_nd_gpu_kernel<Dtype, 3>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
                                                                              num_kernels, data_im, im_shape, col_shape,
                                                                              kernel_shape, pad, stride, dilation, data_col);
        break;
    case 4:
        im2col_nd_gpu_kernel<Dtype, 4>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
                                                                              num_kernels, data_im, im_shape, col_shape,
                                                                              kernel_shape, pad, stride, dilation, data_col);
        break;
    case 5:
        im2col_nd_gpu_kernel<Dtype, 5>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
                                                                              num_kernels, data_im, im_shape, col_shape,
                                                                              kernel_shape, pad, stride, dilation, data_col);
        break;
    case 6:
        im2col_nd_gpu_kernel<Dtype, 6>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
                                                                              num_kernels, data_im, im_shape, col_shape,
                                                                              kernel_shape, pad, stride, dilation, data_col);
        break;
    case 7:
        im2col_nd_gpu_kernel<Dtype, 7>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
                                                                              num_kernels, data_im, im_shape, col_shape,
                                                                              kernel_shape, pad, stride, dilation, data_col);
        break;
    case 8:
        im2col_nd_gpu_kernel<Dtype, 8>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
                                                                              num_kernels, data_im, im_shape, col_shape,
                                                                              kernel_shape, pad, stride, dilation, data_col);
        break;
    case 9:
        im2col_nd_gpu_kernel<Dtype, 9>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
                                                                              num_kernels, data_im, im_shape, col_shape,
                                                                              kernel_shape, pad, stride, dilation, data_col);
        break;
    case 10:
        im2col_nd_gpu_kernel<Dtype, 10>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
                                                                              num_kernels, data_im, im_shape, col_shape,
                                                                              kernel_shape, pad, stride, dilation, data_col);
        break;
    default:
        LOG(FATAL) << "im2col_nd_gpu does not support computation with "
                   << num_spatial_axes << " spatial axes";
    }
    CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_nd_gpu<float>(const float* data_im,
const int num_spatial_axes, const int col_size,
const int* im_shape, const int* col_shape,
const int* kernel_shape, const int* pad, const int* stride,
const int* dilation, float* data_col);
template void im2col_nd_gpu<double>(const double* data_im,
const int num_spatial_axes, const int col_size,
const int* im_shape, const int* col_shape,
const int* kernel_shape, const int* pad, const int* stride,
const int* dilation, double* data_col);

template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
                                  const int height, const int width, const int channels,
                                  const int kernel_h, const int kernel_w,
                                  const int pad_h, const int pad_w,
                                  const int stride_h, const int stride_w,
                                  const int dilation_h, const int dilation_w,
                                  const int height_col, const int width_col,
                                  Dtype* data_im,  const int accumulate,
                                  int lcm_by_stride_h, int lcm_by_stride_w) {
    CUDA_KERNEL_LOOP(index, n) {
        Dtype val = 0;
        const int w_im = index % width + pad_w;
        const int h_im = (index / width) % height + pad_h;
        const int c_im = index / (width * height);
        int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
        int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
        // compute the start and end of the output
        int w_col_start =
                (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
        const int w_col_end = min(w_im / stride_w + 1, width_col);
        int h_col_start =
                (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
        const int h_col_end = min(h_im / stride_h + 1, height_col);
        // TODO: use LCM of stride and dilation to avoid unnecessary loops

        int h_k_start = (h_im - h_col_start * stride_h);
        int w_k_start = (w_im - w_col_start * stride_w);

        // obtain the valid starting index in col to employ lcm of column and stride
        // otherwise the shift will only encounter invalid locations

//        if (h_k_start % dilation_h)
        {
            while(h_k_start % dilation_h)
            {
                h_col_start++;
                h_k_start = (h_im - h_col_start * stride_h);
            }
        }

//        if (w_k_start % dilation_w)
        {
            while(w_k_start % dilation_w)
            {
                w_col_start++;
                w_k_start = (w_im - w_col_start * stride_w);
            }
        }

//                        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
//                            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {

        for (int h_col = h_col_start; h_col < h_col_end; h_col += lcm_by_stride_h) {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += lcm_by_stride_w) {
                int h_k = (h_im - h_col * stride_h);
                int w_k = (w_im - w_col * stride_w);
//                if (h_k % dilation_h == 0 && w_k % dilation_w == 0)
                {
                    h_k /= dilation_h;
                    w_k /= dilation_w;
                    int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                          height_col + h_col) * width_col + w_col;
                    val += data_col[data_col_index];
                }
            }
        }
        data_im[index] = val + data_im[index] * accumulate;
    }
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const int dilation_h, const int dilation_w,
                Dtype* data_im, const int accumulate) {
    int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
            stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
            stride_w + 1;
    int num_kernels = channels * height * width;
    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    // NOLINT_NEXT_LINE(whitespace/operators)

    //    if(accumulate)
    //    std::cout << "col 22 im accumulate = " << accumulate << "\n";

    // in the input space diff will be back propagated at lcm of dilation and stride
    // but in the output space, locations will be divided by stride

    const int lcm_dilation_stride_h = boost::math::lcm(dilation_h, stride_h);
    const int lcm_dilation_stride_w = boost::math::lcm(dilation_w, stride_w);

    const int lcm_by_stride_h = lcm_dilation_stride_h / stride_h;
    const int lcm_by_stride_w = lcm_dilation_stride_w / stride_w;

    col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
            CAFFE_CUDA_NUM_THREADS>>>(
                                        num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
                                        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                        height_col, width_col, data_im, accumulate,
                                        lcm_by_stride_h, lcm_by_stride_w);
    CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
const int height, const int width, const int kernel_h, const int kernel_w,
const int pad_h, const int pad_w, const int stride_h,
const int stride_w, const int dilation_h, const int dilation_w,
float* data_im, const int accumulate);
template void col2im_gpu<double>(const double* data_col, const int channels,
const int height, const int width, const int kernel_h, const int kernel_w,
const int pad_h, const int pad_w, const int stride_h,
const int stride_w, const int dilation_h, const int dilation_w,
double* data_im, const int accumulate);

template <typename Dtype, int num_axes>
__global__ void col2im_nd_gpu_kernel(const int n, const Dtype* data_col,
                                     const int* im_shape, const int* col_shape,
                                     const int* kernel_shape, const int* pad, const int* stride,
                                     const int* dilation, Dtype* data_im) {
    int d_im[num_axes];  // NOLINT(runtime/arrays)
    int d_col_iter[num_axes];  // NOLINT(runtime/arrays)
    int d_col_start[num_axes];  // NOLINT(runtime/arrays)
    int d_col_end[num_axes];  // NOLINT(runtime/arrays)

    __shared__ int shared_dilation[num_axes];
    __shared__ int shared_kernel_shape[num_axes];
    __shared__ int shared_pad[num_axes];
    __shared__ int shared_stride[num_axes];
    __shared__ int shared_col_shape[num_axes + 1];
    __shared__ int shared_im_shape[num_axes + 1];

    if (threadIdx.x < num_axes) {
        shared_dilation[threadIdx.x] = dilation[threadIdx.x];
        shared_kernel_shape[threadIdx.x] = kernel_shape[threadIdx.x];
        shared_pad[threadIdx.x] = pad[threadIdx.x];
        shared_stride[threadIdx.x] = stride[threadIdx.x];
    }
    if (threadIdx.x < num_axes + 1) {
        shared_col_shape[threadIdx.x] = col_shape[threadIdx.x];
        shared_im_shape[threadIdx.x] = im_shape[threadIdx.x];
    }
    __syncthreads();

    CUDA_KERNEL_LOOP(index, n) {
        // Initialize channel_in, computed in the loop below, with intermediate
        // computations used to compute the spatial indices.
        int c_im = index;
        // Calculate d_im (image dimensions).
        for (int i = num_axes - 1; i >= 0; --i) {
            d_im[i] = c_im % shared_im_shape[i + 1] + shared_pad[i];
            c_im /= shared_im_shape[i + 1];
        }
        // Calculate col start/end indices.
        bool done = false;
        for (int i = 0; i < num_axes; ++i) {
            const int kernel_extent =
                    shared_dilation[i] * (shared_kernel_shape[i] - 1) + 1;
            d_col_start[i] = d_col_iter[i] =
                    (d_im[i] < kernel_extent) ? 0 :
                                                (d_im[i] - kernel_extent) / shared_stride[i] + 1;
            d_col_end[i] =
                    min(d_im[i] / shared_stride[i] + 1, shared_col_shape[i + 1]);
            if (d_col_start[i] >= d_col_end[i]) {
                // Skip computation if the dimension is 0 at any spatial axis --
                // final val will be 0.
                data_im[index] = 0;
                done = true;
                break;  // for (int i = 0; i < num_axes; ++i)
            }
        }
        if (done) {
            continue;  // CUDA_KERNEL_LOOP(index, n)
        }
        // Loop over the col to compute the output val.
        Dtype val = 0;
        bool incremented = true;
        bool skip = false;
        do {
            // Compute the final offset.
            int final_offset = 0;
            int kernel_shape_prod = 1;
            int kernel_index;
            for (int i = num_axes - 1; i >= 0; --i) {
                kernel_index = d_im[i] - d_col_iter[i] * shared_stride[i];
                if (kernel_index % shared_dilation[i]) {
                    skip = true;
                    break;
                } else {
                    kernel_index /= shared_dilation[i];
                    final_offset += kernel_index * kernel_shape_prod;
                    kernel_shape_prod *= shared_kernel_shape[i];
                }
            }
            if (!skip) {
                final_offset += kernel_shape_prod * c_im;
                for (int i = 0; i < num_axes; ++i) {
                    final_offset *= shared_col_shape[i + 1];
                    final_offset += d_col_iter[i];
                }
                val += data_col[final_offset];
            }
            skip = false;
            incremented = false;
            for (int i = num_axes - 1; i >= 0; --i) {
                const int d_max = d_col_end[i];
                if (d_col_iter[i] == d_max - 1) {
                    d_col_iter[i] = d_col_start[i];
                } else {  // d_col_iter[i] < d_max - 1
                    ++d_col_iter[i];
                    incremented = true;
                    break;  // for (int i = num_axes - 1; i >= 0; --i)
                }
            }  // for (int i = num_axes - 1; i >= 0; --i)
        }  while (incremented);
        data_im[index] = val;
    }  // CUDA_KERNEL_LOOP(index, n)
}

template <typename Dtype>
void col2im_nd_gpu(const Dtype* data_col, const int num_spatial_axes,
                   const int im_size, const int* im_shape, const int* col_shape,
                   const int* kernel_shape, const int* pad, const int* stride,
                   const int* dilation, Dtype* data_im) {
    // num_axes should be smaller than block size
    DCHECK_LT(num_spatial_axes, CAFFE_CUDA_NUM_THREADS);
    switch (num_spatial_axes) {
    case 1:
        col2im_nd_gpu_kernel<Dtype, 1>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
                                                                          im_size, data_col, im_shape, col_shape,
                                                                          kernel_shape, pad, stride, dilation, data_im);
        break;
    case 2:
        col2im_nd_gpu_kernel<Dtype, 2>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
                                                                          im_size, data_col, im_shape, col_shape,
                                                                          kernel_shape, pad, stride, dilation, data_im);
        break;
    case 3:
        col2im_nd_gpu_kernel<Dtype, 3>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
                                                                          im_size, data_col, im_shape, col_shape,
                                                                          kernel_shape, pad, stride, dilation, data_im);
        break;
    case 4:
        col2im_nd_gpu_kernel<Dtype, 4>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
                                                                          im_size, data_col, im_shape, col_shape,
                                                                          kernel_shape, pad, stride, dilation, data_im);
        break;
    case 5:
        col2im_nd_gpu_kernel<Dtype, 5>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
                                                                          im_size, data_col, im_shape, col_shape,
                                                                          kernel_shape, pad, stride, dilation, data_im);
        break;
    case 6:
        col2im_nd_gpu_kernel<Dtype, 6>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
                                                                          im_size, data_col, im_shape, col_shape,
                                                                          kernel_shape, pad, stride, dilation, data_im);
        break;
    case 7:
        col2im_nd_gpu_kernel<Dtype, 7>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
                                                                          im_size, data_col, im_shape, col_shape,
                                                                          kernel_shape, pad, stride, dilation, data_im);
        break;
    case 8:
        col2im_nd_gpu_kernel<Dtype, 8>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
                                                                          im_size, data_col, im_shape, col_shape,
                                                                          kernel_shape, pad, stride, dilation, data_im);
        break;
    case 9:
        col2im_nd_gpu_kernel<Dtype, 9>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
                                                                          im_size, data_col, im_shape, col_shape,
                                                                          kernel_shape, pad, stride, dilation, data_im);
        break;
    case 10:
        col2im_nd_gpu_kernel<Dtype, 10>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
                                                                          im_size, data_col, im_shape, col_shape,
                                                                          kernel_shape, pad, stride, dilation, data_im);
        break;
    default:
        LOG(FATAL) << "col2im_nd_gpu does not support computation with "
                   << num_spatial_axes << " spatial axes";
    }
    CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_nd_gpu<float>(const float* data_col,
const int num_spatial_axes, const int im_size,
const int* im_shape, const int* col_shape,
const int* kernel_shape, const int* pad, const int* stride,
const int* dilation, float* data_im);
template void col2im_nd_gpu<double>(const double* data_col,
const int num_spatial_axes, const int im_size,
const int* im_shape, const int* col_shape,
const int* kernel_shape, const int* pad, const int* stride,
const int* dilation, double* data_im);

}  // namespace caffe
