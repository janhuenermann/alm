#include <torch/extension.h>
#include <optional>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "sutherland_hodgman.hpp"

using namespace std;
using namespace torch;

constexpr int num_threads = 2 * C10_WARP_SIZE;
constexpr int thread_work_size = 1024;
constexpr int block_work_size = num_threads * thread_work_size;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__global__ void sutherland_hodgman_gpu_kernel(scalar_t *result,
                                              const scalar_t *poly1, const scalar_t *poly2,
                                              const int out_count, const int out_len,
                                              const int poly1_len, const int poly2_len) {
   int index = thread_work_size * (blockIdx.x * blockDim.x + threadIdx.x);
   const int end = min(index + thread_work_size, out_count);

   if (index >= end) {
      return ;
   }

   scalar_t * tmp = reinterpret_cast<scalar_t *>(__nv_aligned_device_malloc(2 * out_len * sizeof(scalar_t), 2));

   const int result_stride = out_len * 2;
   const int poly1_stride = poly1_len * 2;
   const int poly2_stride = poly2_len * 2;

   for (; index < end; ++index) {
      polygon_clip(
         result+index*result_stride,
         tmp, 
         poly1+index*poly1_stride,
         poly2+index*poly2_stride,
         poly1_len, poly2_len);
   }

   free(reinterpret_cast<void *>(tmp));
}

Tensor sutherland_hodgman_gpu(const Tensor &poly1, const Tensor &poly2) {
   CHECK_INPUT(poly1);
   CHECK_INPUT(poly2);
   CHECK_INPUT_POLY_AND_PREPARE(poly1, poly2);

   const dim3 blocks((out_count + block_work_size - 1) / block_work_size);
   auto stream = at::cuda::getCurrentCUDAStream();

   AT_DISPATCH_FLOATING_TYPES_AND_HALF(result.scalar_type(), "sutherland_hodgman_gpu", [&] {
      sutherland_hodgman_gpu_kernel<<<blocks, num_threads, 0, stream>>>(
         result.data_ptr<scalar_t>(),
         poly1.data_ptr<scalar_t>(),
         poly2.data_ptr<scalar_t>(),
         out_count,
         out_len,
         poly1_len,
         poly2_len);
   });

   return result;
}

