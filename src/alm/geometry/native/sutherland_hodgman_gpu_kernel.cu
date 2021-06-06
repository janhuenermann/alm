#include <torch/extension.h>
#include <optional>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>

#include "sutherland_hodgman.hpp"


using namespace std;
using namespace torch;


constexpr int num_threads = 4 * C10_WARP_SIZE;
constexpr int thread_work_size = 1024;
constexpr int block_work_size = num_threads * thread_work_size;


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/** -------------- KERNELS -------------- */


template <typename scalar_t>
__global__ void sutherland_hodgman_gpu_kernel(
   point<scalar_t> * result_data,
   const point<scalar_t> * poly1_data,
   const point<scalar_t> * poly2_data,
   const int result_count, const int result_len,
   const int poly1_len, const int poly2_len)
{
   int index = thread_work_size * (blockIdx.x * blockDim.x + threadIdx.x);
   const int end = min(index + thread_work_size, result_count);

   if (index >= end) {
      return ;
   }

   point<scalar_t> * tmp_data = malloc_points(result_len);
   
   assert(tmp_data != NULL);

   result_data += index*result_len;
   poly1_data += index*poly1_len;
   poly2_data += index*poly2_len;

   for (; index < end; ++index) {
      polygon_clip(result_data, tmp_data, poly1_data, poly2_data, poly1_len, poly2_len, result_len);

      result_data += result_len;
      poly1_data += poly1_len;
      poly2_data += poly2_len;
   }

   free(reinterpret_cast<void *>(tmp_data));
}


template <typename scalar_t>
__global__ void compute_intersection_area_gpu_kernel(
   scalar_t * result_data,
   const point<scalar_t> * poly1_data,
   const point<scalar_t> * poly2_data,
   const int result_count, const int result_len,
   const int poly1_len, const int poly2_len)
{
   int index = thread_work_size * (blockIdx.x * blockDim.x + threadIdx.x);
   const int end = min(index + thread_work_size, result_count);

   if (index >= end) {
      return ;
   }

   int64_t npoly;
   point<scalar_t> * vertex_data = malloc_points(2 * result_len);

   assert(vertex_data != NULL);

   point<scalar_t> * tmp_data = vertex_data + result_len;

   result_data += index;
   poly1_data += index*poly1_len;
   poly2_data += index*poly2_len;

   for (; index < end; ++index) {
      npoly = polygon_clip(vertex_data, tmp_data, poly1_data, poly2_data, poly1_len, poly2_len, result_len);

      assert(npoly <= result_len);

      (*result_data) = shoelace(vertex_data, npoly);

      result_data += 1;
      poly1_data += poly1_len;
      poly2_data += poly2_len;
   }

   free(reinterpret_cast<void *>(vertex_data));
}

template <typename scalar_t>
__global__ void convex_hull_gpu_kernel(
   int64_t * result_data,
   const point<scalar_t> * poly_data,
   const int64_t result_count,
   const int64_t poly_len)
{
   int64_t index = thread_work_size * (blockIdx.x * blockDim.x + threadIdx.x);
   const int64_t end = min(index + thread_work_size, result_count);

   result_data += index*poly_len;
   poly_data += index*poly_len;

   for (; index < end; ++index) {
      graham_scan(result_data, poly_data, poly_len);
      result_data += poly_len;
      poly_data += poly_len;
   }
}


/** -------------- DISPATCHERS -------------- */


Tensor sutherland_hodgman_gpu(const Tensor & poly1, const Tensor & poly2) {
   CHECK_INPUT(poly1);
   CHECK_INPUT(poly2);
   CHECK_INPUT_POLY_AND_PREPARE(poly1, poly2);
   
   int64_t result_len = get_max_intersection_count(poly1_len, poly2_len);
   int64_t result_count = poly1.numel() / 2 / poly1_len;

   result_shape.push_back(result_len);
   result_shape.push_back(2);

   torch::Tensor result = at::zeros(result_shape, poly1.options());
   AT_DISPATCH_FLOATING_TYPES_AND_HALF(result.scalar_type(), "sutherland_hodgman_gpu", [&] {
      const dim3 blocks((result_count + block_work_size - 1) / block_work_size);
      auto stream = at::cuda::getCurrentCUDAStream();
      sutherland_hodgman_gpu_kernel<<<blocks, num_threads, 0, stream>>>(
         point_cast(result.data_ptr<scalar_t>()),
         const_point_cast(poly1.data_ptr<scalar_t>()),
         const_point_cast(poly2.data_ptr<scalar_t>()),
         result_count,
         result_len,
         poly1_len,
         poly2_len);
   });

   return result;
}


Tensor compute_intersection_area_gpu(const Tensor & poly1, const Tensor & poly2) {
   CHECK_INPUT(poly1);
   CHECK_INPUT(poly2);
   CHECK_INPUT_POLY_AND_PREPARE(poly1, poly2);

   int64_t intersection_len = get_max_intersection_count(poly1_len, poly2_len);
   int64_t result_count = poly1.numel() / 2 / poly1_len;

   torch::Tensor result = at::empty(result_shape, poly1.options());
   AT_DISPATCH_FLOATING_TYPES_AND_HALF(result.scalar_type(), "compute_intersection_area_gpu", [&] {
      const dim3 blocks((result_count + block_work_size - 1) / block_work_size);
      auto stream = at::cuda::getCurrentCUDAStream();
      compute_intersection_area_gpu_kernel<<<blocks, num_threads, 0, stream>>>(
         result.data_ptr<scalar_t>(),
         const_point_cast(poly1.data_ptr<scalar_t>()),
         const_point_cast(poly2.data_ptr<scalar_t>()),
         result_count,
         intersection_len,
         poly1_len,
         poly2_len);
   });

   return result;
}


Tensor convex_hull_gpu(const Tensor & poly) {
   CHECK_INPUT(poly);
   TORCH_CHECK(poly.dim() >= 2, "Polygon tensors must have dim >= 2.");
   TORCH_CHECK(poly.size(-1) == 2, "You provided a tensor with invalid shape. size(-1) must be 2. Here it is ", poly.size(-1), ".");
   TORCH_CHECK(poly.size(-2) > 2, "You provided a tensor with invalid shape. size(-2) must be greater than 2. Here it is ", poly.size(-2), ".");
   TORCH_CHECK(poly.stride(-1) == 1, "Given polygon is not stored in column minor format");
   
   int64_t poly_len = poly.size(-2);
   int64_t result_count = poly.numel() / 2 / poly_len;

   std::vector<int64_t> result_shape;
   for (int k = 0; k < poly.dim() - 2; ++k) {
      result_shape.push_back(poly.size(k));
   }

   result_shape.push_back(poly_len);

   torch::Tensor result = at::empty(result_shape, poly.options().dtype(torch::kInt64));
   AT_DISPATCH_FLOATING_TYPES_AND_HALF(poly.scalar_type(), "convex_hull_gpu", [&] {
      const dim3 blocks((result_count + block_work_size - 1) / block_work_size);
      auto stream = at::cuda::getCurrentCUDAStream();
      convex_hull_gpu_kernel<<<blocks, num_threads, 0, stream>>>(
         result.data_ptr<int64_t>(),
         const_point_cast(poly.data_ptr<scalar_t>()),
         result_count,
         poly_len);
   });

   return result;
}

