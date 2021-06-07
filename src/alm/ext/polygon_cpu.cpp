#include <torch/extension.h>
#include <optional>
#include <vector>

#include "sutherland_hodgman.hpp"
#include "graham_scan.hpp"


using namespace std;
using namespace torch;
using namespace torch::indexing;


Tensor sutherland_hodgman_cpu(const Tensor & poly1, const Tensor & poly2, const double pad_value) {
   CHECK_INPUT_POLY_AND_PREPARE(poly1, poly2);

   int64_t result_len = get_max_intersection_count(poly1_len, poly2_len);
   result_shape.push_back(result_len);
   result_shape.push_back(2);

   torch::Tensor result = at::zeros(result_shape, poly1.options());

   at::TensorIteratorConfig iter_config;
   auto iter = iter_config
     .check_all_same_dtype(false)
     .resize_outputs(false)
     .declare_static_shape(result.sizes(), { result.dim()-2, result.dim()-1 })
     .add_output(result)
     .add_input(poly1)
     .add_input(poly2)
     .build();

   AT_DISPATCH_FLOATING_TYPES_AND_HALF(result.scalar_type(), "sutherland_hodgman_cpu", [&] {
      iter.for_each([&](char ** data, const int64_t* strides, int64_t n) {
         const char * poly1_data = data[1];
         const char * poly2_data = data[2];
         char * result_data = data[0];
         point<scalar_t> * tmp_data = malloc_points(result_len);
         for (int k = 0; k < n; ++k) {
            polygon_clip(
               reinterpret_cast<point<scalar_t> *>(result_data),
               tmp_data,
               reinterpret_cast<const point<scalar_t> *>(poly1_data),
               reinterpret_cast<const point<scalar_t> *>(poly2_data),
               poly1_len, poly2_len, result_len, (scalar_t) pad_value);
            result_data += strides[0];
            poly1_data += strides[1];
            poly2_data += strides[2];
         }
         free(tmp_data);
      });
   });
   
   return result;
}


Tensor compute_intersection_area_cpu(const Tensor & poly1, const Tensor & poly2, const double pad_value) {
   CHECK_INPUT_POLY_AND_PREPARE(poly1, poly2);

   int64_t result_len = get_max_intersection_count(poly1_len, poly2_len);
   torch::Tensor result = at::zeros(result_shape, poly1.options());

   at::TensorIteratorConfig iter_config;
   auto iter = iter_config
     .check_all_same_dtype(false)
     .resize_outputs(false)
     .declare_static_shape(result.sizes())
     .add_output(result)
     .add_input(poly1)
     .add_input(poly2)
     .build();

   AT_DISPATCH_FLOATING_TYPES_AND_HALF(result.scalar_type(), "sutherland_hodgman_cpu", [&] {
      iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
         point<scalar_t> * vertex_data = malloc_points(2 * result_len); // 2 here is for result and tmp
         point<scalar_t> * tmp_data = vertex_data + result_len; // 2 here is for x and y
         int64_t npoly;
         const char * poly1_data = data[1];
         const char * poly2_data = data[2];
         char * result_data = data[0];
         for (int k = 0; k < n; ++k) {
            npoly = polygon_clip(
               vertex_data, tmp_data,
               reinterpret_cast<const point<scalar_t> *>(poly1_data),
               reinterpret_cast<const point<scalar_t> *>(poly2_data),
               poly1_len, poly2_len, result_len, (scalar_t) pad_value);
            result_data[0] = shoelace(vertex_data, npoly);
            result_data += strides[0];
            poly1_data += strides[1];
            poly2_data += strides[2];
         }
         free(vertex_data);
      });
   });
   
   return result;
}


Tensor convex_hull_cpu(const Tensor & poly) {
   TORCH_CHECK(poly.dim() >= 2, "Polygon tensors must have dim >= 2.");
   TORCH_CHECK(poly.size(-1) == 2, "You provided a tensor with invalid shape. size(-1) must be 2. Here it is ", poly.size(-1), ".");
   TORCH_CHECK(poly.size(-2) > 2, "You provided a tensor with invalid shape. size(-2) must be greater than 2. Here it is ", poly.size(-2), ".");
   TORCH_CHECK(poly.stride(-1) == 1, "Polygon is not stored in column minor format");
   
   std::vector<int64_t> result_shape;
   for (int k = 0; k < poly.dim() - 2; ++k) {
      result_shape.push_back(poly.size(k));
   }

   int64_t poly_len = poly.size(-2);
   result_shape.push_back(poly_len);

   torch::Tensor result = at::empty(result_shape, poly.options().dtype(torch::kInt64));
   at::TensorIteratorConfig iter_config;
   auto iter = iter_config
     .check_all_same_dtype(false)
     .resize_outputs(false)
     .declare_static_shape(result.sizes(), { result.dim()-1 })
     .add_output(result)
     .add_input(poly)
     .build();

   AT_DISPATCH_FLOATING_TYPES_AND_HALF(poly.scalar_type(), "convex_hull", [&] {
      iter.for_each([&](char ** data, const int64_t * strides, int64_t n) {
         const char * poly_data = data[1];
         char * result_data = data[0];
         for (int k = 0; k < n; ++k) {
            graham_scan(
               reinterpret_cast<int64_t *>(result_data),
               reinterpret_cast<const point<scalar_t> *>(poly_data),
               poly_len);
            poly_data += strides[1];
            result_data += strides[0];
         }
      });
   });
   
   return result;
}


TORCH_LIBRARY(alm_ext, m) {
  m.def("sutherland_hodgman", &sutherland_hodgman_cpu);
  m.def("compute_intersection_area", &compute_intersection_area_cpu);
  m.def("convex_hull", &convex_hull_cpu);
}


// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//    m.def("sutherland_hodgman", &sutherland_hodgman, "Sutherland Hodgman Polygon Clipping Forward Pass");
//    m.def("compute_intersection_area", &compute_intersection_area, "Computer Area of Intersection of Two Polygons");
//    m.def("convex_hull", &convex_hull, "Convex Hull of 2D Points");
// }
