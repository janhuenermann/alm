#include <torch/extension.h>
#include <optional>
#include <vector>

#include "sutherland_hodgman.hpp"

using namespace std;
using namespace torch;
using namespace torch::indexing;

Tensor sutherland_hodgman(const Tensor &poly1, const Tensor &poly2) {
   CHECK_INPUT_POLY_AND_PREPARE(poly1, poly2);

   at::TensorIteratorConfig iter_config;
   auto iter = iter_config
     .check_all_same_dtype(false)
     .resize_outputs(false)
     .declare_static_shape(result.sizes(), { result.dim()-2, result.dim()-1 })
     .add_output(result)
     .add_input(poly1)
     .add_input(poly2)
     .build();

   iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      float *tmp = new float[2 * out_len];
      char *result_data = data[0];
      const char *poly1_data = data[1];
      const char *poly2_data = data[2];
      for (int k = 0; k < n; ++k) {
         polygon_clip(
            reinterpret_cast<float *>(result_data), tmp, 
            reinterpret_cast<const float *>(poly1_data),
            reinterpret_cast<const float *>(poly2_data),
            poly1_len, poly2_len);

         result_data += strides[0];
         poly1_data += strides[1];
         poly2_data += strides[2];
      }
      delete [] tmp;
   });

   return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("sutherland_hodgman", &sutherland_hodgman, "Sutherland Hodgman Polygon Clipping Forward Pass");
}