#include <torch/extension.h>

#include <vector>

using namespace torch;


Tensor sutherland_hodgman_gpu(const Tensor & poly1, const Tensor & poly2, const double pad_value);
Tensor compute_intersection_area_gpu(const Tensor & poly1, const Tensor & poly2, const double pad_value);
Tensor convex_hull_gpu(const Tensor & poly);


Tensor sutherland_hodgman_cpu(const Tensor & poly1, const Tensor & poly2, const double pad_value);
Tensor compute_intersection_area_cpu(const Tensor & poly1, const Tensor & poly2, const double pad_value);
Tensor convex_hull_cpu(const Tensor & poly);


Tensor sutherland_hodgman(const Tensor & poly1, const Tensor & poly2, const double pad_value) {
   if (poly1.is_cuda) {
      return sutherland_hodgman_gpu(poly1, poly2, pad_value);
   }
   return sutherland_hodgman_cpu(poly1, poly2, pad_value);
};


Tensor compute_intersection_area(const Tensor & poly1, const Tensor & poly2, const double pad_value) {
   if (poly1.is_cuda) {
      return compute_intersection_area_gpu(poly1, poly2, pad_value);
   }
   return compute_intersection_area_cpu(poly1, poly2, pad_value);
};


Tensor convex_hull(const Tensor & poly) {
   if (poly.is_cuda) {
      return convex_hull_gpu(poly);
   }
   return convex_hull_cpu(poly);
};


TORCH_LIBRARY(alm_ext, m) {
  m.def("sutherland_hodgman", &sutherland_hodgman);
  m.def("compute_intersection_area", &compute_intersection_area);
  m.def("convex_hull", &convex_hull);
}
