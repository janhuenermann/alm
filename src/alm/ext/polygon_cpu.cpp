#include <torch/extension.h>
#include <optional>
#include <vector>

#include "sutherland_hodgman.hpp"
#include "graham_scan.hpp"


using namespace std;
using namespace torch;
using namespace torch::indexing;


Tensor sutherland_hodgman_cpu(const Tensor & poly1, const Tensor & poly2, const double pad_value);
Tensor compute_intersection_area_cpu(const Tensor & poly1, const Tensor & poly2, const double pad_value);
Tensor convex_hull_cpu(const Tensor & poly);


TORCH_LIBRARY(alm_ext, m) {
  m.def("sutherland_hodgman", &sutherland_hodgman_cpu);
  m.def("compute_intersection_area", &compute_intersection_area_cpu);
  m.def("convex_hull", &convex_hull_cpu);
}
