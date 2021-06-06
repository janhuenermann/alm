#include <torch/extension.h>

#include <vector>

using namespace torch;


Tensor sutherland_hodgman_gpu(const Tensor & poly1, const Tensor & poly2);
Tensor compute_intersection_area_gpu(const Tensor & poly1, const Tensor & poly2);


Tensor sutherland_hodgman(const Tensor & poly1, const Tensor & poly2) {
   return sutherland_hodgman_gpu(poly1, poly2);
};


Tensor compute_intersection_area(const Tensor & poly1, const Tensor & poly2) {
   return compute_intersection_area_gpu(poly1, poly2);
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("sutherland_hodgman", &sutherland_hodgman, "Sutherland Hodgman Polygon Clipping Forward Pass (GPU)");
   m.def("compute_intersection_area", &compute_intersection_area, "Computer Area of Intersection of Two Polygons (GPU)");
}
