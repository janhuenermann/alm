#pragma once
#include "math.hpp"


template <typename scalar_t>
SHARED void edge_clip(
   point<scalar_t> * result, const point<scalar_t> * polygon,
   const point<scalar_t> & a, const point<scalar_t> & b,
   int64_t & n, const int64_t & qmax)
{
   if (n <= 1) {
      return ;
   }

   // Precompute slope of clipping line a -> b
   point<scalar_t> r(b.x - a.x, b.y - a.y);
   scalar_t det_ar = a.x * r.y - a.y * r.x; // det([a|r])

   // Line to be clipped p0 -> p1
   const point<scalar_t> * p0 = polygon + n - 1;
   const point<scalar_t> * p1 = polygon;

   // Output size m
   int64_t m = 0;
   // Is p0 or p1 outside?
   bool out_p0, out_p1;

   #pragma unroll 4
   for(int64_t i = 0; i < n && m < qmax; i++) {
      out_p0 = is_outside(*p0, r, det_ar);
      out_p1 = is_outside(*p1, r, det_ar);

      if (out_p0 != out_p1) {
         intersect(result[m], a, r, *p0, *p1, det_ar);
         ++m;
      }

      if (!out_p1) {
         result[m].x = p1->x;
         result[m].y = p1->y;
         ++m;
      }

      p0 = p1;
      ++p1;
   }

   n = m;
}


template <typename scalar_t>
SHARED int64_t polygon_clip(
   point<scalar_t> * result, point<scalar_t> * tmp,
   const point<scalar_t> * polygon1, const point<scalar_t> * polygon2,
   const int64_t & n, const int64_t & m, const int64_t & qmax)
{
   if (m <= 1) {
      return 0;
   }

   int64_t l = n;
   point<scalar_t> * arr0 = tmp, * arr1 = result, * ___tmp;

   // For m = 4 (m is even):
   // polygon1 -> tmp, tmp -> result, result -> tmp, tmp -> result
   // For m = 3 (m is odd):
   // polygon1 -> result, result -> tmp, tmp -> result
   if (m % 2 == 1) {
      swap(arr0, arr1);
   }

   // Copy first
   edge_clip(arr0, polygon1, polygon2[m-1], polygon2[0], l, qmax);

   for (int i = 1; i < m; ++i, ++polygon2) {
      edge_clip(arr1, arr0, polygon2[0], polygon2[1], l, qmax);
      swap(arr1, arr0);
   }

   // Pad with zeros
   result += l;
   for (int64_t i = l; i < qmax; ++i, ++result) {
      result->x = 0.0F;
      result->y = 0.0F;
   }

   return l;
}


// For bound, see https://resources.mpi-inf.mpg.de/departments/d1/teaching/ws09_10/CGGC/Notes/Polygons.pdf
int64_t get_max_intersection_count(int64_t p, int64_t q) {
   return std::min(2 * p, 2 * q);
}


// Check input format and shapes
#define CHECK_INPUT_POLY_AND_PREPARE(poly1, poly2) \
   TORCH_CHECK(poly1.dim() == poly2.dim(), "Both polygons must have the same number of dims. ", poly1.dim(), " != ", poly2.dim());\
   TORCH_CHECK(poly1.dim() >= 2, "Polygon tensors must have dim >= 2.");\
   TORCH_CHECK(poly1.size(-1) == 2, "You provided a tensor for polygon 1 with invalid shape. size(-1) must be 2. Here it is ", poly1.size(-1), ".");\
   TORCH_CHECK(poly2.size(-1) == 2, "You provided a tensor for polygon 2 with invalid shape. size(-1) must be 2. Here it is ", poly2.size(-1), ".");\
   TORCH_CHECK(poly1.size(-2) > 2, "You provided a tensor for polygon 1 with invalid shape. size(-2) must be greater than 2. Here it is ", poly1.size(-2), ".");\
   TORCH_CHECK(poly2.size(-2) > 2, "You provided a tensor for polygon 2 with invalid shape. size(-2) must be greater than 2. Here it is ", poly2.size(-2), ".");\
   TORCH_CHECK(poly1.stride(-1) == 1, "Polygon 1 is not stored in column minor format");\
   TORCH_CHECK(poly2.stride(-1) == 1, "Polygon 2 is not stored in column minor format");\
   std::vector<int64_t> result_shape;\
   for (int k = 0; k < poly1.dim() - 2; ++k) {\
      TORCH_CHECK(poly1.size(k) == poly2.size(k), "Dimension ", k, " must match for both polygon tensors. [poly1.size(", k, ") = ", poly1.size(k), "] != [", poly2.size(k), " = poly2.size(", k, ")]");\
      result_shape.push_back(poly1.size(k));\
   }\
   int64_t poly1_len = poly1.size(-2);\
   int64_t poly2_len = poly2.size(-2);
