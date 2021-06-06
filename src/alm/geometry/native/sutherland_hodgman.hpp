#pragma once

#ifndef SUTHERLAND_HODGMAN
#define SUTHERLAND_HODGMAN

#ifdef __CUDACC__
#include <thrust/sort.h>
#define SHARED __device__
#define sort(...) thrust::sort(__VA_ARGS__)
#else
#include <algorithm>
#define SHARED
#define sort(...) std::sort(__VA_ARGS__)
#endif
#include <iostream>


#define swap(x, y) ___tmp = y; y = x; x = ___tmp

template <typename scalar_t>
struct point {
   SHARED point(scalar_t x, scalar_t y) : x(x), y(y) {
      static_assert(sizeof(point<scalar_t>) == 2 * sizeof(scalar_t), "Arch not supported");
   }

   SHARED scalar_t sq_dist_to(const point<scalar_t> & other) const {
      scalar_t dx = other.x - x, dy = other.y - y;
      return dx*dx + dy*dy;
   }

   scalar_t x;
   scalar_t y;
};


template <typename scalar_t>
bool operator==(const point<scalar_t> & lhs, const point<scalar_t> & rhs) {
    return lhs.x == rhs.x && rhs.y == lhs.y;
}


#define malloc_points_no_cast(n) (malloc((n) * sizeof(point<scalar_t>)))
#define malloc_points(n) (reinterpret_cast<point<scalar_t> *>(malloc_points_no_cast(n)))
#define point_cast(x) (reinterpret_cast<point<scalar_t> *>(x))
#define const_point_cast(x) (reinterpret_cast<const point<scalar_t> *>(x))


template <typename scalar_t>
SHARED inline bool is_outside(const point<scalar_t> & p, const point<scalar_t> & r, const scalar_t & det_ar) {
   return p.x * r.y - p.y * r.x > det_ar;
}


template <typename scalar_t>
SHARED inline void intersect(
   point<scalar_t> & p, const point<scalar_t> & a, const point<scalar_t> & r,
   const point<scalar_t> & c, const point<scalar_t> & d, const scalar_t & det_ar)
{
   point<scalar_t> s(d.x - c.x, d.y - c.y);
   scalar_t u = (c.x * r.y - c.y * r.x - det_ar) / (r.x * s.y - r.y * s.x);
   p.x = c.x + s.x * u;
   p.y = c.y + s.y * u;
}


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


template <typename scalar_t>
SHARED scalar_t shoelace(const point<scalar_t> * polygon, const int64_t & n) {
   if (n <= 2) {
      return 0.0F;
   }
   // x_n y_1 - x_1 y_n
   scalar_t area2 = polygon[n-1].x * polygon[0].y - polygon[0].x * polygon[n-1].y;
   for (int k = 1; k < n; ++k, ++polygon) {
      // x_{n-1} y_n - x_n y_{n-1}
      area2 += polygon[0].x * polygon[1].y;
      area2 -= polygon[1].x * polygon[0].y;
   }
   return abs(area2) / 2.0F;
}


template <typename scalar_t>
SHARED inline int ccw(const point<scalar_t> & a, const point<scalar_t> & b, const point<scalar_t> & c) {
   scalar_t area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
   return (area < 0.0F) - (area > 0.0F);
}


// https://en.wikipedia.org/wiki/Graham_scan
template <typename scalar_t>
SHARED void graham_scan(int64_t * result, const point<scalar_t> * points, const int64_t & n) {
   if (n <= 2) {
      return ;
   }

   // init ref list with range zero to n-1
   for (int i = 0; i < n; ++i) {
      result[i] = i;
   }

   // find the point having the lowest y coordinate (pivot)
   point<scalar_t> lowest = points[0];
   int64_t lowest_index = 0;
   for (int64_t i = 1; i < n; ++i) {
      if (lowest.y > points[i].y || (lowest.y == points[i].y && lowest.x > points[i].x)) {
         lowest_index = i;
         lowest = points[i];
      }
   }

   // make least y the pivot element
   result[0] = lowest_index;
   result[lowest_index] = 0;

   const point<scalar_t> & pivot = points[lowest_index];

   // sort the remaining point according to polar order about the pivot
   sort(result + 1, result + n, [&] (const auto & lhs, const auto & rhs) {
      int order = ccw(pivot, points[lhs], points[rhs]);
      if (order ==  1) return true;
      if (order == -1) return false;
      return pivot.sq_dist_to(points[lhs]) < pivot.sq_dist_to(points[rhs]);
   });

   int64_t m = 0;
   for (int64_t i = 0; i < n; i++) {
      while (m > 1 && ccw(points[result[m-2]], points[result[m-1]], points[result[i]]) <= 0) {
         --m;
      }
      result[m++] = result[i];
   }

   if (m == 2 && points[result[1]] == points[result[0]]) {
      m = 1;
   }

   for (int64_t i = m; i < n; ++i) {
      result[i] = -1;
   }
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



#endif