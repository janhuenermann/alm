#pragma once

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
#define malloc_points_no_cast(n) (malloc((n) * sizeof(point<scalar_t>)))
#define malloc_points(n) (reinterpret_cast<point<scalar_t> *>(malloc_points_no_cast(n)))
#define point_cast(x) (reinterpret_cast<point<scalar_t> *>(x))
#define const_point_cast(x) (reinterpret_cast<const point<scalar_t> *>(x))


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
SHARED bool operator==(const point<scalar_t> & lhs, const point<scalar_t> & rhs) {
    return lhs.x == rhs.x && rhs.y == lhs.y;
}


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

