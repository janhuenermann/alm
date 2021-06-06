#pragma once
#include "math.hpp"


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