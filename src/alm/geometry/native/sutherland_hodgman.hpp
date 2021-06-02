#pragma once

#ifndef SUTHERLAND_HODGMAN
#define SUTHERLAND_HODGMAN

#ifdef __CUDACC__
#define SHARED __device__
#else
#define SHARED
#endif

template <typename scalar_t>
SHARED bool outside(const scalar_t *p, const scalar_t *a, const scalar_t *r) {
   return r[1] * (p[0] - a[0]) - r[0] * (p[1] - a[1]) > 0;
}

template <typename scalar_t>
SHARED void intersect(scalar_t *p, const scalar_t *a, const scalar_t *r, const scalar_t *c, const scalar_t *d) {
   scalar_t sx = d[0] - c[0], sy = d[1] - c[1];
   scalar_t t = ((c[0] - a[0]) * sy - (c[1] - a[1]) * sx) / (r[0] * sy - r[1] * sx);
   p[0] = a[0] + r[0] * t;
   p[1] = a[1] + r[1] * t;
}

template <typename scalar_t>
SHARED void edge_clip(scalar_t *result, const scalar_t *polygon, const scalar_t *a, const scalar_t *b, int64_t &n, const int64_t &qmax) {
   if (n <= 1) {
      return ;
   }

   // precompute slope of line a -> b
   scalar_t r[2] = {
      b[0] - a[0],
      b[1] - a[1]
   };

   const scalar_t * point0 = polygon + 2 * n - 2;
   const scalar_t * point1 = polygon;

   int64_t size = 0;
   bool out0, out1;

   #pragma unroll 4
   for(int64_t i = 0; i < n; i++) {
      out0 = outside(point0, a, r);
      out1 = outside(point1, a, r);

      if (out0 != out1) {
         intersect(result, a, r, point0, point1);
         result += 2;
         // Check that we are not out of bounds
         if ((++size) >= qmax) break ;
      }

      if (!out1) {
         result[0] = point1[0];
         result[1] = point1[1];
         result += 2;
         // Check that we are not out of bounds
         if ((++size) >= qmax) break ;
      }

      point0 = point1;
      point1 += 2;
   }

   n = size;
}

#define swap(x, y) ___tmp = y; y = x; x = ___tmp

template <typename scalar_t>
SHARED int64_t polygon_clip(scalar_t *result, scalar_t *tmp, const scalar_t *polygon1, const scalar_t *polygon2, const int64_t &n, const int64_t &m, const int64_t &qmax) {
   if (m <= 1) {
      return 0;
   }

   int64_t l = n;
   scalar_t * a = tmp, * b = result, * ___tmp;
   if (m % 2 == 1) {
      swap(a, b);
   }

   // For m = 4 (m is even):
   // polygon1 -> tmp
   // tmp -> result
   // result -> tmp
   // tmp -> result

   // For m = 5 (m is odd):
   // polygon1 -> result
   // result -> tmp
   // tmp -> result
   // result -> tmp
   // tmp -> result

   // Wrap around
   edge_clip(a, polygon1, polygon2 + 2 * m - 2, polygon2, l, qmax);

   // All lines going forward
   for (int i = 1; i < m; ++i) {
      edge_clip(b, a, polygon2, polygon2 + 2, l, qmax);
      swap(a, b);
      polygon2 += 2;
   }

   return l;
}

template <typename scalar_t>
SHARED scalar_t shoelace(const scalar_t *polygon, const int64_t &n) {
   if (n <= 2) {
      return 0.0F;
   }
   // x_n y_1 - x_1 y_n
   scalar_t area2 = polygon[2*n-2] * polygon[1] - polygon[0] * polygon[2*n-1];
   for (int k = 1; k < n; ++k) {
      // x_{n-1} y_n - x_n y_{n-1}
      area2 += polygon[0] * polygon[3] - polygon[2] * polygon[1];
      polygon += 2;
   }
   return area2 / 2.0F;
}

// For bound, see https://resources.mpi-inf.mpg.de/departments/d1/teaching/ws09_10/CGGC/Notes/Polygons.pdf
int64_t get_max_intersection_count(int64_t p, int64_t q) {
   return std::min(2 * p, 2 * q);
}

#define CHECK_INPUT_POLY_AND_PREPARE(poly1, poly2) \
   TORCH_CHECK(poly1.dim() == poly2.dim(), "Both polygons must have the same number of dims. ", poly1.dim(), " != ", poly2.dim());\
   TORCH_CHECK(poly1.dim() >= 2, "Polygon tensors must have dim >= 2.");\
   TORCH_CHECK(poly1.size(-1) == 2, "You provided a tensor for polygon 1 with invalid shape. size(-1) must be 2. Here it is ", poly1.size(-1), ".");\
   TORCH_CHECK(poly2.size(-1) == 2, "You provided a tensor for polygon 2 with invalid shape. size(-1) must be 2. Here it is ", poly2.size(-1), ".");\
   TORCH_CHECK(poly1.size(-2) > 2, "You provided a tensor for polygon 1 with invalid shape. size(-2) must be greater than 2. Here it is ", poly1.size(-2), ".");\
   TORCH_CHECK(poly2.size(-2) > 2, "You provided a tensor for polygon 2 with invalid shape. size(-2) must be greater than 2. Here it is ", poly2.size(-2), ".");\
   TORCH_CHECK(poly1.stride(-1) == 1, "Polygon 1 is not stored in column minor format");\
   TORCH_CHECK(poly2.stride(-1) == 1, "Polygon 2 is not stored in column minor format");\
   std::vector<int64_t> out_shape;\
   for (int k = 0; k < poly1.dim() - 2; ++k) {\
      TORCH_CHECK(poly1.size(k) == poly2.size(k), "Dimension ", k, " must match for both polygon tensors. [poly1.size(", k, ") = ", poly1.size(k), "] != [", poly2.size(k), " = poly2.size(", k, ")]");\
      out_shape.push_back(poly1.size(k));\
   }\
   int64_t poly1_len = poly1.size(-2);\
   int64_t poly2_len = poly2.size(-2);\
   int64_t out_len = get_max_intersection_count(poly1_len, poly2_len);\
   int64_t out_count = poly1.numel() / (2 * poly1_len);



#endif