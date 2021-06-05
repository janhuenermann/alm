#pragma once

#ifndef SUTHERLAND_HODGMAN
#define SUTHERLAND_HODGMAN

#define swap(x, y) ___tmp = y; y = x; x = ___tmp

#ifdef __CUDACC__
#define SHARED __device__
#else
#define SHARED
#endif


template <typename scalar_t>
SHARED inline bool outside(const scalar_t * p, const scalar_t * r, const scalar_t & det_ar) {
   return p[0] * r[1] - p[1] * r[0] > det_ar;
}


template <typename scalar_t>
SHARED inline void intersect(
   scalar_t * p, const scalar_t * a, const scalar_t * r,
   const scalar_t * c, const scalar_t * d, const scalar_t & det_ar)
{
   scalar_t s[2] = {
      d[0] - c[0],
      d[1] - c[1]
   };

   // scalar_t t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / (r[0] * s[1] - r[1] * s[0]);
   // p[0] = a[0] + r[0] * t;
   // p[1] = a[1] + r[1] * t;

   scalar_t u = (c[0] * r[1] - c[1] * r[0] - det_ar) / (r[0] * s[1] - r[1] * s[0]);
   p[0] = c[0] + s[0] * u;
   p[1] = c[1] + s[1] * u;
}


template <typename scalar_t>
SHARED void edge_clip(
   scalar_t * result, const scalar_t * polygon,
   const scalar_t * a, const scalar_t * b,
   int64_t & n, const int64_t &qmax)
{
   if (n <= 1) {
      return ;
   }

   // Precompute slope of line a -> b
   scalar_t r[2] = {
      b[0] - a[0],
      b[1] - a[1]
   };

   scalar_t det_ar = a[0] * r[1] - a[1] * r[0]; // det([a|r])

   const scalar_t * p0 = polygon + 2 * n - 2;
   const scalar_t * p1 = polygon;

   int64_t size = 0;
   bool out0, out1;

   #pragma unroll 4
   for(int64_t i = 0; i < n; i++) {
      out0 = outside(p0, r, det_ar);
      out1 = outside(p1, r, det_ar);

      if (out0 != out1) {
         size++;
         intersect(result, a, r, p0, p1, det_ar);
         result += 2;
         if (size >= qmax) break ;
      }

      if (!out1) {
         size++;
         *(result++) = p1[0];
         *(result++) = p1[1];
         if (size >= qmax) break ;
      }

      p0 = p1;
      p1 += 2;
   }

   n = size;
}


template <typename scalar_t>
SHARED int64_t polygon_clip(
   scalar_t * result, scalar_t * tmp,
   const scalar_t * polygon1, const scalar_t * polygon2,
   const int64_t & n, const int64_t & m, const int64_t & qmax)
{
   if (m <= 1) {
      return 0;
   }

   int64_t l = n;
   scalar_t * arr0 = tmp, * arr1 = result, * ___tmp;

   // For m = 4 (m is even):
   // polygon1 -> tmp, tmp -> result, result -> tmp, tmp -> result
   // For m = 3 (m is odd):
   // polygon1 -> result, result -> tmp, tmp -> result
   if (m % 2 == 1) {
      swap(arr0, arr1);
   }

   // Wrap around
   edge_clip(arr0, polygon1, polygon2 + 2 * m - 2, polygon2, l, qmax);

   // All lines going forward
   for (int i = 1; i < m; ++i) {
      edge_clip(arr1, arr0, polygon2, polygon2 + 2, l, qmax);
      swap(arr0, arr1);
      polygon2 += 2;
   }

   // Pad with zeros
   result += 2 * l;
   for (int64_t i = l; i < qmax; ++i) {
      *(result++) = 0.0F;
      *(result++) = 0.0F;
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
      area2 += polygon[0] * polygon[3];
      area2 -= polygon[2] * polygon[1];
      polygon += 2;
   }
   return abs(area2) / 2.0F;
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