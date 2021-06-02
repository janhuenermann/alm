#pragma once

#ifndef SUTHERLAND_HODGMAN
#define SUTHERLAND_HODGMAN

#ifdef __CUDACC__
#define SHARED __device__
#else
#define SHARED
#endif

template <typename scalar_t>
SHARED bool outside(const scalar_t *&p, const scalar_t *&a, const scalar_t *r) {
   return r[0] * (p[1] - a[1]) < r[1] * (p[0] - a[0]);
}

template <typename scalar_t>
SHARED void intersect(scalar_t *&p, const scalar_t *&a, const scalar_t *r, const scalar_t *&c, const scalar_t *&d) {
   scalar_t sx = d[0] - c[0], sy = d[1] - c[1];
   scalar_t t = ((c[0] - a[0]) * sy - (c[1] - a[1]) * sx) / (r[0] * sy - r[1] * sx);
   p[0] = a[0] + r[0] * t;
   p[1] = a[1] + r[1] * t;
}

template <typename scalar_t>
SHARED int64_t edge_clip(scalar_t *result, const scalar_t *polygon, const scalar_t *a, const scalar_t *b, const int64_t &n) {
   if (n <= 2) {
      return 0;
   }

   scalar_t r[2]; // precompute slope of line a -> b
   r[0] = b[0] - a[0];
   r[1] = b[1] - a[1];

   const scalar_t *v0, *v1;
   scalar_t *vout = result;
   bool p1, p2;

   #pragma unroll 4
   for(int i = 0; i < n; i++) {
      v0 = polygon + i*2;
      v1 = polygon + ((i+1)%n)*2;

      p1 = outside(v0, a, r);
      p2 = outside(v1, a, r);

      if (p1 != p2) {
         intersect(vout, a, r, v0, v1);
         vout += 2;
      }

      if (!p2) {
         vout[0] = v1[0];
         vout[1] = v1[1];
         vout += 2;
      }
   }

   return (vout - result)/2;
}

#define swap(x, y) ___tmp = b; b = a; a = ___tmp

template <typename scalar_t>
SHARED int64_t polygon_clip(scalar_t *result, scalar_t *tmp, const scalar_t *polygon1, const scalar_t *polygon2, const int64_t &n, const int64_t &m) {
   int64_t l = n;
   if (l <= 2) {
      return 0;
   }

   scalar_t *a = tmp, *b = result, *___tmp;
   if (m % 2 == 1) { // m - 1 is even => copy in result first
      swap(a, b);
   }

   // Copy from polygon1 into tmp
   l = edge_clip(a, polygon1, polygon2, polygon2+2, l);
   for (int i = 1; i < m - 1; ++i) {
      l = edge_clip(b, a, polygon2+2*i, polygon2+2*i+2, l);
      swap(a, b);
   }

   return edge_clip(b, a, polygon2+2*(m-1), polygon2, l);
}

template <typename scalar_t>
SHARED scalar_t shoelace(const scalar_t *polygon, const int64_t &n) {
   scalar_t area2 = 0;
   // x_n y_1 - x_1 y_n
   area2 += polygon[2*n] * polygon[1] - polygon[0] * polygon[2*n+1];
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