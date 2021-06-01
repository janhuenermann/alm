int64_t d_edge_clip(float *result, const float *polygon, const float *&a, const float *&b,
                    const float *d_result, float *d_polygon, float *&d_a, float *&d_b, const int64_t &n) {
   const float *v0, *v1, *d_vout = d_result;
   float *vout = result, *d_v0, *d_v1;
   bool p1, p2;

   for(int i = 0; i < n; i++) {
      v0 = polygon + i*2;
      v1 = polygon + ((i+1)%n)*2;
      
      d_v0 = d_polygon + i*2;
      d_v1 = d_polygon + ((i+1)%n)*2;

      p1 = outside(v0, a, b);
      p2 = outside(v1, a, b);

      if (p1 != p2) {
         d_intersect(vout, v0, v1, a, b, d_vout, d_v0, d_v1, d_a, d_b);
         vout += 2;
         d_vout += 2;
      }

      if (!p2) {
         vout[0] = v1[0];
         vout[1] = v1[1];
         d_v1[0] += d_vout[0];
         d_v1[1] += d_vout[1];
         vout += 2;
         d_vout += 2;
      }
   }

   return (vout - result)/2;
}


int64_t d_polygon_clip(float *tmp1, float *tmp2, const float *polygon1, const float *polygon2,
                       const float *d_result, float *d_polygon1, float *d_polygon2,
                       const int64_t &n, const int64_t &m) {
   float *out, tmp;
   const float *d_tmp;
   int64_t a_i, b_i;
   int64_t l = n;
   for (int i = 0; i < m; ++i) {
      if (l == 0) {
         return 0;
      }

      if (i == 0) {
         out = tmp1;
         tmp = polygon1;
         d_tmp = d_result;
      }
      else if (i % 2 == 1) {
         out = tmp2;
         tmp = tmp1;
      }
      else {
         out = tmp1;
         tmp = tmp2;
      }

      a_i = 2*i;
      b_i = 2*((i+1)%m);

      l = d_edge_clip(out, tmp, polygon2+a_i, polygon2+b_i,
                      d_tmp, d_polygon1, d_polygon2+a_i, d_polygon2+b_i, l);
   }

   return l;
}


void d_intersect(float *&p, const float *&a, const float *&b, const float *&c, const float *&d,
                 const float *&d_p, float *&d_a, float *&d_b, float *&d_c, float *&d_d) {
   float rx = b[0] - a[0], ry = b[1] - a[1];
   float sx = d[0] - c[0], sy = d[1] - c[1];
   float rs = rx * sy - ry * sx;
   float acx = c[0] - a[0], acy = c[1] - a[1];
   float t = (acx * sy - acy * sx) / rs;

   p[0] = a[0] + rx * t;
   p[1] = a[1] + ry * t;

   float dt  = rx * d_p[0] + ry * d_p[1];
   float drs = - dt * (acx * sy - acy * sx) / (rs * rs);
   float dsx = - dt * acy / rs + ry * drs;
   float dsy = dt * acx / rs + rx * drs;
   float drx = t * d_p[0] + sy * drs;
   float dry = t * d_p[1] + sx * drs;

   d_a[0] += - drx - dt * sy / rs + d_p[0];
   d_a[1] += - dry + dt * sx / rs + d_p[1];

   d_b[0] += drx;
   d_b[1] += dry;
   
   d_c[0] += - dsx + dt * sy / rs;
   d_c[1] += - dsy - dt * sx / rs;

   d_d[0] += dsx;
   d_d[1] += dsy;
}