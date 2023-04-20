#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], j[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    j[i] = i;
  }
  for(int i=0; i<N; i++) {
    __m256 ivec = _mm256_set1_ps(i);
    __m256 jvec = _mm256_load_ps(j);
    __m256 mask = _mm256_cmp_ps(jvec,ivec,_CMP_NEQ_UQ);

    __m256 xvec = _mm256_load_ps(x);
    __m256 yvec = _mm256_load_ps(y);
    __m256 xivec = _mm256_set1_ps(x[i]);
    __m256 yivec = _mm256_set1_ps(y[i]);
    __m256 rxvec = _mm256_sub_ps(xivec,xvec);
    __m256 ryvec = _mm256_sub_ps(yivec,yvec);

    __m256 rx2vec = _mm256_mul_ps(rxvec,xvec);
    __m256 ry2vec = _mm256_mul_ps(ryvec,yvec);
    __m256 rvec = _mm256_add_ps(rx2vec,ry2vec);
    rvec = _mm256_rsqrt_ps(rvec);
    __m256 mvec = _mm256_load_ps(m);
    for (int k=0; k<3; k++) {
      mvec = _mm256_mul_ps(mvec,rvec);
    }
    __m256 mrx = _mm256_mul_ps(mvec,rxvec);
    __m256 mry = _mm256_mul_ps(mvec,ryvec);
    __m256 zero = _mm256_setzero_ps();
    mrx = _mm256_blendv_ps(zero,mrx,mask);
    mry = _mm256_blendv_ps(zero,mry,mask);

    __m256 mrx_p, mry_p;
    for (int k=0; k<3; k++) {
      mrx_p = _mm256_permute2f128_ps(mrx,mrx,1);
      mry_p = _mm256_permute2f128_ps(mry,mry,1);
      mrx = _mm256_hadd_ps(mrx,mrx_p);
      mry = _mm256_hadd_ps(mry,mry_p);
    }
    __m256 fxvec = _mm256_sub_ps(zero,mrx);
    __m256 fyvec = _mm256_sub_ps(zero,mry);

    float fxi[N], fyi[N];
    _mm256_store_ps(fxi,fxvec);
    _mm256_store_ps(fyi,fyvec);
    fx[i] = fxi[i];
    fy[i] = fyi[i];

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
