#include <vector>
#include <cmath>
#include <cstdio>
#include <omp.h>
#include <immintrin.h>
using namespace std;
typedef vector<vector<float>> matrix;

int main() {
    const int nx = 41;
    const int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2. / (nx - 1);
    double dy = 2. / (ny - 1);
    double dt = 0.01;
    double rho = 1;
    double nu = 0.02;

    vector<float> x(nx,0);
    vector<float> y(ny,0);

    float u[ny][nx];
    float v[ny][nx];
    float p[ny][nx];
    float b[ny][nx];
    
    matrix pn(ny, vector<float>(nx,0));
    matrix un(ny, vector<float>(nx,0));
    matrix vn(ny, vector<float>(nx,0));

#pragma omp parallel for
    for (int i = 0; i < nx; i++) {
        x[i] = i * dx;
    }

#pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        y[j] = j * dy;
    }

    __m256 dtvec  = _mm256_set1_ps(dt);
    __m256 dxvec  = _mm256_set1_ps(dx);
    __m256 dyvec  = _mm256_set1_ps(dy);
    __m256 one    = _mm256_set1_ps(1);
    __m256 two    = _mm256_set1_ps(2);
    __m256 rhovec = _mm256_set1_ps(rho);
    __m256 nuvec  = _mm256_set1_ps(nu);

    __m256 dxx2   = _mm256_mul_ps(two, dxvec);
    __m256 dyx2   = _mm256_mul_ps(two, dyvec);

    __m256 dx2   = _mm256_pow_ps(dxvec, two);
    __m256 dy2   = _mm256_pow_ps(dyvec, two);

    __m256 dtdx  = _mm256_div_ps(dtvec, dxvec);
    __m256 dtdy  = _mm256_div_ps(dtvec, dyvec);
    __m256 dtrx  = _mm256_div_ps(dtvec, _mm256_mul_ps(_mm256_mul_ps(two, rhovec), dxvec));
    __m256 dtdx2 = _mm256_div_ps(dtvec, dx2);
    __m256 dtdy2 = _mm256_div_ps(dtvec, dy2);
    __m256 nux2  = _mm256_mul_ps(nuvec, dtdx2);
    __m256 nuy2  = _mm256_mul_ps(nuvec, dtdy2);

    for (int n = 0; n < nt; n++) {
#pragma omp parallel for
        for (int j = 1; j < ny-1; j++) {
            for (int i = 1; i < nx-1; i+=8) {
                __m256 ujn1  = _mm256_load_ps(&un[j-1][i]);
                __m256 ujp1  = _mm256_load_ps(&un[j+1][i]);
                __m256 uin1  = _mm256_load_ps(&un[j][i-1]);
                __m256 uip1  = _mm256_load_ps(&un[j][i+1]);

                __m256 vjn1  = _mm256_load_ps(&vn[j-1][i]);
                __m256 vjp1  = _mm256_load_ps(&vn[j+1][i]);
                __m256 vin1  = _mm256_load_ps(&vn[j][i-1]);
                __m256 vip1  = _mm256_load_ps(&vn[j][i+1]);

                __m256 uip1_n_uin1 = _mm256_sub_ps(uip1, uin1);
                __m256 ujp1_n_ujn1 = _mm256_sub_ps(ujp1, ujn1);
                __m256 vip1_n_vin1 = _mm256_sub_ps(vip1, vin1);
                __m256 vjp1_n_vjn1 = _mm256_sub_ps(vjp1, vjn1);

                __m256 bvec;
                bvec = _mm256_add_ps(_mm256_div_ps(uip1_n_uin1, dxx2), _mm256_div_ps(vjp1_n_vjn1, dyx2));
                bvec = _mm256_sub_ps(bvec, _mm256_pow_ps(_mm256_div_ps(uip1_n_uin1, dxx2), two));
                bvec = _mm256_sub_ps(bvec, _mm256_mul_ps(two, _mm256_mul_ps(_mm256_div_ps(ujp1_n_ujn1, dyx2), _mm256_div_ps(vip1_n_vin1, dxx2))));
                bvec = _mm256_sub_ps(bvec, _mm256_pow_ps(_mm256_div_ps(vjp1_n_vjn1, dyx2), two));
                bvec = _mm256_mul_ps(rhovec, _mm256_mul_ps(_mm256_rcp_ps(dtvec), bvec));
                
                _mm256_store_ps(b[j]+i,bvec);
            }
        }
        for (int it = 0; it < nit; it++) {
#pragma omp parallel for
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    pn[j][i] = p[j][i];
                }
            }
#pragma omp parallel for
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i+=8) {
                    __m256 bvec  = _mm256_load_ps(&b[j][i]);

                    __m256 pin1  = _mm256_load_ps(&pn[j][i-1]);
                    __m256 pip1  = _mm256_load_ps(&pn[j][i+1]);
                    __m256 pjn1  = _mm256_load_ps(&pn[j-1][i]);
                    __m256 pjp1  = _mm256_load_ps(&pn[j+1][i]);
                    __m256 pvec;
                    pvec =                     _mm256_mul_ps(dy2, _mm256_add_ps(pip1, pin1));
                    pvec = _mm256_add_ps(pvec, _mm256_mul_ps(dx2, _mm256_add_ps(pjp1, pjn1)));
                    pvec = _mm256_sub_ps(pvec, _mm256_mul_ps(bvec, _mm256_mul_ps(dx2, dy2)));
                    pvec = _mm256_div_ps(pvec, _mm256_mul_ps(two, _mm256_add_ps(dx2, dy2)));
                
                    _mm256_store_ps(p[j]+i,pvec);
                }
            }
#pragma omp parallel for
            for (int j = 0; j < ny; j++) {
                p[j][nx-1] = p[j][nx-2];
            }
#pragma omp parallel for
            for (int i = 0; i < nx; i++) {
                p[0][i] = p[1][i];
            }
#pragma omp parallel for
            for (int j = 0; j < ny; j++) {
                p[j][0] = p[j][1];
            }
#pragma omp parallel for
            for (int i = 0; i < nx; i++) {
                p[ny-1][i] = 0;
            }
        }
#pragma omp parallel for
        for (int j = 1; j < ny-1; j++) {
            for (int i = 1; i < nx-1; i++) {
                un[j][i] = u[j][i];
                vn[j][i] = v[j][i];
            }
        }
        
#pragma omp parallel for
        for (int j = 1; j < ny-1; j++) {
            for (int i = 1; i < nx-1; i+=8) {
                __m256 pin1  = _mm256_load_ps(&un[j][i-1]);
                __m256 pip1  = _mm256_load_ps(&un[j][i+1]);
                __m256 pjn1  = _mm256_load_ps(&un[j-1][i]);
                __m256 pjp1  = _mm256_load_ps(&un[j+1][i]);

                __m256 unvec = _mm256_load_ps(&un[j][i]);
                __m256 ujn1  = _mm256_load_ps(&un[j-1][i]);
                __m256 ujp1  = _mm256_load_ps(&un[j+1][i]);
                __m256 uin1  = _mm256_load_ps(&un[j][i-1]);
                __m256 uip1  = _mm256_load_ps(&un[j][i+1]);
                __m256 uvec;
                uvec = _mm256_sub_ps(unvec, _mm256_mul_ps(_mm256_mul_ps(unvec, dtdx), _mm256_sub_ps(unvec, uin1)));
                uvec = _mm256_sub_ps(uvec,  _mm256_mul_ps(_mm256_mul_ps(unvec, dtdy), _mm256_sub_ps(unvec, ujn1)));
                uvec = _mm256_sub_ps(uvec,  _mm256_mul_ps(dtrx, _mm256_sub_ps(pip1, pin1)));
                uvec = _mm256_add_ps(uvec,  _mm256_mul_ps(nux2, _mm256_add_ps(_mm256_sub_ps(uip1, _mm256_mul_ps(two, unvec)), uin1)));
                uvec = _mm256_add_ps(uvec,  _mm256_mul_ps(nuy2, _mm256_add_ps(_mm256_sub_ps(ujp1, _mm256_mul_ps(two, unvec)), ujn1)));
                
                __m256 vnvec = _mm256_load_ps(&vn[j][i]);
                __m256 vjn1  = _mm256_load_ps(&vn[j-1][i]);
                __m256 vjp1  = _mm256_load_ps(&vn[j+1][i]);
                __m256 vin1  = _mm256_load_ps(&vn[j][i-1]);
                __m256 vip1  = _mm256_load_ps(&vn[j][i+1]);
                __m256 vvec;
                vvec = _mm256_sub_ps(vnvec, _mm256_mul_ps(_mm256_mul_ps(vnvec, dtdx), _mm256_sub_ps(vnvec, vin1)));
                vvec = _mm256_sub_ps(vvec,  _mm256_mul_ps(_mm256_mul_ps(vnvec, dtdy), _mm256_sub_ps(vnvec, vjn1)));
                vvec = _mm256_sub_ps(vvec,  _mm256_mul_ps(dtrx, _mm256_sub_ps(pjp1, pjn1)));
                vvec = _mm256_add_ps(vvec,  _mm256_mul_ps(nux2, _mm256_add_ps(_mm256_sub_ps(vip1, _mm256_mul_ps(two, vnvec)), vin1)));
                vvec = _mm256_add_ps(vvec,  _mm256_mul_ps(nuy2, _mm256_add_ps(_mm256_sub_ps(vjp1, _mm256_mul_ps(two, vnvec)), vjn1)));
                
                _mm256_store_ps(u[j]+i,uvec);
                _mm256_store_ps(v[j]+i,vvec);
            }
        }
#pragma omp parallel for
        for (int j = 0; j < ny; j++) {
            u[j][0] = 0;
            u[j][nx-1] = 0;
            v[j][0] = 0;
            v[j][nx-1] = 0;
        }
#pragma omp parallel for
        for (int i = 0; i < nx; i++) {
            u[0][i] = 0;
            u[ny-1][i] = 1;
            v[0][i] = 0;
            v[ny-1][i] = 0;
        }
    }
}
