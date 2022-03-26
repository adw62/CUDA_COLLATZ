#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cassert>

__global__
void henon_kernel(double *xd_points, double *yd_points, double a, double b, int steps) {
    //2D log map using doubles
    int i = blockIdx.x;
    double xn = xd_points[i];
    double yn = yd_points[i];
    double xn1 = xd_points[i];
    double yn1 = yd_points[i];
    for (int s = 0; s<steps; ++s) {
    xn1 = 1 - xn*xn*a + yn;
    yn1 = b*xn;
    xn = xn1;
    yn = yn1;
    }
    xd_points[i] = xn;
    yd_points[i] = yn;
}

__global__
void tinkerbell_kernel(double *xd_points, double *yd_points, double a, double b,
 double c, double d, int steps) {
    //2D log map using doubles
    int i = blockIdx.x;
    double xn = xd_points[i];
    double yn = yd_points[i];
    double xn1 = xd_points[i];
    double yn1 = yd_points[i];
    for (int s = 0; s<steps; ++s) {
    xn1 = xn*xn - yn*yn + a*xn + b*yn;
    yn1 = 2*xn*yn + c*xn + d*yn;
    xn = xn1;
    yn = yn1;
    }
    xd_points[i] = xn;
    yd_points[i] = yn;
}

__global__
void bogdanov_kernel(double *xd_points, double *yd_points, double eps, double k,
 double mew, int steps) {
    //2D log map using doubles
    int i = blockIdx.x;
    double xn = xd_points[i];
    double yn = yd_points[i];
    double xn1 = xd_points[i];
    double yn1 = yd_points[i];
    for (int s = 0; s<steps; ++s) {
    yn1 = yn + eps*yn + k*xn*(xn-1) + mew*xn*yn;
    xn1 = xn+yn1;

    xn = xn1;
    yn = yn1;
    }
    xd_points[i] = xn;
    yd_points[i] = yn;
}

__global__
void logistics_kernel(double *xd_points, double r, int steps) {
    //1D log map using doubles
    int i = blockIdx.x;
    double xn = xd_points[i];
    double xn1 = xd_points[i];
    for (int s = 0; s<steps; ++s) {
    xn1 = r*xn*(1-xn);
    xn = xn1;
    }
    xd_points[i] = xn;
}

__global__
void coll_kernel(int *a, int *b, int N) {
    //1D log map using ints with extra rules
    int i = blockIdx.x;
    if (i<N) {
	int count = 1;
	int res = a[i];
	bool running = true;
	while (running){
	    count += 1;
	    if (res%2 == 0) { 
		res = res/2;
	    } else {
		res = 3*res+1;
	    }
            if (res == 1){
                b[i] = count;
		running = false;
	    }
	    if (count >= 10000){
	        b[i] = 888888;
	        running = false;
	    } 
        }
    }
}

void collatz(int *start_ints, int N) {
    int ha[N], hb[N];
    int *da, *db;
    cudaMalloc((void **)&da, N*sizeof(int));
    cudaMalloc((void **)&db, N*sizeof(int));

    for (int i = 0; i<N; ++i) {
        ha[i] = start_ints[i];
    }

    cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);
    coll_kernel<<<N, 1>>>(da, db, N);
    cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i<N; ++i) {
        start_ints[i] = hb[i];
    }
    cudaFree(da);
    cudaFree(db);
}

void bogdanov(double *x_points, int N, double *y_points, int M, double eps, double k,
 double mew, int steps) {
    assert(N==M);
    double* xh_points = new double[N]();
    double* yh_points = new double[M]();

    double* xd_points = new double[N]();
    double* yd_points = new double[M]();

    cudaMalloc((void **)&xd_points, N*sizeof(double));
    cudaMalloc((void **)&yd_points, M*sizeof(double));

    for (int i = 0; i<N; ++i) {
	    xh_points[i] = x_points[i];
    }
    for (int i = 0; i<M; ++i) {
	    yh_points[i] = y_points[i];
    }

    cudaMemcpy(xd_points, xh_points, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(yd_points, yh_points, M*sizeof(double), cudaMemcpyHostToDevice);

    bogdanov_kernel<<<N, 1>>>(xd_points, yd_points, eps, k, mew, steps);

    cudaMemcpy(xh_points, xd_points, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(yh_points, yd_points, N*sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i<N; ++i) {
        x_points[i] = xh_points[i];
    }
    for (int i = 0; i<N; ++i) {
	y_points[i] = yh_points[i];
    }

    cudaFree(xd_points);
    cudaFree(yd_points);
}

void tinkerbell(double *x_points, int N, double *y_points, int M, double a, double b,
 double c, double d, int steps) {
    assert(N==M);
    double* xh_points = new double[N]();
    double* yh_points = new double[M]();

    double* xd_points = new double[N]();
    double* yd_points = new double[M]();

    cudaMalloc((void **)&xd_points, N*sizeof(double));
    cudaMalloc((void **)&yd_points, M*sizeof(double));

    for (int i = 0; i<N; ++i) {
	    xh_points[i] = x_points[i];
    }
    for (int i = 0; i<M; ++i) {
	    yh_points[i] = y_points[i];
    }

    cudaMemcpy(xd_points, xh_points, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(yd_points, yh_points, M*sizeof(double), cudaMemcpyHostToDevice);

    tinkerbell_kernel<<<N, 1>>>(xd_points, yd_points, a, b, c, d, steps);

    cudaMemcpy(xh_points, xd_points, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(yh_points, yd_points, N*sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i<N; ++i) {
        x_points[i] = xh_points[i];
    }
    for (int i = 0; i<N; ++i) {
	y_points[i] = yh_points[i];
    }

    cudaFree(xd_points);
    cudaFree(yd_points);
}

void logistics_map(double *x_points, int N, double r, int steps) {
    double* xh_points = new double[N]();

    double* xd_points = new double[N]();

    cudaMalloc((void **)&xd_points, N*sizeof(double));

    for (int i = 0; i<N; ++i) {
	    xh_points[i] = x_points[i];
    }

    cudaMemcpy(xd_points, xh_points, N*sizeof(double), cudaMemcpyHostToDevice);

    logistics_kernel<<<N, 1>>>(xd_points, r, steps);

    cudaMemcpy(xh_points, xd_points, N*sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i<N; ++i) {
        x_points[i] = xh_points[i];
    }

    cudaFree(xd_points);
}

void henon(double *x_points, int N, double *y_points, int M, double a, double b, int steps) {
    assert(N==M);
    double* xh_points = new double[N]();
    double* yh_points = new double[M]();

    double* xd_points = new double[N]();
    double* yd_points = new double[M]();

    cudaMalloc((void **)&xd_points, N*sizeof(double));
    cudaMalloc((void **)&yd_points, M*sizeof(double));

    for (int i = 0; i<N; ++i) {
	    xh_points[i] = x_points[i];
    }
    for (int i = 0; i<M; ++i) {
	    yh_points[i] = y_points[i];
    }

    cudaMemcpy(xd_points, xh_points, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(yd_points, yh_points, M*sizeof(double), cudaMemcpyHostToDevice);

    henon_kernel<<<N, 1>>>(xd_points, yd_points, a, b, steps);

    cudaMemcpy(xh_points, xd_points, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(yh_points, yd_points, N*sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i<N; ++i) {
        x_points[i] = xh_points[i];
    }
    for (int i = 0; i<N; ++i) {
	y_points[i] = yh_points[i];
    }

    cudaFree(xd_points);
    cudaFree(yd_points);
}


