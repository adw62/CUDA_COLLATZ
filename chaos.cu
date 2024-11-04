#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cassert>

__global__ void henon_kernel(double *xd_points, double *yd_points, double a, double b, int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform Henon map computation
    double x = xd_points[idx];
    double y = yd_points[idx];
    for (int i = 0; i < steps; ++i) {
        double x_new = 1 - a * x * x + y;
        double y_new = b * x;
        x = x_new;
        y = y_new;
    }
    xd_points[idx] = x;
    yd_points[idx] = y;
}

__global__ void tinkerbell_kernel(double *xd_points, double *yd_points, double a, double b, double c, double d, int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Define bounding box parameters
    double xmin = -200.0;
    double xmax = 200.0;
    double ymin = -200.0;
    double ymax = 200.0;

    // Perform Tinkerbell map computation
    double x = xd_points[idx];
    double y = yd_points[idx];
    for (int i = 0; i < steps; ++i) {
        double x_new = x * x - y * y  + a * x + b * y;
        double y_new = 2 * x * y + c * x + d * y;
        
        // Check if values exceed threshold
        if (fabs(x_new) > 1000000000 || fabs(y_new) > 1000000000) {
            // Set output values to 0 and return
            xd_points[idx] = 0.0;
            yd_points[idx] = 0.0;
            return;
        }

        x = x_new;
        y = y_new;
    }
    
    // Check if final values fall within bounding box
    if (x < xmin || x > xmax || y < ymin || y > ymax) {
        xd_points[idx] = 0.0;
        yd_points[idx] = 0.0;
    } else {
        xd_points[idx] = x;
        yd_points[idx] = y;
    }
}

__global__ void bogdanov_kernel(double *xd_points, double *yd_points, double eps, double k, double mew, int steps) {
    // 2D log map using doubles
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double xn = xd_points[idx];
    double yn = yd_points[idx];
    double xn1 = xd_points[idx];
    double yn1 = yd_points[idx];
    for (int s = 0; s < steps; ++s) {
        yn1 = yn + eps * yn + k * xn * (xn - 1) + mew * xn * yn;
        xn1 = xn + yn1;

        xn = xn1;
        yn = yn1;
    }
    xd_points[idx] = xn;
    yd_points[idx] = yn;
}

__global__
void logistics_kernel(double *xd_points, double r, int steps) {
    //1D log map using doubles
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double xn = xd_points[idx];
    double xn1 = xd_points[idx];
    for (int s = 0; s<steps; ++s) {
    xn1 = r*xn*(1-xn);
    xn = xn1;
    }
    xd_points[idx] = xn;
}

__global__ void coll_kernel(int *a, int *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int res = a[idx];
        int count = 1;
        while (res != 1 && count < 10000) {
            count++;
            res = (res % 2 == 0) ? res / 2 : 3 * res + 1;
        }
        b[idx] = (res == 1) ? count : 888888;
    }
}

void collatz(int *start_ints, int N) {
    int *d_start_ints, *d_result_ints;

    cudaMalloc((void **)&d_start_ints, N * sizeof(int));
    cudaMalloc((void **)&d_result_ints, N * sizeof(int));

    cudaMemcpy(d_start_ints, start_ints, N * sizeof(int), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    coll_kernel<<<gridSize, blockSize>>>(d_start_ints, d_result_ints, N);

    cudaMemcpy(start_ints, d_result_ints, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_start_ints);
    cudaFree(d_result_ints);
}

void bogdanov(double *x_points, int N, double *y_points, int M, double eps, double k,
 double mew, int steps) {
    assert(N==M);

    double* xd_points;
    double* yd_points;

    cudaMalloc((void **)&xd_points, N*sizeof(double));
    cudaMalloc((void **)&yd_points, M*sizeof(double));

    cudaMemcpy(xd_points, x_points, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(yd_points, y_points, M*sizeof(double), cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    bogdanov_kernel<<<gridSize, blockSize>>>(xd_points, yd_points, eps, k, mew, steps);

    cudaMemcpy(x_points, xd_points, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_points, yd_points, N*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(xd_points);
    cudaFree(yd_points);
}

void tinkerbell(double *x_points, int N, double *y_points, int M, double a, double b, double c, double d, int steps) {
    assert(N == M);
    
    double* xd_points;
    double* yd_points;

    cudaMalloc((void **)&xd_points, N * sizeof(double));
    cudaMalloc((void **)&yd_points, M * sizeof(double));

    cudaMemcpy(xd_points, x_points, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(yd_points, y_points, M * sizeof(double), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    tinkerbell_kernel<<<gridSize, blockSize>>>(xd_points, yd_points, a, b, c, d, steps);

    cudaMemcpy(x_points, xd_points, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_points, yd_points, M * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(xd_points);
    cudaFree(yd_points);
}

void logistics_map(double *x_points, int N, double r, int steps) {

    double* xd_points;

    cudaMalloc((void **)&xd_points, N*sizeof(double));
    
    cudaMemcpy(xd_points, x_points, N*sizeof(double), cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    logistics_kernel<<<gridSize, blockSize>>>(xd_points, r, steps);

    cudaMemcpy(x_points, xd_points, N*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(xd_points);
}

void henon(double *x_points, int N, double *y_points, int M, double a, double b, int steps) {
    assert(N == M);

    double* xd_points;
    double* yd_points;

    cudaMalloc((void **)&xd_points, N * sizeof(double));
    cudaMalloc((void **)&yd_points, M * sizeof(double));

    cudaMemcpy(xd_points, x_points, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(yd_points, y_points, M * sizeof(double), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    henon_kernel<<<gridSize, blockSize>>>(xd_points, yd_points, a, b, steps);

    cudaMemcpy(x_points, xd_points, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_points, yd_points, M * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(xd_points);
    cudaFree(yd_points);
}


