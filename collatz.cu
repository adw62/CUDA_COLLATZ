#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__
void coll_kernel(int *a, int *b, int N) {
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

void collatz(int *invec, int N) {
    int ha[N], hb[N];

    //
    // Create corresponding int arrays on the GPU.
    // ('d' stands for "device".)
    //
    int *da, *db;
    cudaMalloc((void **)&da, N*sizeof(int));
    cudaMalloc((void **)&db, N*sizeof(int));

    //
    // Initialise the input data on the CPU.
    //
    for (int i = 0; i<N; ++i) {
        ha[i] = invec[i];
    }

    //
    // Copy input data to array on GPU.
    //
    cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice); 

    //
    // Launch GPU code with N threads, one per
    // array element.
    //
    coll_kernel<<<N, 1>>>(da, db, N);

    //
    // Copy output array from GPU back to CPU.
    //
    cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i<N; ++i) {
        invec[i] = hb[i];
    }

    //
    // Free up the arrays on the GPU.
    //
    cudaFree(da);
    cudaFree(db);

}


