////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Template project which demonstrates the basics on how to setup a project
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);
typedef float DTYPE;
extern "C"
void computeGold(float *reference, float *idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel(DTYPE *out_data, unsigned int nsteps)
{
	const unsigned int tidb = threadIdx.x;
	const unsigned int tid  = blockIdx.x * blockDim.x +tidb;

	DTYPE step ;
	if(tid < nsteps ){
		step = (1.0)/((DTYPE)nsteps);
		DTYPE x = ((DTYPE) tid +0.5) * step ;
		out_data[tid] = (DTYPE) 1.0 / (1.0 + x*x);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
    bool bTestResult = true;

    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
//    int devID = findCudaDevice(argc, (const char **)argv);

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);


    unsigned int threads_per_block = 1024;
    unsigned int nsteps = atoi(argv[1]);
    unsigned int mem_size = sizeof(DTYPE) * nsteps;

    DTYPE *d_vector;
    checkCudaErrors(cudaMalloc((void **) &d_vector,mem_size)); 
	
    // setup execution parameter
    dim3  grid((nsteps+threads_per_block-1)/threads_per_block, 1, 1);
    dim3  threads(threads_per_block, 1, 1);

    // execute the kernel
    testKernel<<< grid, threads, 0 >>>(d_vector, nsteps);
    
    // create a variable in host side 
    DTYPE *h_vector;
    h_vector = (DTYPE *) malloc(mem_size);
    
    DTYPE sum=0;
    DTYPE step = (1.0)/((DTYPE)nsteps);


    cudaMemcpy(h_vector,d_vector,mem_size,cudaMemcpyDeviceToHost);
    for(int i =0 ;i< nsteps ; i++){
	    //sum all of value 
	    sum += h_vector[i];    	
    }
    
    
    printf("The values of Pi  is %f",sum*step*4.0);


    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

   
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
    
    //cleanup memory
    free(h_vector);
    checkCudaErrors(cudaFree(d_vector));

    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
