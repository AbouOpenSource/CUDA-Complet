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
#include <iostream>
#include <iomanip>
#include <cstdlib>

using std::cout;

// includes CUDA
#include <cuda_runtime.h>
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

extern "C"
void computeGold(float *reference, float *idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
//Here my function will be define 

__global__ void integral(int argc, char **argv){
	long i, nsteps;
	double pi, step, sum =0.0;
	nsteps = 0; 

	if(nsteps <= 0 )
		nsteps = 100;
	step = (1.0)/((double)nsteps);
	for(i = 0; i < nsteps; ++i)
		{
			double x = ((double)i+0.5)*step;
			sum +=1.0/(1.0 + x * x);
		}
	// in this step he resume the formula by doing the operation with the width one time with the sum of the height 	
	pi = 4.0 *step *sum;
 }



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
		
		int *a, *b, *c;
		int *dev_a, *dev_b, *dev_c;
		
		int size = sizeof( double );
		printf("The size is  : %d",size);
		integral<<<1,1>>> (argc, argv);
}

