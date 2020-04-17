#include "stdio.h"
#define THREADS_PER_BLOCK 512
#define N (2048*2048)
__global__ void add(int *a, int *b, int *c){

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < N)
	c[index] = a[index] + b[index];
	printf('index: %d',threadIdx.x);
}
int main(void ){

	int *a,*b,*c; // host copies of a, b and c
	int *dev_a,*dev_b, *dev_c; // device copies of a, b and c
	int size = N * sizeof(int); // we need space for an integer


	//allocate device copies of a, b , c
	cudaMalloc((void**) &dev_a, size);
	cudaMalloc((void**) &dev_b, size);
	cudaMalloc((void**) &dev_c, size);

	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);

	//random_ints(a,N);
	//random_ints(b,N);
	for (int i= 0; i<N ; i++){
		a[i]=i;
		b[i]=i*2;
		}

	//copy inputs to device (GPU)
	cudaMemcpy(dev_a, a, size , cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
	
	// launch add() kernel on GPU, passing parameters
	add<<<N/THREADS_PER_BLOCK  , THREADS_PER_BLOCK >>> (dev_a,dev_b,dev_c);
	
	//copy device result back to host copy of c 
	cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
	/*for(int i =0; i<N; i++){
		printf("The value of the %d plus %d is : %d\n", a[i], b[i], c[i]);
		}*/


	free(a);
	free(b);
	free(c);
	
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
		
	return 0;
}
