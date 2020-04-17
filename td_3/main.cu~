#include "stdio.h"
__global__ void add  (int *a, int *b, int *c){
        *c =  *a + *b;
}


int main(void ){

	int a,b,c; // host copies of a, b and c
	int *dev_a,*dev_b, *dev_c; // device copies of a, b and c
	int size = sizeof(int); // we need space for an integer


	//allocate device copies of a, b , c
	cudaMalloc((void**) &dev_a, size);
	cudaMalloc((void**) &dev_b, size);
	cudaMalloc((void**) &dev_c, size);

	a = 2;
	b = 7;

	//copy inputs to device (GPU)
	cudaMemcpy(dev_a, &a, size , cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice);
	
	// launch add() kernel on GPU, passing parameters
	add<<<1,1>>> (dev_a,dev_b,dev_c);
	
	//copy device result back to host copy of c 
	cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);
	
	printf("The value of c is %d\n",c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
		
 
}
