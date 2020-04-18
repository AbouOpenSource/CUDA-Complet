#include <stdio.h>
#define FIRST_CHAR 32
#define LAST_CHAR 128
#define NBR 96
__global__ void histo_kernel(unsigned char *buffer,long size, unsigned int *histo){
	
	__shared__ unsigned int temp[256];
	temp[threadIdx.x]=0;
	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int offset = blockDim.x *gridDim.x;
	while(i<size){
		atomicAdd(&(histo[buffer[i]]),1);
		i+=offset;
	}
	
	__syncthreads();
	atomicAdd( &(histo[threadIdx.x]), temp[threadIdx.x] );

}



int main(int argc, char *argv[]){

//	unsigned char *buffer = (unsigned char *) big_random_block(SIZE);
	
	
	
	        if(argc <= 2){
        fprintf(stderr, "Arguments non valide");
        return 1;
        }
        FILE *f_input;
        FILE *f_output;

        long lSize;
        char *buffer;

        f_input = fopen ( argv[1] , "r" );
        f_output = fopen( argv[2],"w");
        if( !f_input ) perror(argv[1]),exit(1);


        fseek( f_input , 0L , SEEK_END);
        lSize = ftell( f_input );
        rewind( f_input );

        printf("The size is : %li", lSize);

        //buffer = calloc( 1, lSize+1 );
        buffer =(char*) malloc(lSize);
        if( !buffer ) fclose(f_input),fputs("memory alloc fails",stderr),exit(1);


       if( 1!=fread( buffer , lSize, 1 , f_input) )
          fclose(f_input),free(buffer),fputs("entire read fails",stderr),exit(1);	
	
	
	
	
	
	/*Create event for co;pute running time*/
	cudaEvent_t start, stop;
        cudaEventCreate( &start );
	cudaEventCreate( &stop );    
        /*Launch event to specify the start of running*/
	cudaEventRecord( start, 0);


	/*allocate device memory*/
	unsigned char *dev_buffer;
	unsigned int *dev_histo;
	/*Give space in Global memory of GPU to store different variable*/
	cudaMalloc( (void**)&dev_buffer, lSize);
	/*Copy from CPU Global memory to GPU Global memory*/
	cudaMemcpy( dev_buffer, buffer, lSize, cudaMemcpyHostToDevice  );    
	/*Create space for histo variable and initialize at 0 each slopt*/
	cudaMalloc( (void**)&dev_histo, 256 * sizeof( long));    
	cudaMemset( dev_histo, 0, 256 * sizeof( int ));
	


	cudaDeviceProp  prop;    
	cudaGetDeviceProperties( &prop, 0  );
	int blocks = prop.multiProcessorCount;    
	histo_kernel<<<blocks*2,256>>>( dev_buffer, lSize, dev_histo );


	/*Define histo vqriqble and copy on GPU global memory*/
	unsigned int histo[256];    
	cudaMemcpy( histo, dev_histo,256 * sizeof( int ),cudaMemcpyDeviceToHost);
	
	for(int i =FIRST_CHAR;i< LAST_CHAR;i++){
            printf("%c:%d\n",i,histo[i]);
            fprintf(f_output, "%c:%d\n",i,histo[i]);
        }
	
	/*Get event at the end of loop*/
	cudaEventRecord( stop, 0  );    
	cudaEventSynchronize( stop );
	float   elapsedTime;    
        cudaEventElapsedTime( &elapsedTime, start, stop );    
	printf( "Time to generate:  %3.1f ms\n", elapsedTime );
		
	/*Destroy event for running time*/
	cudaEventDestroy( start );    
	cudaEventDestroy( stop );    
	
	
	/*Free memory and close the files**/
	cudaFree( dev_histo );    
	cudaFree( dev_buffer );    
	fclose(f_input);
	fclose(f_output);
	free( buffer );
	return 0; 


	}



