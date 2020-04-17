#include "util.h"

#define SIZE (100*1024*1024)

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
	
	
	
	
	
	
	cudaEvent_t start, stop;
	HANDLE_ERROR( cudaEventCreate( &start ));
	HANDLE_ERROR( cudaEventCreate( &stop ));    
	HANDLE_ERROR( cudaEventRecord( start, 0));

	unsigned char *dev_buffer;
	unsigned int *dev_histo;

	HANDLE_ERROR( cudaMalloc( (void**)&dev_buffer, lSize));
	HANDLE_ERROR( cudaMemcpy( dev_buffer, buffer, lSize, cudaMemcpyHostToDevice ) );    
	HANDLE_ERROR( cudaMalloc( (void**)&dev_histo, 256 * sizeof( long )));    
	HANDLE_ERROR( cudaMemset( dev_histo, 0, 256 * sizeof( int )));

	cudaDeviceProp  prop;    
	HANDLE_ERROR( cudaGetDeviceProperties( &prop, 0 ) );
	int blocks = prop.multiProcessorCount;    
	histo_kernel<<<blocks*2,256>>>( dev_buffer, lSize, dev_histo );



	unsigned int histo[256];    
	HANDLE_ERROR( cudaMemcpy( histo, dev_histo,256 * sizeof( int ),cudaMemcpyDeviceToHost));
	for(int i =32;i< 128;i++){
            printf("%c:%d\n",i,histo[i]);
            fprintf(f_output, "%c:%d\n",i,histo[i]);
        }

	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );    
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	float   elapsedTime;    
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );    
	printf( "Time to generate:  %3.1f ms\n", elapsedTime );
		

	HANDLE_ERROR( cudaEventDestroy( start ) );    
	HANDLE_ERROR( cudaEventDestroy( stop ) );    
	
	cudaFree( dev_histo );    
	cudaFree( dev_buffer );    
	fclose(f_input);
	fclose(f_output);
	free( buffer );
	return 0; 


	}



