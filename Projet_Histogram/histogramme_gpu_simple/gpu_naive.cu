#include <stdio.h>

#define START 32 
#define END 126
#define NBR 68 

__global__ void histo_kernel(unsigned char *buffer,long size, unsigned int *histo){

	int dt = 32;
	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int stride = blockDim.x *gridDim.x;
	while(i<size){
		
		if (buffer[i] >= 32 && buffer[i] < 97)
                
            		atomicAdd(&(histo[buffer[i]-dt]),1);	
		if (buffer[i] >=97 && buffer[i] <= 122)
                	atomicAdd(&(histo[buffer[i]-dt-32]),1);
	
            	if (buffer[i] > 122 && buffer[i] <= 127 )
                	atomicAdd(&(histo[buffer[i]-dt -32 -26]),1);
			
		
			i+=stride;
	}

}



int main(int argc, char *argv[]){


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
 	cudaEventCreate( &start );
	cudaEventCreate( &stop );    
	cudaEventRecord( start, 0);

	unsigned char *dev_buffer;
	unsigned int *dev_histo;

	 cudaMalloc( (void**)&dev_buffer, lSize);
	 cudaMemcpy( dev_buffer, buffer, lSize, cudaMemcpyHostToDevice );    
	 cudaMalloc( (void**)&dev_histo, 256 * sizeof( long ));    
	 cudaMemset( dev_histo, 0, 256 * sizeof( int ));

	cudaDeviceProp  prop;    
	cudaGetDeviceProperties( &prop, 0  );
	int multiproc = prop.multiProcessorCount;    
        dim3  blocks(multiproc*2,1,1);
        dim3  threads(NBR, 1, 1);

	histo_kernel<<<blocks,threads>>>( dev_buffer, lSize, dev_histo );



	unsigned int histo[NBR];    

	cudaMemcpy( histo, dev_histo,NBR * sizeof( int ),cudaMemcpyDeviceToHost);
	int dt =32;
	for(int i =0;i< 68;i++){
		
        if(i>=0 && i<= 31&& (i+dt != 42) && (i+dt != 36)){
            printf("%c:%d\n",i+dt,histo[i]);
        }

        if(i>31 && i<= 57 ){
            printf("%c:%d\n",i+dt+32,histo[i]);
        }

        if(i> 57 && i <=64)
            printf("%c:%d\n",i+dt,histo[i]);

        if(i>64)
            printf("%c:%d\n",i+dt+26,histo[i]);
	
	}
	cudaEventRecord( stop, 0 ) ;    
	cudaEventSynchronize( stop );
	float   elapsedTime;    
	cudaEventElapsedTime( &elapsedTime, start, stop  );    
	printf( "Time to generate:  %3.1f ms\n", elapsedTime );


	cudaEventDestroy( start ) ;    
	cudaEventDestroy( stop );    


	/*Free space*/
	cudaFree( dev_histo );    
	cudaFree( dev_buffer );
	fclose(f_input);
	fclose(f_output);
	free(buffer);
	return 0; 


}
