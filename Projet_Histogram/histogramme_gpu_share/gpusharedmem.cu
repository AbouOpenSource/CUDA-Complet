#include <stdio.h>
#define NBR 68
__global__ void histo_kernel(unsigned char *buffer,long size, unsigned int *histo){
	
	 __shared__ unsigned int temp[68];
        int dt = 32;
	temp[threadIdx.x]=0;
        int i = threadIdx.x + blockIdx.x *blockDim.x;
        int offset = blockDim.x *gridDim.x;
        while(i<size){
                
            if (buffer[i] >= 32 && buffer[i] < 97)
                   //  histo[buffer[i]-dt]++;
            	atomicAdd(&temp[buffer[i]-dt],1);
	    if (buffer[i] >=97 && buffer[i] <= 122)
                atomicAdd(&temp[buffer[i] -dt -32],1);
		   // histo[buffer[i] - dt - 32]++;
            if (buffer[i] > 122 && buffer[i] <= 127 )
                   // histo[buffer[i] - dt - 32 - 26]++;
		atomicAdd(&temp[buffer[i]-dt -32-26],1);
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
        /*For file input file and output file*/
	FILE *f_input;
        FILE *f_output;
	/*Will content the number of char in the input file*/
        long lSize;
        /*will content the file in char format*/
	char *buffer;
	/*Open the */
        f_input = fopen ( argv[1] ,"r" );
        f_output = fopen( argv[2],"w");


        if( !f_input ) perror(argv[1]),exit(1);


        fseek( f_input , 0L , SEEK_END);
        lSize = ftell( f_input );
        rewind( f_input );

       

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
	cudaMalloc( (void**)&dev_histo, NBR * sizeof( long));    
	cudaMemset( dev_histo, 0, NBR * sizeof( int ));    

	/*Define of the configuration for kernel running*/
 	cudaDeviceProp  proprieties;
        cudaGetDeviceProperties( &proprieties, 0  );
        int multiproc = proprieties.multiProcessorCount;
        dim3  blocks(multiproc*2,1,1);
        dim3  threads(NBR, 1, 1);



	histo_kernel<<<blocks,threads>>>( dev_buffer, lSize, dev_histo );


	/*Define histo vqriqble and copy on GPU global memory*/
	unsigned int histo[NBR];    
	cudaMemcpy( histo, dev_histo,NBR * sizeof( int ),cudaMemcpyDeviceToHost);
	int dt =32;	
	for(int i =0;i< NBR;i++){

        if((i>=0 && i<= 31 && (i+dt != 42) && (i+dt != 36)) || (i>58 && i<=64) )
            fprintf(f_output, "%c:%d\n",i+dt,histo[i]);

        if(i>31 && i<= 58 )
	    fprintf(f_output, "%c:%d\n",i+dt+32,histo[i]);
	
       // if(i> 58 && i <=64)
         //   fprintf(f_output, "%c:%d\n",i+dt,histo[i]);	
        
	if(i>64)
            fprintf(f_output, "%c:%d\n",i+dt+26,histo[i]);
 

	}
	
	/*Get event at the end of loop*/
	cudaEventRecord( stop, 0  );    
	cudaEventSynchronize( stop );
	float   elapsedTime;    
        cudaEventElapsedTime( &elapsedTime, start, stop );    
	printf( "Time of running :  %3.1f ms\n", elapsedTime );
		
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



