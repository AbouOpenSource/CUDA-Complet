#include<stdio.h>
#define START 32 //first char to make hist ascii code
#define STOP 127 //last char to make hist ascii code 
int main(int argc, char** argv){

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

        buffer =(char*) malloc(lSize);
        if( !buffer ) fclose(f_input),fputs("memory alloc fails",stderr),exit(1);


        if( 1!=fread( buffer , lSize, 1 , f_input) )
          fclose(f_input),free(buffer),fputs("entire read fails",stderr),exit(1);

	unsigned int histo[256];

	for(int i=0; i<256; i++)
        	histo[i]=0;
	
	/*Computing of the time*/
//	std::clock_t c_start = std::clock();
	
	
	/*Core of Algo compute a hist by increase hist array if char is occured**/
	for(int i=0; i< lSize ;i++){
		histo[buffer[i]]++;
	}
	/*End of excution*/
	
//	std::clock_t c_end = std::clock();
  //  	long time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
   //     std::cout << "CPU time used: " << time_elapsed_ms << " ms\n";



	for(int i=START; i<STOP;i++){
		printf("%c:%d\n",i,histo[i]);
	        fprintf(f_output, "%c:%d\n",i,histo[i]);

	}
	
        fclose(f_input);
        fclose(f_output);
        free(buffer);
	return 0;
}
