#include<stdio.h>
#define START 32 //first char to make hist ascii code
#define STOP 127 //last char to make hist ascii code 
#define NBR_CHAR 68
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

//        printf("The size is : %li\n", lSize);

        buffer =(char*) malloc(lSize);
        if( !buffer ) fclose(f_input),fputs("memory alloc fails",stderr),exit(1);


        if( 1!=fread( buffer , lSize, 1 , f_input) )
          fclose(f_input),free(buffer),fputs("entire read fails",stderr),exit(1);

	unsigned int histo[NBR_CHAR];
	unsigned int dt =32; 
	for(int i=0; i<NBR_CHAR; i++)
        
		histo[i]=0;	
	/*Core of Algo compute a hist by increase hist array if char is occured**/
	for(int i=0; i< lSize ;i++){
		
            if (buffer[i] >= 32 && buffer[i] < 97)
                histo[buffer[i]-dt]++;
            if (buffer[i] >=97 && buffer[i] <= 122)
                histo[buffer[i] - dt - 32]++;
            if (buffer[i] > 122 && buffer[i] <= 127 )
                histo[buffer[i] - dt - 32 - 26]++;

	}	


	for(int i=0 ; i<NBR_CHAR ; i++){
	
        	if(i>=0 && i<= 31&& (i+dt != 42) && (i+dt != 36)){
			fprintf(f_output, "%c:%d\n",i+dt,histo[i]);
		//	printf("%d:%c:%d\n",i+dt,i+dt,histo[i]);
 
		}

        	if(i>31 && i<= 57 ){
            		fprintf(f_output, "%c:%d\n",i+dt+32,histo[i]);
			//printf("%d:%c:%d\n",i+dt+32,i+dt+32,histo[i]);
        	}

        	if(i> 57 && i <=64)
            		fprintf(f_output, "%c:%d\n",i+dt,histo[i]);
			//printf("%d:%c:%d\n",i+dt,i+dt,histo[i]);

        	if(i>64)
            		fprintf(f_output, "%c:%d\n",i+dt+26,histo[i]);
			//printf("%d:%c:%d\n",i+dt+26,i+dt+26,histo[i]);

	}
	
        fclose(f_input);
        fclose(f_output);
        free(buffer);
	return 0;
}
