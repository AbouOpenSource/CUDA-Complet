#include <iostream>
#include <iomanip>
#include <cstdlib>

using std::cout;
int main( int argc , char* argv[]){
	long i, nsteps;
	double pi, step, sum =0.0;
	nsteps = 0; 
	if(argc> 1)
		nsteps = atol(argv[1]);
	if(nsteps <= 0 )
		nsteps = 100;
	step = (1.0)/((double)nsteps);
	for(i = 0; i < nsteps; ++i)
		{
			double x = ((double)i+0.5)*step;
			sum +=1.0/(1.0 + x * x);
		}
	pi = 4.0 *step *sum;
	cout << std::fixed;
	cout << "Pi is "<< std::setprecision(17) << pi << "\n";
}
