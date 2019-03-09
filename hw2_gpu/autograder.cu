#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include "common.h"


#define MAX_ENTRIES 100 

//
//  benchmarking program
//
int main( int argc, char **argv )
{
    int n[MAX_ENTRIES],i,count=0,num;
    double t[MAX_ENTRIES],slope[MAX_ENTRIES-1],ss[MAX_ENTRIES],grade,ssgrade,ss_avg;
    double lt[MAX_ENTRIES],ln[MAX_ENTRIES],b2,sx=0.0,sx2=0.0,sxy=0.0,sy=0.0;

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help \n" );
        printf( "-s <filename> to specify name of summary file \n");
        printf( "-v to specify what to autograde (gpu) \n" );
        return 0;
    }
    
    char *savename = read_string( argc, argv, "-s", NULL );
    FILE *fread = savename ? fopen( savename, "r" ) : NULL;

    char *autoname = read_string( argc, argv, "-v", NULL );
     
    if (strcmp(autoname,"gpu")==0){
      if(fread) 
        while( fscanf (fread,"%d %lf",&n[count],&t[count]) != EOF ) 
          count++;
      num = count/2;     

      for (i=0; i<num-1;i++) {
        slope [i] = ( log(t[i+1]) - log(t[i]) ) / ( log(n[i+1]*1.0) - log(n[i]*1.0) );
      }
      for (i=0; i<num; i++) {
        lt[i] = log(t[i]);
        ln[i] = log(n[i]*1.0);
      }
      for (i=0; i<num; i++) {
        sx += ln[i];
        sy += lt[i];
        sxy += ln[i]*lt[i];
        sx2 += ln[i]*ln[i];
      }

      b2 = (sxy - (sx * sy)/ (num * 1.0) ) / (sx2 - (sx * sx)/ (num * 1.0));
 
      printf("\nGPU code is O(N^slope)");
      printf("\nSlope estimates are :");
      for (i=0; i<num-1; i++){
        printf(" %lf",slope[i]);
      }
      printf("\nSlope estimate for line fit is: %lf\n", b2);
	  
	  if (b2 < 1.2) grade = 40.00;
	  else if (b2 < 1.4) grade = (1.4-b2)/(0.2) * 40.00;
	  else grade =0.0;
	  
      for (i=0; i<=num;i++) {
         ss[i] = t[i+num]/t[i];
      }
 
      printf("\nGPU speedup estimates are :\n");
      for (i=0; i<num; i++){
        printf(" %7.2lf",ss[i]);
      }
      printf(" (speedup)\n");

	ss_avg=0.0;
	  for (i=0; i<num; i++){
        ss_avg+=ss[i];
      }
	  ss_avg/=num;
	  
	  printf("Average speedup: %7.2lf \n\n",ss_avg);

	  if (ss_avg > 26) ssgrade=60.00;
	  else if (ss_avg > 18) ssgrade=40.00 + 20.00 * (ss_avg-18)/18; 
	  else if (ss_avg > 14) ssgrade= (ss_avg-14)/14 * 40.00;
	  else ssgrade =0;
	  
	if (grade > 0)
	  grade= grade + ssgrade;
	  
	  printf("\nGPU Total Grade = %7.2f\n\n",grade);
    }

    fclose( fread );
    
    return 0;
}
