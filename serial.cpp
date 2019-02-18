#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"

#define density 0.0005
#define mass    0.01
#define cutoff  0.01

#define min_r   (cutoff / 100)
#define dt      0.0005

double size2;
int bin_size;
int num_bins;
int * bin_Ids;

typedef struct{
    int num_par;
    int num_nei;
    int * nei_id;
    int * par_id;
};

void init_bins( bin_dict * bins ) {
    int dx[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    int dy[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    for(int i = 0; i < num_bins; i++){
        bins[i].num_nei = 0;
        bins[i].nei_id = (int*) malloc(9 * sizeof(int));
        int x = i % bin_size;
        int y = (i - x) / bin_size;
        for(int k = 0; k < 9; k++){
            int new_x = x + dx[k];
            int new_y = y + dy[k];
            if (new_x >= 0 && new_y >= 0 && new_x < bin_size && new_y < bin_size) {
                int new_id = new_x + new_y * bin_size;
                bins[i].nei_id[bins[i].num_nei] = new_id;
                bins[i].num_nei++;
            }
        }
    }
}

void binning(bin_dict * bins, int n) {
    int i, id, idx;
    for(i = 0; i < num_bins; i++){
        bins[i].num_par = 0;
    }

    for(i = 0; i < n; i++){
        id = bin_Ids[i];
        idx = bins[id].num_par;
        bins[id].par_id[idx] = i;
        bins[id].num_par++;
    }

    return;
}

void apply_force_bin(particle_t * _particles, bin_dict * bins, int _binId, double * dmin, double * davg, int * navg) {
    bin_dict * bin = bins + _binId;         // make program work on specific bin with ID _binID

    for(int i = 0; i < bin->num_par; i++){
        for(int k = 0; k < bin->num_nei; k++){
            bin_dict * new_bin = bins + bin->nei_id[k];
            for(int j = 0; j < new_bin->num_par; j++){
                apply_force(_particles[bin->par_id[i]], _particles[new_bin->par_id[j]], dmin, davg, navg);
            }
        }
    }
}

int main( int argc, char **argv )
{
    int navg,nabsavg=0;
    double davg,dmin, absmin=1.0, absavg=0.0;

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    set_size(n);

    size2 = sqrt(density * n);
    bin_size = (int)ceil(size2 / cutoff);      // use cutoff to divide bins each with size cutoff
    num_bins = bin_size * bin_size;           // total number of bins in domain
    bin_Ids =  (int *) malloc(n * sizeof(int));

    bin_dict * bins = (bin_dict *) malloc(num_bins * sizeof(bin_dict));

    for(int i = 0; i < num_bins; i++){
        bins[i].par_id = (int*) malloc(n*sizeof(int));
    }

    init_bins(bins);

    init_particles( n, particles );
    for(int i = 0; i < n; i++){
        move(particles[i]);   // assign each particle to a bin
        particles[i].ax = 0;
        particles[i].ay = 0;
        bin_Ids[i] = (int)(floor(particles[i].x / cutoff) * bin_size
                             + floor(particles[i].y / cutoff));

    }

    binning(bins, n);    // calculate number of particles in each bin

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( ); // reads time using function in commons.cpp

    for(int step = 0; step < NSTEPS; step++){

        navg = 0;
        davg = 0.0;
        dmin = 1.0;
        //
        //  compute forces
        //
        for(int i = 0; i < n; i++){
            particles[i].ax = particles[i].ay = 0;    // initialize acceleration after each step
        }

        for(int i = 0; i < num_bins; i++){
            apply_force_bin(particles, bins, i, &dmin, &davg, &navg);    // apply forces in particles for bin by bin with only neighboring bins
        }

        //
        //  move particles
        //
        for(int i = 0; i < n; i++){
            move(particles[i]);
            particles[i].ax = particles[i].ay = 0;
            bin_Ids[i] = (int)(floor(particles[i].x / cutoff) * bin_size
                                 + floor(particles[i].y / cutoff));
        }

        binning(bins, n);            // reset number of particles in each bin and calculate again

        if(find_option( argc, argv, "-no" ) == -1)
        {
          //
          // Computing statistical data
          //
          if (navg) {
            absavg +=  davg/navg;
            nabsavg++;
          }
          if (dmin < absmin) absmin = dmin;

        }
    }
    simulation_time = read_timer( ) - simulation_time;

    printf( "n = %d, simulation time = %g seconds", n, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
      if (nabsavg) absavg /= nabsavg;
    //
    //  -The minimum distance absmin between 2 particles during the run of the simulation
    //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
    //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
    //
    //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
    //
    printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
    if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
    if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");

    //
    // Printing summary data
    //
    if( fsum)
        fprintf(fsum,"%d %g\n",n,simulation_time);

    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );
    free( particles );
    if( fsave )
        fclose( fsave );

    return 0;
}
