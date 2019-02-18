#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"

#define density 0.0005
#define cutoff  0.01
#define PARICLE_BIN(p) (int)(floor(p.x / cutoff) * bin_size
                             + floor(p.y / cutoff));

double size2;
int bin_size;
int num_bins;
int * bin_Ids;

typedef struct{
    int num_par;
    int num_nei;
    int * nei_id;
    int * par_id;
} bin;

void init_bins( bin * bins ) {
    int dx[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    int dy[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    int x, y, i, k, next_x, next_y, new_id;
    for(i = 0; i < num_bins; i++){
        bins[i].num_nei = 0;
        bins[i].nei_id = (int *) malloc(9 * sizeof(int));
        x = i % bin_size;
        y = (i - x) / bin_size;
        for(k = 0; k < 9; k++){
            next_x = x + dx[k];
            next_y = y + dy[k];
            if (next_x >= 0 && next_y >= 0 && next_x < bin_size && next_y < bin_size) {
                new_id = next_x + next_y * bin_size;
                bins[i].nei_id[bins[i].num_nei] = new_id;
                bins[i].num_nei++;
            }
        }
    }

    return;
}

void binning(bin * bins, int n) {
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

void apply_force_bin(particle_t * _particles, bin * bins, int _binId, double * dmin, double * davg, int * navg) {
    bin * cur_bin = bins + _binId;
    bin * new_bin;
    int i, k, j, par_cur, par_nei;
    for(i = 0; i < cur_bin->num_par; i++){
        for(k = 0; k < cur_bin->num_nei; k++){
            new_bin = bins + cur_bin->nei_id[k];
            for(j = 0; j < new_bin->num_par; j++){
                par_cur = cur_bin->par_id[i];
                par_nei = new_bin->par_id[j];
                apply_force(_particles[par_cur], _particles[par_nei], dmin, davg, navg);
            }
        }
    }

    return;
}

int main( int argc, char **argv )
{
    int navg, nabsavg = 0;
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

    particle_t * particles = (particle_t *) malloc(n * sizeof(particle_t));

    set_size(n);

    size2 = sqrt(density * n);
    bin_size = (int)ceil(size2 / cutoff);
    num_bins = bin_size * bin_size;
    bin_Ids =  (int *) malloc(n * sizeof(int));
    bin * bins = (bin *) malloc(num_bins * sizeof(bin));

    for(int i = 0; i < num_bins; i++){
        bins[i].par_id = (int *) malloc(n * sizeof(int));
    }

    init_bins(bins);
    init_particles(n, particles);

    for(int i = 0; i < n; i++){
        move(particles[i]);
        particles[i].ax = particles[i].ay = 0;
        bin_Ids[i] = PARICLE_BIN(particles[i]);
    }

    binning(bins, n);

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
            particles[i].ax = particles[i].ay = 0;
        }

        for(int i = 0; i < num_bins; i++){
            apply_force_bin(particles, bins, i, &dmin, &davg, &navg);
        }

        //
        //  move particles
        //
        for(int i = 0; i < n; i++){
            move(particles[i]);
            particles[i].ax = particles[i].ay = 0;
            bin_Ids[i] = PARICLE_BIN(particles[i]);
        }

        binning(bins, n);            // reset number of particles in each bin and calculate again

        if(find_option( argc, argv, "-no" ) == -1)
        {
          //
          // Computing statistical data
          //
          if(navg){
            absavg +=  davg/navg;
            nabsavg++;
          }

          if(dmin < absmin) absmin = dmin;

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
