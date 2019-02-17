#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"

//definition from commons
#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

int bin_size;
int bin_num;
int * particle_bin; //which particle in which bin

double width;

typedef struct{
    int num_par;
    int num_nei;
    int * par_id;
    int * nei_id; //neighbor bin
} bin; //which bin has which particle


//map bins back to particles after each move
void set_bin(int n, bin * bins){

    int i = 0;
    //clear all bins
    for(i = 0; i < bin_num; i++){
        bins[i].num_par = 0;
    }

    //fill the particles in bins
    for(i = 0; i < n; i++){
        int par_idx = particle_bin[i]; //get particles by particle id
        int cur_par = bins[par_idx].num_par; //get numbers of particle in bin
        bins[par_idx].par_id[cur_par] = i;
        bins[par_idx].num_par++;
    }

    return;
}

void init_bins(int n, bin * bins){
    width = sqrt( density * n);
    bin_size = (int)ceil(width / cutoff);
    bin_num = bin_size * bin_size;
    particle_bin = (int *) malloc(n * sizeof(int));

    int x_dir[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    int y_dir[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    int i, j, x, y, _x, _y;
    bins = (bin *) malloc(bin_num * sizeof(bin));

    for(i = 0; i < bin_num; i++){
        //for each bins
        bins[i].num_par = 0;
        bins[i].nei_id = (int *)malloc(9 * sizeof(int));
        bins[i].par_id = (int *)malloc(n * sizeof(int));

        x = i % bin_size;
        y = (i - x) / bin_size;
        for(j = 0; j < 9; j++){
            //for bins neighbor
            _x = x + x_dir[j];
            _y = y + y_dir[j];
            if(_x >= 0 && _y >= 0 && _x < bin_size && _y < bin_size){
                bins[i].nei_id[j] = _x + _y * bin_size;
                bins[i].num_nei++;
            }
        }

    }

    return;
}


void apply_force_bin(particle_t * p, bin * bins, int i, double *dmin, double *davg, int *navg){

    //for each particles in one bin
    bin * cur_bin = bins + i;
    int num_par = cur_bin->num_par;
    int num_neigh = cur_bin->num_nei;

    //for every paticle in this bin
    for(int i = 0; i < num_par; i++){
        //check the neighbor for this particle
        for(int j = 0; j < num_neigh; j++){
            bin * bin_nei = bins + cur_bin->nei_id[j];
            for(int k = 0; k < bin_nei->num_par; k++){
                int par_idx = cur_bin->par_id[i]; //the center bin
                int par_nei_idx = bin_nei->par_id[k];
                apply_force(p[par_idx], p[par_nei_idx], dmin, davg, navg);
            }
        }
    }

    return;
}


//
//  benchmarking program
//
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

    //
    // init
    //
    printf("start initialization\n");
    particle_t * particles = (particle_t *) malloc( n * sizeof(particle_t) );
    set_size(n);

    bin * bins;
    init_bins(n, bins);
    init_particles(n, particles);

    printf("finish bin initialization\n");

    for(int i = 0; i < n; i++){
        move(particles[i]);
        particles[i].ax = 0;
        particles[i].ay = 0;
        //assign the particle to the bin
        particle_bin[i] = (int) (floor(particles[i].x / cutoff) * bin_size +
                                 floor(particles[i].y / cutoff));
    }

    set_bin(n, bins);

    printf("finish particle initialization\n");
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
	      navg = 0;
        davg = 0.0;
	      dmin = 1.0;
        //
        //  compute forces
        //
        // for( int i = 0; i < n; i++ )
        // {
        //     particles[i].ax = particles[i].ay = 0;
        //     for (int j = 0; j < n; j++ )
				//         apply_force( particles[i], particles[j],&dmin,&davg,&navg);
        // }

        for(int i = 0; i < n; i++){
            particles[i].ax = particles[i].ay = 0;
        }

        for(int i = 0; i < bin_num; i++){
            apply_force_bin(particles, bins, i, &dmin, &davg, &navg);
        }

        //
        //  move particles
        //
        for(int i = 0; i < n; i++){
            move(particles[i]);
            particles[i].ax = 0;
            particles[i].ay = 0;
            //get there new position
            particle_bin[i] = (int)(floor(particles[i].x / cutoff) * bin_size
                               + floor(particles[i].y / cutoff));
        }

        //let bin get the new particle
        set_bin(n, bins);

        if( find_option( argc, argv, "-no" ) == -1 )
        {
            //
            // Computing statistical data
            //
            if (navg) {
              absavg +=  davg/navg;
              nabsavg++;
            }
            if (dmin < absmin) absmin = dmin;

            //
            //  save if necessary
            //
            if( fsave && (step%SAVEFREQ) == 0 )
                save( fsave, n, particles );
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
