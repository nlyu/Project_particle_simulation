#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "common.h"

#define density 0.0005
#define cutoff  0.01
#define PARICLE_BIN(p) (int)(floor(p.x / cutoff) * bin_size + floor(p.y / cutoff))

int particle_num;
int bin_size;
int num_bins;
int * bin_Ids;

class bin{
public:
    int num_nei;   //counter
    std::vector<int> nei_id;           //neighboring bins
    std::vector<int> par_id;           //paricles in the bins

    bin(): nei_id(9, 0){
        num_nei = 0;
    }
};                          //the bin that separate the zone

/*
    initialize the bins
*/
void init_bins(bin * bins){
    int x, y, i, k, next_x, next_y, new_id;
    int dx[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    int dy[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};

    //for each bins
    for(i = 0; i < num_bins; ++i){
        x = i % bin_size;
        y = (i - x) / bin_size;
        //for bin's neighbor
        for(k = 0; k < 9; ++k){
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

/*
   update the particles in bins
*/
void binning(bin * bins){
    int i, id, idx;
    //clear particle counter
    for(i = 0; i < num_bins; ++i){
        bins[i].par_id.clear();
    }

    //set particles into bin
    for(i = 0; i < particle_num; ++i){
        id = bin_Ids[i];
        bins[id].par_id.push_back(i);
    }
    return;
}

/*
  apply particle force in each bin
*/
void apply_force_bin(particle_t & local, bin * bins, int id, particle_t * _particles, double * dmin, double * davg, int * navg){
    bin * cur_bin = bins + PARICLE_BIN(local);
    bin * new_bin;
    int i, j, par_nei;
    for(i = 0; i < cur_bin->num_nei; ++i){
        //for each neighbors including itself
        new_bin = bins + cur_bin->nei_id[i];
        for(j = 0; j < new_bin->par_id.size(); ++j){
            par_nei = new_bin->par_id[j];
            apply_force(local,
                        _particles[par_nei],
                        dmin, davg, navg);
        }
    }
    return;
}

int main( int argc, char **argv )
{
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg;

    //
    //  process command line parameters
    //
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
    particle_num = n;
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );

    //
    //  set up the data partitioning across processors
    //
    int particle_per_proc = (n + n_proc - 1) / n_proc;
    int *partition_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
    for( int i = 0; i < n_proc+1; i++ )
        partition_offsets[i] = min( i * particle_per_proc, n );

    int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
    for( int i = 0; i < n_proc; i++ )
        partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];

    //
    //  allocate storage for local partition
    //
    int nlocal = partition_sizes[rank];
    particle_t *local = (particle_t*) malloc( nlocal * sizeof(particle_t) );

    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size(n);
    //initialize of global var and bin
    bin_size = (int) ceil(sqrt(density * particle_num) / cutoff);
    num_bins = bin_size * bin_size;
    bin_Ids =  new int[particle_num];
    bin * bins = new bin[num_bins];

    init_bins(bins);
    //if( rank == 0 )
    init_particles( n, particles );
    printf("We have particle: %d\n", particle_num);

    return 0;
    // for(int i = 0; i < particle_num; ++i){
    //     move(particles[i]);
    //     particles[i].ax = particles[i].ay = 0;
    //     bin_Ids[i] = PARICLE_BIN(particles[i]);
    // }

    MPI_Scatterv( particles, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD );

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;
        //
        //  collect all global data locally (not good idea to do)
        //
        MPI_Allgatherv( local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD );

        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if( find_option( argc, argv, "-no" ) == -1 )
          if( fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );

        //
        //  compute all forces
        //
        for( int i = 0; i < nlocal; i++ )
        {
            local[i].ax = local[i].ay = 0;
            for (int j = 0; j < n; j++ )
                apply_force( local[i], particles[j], &dmin, &davg, &navg );
        }

        if( find_option( argc, argv, "-no" ) == -1 )
        {

          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);


          if (rank == 0){
            //
            // Computing statistical data
            //
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) absmin = rdmin;
          }
        }

        //
        //  move particles
        //
        for( int i = 0; i < nlocal; i++ )
            move( local[i] );
    }
    simulation_time = read_timer( ) - simulation_time;

    if (rank == 0) {
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
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }

    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    free( partition_offsets );
    free( partition_sizes );
    free( local );
    free( particles );
    if( fsave )
        fclose( fsave );

    MPI_Finalize( );

    return 0;
}
