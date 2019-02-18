#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define cutoff2 (cutoff * cutoff)
#define min_r   (cutoff/100)
#define dt      0.0005
#define FOR(i,n) for( int i=0; i<n; i++ )

double size2;
int bin_size;           // number of bins in one direction
int num_bins;           // total number of bins in the domain
int shift[9];
int* bin_Ids;           // bin Ids

typedef struct{
    int num_particles;
    int num_neigh;
    int* neighbors_ids;
    int* particle_ids;
} bin_dict;

//
//  benchmarking program
//
void move_v2( particle_t &p, int _id)
{
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x  += p.vx * dt;
    p.y  += p.vy * dt;

    //
    //  bounce from walls
    //
    while( p.x < 0 || p.x > size2 )
    {
        p.x  = p.x < 0 ? -p.x : 2*size2-p.x;
        p.vx = -p.vx;
    }
    while( p.y < 0 || p.y > size2 )
    {
        p.y  = p.y < 0 ? -p.y : 2*size2-p.y;
        p.vy = -p.vy;
    }

    p.ax = 0;
    p.ay = 0;
    //int id = ;
    bin_Ids[_id] = (int)(floor(p.x / cutoff) * bin_size
                         + floor(p.y / cutoff));          // save bin location for each particle
}


void apply_force_v2( particle_t &particle, particle_t &neighbor, double *dmin, double *davg, int *navg)
{

    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if( r2 > cutoff*cutoff )
        return;
	if (r2 != 0)
        {
	   if (r2/(cutoff*cutoff) < *dmin * (*dmin))
	      *dmin = sqrt(r2)/cutoff;
           (*davg) += sqrt(r2)/cutoff;
           (*navg) ++;
        }

    r2 = fmax( r2, min_r*min_r );
    double r = sqrt( r2 );

    //
    //  very simple short-range repulsive force
    //
    double coef = ( 1 - cutoff / r ) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

void init_bins( bin_dict* _bins ) {
    int dx[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    int dy[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};  // x and y coordinates of possible neighbors
    FOR (i, num_bins) {          // for loop to count number of bin neighbors
        _bins[i].num_neigh = 0;
        _bins[i].neighbors_ids = (int*) malloc(9 * sizeof(int));
        int x = i % bin_size;        // x value for bin location (0,1,2,...,49,0,1,2,...,49...)
        int y = (i - x) / bin_size;  // y value for bin location (0,0,0,...,0 ,1,1,1,...,0 ...)
        FOR (k, 9) {                // for loop to place neighbor and check neighbor is inbound
            int new_x = x + dx[k];
            int new_y = y + dy[k];
            if (new_x >= 0 && new_y >= 0 && new_x < bin_size && new_y < bin_size) {
                int new_id = new_x + new_y * bin_size;
                _bins[i].neighbors_ids[_bins[i].num_neigh] = new_id;
                _bins[i].num_neigh++;
            }
        }

    }
}


void binning(bin_dict* _bins, int _num) {
    FOR (i, num_bins)                   // delete particles in each bin
        _bins[i].num_particles = 0;

    FOR (i, _num) {
        int id = bin_Ids[i];          // get bin location for each particle from last move
        _bins[id].particle_ids[_bins[id].num_particles] = i;    // link particle index id to bin index id
        _bins[id].num_particles++;      // add one particle to the specific bin id
    }
}

void apply_force_bin(particle_t* _particles, bin_dict* _bins, int _binId, double *dmin, double *davg, int *navg) {
    bin_dict* bin = _bins + _binId;         // make program work on specific bin with ID _binID

    FOR (i, bin->num_particles) {           // for loop that goes through all particles in specific bin
        FOR (k, bin->num_neigh) {           // for loop to apply force on the surrounding bins
            bin_dict* new_bin = _bins + bin->neighbors_ids[k];
            for(int j = 0; j < new_bin->num_particles; j++)
                apply_force(_particles[bin->particle_ids[i]], _particles[new_bin->particle_ids[j]], dmin, davg, navg);
        }
    }
}

void set_size2( int n )
{
    size2 = sqrt( density * n );
    bin_size = (int)ceil(size2 / cutoff);      // use cutoff to divide bins each with size cutoff
    num_bins = bin_size * bin_size;           // total number of bins in domain
    bin_Ids =  (int*) malloc(n * sizeof(int));

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

    set_size( n );
    set_size2( n );                          // set size to sqrt(density==0.0005*n==500) used to initialize position of the particles in next step
    bin_dict* bins = (bin_dict*) malloc(num_bins * sizeof(bin_dict));
    FOR (i, num_bins)
        bins[i].particle_ids = (int*) malloc(n*sizeof(int));
    init_bins(bins);


    init_particles( n, particles );
    FOR (i, n)
        move_v2(particles[i], i);   // assign each particle to a bin

    binning(bins, n);    // calculate number of particles in each bin

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( ); // reads time using function in commons.cpp
	
    FOR(step, NSTEPS)
    {

        navg = 0;
        davg = 0.0;
        dmin = 1.0;
        //
        //  compute forces
        //
	FOR(i, n){
            particles[i].ax = 0;    // initialize acceleration after each step
            particles[i].ay = 0;    // initialize acceleration after each step
        }

        FOR(i, num_bins){
            apply_force_bin(particles, bins, i, &dmin, &davg, &navg);    // apply forces in particles for bin by bin with only neighboring bins
        }
 
        //
        //  move particles
        //
        for( int i = 0; i < n; i++ ) 
            move_v2( particles[i], i );

        binning( bins, n);            // reset number of particles in each bin and calculate again

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

