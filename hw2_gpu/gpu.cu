#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#define NUM_THREADS 16
#define NUM_PARTICLE_BIN 32 //hardcoded looks fine. We assume there would be at most 64 particle in each bin
#define get_bin(p, bins_num, s)   (int)(p.x / (double)(s / bins_num)) + (int)(p.y / (double)(s / bins_num)) * bins_num

extern double size;
int n_bins_side;
int n_bins;

class bin_t {
public:
    int n_particles;
    int n_no_change;
    int n_change;
    //particles in bin
    int particles[];
    //exchange bin recorder
    int no_change[NUM_PARTICLE_BIN];
    int change[NUM_PARTICLE_BIN];

    bin_t(){
        this->n_no_change = this->n_change = this->n_particles = 0;
    }

    void add(int par_id){
        this->particles[this->n_particles] = par_id;
        this->n_particles++;
    }

    CUDA_CALLABLE_MEMBER void update(int new_bin, int cur_bin, int p_id){
        if (new_bin != cur_bin) {
            this->change[this->n_change++] = p_id;
        } else {
            this->no_change[this->n_no_change++] = p_id;
        }
    }

    ~bin_t(){
        //no memory leak here
    }
};

//
//  benchmarking program
//
__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if( r2 > cutoff*cutoff )
        return;
    //r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
    double r = sqrt( r2 );

    //
    //  very simple short-range repulsive force
    //
    double coef = ( 1 - cutoff / r ) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}


__device__ void move_particle_gpu(particle_t &p, double d_size) {
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
    while( p.x < 0 || p.x > d_size )
    {
        p.x  = p.x < 0 ? -(p.x) : 2*d_size-p.x;
        p.vx = -(p.vx);
    }
    while( p.y < 0 || p.y > d_size )
    {
        p.y  = p.y < 0 ? -(p.y) : 2*d_size-p.y;
        p.vy = -(p.vy);
    }
}

__global__ void compute_forces_gpu(particle_t *particles,
                                   bin_t *d_bins,
                                   int d_n_bins, int d_n_bins_side) {
    // Get thread (bin) ID
    int cur_b = threadIdx.x + blockIdx.x * blockDim.x;
    if (cur_b >= d_n_bins) return;

    int cur_b_row = cur_b % d_n_bins_side;
    int cur_b_col = cur_b / d_n_bins_side;

    //clear the acceleration to zero
    for (int p1 = 0; p1 < d_bins[cur_b].n_particles; p1++) {
        particles[d_bins[cur_b].particles[p1]].ax = particles[d_bins[cur_b].particles[p1]].ay = 0;
    }

    //nine bins, left to right, top do down
    int x_idx[] = {cur_b_row-1, cur_b_row-1, cur_b_row-1, cur_b_row,   cur_b_row, cur_b_row,   cur_b_row+1, cur_b_row+1, cur_b_row+1};
    int y_idx[] = {cur_b_col-1, cur_b_col,   cur_b_col+1, cur_b_col-1, cur_b_col, cur_b_col+1, cur_b_col-1,  cur_b_col,  cur_b_col+1};
    for(int i = 0; i < 9; i++){
        if(x_idx[i] >= 0 && x_idx[i] < d_n_bins_side && y_idx[i] >= 0 && y_idx[i] < d_n_bins_side){
            int nei_b = x_idx[i] + y_idx[i] * d_n_bins_side; //get the neighbor bin
            for (int p1 = 0; p1 < d_bins[cur_b].n_particles; p1++) {
                for (int p2 = 0; p2 < d_bins[nei_b].n_particles; p2++) {
                    //compute force between cur bin and neighbor bin
                    apply_force_gpu(particles[d_bins[cur_b].particles[p1]],
                                    particles[d_bins[nei_b].particles[p2]]);
                }
            }
        }
    }
}


__global__ void move_gpu (particle_t *particles,
                                bin_t *d_bins,
                                double d_size, int d_n_bins_side, int d_n_bins) {
    // Get thread (bin) ID
    int cur_b = threadIdx.x + blockIdx.x * blockDim.x;
    if (cur_b >= d_n_bins) return;

    d_bins[cur_b].n_no_change = d_bins[cur_b].n_change = 0;

    // move the particle
    for (int p_id = 0; p_id < d_bins[cur_b].n_particles; p_id++) {
        //move the particle to new bin
        int p_new = d_bins[cur_b].particles[p_id];
        particle_t &p = particles[p_new];
        move_particle_gpu(p, d_size);
        // record the state of the partivle
        int new_b_idx = get_bin(p, d_n_bins_side, d_size);
        // check whether the partivle move to a new bin
        if (new_b_idx != cur_b) {
            d_bins[cur_b].change[d_bins[cur_b].n_change++] = p_new;
        } else {
            d_bins[cur_b].no_change[d_bins[cur_b].n_no_change++] = p_new;
        }
        //d_bins[cur_b].update(new_b_idx, cur_b, p_new);
    }
}

__global__ void binning (particle_t *particles,
                                bin_t *d_bins,
                                double d_size, int d_n_bins_side, int d_n_bins) {
    // Get thread bin ID
    int cur_b = threadIdx.x + blockIdx.x * blockDim.x;
    if (cur_b >= d_n_bins) return;

    // Saves the particle that stays in the bin
    d_bins[cur_b].n_particles = d_bins[cur_b].n_no_change;
    for (int p1 = 0; p1 < d_bins[cur_b].n_particles; p1++) {
        d_bins[cur_b].particles[p1] = d_bins[cur_b].no_change[p1];
    }

    // accept the incoming particle to the bin
    int cur_b_row = cur_b % d_n_bins_side;
    int cur_b_col = cur_b / d_n_bins_side;

    int x_idx[] = {cur_b_row-1, cur_b_row-1, cur_b_row-1, cur_b_row,   cur_b_row, cur_b_row,   cur_b_row+1, cur_b_row+1, cur_b_row+1};
    int y_idx[] = {cur_b_col-1, cur_b_col,   cur_b_col+1, cur_b_col-1, cur_b_col, cur_b_col+1, cur_b_col-1,  cur_b_col,  cur_b_col+1};
    for(int i = 0; i < 9; i++){
        //check out of border
        if(x_idx[i] >= 0 && x_idx[i] < d_n_bins_side && y_idx[i] >= 0 && y_idx[i] < d_n_bins_side){
            int nei_bin = x_idx[i] + y_idx[i] * d_n_bins_side;
            //get the incoming particles to the current bin
            for (int p2 = 0; p2 < d_bins[nei_bin].n_change; p2++) {
                int par_comming = d_bins[nei_bin].change[p2];
                particle_t &p = particles[par_comming];
                if (get_bin(p, d_n_bins_side, d_size == cur_b)) { //find the particle from the neighbor that arrives in the cur bin
                    d_bins[cur_b].particles[d_bins[cur_b].n_particles++] = par_comming;
                }
            }
        }
    }
}

int main( int argc, char **argv )
{
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize();

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );

    init_particles( n, particles );

    cudaThreadSynchronize();

    //set bins
    n_bins_side = size / 0.03;
    n_bins = n_bins_side * n_bins_side;

    bin_t * d_bins;
    cudaMalloc((void **) &d_bins, n_bins * sizeof(bin_t));

    //initialize bins
    bin_t * bins = new bin_t[n_bins];
    //distribute each particle to a bin
    for (int i = 0; i < n; i++) {
        int b_idx = get_bin(particles[i], n_bins_side, size);
        bins[b_idx].add(i);
    }
    // Copy host bins to device
    cudaMemcpy(d_bins, bins, n_bins * sizeof(bin_t), cudaMemcpyHostToDevice);

    delete[] bins;

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;

    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  compute forces
        //
        int blks = (n_bins + NUM_THREADS - 1) / NUM_THREADS;
        compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, d_bins,
                                                      n_bins, n_bins_side);

        //
        //  move particles
        //
        //printf("I have %d\n", blks);
        move_gpu <<< blks, NUM_THREADS >>> (d_particles, d_bins, size, n_bins_side, n_bins);

        //cannot combine move_gpu() and bining() into one step...need block sync which __syncthreads() cannot achieve
        binning <<< blks, NUM_THREADS >>> (d_particles, d_bins, size, n_bins_side, n_bins);
        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
            // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
        }
    }

    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;

    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    free( particles );
    cudaFree(d_particles);
    cudaFree(d_bins);
    if( fsave )
        fclose( fsave );

    return 0;
}
