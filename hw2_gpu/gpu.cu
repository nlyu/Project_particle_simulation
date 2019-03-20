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

#define NUM_THREADS 256
#define NUM_PARTICLE_BIN 16 //hardcoded looks fine. We assume there would be at most 64 particle in each bin. >=4 is good!
#define get_bin(p, bins_num, s)   (int)(p.x / (double)(s / bins_num)) + (int)(p.y / (double)(s / bins_num)) * bins_num

extern double size; //size of the canvas
int n_bins_side;    //numbers of bins per side
int n_bins;         //numbers of bins

class bin_t {
public:
    int par_counter;
    int par_after_counter;
    int par_remove_counter;
    //particles in bin
    int particles[];
    //exchange bin recorder
    int particles_after[NUM_PARTICLE_BIN];
    int particles_remove[NUM_PARTICLE_BIN];

    bin_t(){
        this->par_after_counter = this->par_remove_counter = this->par_counter = 0;
    }

    //add new particle to the bin
    CUDA_CALLABLE_MEMBER void add(int par_id){
        this->particles[this->par_counter] = par_id;
        this->par_counter++;
    }

    //record the particle state in new step
    CUDA_CALLABLE_MEMBER void update(int new_bin, int cur_bin, int p_id){
        if (new_bin != cur_bin) {
            this->particles_remove[this->par_remove_counter++] = p_id;
        } else {
            this->particles_after[this->par_after_counter++] = p_id;
        }
    }

    //zero the counters
    CUDA_CALLABLE_MEMBER void clear_counter(){
        this->par_after_counter = this->par_remove_counter = 0;
    }

    //exchange the particle from previous step to current step
    CUDA_CALLABLE_MEMBER void exchange(int p_id){
        this->particles[p_id] = this->particles_after[p_id];
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

//computer the force for each particle using serial method
__global__ void compute_forces_gpu(particle_t *particles,
                                   bin_t *d_bins,
                                   int d_n_bins, int d_n_bins_side) {
    // Get thread (bin) ID
    int cur_bin = threadIdx.x + blockIdx.x * blockDim.x;
    if (cur_bin >= d_n_bins) return;

    int b1_row = cur_bin % d_n_bins_side;
    int b1_col = cur_bin / d_n_bins_side;

    for (int p1 = 0; p1 < d_bins[cur_bin].par_counter; ++p1) {
        particles[d_bins[cur_bin].particles[p1]].ax = particles[d_bins[cur_bin].particles[p1]].ay = 0;
    }

    //nine bins, left to right, top do down
    int x_idx[] = {b1_row-1, b1_row-1, b1_row-1, b1_row,   b1_row, b1_row,   b1_row+1, b1_row+1, b1_row+1};
    int y_idx[] = {b1_col-1, b1_col,   b1_col+1, b1_col-1, b1_col, b1_col+1, b1_col-1,  b1_col,  b1_col+1};
    for(int i = 0; i < 9; ++i){
        if(x_idx[i] >= 0 && x_idx[i] < d_n_bins_side && y_idx[i] >= 0 && y_idx[i] < d_n_bins_side){
            int nei_bin = x_idx[i] + y_idx[i] * d_n_bins_side; //get the neighbor bin
            for(int p1 = 0; p1 < d_bins[cur_bin].par_counter; ++p1) {
                for (int p2 = 0; p2 < d_bins[nei_bin].par_counter; ++p2) {
                    //compute force between cur bin and neighbor bin
                    apply_force_gpu(particles[d_bins[cur_bin].particles[p1]],
                                    particles[d_bins[nei_bin].particles[p2]]);
                }
            }
        }
    }
}


//move the particles according to their force
__global__ void move_gpu (particle_t *particles,
                                bin_t *d_bins,
                                double d_size, int d_n_bins_side, int d_n_bins) {
    // Get thread (bin) ID
    int cur_bin = threadIdx.x + blockIdx.x * blockDim.x;
    if (cur_bin >= d_n_bins) return;

    // initialize the exchange counter
    d_bins[cur_bin].clear_counter();

    // Move this bin's particles to either leaving or staying
    for (int p1 = 0; p1 < d_bins[cur_bin].par_counter; ++p1) {
        //move the particle to new bin
        int p_new = d_bins[cur_bin].particles[p1];
        particle_t &p = particles[p_new];
        move_particle_gpu(p, d_size);
        // record the state of the partivle
        int new_b_idx = get_bin(p, d_n_bins_side, d_size);
        d_bins[cur_bin].update(new_b_idx, cur_bin, p_new); // check whether the partivle move to a new bin
    }
}

//allocate the particle after moved, each particle is in new position. Assign the bins again for the particles.
__global__ void binning (particle_t *particles,
                                bin_t *d_bins,
                                double d_size, int d_n_bins_side, int d_n_bins) {
    // Get thread bin ID
    int cur_bin = threadIdx.x + blockIdx.x * blockDim.x;
    if (cur_bin >= d_n_bins) return;

    // Saves the particle that stays in the bin
    d_bins[cur_bin].par_counter = d_bins[cur_bin].par_after_counter;
    for (int p1 = 0; p1 < d_bins[cur_bin].par_counter; ++p1) {
        d_bins[cur_bin].exchange(p1);
    }


    // accept the incoming particle to the bin
    int cur_b_row = cur_bin % d_n_bins_side;
    int cur_b_col = cur_bin / d_n_bins_side;
    int x_idx[] = {cur_b_row-1, cur_b_row-1, cur_b_row-1, cur_b_row,   cur_b_row, cur_b_row,   cur_b_row+1, cur_b_row+1, cur_b_row+1};
    int y_idx[] = {cur_b_col-1, cur_b_col,   cur_b_col+1, cur_b_col-1, cur_b_col, cur_b_col+1, cur_b_col-1,  cur_b_col,  cur_b_col+1};

    for(int i = 0; i < 9; ++i){
      //check out of border
        if(x_idx[i] >= 0 && x_idx[i] < d_n_bins_side && y_idx[i] >= 0 && y_idx[i] < d_n_bins_side){
            int b2 = x_idx[i] + y_idx[i] * d_n_bins_side;
            for (int p2 = 0; p2 < d_bins[b2].par_remove_counter; ++p2) {
                int par_comming = d_bins[b2].particles_remove[p2];
                particle_t &p = particles[par_comming];
                if (get_bin(p, d_n_bins_side, d_size) == cur_bin) {//find the particle from the neighbor that arrives in the cur bin
                    d_bins[cur_bin].add(par_comming);
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
    n_bins_side = size / 0.03;
    //n_bins = n_bins_side * n_bins_side;
    n_bins = pow(n_bins_side, 2);


    bin_t * d_bins;
    cudaMalloc((void **) &d_bins, n_bins * sizeof(bin_t));

    bin_t * bins = new bin_t[n_bins];
    //distribute each particle to a bin
    for (int i = 0; i < n; ++i) {
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

        //
        //  recalulate the particle in bins
        // cannot combine move_gpu() and bining() into one step...need block sync which __syncthreads() cannot achieve
        //
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
