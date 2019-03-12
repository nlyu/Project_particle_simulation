#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 256

int bins_per_side;
int n_bins;

extern double size;

struct bin_t {
    int particles[32];
    int n_particles;
};

__host__ int bin_of_particle(particle_t &particle) {
    double sidelength = size / bins_per_side;
    int b_row = (int)(particle.x / sidelength);
    int b_col = (int)(particle.y / sidelength);
    return b_row + b_col * bins_per_side;
}

__device__ int bin_of_particle_gpu(particle_t &particle, double d_size, int d_bins_per_side) {
    double sidelength = d_size / d_bins_per_side;
    int b_row = (int)(particle.x / sidelength);
    int b_col = (int)(particle.y / sidelength);
    return b_row + b_col * d_bins_per_side;
}

__host__ void init_bins(int n, particle_t *particles,
                  bin_t *d_bins) {
    // Create bins on host
    bin_t *bins = new bin_t[n_bins];
    for (int b = 0; b < n_bins; b++) {
        bins[b].n_particles = 0;
    }
    // Assign each particle to a bin
    for (int k = 0; k < n; k++) {
        int b_idx = bin_of_particle(particles[k]);
        bins[b_idx].particles[bins[b_idx].n_particles++] = k;
    }
    // Copy host bins to device
    cudaMemcpy(d_bins, bins, n_bins * sizeof(bin_t), cudaMemcpyHostToDevice);

    delete[] bins;
}

__host__ void init_bins_id(int n, particle_t *particles,
                  int * d_bins_id) {
    // Create bins on host
    int * bins_id = new int[n];
    for (int p = 0; p < n; p++) {
        bins_id[p] = bin_of_particle(particles[p]);
    }

    // Copy host bins to device
    cudaMemcpy(d_bins_id, bins_id, n * sizeof(int), cudaMemcpyHostToDevice);

    delete[] bins_id;
}

//
//  benchmarking program
//
__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if( r2 > cutoff * cutoff )
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

__global__ void compute_forces_gpu(particle_t *particles,
                                   bin_t *d_bins,
                                   int *d_bins_id,
                                   int d_n_bins, int d_bins_per_side) {
    // Get thread (bin) ID
    int b1 = threadIdx.x + blockIdx.x * blockDim.x;
    if (b1 >= d_n_bins) return;

    int b1_row = b1 % d_bins_per_side;
    int b1_col = b1 / d_bins_per_side;

    for (int p1 = 0; p1 < d_bins[b1].n_particles; p1++) {
        particles[d_bins[b1].particles[p1]].ax = particles[d_bins[b1].particles[p1]].ay = 0;
    }

    for (int b2_row = max(0, b1_row - 1);
         b2_row <= min(d_bins_per_side - 1, b1_row + 1);
         b2_row++) {
        for (int b2_col = max(0, b1_col - 1);
             b2_col <= min(d_bins_per_side - 1, b1_col + 1);
             b2_col++) {
            int b2 = b2_row + b2_col * d_bins_per_side;
            for (int p1 = 0; p1 < d_bins[b1].n_particles; p1++) {
                for (int p2 = 0; p2 < d_bins[b2].n_particles; p2++) {
                    apply_force_gpu(particles[d_bins[b1].particles[p1]],
                                    particles[d_bins[b2].particles[p2]]);
                }
            }
        }
    }

    // Clear staying and leaving from previous iteration in preparation for move_gpu_step1
    // _bins[b1].n_staying = d_bins[b1].n_leaving = 0;
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

__global__ void move_gpu_my1 (particle_t *particles,
                                bin_t *d_bins,
                                double d_size,
                                int *d_bins_id, int d_bins_per_side, int d_n_bins) {
    // Get thread (bin) ID
    int b = threadIdx.x + blockIdx.x * blockDim.x;
    if (b >= d_n_bins) return;

    // Move this bin's particles to either leaving or staying
    for (int p1 = 0, p_id = 0; p1 < d_bins[b].n_particles; p1++) {
        p_id = d_bins[b].particles[p1];
        particle_t &p = particles[p_id];
        move_particle_gpu(p, d_size);
        int new_b_idx = bin_of_particle_gpu(p, d_size, d_bins_per_side);
        if (new_b_idx != b) {
            d_bins_id[p_id] = new_b_idx;
        }
    }
}

__global__ void binning (particle_t *particles,
                         bin_t *d_bins, int * d_bins_id, int d_n){
    int b = threadIdx.x + blockIdx.x * blockDim.x;

    d_bins[b].n_particles = 0;
    for (int p = 0; p < d_n; p++) {
        if(d_bins_id[p] == b){
            d_bins[b].particles[d_bins[b].n_particles++] = p;
        }
    }
}

//
// __global__ void move_gpu_step1 (particle_t *particles,
//                                 bin_t *d_bins,
//                                 double d_size, int d_bins_per_side, int d_n_bins) {
//     // Get thread (bin) ID
//     int b = threadIdx.x + blockIdx.x * blockDim.x;
//     if (b >= d_n_bins) return;
//
//     // Move this bin's particles to either leaving or staying
//     for (int p1 = 0; p1 < d_bins[b].n_particles; p1++) {
//         particle_t &p = particles[d_bins[b].particles[p1]];
//         move_particle_gpu(p, d_size);
//         int new_b_idx = bin_of_particle_gpu(p, d_size, d_bins_per_side);
//         if (new_b_idx != b) {
//             d_bins[b].leaving[d_bins[b].n_leaving++] = d_bins[b].particles[p1];
//         } else {
//             d_bins[b].staying[d_bins[b].n_staying++] = d_bins[b].particles[p1];
//         }
//     }
//     assert(d_bins[b].n_leaving < 32);
//     assert(d_bins[b].n_staying < 32);
// }
//
// __global__ void move_gpu_step2 (particle_t *particles,
//                                 bin_t *d_bins,
//                                 double d_size, int d_bins_per_side, int d_n_bins) {
//     // Get thread (bin) ID
//     int b = threadIdx.x + blockIdx.x * blockDim.x;
//     if (b >= d_n_bins) return;
//
//     // Consolidate staying and particles from neighbor bins' leaving
//     // lists. Assumes particles don't go so fast that they jump over bins.
//     for (int p1 = 0; p1 < d_bins[b].n_staying; p1++) {
//         d_bins[b].particles[p1] = d_bins[b].staying[p1];
//     }
//     d_bins[b].n_particles = d_bins[b].n_staying;
//
//     int b1_row = b % d_bins_per_side;
//     int b1_col = b / d_bins_per_side;
//     for (int b2_row = max(0, b1_row - 1);
//          b2_row <= min(d_bins_per_side - 1, b1_row + 1);
//          b2_row++) {
//         for (int b2_col = max(0, b1_col - 1);
//              b2_col <= min(d_bins_per_side - 1, b1_col + 1);
//              b2_col++) {
//             int b2 = b2_row + b2_col * d_bins_per_side;
//             for (int p2 = 0; p2 < d_bins[b2].n_leaving; p2++) {
//                 particle_t &p = particles[d_bins[b2].leaving[p2]];
//                 int new_b_idx = bin_of_particle_gpu(p, d_size, d_bins_per_side);
//                 if (new_b_idx == b) {
//                     d_bins[b].particles[d_bins[b].n_particles++] = d_bins[b2].leaving[p2];
//                 }
//             }
//         }
//     }
//     assert(d_bins[b].n_particles < 32);
// }



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

    // Initialize bins
    bins_per_side = read_int(argc, argv, "-b", size / (0.01*3));
    n_bins = bins_per_side * bins_per_side;
    bin_t *d_bins;
    cudaMalloc((void **) &d_bins, n_bins * sizeof(bin_t));
    init_bins(n, particles, d_bins);

    int * d_bins_id;
    cudaMalloc((void **) &d_bins_id, n * sizeof(int));
    init_bins_id(n, particles, d_bins_id);


    cudaThreadSynchronize();   // Block until all preceeding tasks on all threads are done
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
        compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, d_bins, d_bins_id,
                                                      n_bins, bins_per_side);

        //
        //  move particles
        //

        //move_gpu_step1 <<< blks, NUM_THREADS >>> (d_particles, d_bins, size, bins_per_side, n_bins);
        //move_gpu_step2 <<< blks, NUM_THREADS >>> (d_particles, d_bins, size, bins_per_side, n_bins);
        move_gpu_my1 <<< blks, NUM_THREADS >>> (d_particles, d_bins, size, d_bins_id, bins_per_side, n_bins, n);
        binning <<< blks, NUM_THREADS >>> (d_particles, d_bins, d_bins_id, n);
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
