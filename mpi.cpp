#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <list>
#include <vector>
#include <cmath>
#include <algorithm>
#include <float.h>
#include <string.h>
#include <math.h>
#include "common.h"

#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

MPI_Datatype PARTICLE;

typedef struct bin_t {
    std::list<imy_particle_t*> particles;
    std::list<imy_particle_t*> incoming;
} bin_t;

double size2;

int bins_per_side;
int n_bins;
int n_proc, rank;
int n;
int rows_per_proc;

bool operator<(const imy_particle_t &a, const imy_particle_t &b) {
    return a.index < b.index;
}

void init_bins(int n, double size, imy_particle_t *particles, std::vector<bin_t> &bins) {
    // Create bins (most do not belong to this task)
    for (int b_idx = 0; b_idx < n_bins; b_idx++) {
        bin_t b;
        bins.push_back(b);
    }
    assign_particles_to_bins(n, size, particles, bins);
}

//
//  I/O routines
//
void save2( FILE *f, int n, my_particle_t *p )
{
    static bool first = true;
    if( first )
    {
        fprintf( f, "%d %g\n", n, size2 );
        first = false;
    }
    for( int i = 0; i < n; i++ )
        fprintf( f, "%g %g\n", p[i].x, p[i].y );
}

void move2( my_particle_t &p )
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
}


void apply_force2( my_particle_t &particle, my_particle_t &neighbor , double *dmin, double *davg, int *navg)
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


void init_iparticles(int n, double size, imy_particle_t *p) {
    srand48( time( NULL ) );

    int sx = (int)ceil(sqrt((double)n));
    int sy = (n+sx-1)/sx;

    int *shuffle = (int*)malloc( n * sizeof(int) );
    for( int i = 0; i < n; i++ )
        shuffle[i] = i;

    for( int i = 0; i < n; i++ )
    {
        //
        //  make sure particles are not spatially sorted
        //
        int j = lrand48()%(n-i);
        int k = shuffle[j];
        shuffle[j] = shuffle[n-i-1];

        //
        //  distribute particles evenly to ensure proper spacing
        //
        p[i].particle.x = size*(1.+(k%sx))/(1+sx);
        p[i].particle.y = size*(1.+(k/sx))/(1+sy);

        //
        //  assign random velocities within a bound
        //
        p[i].particle.vx = drand48()*2-1;
        p[i].particle.vy = drand48()*2-1;

        p[i].index = i;
    }
    free( shuffle );
}

int bin_of_particle(double size, imy_particle_t &particle) {
    double sidelength = size / bins_per_side;
    int b_row = (int)(particle.particle.x / sidelength);
    int b_col = (int)(particle.particle.y / sidelength);
    return b_row + b_col * bins_per_side;
}

std::vector<int> neighbors_of_rank(int rank) {
    std::vector<int> neighbor_ranks;
    if (rank > 0) {
        neighbor_ranks.push_back(rank - 1);
    }
    if (rank < n_proc - 1) {
        neighbor_ranks.push_back(rank + 1);
    }
    return neighbor_ranks;
}

void assign_particles_to_bins(int n, double size, imy_particle_t *particles, std::vector<bin_t> &bins) {
    // Put each particle in its bin
    for (int k = 0; k < n; k++) {
        int b_idx = particles[k].particle.bin_idx = bin_of_particle(size, particles[k]);
        bins[b_idx].particles.push_back(&particles[k]);
    }
}

int rank_of_bin(int b_idx) {
    // 2D partitioning (still need to do)
    int b_row = b_idx % bins_per_side;
    return b_row / rows_per_proc;
}

std::vector<int> bins_of_rank(int rank) {
    std::vector<int> result;
    for (int row = rank * rows_per_proc;
         row < min((rank + 1) * rows_per_proc, bins_per_side); row++) {
        for (int col = 0; col < bins_per_side; col++) {
            result.push_back(row + col * bins_per_side);
        }
    }
    return result;
}

/** Returns the particles owned by the current rank in bins bordering on other_rank. */
std::vector<imy_particle_t> border_particles_of_rank(int other_rank, std::vector<bin_t> &bins) {
    int row;
    if (other_rank < rank) {
        row = rank * rows_per_proc; // first row of rank
    } else {
        row = (rank + 1) * rows_per_proc - 1; // last row of rank, possibly out of bounds
    }
    std::vector<imy_particle_t> result;
    if (row >= 0 && row < bins_per_side) {
        for (int col = 0; col < bins_per_side; col++) {
            bin_t &b = bins[row + col * bins_per_side];
            int n_particles = 0;
            for (std::list<imy_particle_t*>::const_iterator it = b.particles.begin();
                 it != b.particles.end(); it++) {
                result.push_back(**it);
                n_particles++;
            }
        }
    }
    return result;
}

void exchange_neighbors(double size, imy_particle_t *local_particles,
                        int *n_local_particles, std::vector<bin_t> &bins) {
    std::vector<int> neighbor_ranks = neighbors_of_rank(rank);
    // Send border particles to neighbors
    for (int i = 0; i < neighbor_ranks.size(); i++) {
        std::vector<imy_particle_t> border_particles = border_particles_of_rank(neighbor_ranks[i], bins);
        MPI_Request request;
        MPI_Ibsend(border_particles.empty() ? 0 : &border_particles[0], border_particles.size(), PARTICLE, neighbor_ranks[i], 0, MPI_COMM_WORLD, &request);
        MPI_Request_free(&request);
    }
    // Receive and bin neighbors' border particles
    imy_particle_t *cur_pos = local_particles + *n_local_particles;
    for (std::vector<int>::const_iterator it = neighbor_ranks.begin();
         it != neighbor_ranks.end(); it++) {
        MPI_Status status;
        MPI_Recv(cur_pos, n, PARTICLE, *it, 0, MPI_COMM_WORLD, &status);
        int num_particles_received;
        MPI_Get_count(&status, PARTICLE, &num_particles_received);
        assign_particles_to_bins(num_particles_received, size, cur_pos, bins);
        cur_pos += num_particles_received;
        *n_local_particles += num_particles_received;
    }
}

void exchange_moved(double size, imy_particle_t **local_particles_ptr,
                    std::vector<bin_t> &bins, std::vector<int> &local_bin_idxs,
                    int *n_local_particles) {
    std::vector<int> neighbor_ranks = neighbors_of_rank(rank);
    // Send moved particles to neighbors
    // Do this scan when moving particles, populating a map from task to particle (still need to do)
    // For each neighbor task
    for (int i = 0; i < neighbor_ranks.size(); i++) {
        std::vector<imy_particle_t> moved_particles;
        // For each bin owned by that task
        for (int b_idx = 0; b_idx < n_bins; b_idx++) {
            if (rank_of_bin(b_idx) == neighbor_ranks[i]) {
                // For each particle in that bin
                for (std::list<imy_particle_t*>::const_iterator p_it = bins[b_idx].incoming.begin();
                     p_it != bins[b_idx].incoming.end(); p_it++) {
                    // Copy the particle into the moved_particles buffer
                    moved_particles.push_back(**p_it);
                }
            }
        }
        // Send the particles to the task
        MPI_Request request;
        MPI_Ibsend(moved_particles.empty() ? 0 : &moved_particles[0], moved_particles.size(), PARTICLE, neighbor_ranks[i], 0, MPI_COMM_WORLD, &request);
        MPI_Request_free(&request);
    }

    // Receive particles moved here
    imy_particle_t *new_local_particles = new imy_particle_t[n];
    imy_particle_t *cur_pos = new_local_particles;
    for (std::vector<int>::const_iterator it = neighbor_ranks.begin();
         it != neighbor_ranks.end(); it++) {
        MPI_Status status;
        MPI_Recv(cur_pos, n, PARTICLE, *it, 0, MPI_COMM_WORLD, &status);
        int num_particles_received;
        MPI_Get_count(&status, PARTICLE, &num_particles_received);
        cur_pos += num_particles_received;
    }

    // Copy out remaining particles into new_local_particles
    for (std::vector<int>::const_iterator b_it = local_bin_idxs.begin();
         b_it != local_bin_idxs.end(); b_it++) {
        for (std::list<imy_particle_t*>::const_iterator p_it = bins[*b_it].particles.begin();
            p_it != bins[*b_it].particles.end(); p_it++) {
            *cur_pos = **p_it;
            cur_pos++;
        }
    }

    // Apply new_local_particles
    delete[] *local_particles_ptr;
    *local_particles_ptr = new_local_particles;
    *n_local_particles = cur_pos - new_local_particles;

    // Rebin all particles
    bins.clear();
    init_bins(*n_local_particles, size2, *local_particles_ptr, bins);
}

#define imy_particle_t_offset(attr) ((size_t)&(((imy_particle_t*)0)->attr))
#define imy_particle_t_particle_offset(attr) ((size_t)&(((imy_particle_t*)0)->particle.attr))

void init_my_particle_type() {
    int lens[5];
    MPI_Aint disp[5];
    MPI_Datatype typs[5];
    MPI_Datatype temp;
    lens[0] = 1;
    disp[0] = imy_particle_t_particle_offset(x); typs[0] = MPI_DOUBLE;
    lens[1] = 1;
    disp[1] = imy_particle_t_particle_offset(y); typs[1] = MPI_DOUBLE;
    lens[2] = 1;
    disp[2] = imy_particle_t_particle_offset(vx); typs[2] = MPI_DOUBLE;
    lens[3] = 1;
    disp[3] = imy_particle_t_particle_offset(vy); typs[3] = MPI_DOUBLE;
    lens[4] = 1;
    disp[4] = imy_particle_t_offset(index); typs[4] = MPI_INT;
    MPI_Type_create_struct(5, lens, disp, typs, &temp);
    MPI_Type_create_resized(temp, 0, sizeof(imy_particle_t), &PARTICLE);
    MPI_Type_commit(&PARTICLE);
}

void scatter_particles(double size, imy_particle_t *particles, imy_particle_t *local_particles,
                       int *n_local_particles) {
    imy_particle_t *particles_by_bin = new imy_particle_t[n];
    int *sendcnts = new int[n_proc];
    int *displs = new int[n_proc];
    if (rank == 0) {
        int i = 0;
        int cur_displs = 0;
        for (int r = 0; r < n_proc; r++) {
            int sendcnt = 0;
            for (int k = 0; k < n; k++) {
                particles[k].particle.bin_idx = bin_of_particle(size, particles[k]);
                int rb = rank_of_bin(particles[k].particle.bin_idx);
                if (rb == r) {
                    particles_by_bin[i] = particles[k];
                    sendcnt++;
                    i++;
                }
            }
            sendcnts[r] = sendcnt;
            displs[r] = cur_displs;
            cur_displs += sendcnt;
        }
    }
    MPI_Bcast(sendcnts, n_proc, MPI_INT, 0, MPI_COMM_WORLD);
    *n_local_particles = sendcnts[rank];
    MPI_Bcast(displs, n_proc, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(particles_by_bin, sendcnts, displs, PARTICLE,
                 local_particles, *n_local_particles, PARTICLE, 0, MPI_COMM_WORLD);

    delete[] sendcnts;
    delete[] displs;
}

int main(int argc, char **argv)
{
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg;

    // Process command line parameters
    if (find_option(argc, argv, "-h") >= 0)
    {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set the number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        printf("-s <filename> to specify a summary file name\n");
        printf("-no turns off all correctness checks and particle output\n");
        return 0;
    }

    n = read_int(argc, argv, "-n", 1000);
    char *savename = read_string(argc, argv, "-o", NULL);
    char *sumname = read_string(argc, argv, "-s", NULL);

    // Set up MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    imy_particle_t *mpi_buffer = new imy_particle_t[3 * n];
    MPI_Buffer_attach(mpi_buffer, 10 * n * sizeof(imy_particle_t));

    // Allocate generic resources
    FILE *fsave = savename && rank == 0 ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen (sumname, "a") : NULL;

    // Initialize and bin particles
    imy_particle_t *particles = (imy_particle_t*) malloc(n * sizeof(imy_particle_t));
    size2 = sqrt(density * n);

    double size = sqrt(0.0005 * n);
    bins_per_side = read_int(argc, argv, "-b", max(1, sqrt(0.0005 * n) / (0.01 * 3)));
    n_bins = bins_per_side * bins_per_side;
    rows_per_proc = ceil(bins_per_side / (float)n_proc);
    if (rank == 0) {
        init_iparticles(n, size, particles);
    }
    // Allocate local particle buffer
    imy_particle_t *local_particles = (imy_particle_t*) malloc(n * sizeof(imy_particle_t));
    int n_local_particles;
    init_my_particle_type();
    // Populate local particle buffers
    scatter_particles(size, particles, local_particles, &n_local_particles);
    // Initialize local bins
    std::vector<bin_t> bins;
    std::vector<int> local_bin_idxs = bins_of_rank(rank);
    init_bins(n_local_particles, size, local_particles, bins);
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer();
    for (int step = 0; step < NSTEPS; step++)
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        // Populate bins with neighboring bins
        exchange_neighbors(size, local_particles, &n_local_particles, bins);

        // Zero out the accelerations
        for (int i = 0; i < n_local_particles; ++i) {
            local_particles[i].particle.ax = local_particles[i].particle.ay = 0;
        }

        // Compute forces between each local bin and its neighbors
        for (std::vector<int>::const_iterator it = local_bin_idxs.begin();
             it != local_bin_idxs.end(); it++) {
            int b1 = *it;
            int b1_row = b1 % bins_per_side;
            int b1_col = b1 / bins_per_side;
            for (int b2_row = max(0, b1_row - 1);
                 b2_row <= min(bins_per_side - 1, b1_row + 1);
                 b2_row++) {
                for (int b2_col = max(0, b1_col - 1);
                     b2_col <= min(bins_per_side - 1, b1_col + 1);
                     b2_col++) {
                    int b2 = b2_row + b2_col * bins_per_side;
                    for (std::list<imy_particle_t*>::const_iterator it1 = bins[b1].particles.begin();
                         it1 != bins[b1].particles.end(); it1++) {
                        for (std::list<imy_particle_t*>::const_iterator it2 = bins[b2].particles.begin();
                             it2 != bins[b2].particles.end(); it2++) {
                            apply_force2((*it1)->particle, (*it2)->particle, &dmin, &davg, &navg);
                        }
                    }
                }
            }
        }

        if (find_option(argc, argv, "-no") == -1) {
            MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

            if (rank == 0) {
                //
                // Computing statistical data
                //
                if (rnavg) {
                    absavg += rdavg/rnavg;
                    nabsavg++;
                }
                if (rdmin < absmin) absmin = rdmin;
            }
        }

        //
        //  move particles
        //
        for (std::vector<int>::const_iterator b_it = local_bin_idxs.begin();
             b_it != local_bin_idxs.end(); b_it++) {
            int b = *b_it;
            std::list<imy_particle_t*>::iterator it = bins[b].particles.begin();
            while (it != bins[b].particles.end()) {
                imy_particle_t *p = *it;
                move2(p->particle);
                int new_b_idx = bin_of_particle(size, *p);
                if (new_b_idx != b) {
                    bin_t *new_bin = &bins[new_b_idx];
                    p->particle.bin_idx = new_b_idx;
                    bins[b].particles.erase(it++);
                    new_bin->incoming.push_back(p);
                } else {
                    it++;
                }
            }
        }

        for (std::vector<int>::const_iterator b_it = local_bin_idxs.begin();
             b_it != local_bin_idxs.end(); b_it++) {
            int b = *b_it;
            bins[b].particles.splice(bins[b].particles.end(), bins[b].incoming);
            bins[b].incoming.clear();
        }

        exchange_moved(size, &local_particles, bins, local_bin_idxs, &n_local_particles);

        //
        //  save current step if necessary
        //
        if (find_option(argc, argv, "-no") == -1) {
            if (savename && (step % SAVEFREQ) == 0) {
                D(printf("rank %d: Gathering particle counts...\n", rank));
                int *local_particle_counts = 0;
                if (rank == 0) {
                    local_particle_counts = new int[n_proc];
                }
                MPI_Gather(&n_local_particles, 1, MPI_INT,
                           local_particle_counts, 1, MPI_INT,
                           0, MPI_COMM_WORLD);
                int *displs = 0;
                if (rank == 0) {
                    displs = new int[n_proc];
                    displs[0] = 0;
                    for (int i = 1; i < n_proc; i++) {
                        displs[i] = displs[i - 1] + local_particle_counts[i - 1];
                    }
                }
                MPI_Gatherv(local_particles, n_local_particles, PARTICLE,
                            particles, local_particle_counts, displs, PARTICLE,
                            0, MPI_COMM_WORLD);
                if (rank == 0) {
                    std::sort(particles, particles + n);
                    my_particle_t *particles_for_save = new my_particle_t[n];
                    for (int i = 0; i < n; i++) {
                        particles_for_save[i].x = particles[i].particle.x;
                        particles_for_save[i].y = particles[i].particle.y;
                    }
                    save2(fsave, n, particles_for_save);
                    delete[] particles_for_save;
                    delete[] local_particle_counts;
                }
            }
        }
    }
    simulation_time = read_timer() - simulation_time;

    if (rank == 0) {
        printf("n = %d, simulation time = %g seconds", n, simulation_time);

        if (find_option(argc, argv, "-no") == -1) {
            if (nabsavg) absavg /= nabsavg;
            //
            //  -the minimum distance absmin between 2 particles during the run of the simulation
            //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
            //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
            //
            //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
            //
            printf(", absmin = %lf, absavg = %lf", absmin, absavg);
            if (absmin < 0.4) printf("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
            if (absavg < 0.8) printf("\nThe average distance is below 0.8 meaning that most particles are not interacting");
        }
        printf("\n");

        //
        // Printing summary data
        //
        if (fsum) {
            fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
        }
    }

    //
    //  release resources
    //
    if (fsum) {
        fclose(fsum);
    }
    free(particles);
    if (fsave) {
        fclose(fsave);
    }

    MPI_Finalize();

    return 0;
}
