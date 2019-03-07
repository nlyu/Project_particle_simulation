#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <list>
#include <vector>
#include <cmath>
#include <algorithm>
#include <float.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "common.h"

#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

#define find_particle_offset(target)          ((size_t)& (((my_particle_index*)0)->particle.target))
#define for_bin(i, j)     for(int i = max(0, j - 1); i <= min(n_bins_side - 1, j + 1); ++i)

int n_bins_side, n_bins, n_proc, rank, n, n_rows_proc;
double size2;

MPI_Datatype PARTICLE;

//original particle type
class my_particle_t{
public:
      double x, y, vx, vy, ax, ay;
};

//wrap the paritcles with index, providing helper function for mpi
class my_particle_index{
public:
    my_particle_t particle;
    int index;
    int bin_idx;

    void move()
    {
        //  slightly simplified Velocity Verlet integration
        //  conserves energy better than explicit Euler method
        this->particle.vx += this->particle.ax * dt;
        this->particle.vy += this->particle.ay * dt;
        this->particle.x  += this->particle.vx * dt;
        this->particle.y  += this->particle.vy * dt;

        //  bounce from walls
        while(this->particle.x < 0 || this->particle.x > size2 )
        {
            this->particle.x  = this->particle.x < 0 ? -this->particle.x : 2 * size2- this->particle.x;
            this->particle.vx = -this->particle.vx;
        }
        while( this->particle.y < 0 || this->particle.y > size2 )
        {
            this->particle.y  = this->particle.y < 0 ? -this->particle.y : 2*size2-this->particle.y;
            this->particle.vy = -this->particle.vy;
        }
    }

    void apply_force(my_particle_t &neighbor , double *dmin, double *davg, int *navg)
    {
        double dx = neighbor.x - this->particle.x;
        double dy = neighbor.y - this->particle.y;
        double r2 = dx * dx + dy * dy;
        if(r2 > cutoff*cutoff)
            return;
    	  if (r2 != 0){
    	      if (r2/(cutoff*cutoff) < *dmin * (*dmin))
    	      *dmin = sqrt(r2)/cutoff;
            (*davg) += sqrt(r2)/cutoff;
            (*navg) ++;
        }

        r2 = fmax( r2, min_r*min_r );
        double r = sqrt( r2 );

        //  very simple short-range repulsive force
        double coef = ( 1 - cutoff / r ) / r2 / mass;
        this->particle.ax += coef * dx;
        this->particle.ay += coef * dy;
    }
};

//bins to divide the canvas into smaller squares
class bin_t{
public:
    std::list<my_particle_index*> particles; //saves the particles in the bins
    std::list<my_particle_index*> newparticles; //saves the particles that joining the bin

    //add one new particles into the bin
    void add_particles(my_particle_index * p){
        this->particles.push_back(p);
    }

    //add all incoming particles into the bin
    void splice(){
        this->particles.splice(this->particles.end(), this->newparticles);
    }

    //remove all particles in the new paritcles buffer
    void clear_newparticles(){
        this->newparticles.clear();
    }

    //remove all particle in the bins
    void clear_particles(){
        this->particles.clear();
    }

    //update the particles in the bin after the move
    void binning(){
        splice();
        clear_newparticles();
    }

    //get neighbor particles
    void neighbor_particles(std::vector<my_particle_index> &res){
        for(auto &p: this->particles){
            res.push_back(*p);
        }
    }

    //move the particles in the bins for one time step
    void moved_particles_in_bin(std::vector<bin_t> &bins, int b_it){
        auto it = this->particles.begin();
        while (it != this->particles.end()) {
            my_particle_index *p = *it;
            p->move();
            double bin_side_len = size2 / n_bins_side;
            int row_b = floor(p->particle.x / bin_side_len), col_b = floor(p->particle.y / bin_side_len);
            int new_b_idx =  row_b + col_b * n_bins_side;
            if (new_b_idx != b_it) { //if particle is not in the same position
                p->bin_idx = new_b_idx;
                this->particles.erase(it++);
                bins[new_b_idx].newparticles.push_back(p);
            } else {
                it++;
            }
        }
    }
};

bool operator<(const my_particle_index &a, const my_particle_index &b) {
    return a.index < b.index;
}


//initialize the position in the particles
void init_particles_mpi(int rank, int n, double size, my_particle_index *p) {
    if(rank != 0)
        return;
    srand48( time( NULL ) );

    int sx = (int)ceil(sqrt((double)n));
    int sy = (n+sx-1)/sx;

    int *shuffle = (int*)malloc( n * sizeof(int) );
    for( int i = 0; i < n; i++ )
        shuffle[i] = i;

    for( int i = 0; i < n; i++ )
    {
        //  make sure particles are not spatially sorted
        int j = lrand48()%(n-i);
        int k = shuffle[j];
        shuffle[j] = shuffle[n-i-1];

        //  distribute particles evenly to ensure proper spacing
        p[i].particle.x = size*(1.+(k%sx))/(1+sx);
        p[i].particle.y = size*(1.+(k/sx))/(1+sy);

        //  assign random velocities within a bound
        p[i].particle.vx = drand48()*2-1;
        p[i].particle.vy = drand48()*2-1;

        p[i].index = i;
    }
    free( shuffle );
}

//get the bin id for a particle
int particle_bin(double canvas_side_len, my_particle_index &p) {
    int bin_row = p.particle.x / (canvas_side_len / n_bins_side);
    int bin_col = p.particle.y / (canvas_side_len / n_bins_side);
    return bin_col * n_bins_side + bin_row;
}

//match the particles to the corresponding bin
void assign_particles_to_bins(int n, double canvas_side_len, my_particle_index *particles, std::vector<bin_t> &bins) {
    for (int i = 0; i < n; ++i) {
        my_particle_index &p = particles[i];
        int b_idx = particle_bin(canvas_side_len, p);
        p.bin_idx = particle_bin(canvas_side_len, p);
        bins[b_idx].add_particles(&p);
    }
}

//initialize the bins in the canvas
void init_bins(int n, double size, my_particle_index *particles, std::vector<bin_t> &bins) {
    for (int b_idx = 0; b_idx < n_bins; b_idx++) {
        bin_t b;
        bins.push_back(b);
    }
    assign_particles_to_bins(n, size, particles, bins);
}

//get the which process the bin is in
int rank_of_bin(int b_idx) {
    int b_row = b_idx % n_bins_side;
    return b_row / n_rows_proc;
}

//get bins in this processor
std::vector<int> bins_of_rank(int rank) {
    std::vector<int> res;
    int row_s = rank * n_rows_proc,
        row_e = min(n_bins_side, n_rows_proc * (rank + 1));
    for (int row = row_s; row < row_e; ++row)
        for (int col = 0; col < n_bins_side; ++col)
            res.push_back(row + col * n_bins_side);
    return res;
}

//get boerder particles around this processor
std::vector<my_particle_index> get_rank_border_particles(int nei_rank, std::vector<bin_t> &bins) {
    int row;
    if (nei_rank < rank) row = rank * n_rows_proc;
    else row = n_rows_proc * (rank + 1) - 1;

    std::vector<my_particle_index> res;
    if (row < 0 || row >= n_bins_side) return res;
    for (int col = 0; col < n_bins_side; ++col) {
        bin_t &b = bins[row + col * n_bins_side];
        b.neighbor_particles(res);
    }
    return res;
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

    // Allocate generic resources
    FILE *fsave = savename && rank == 0 ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen (sumname, "a") : NULL;

    // Allocate receieve & send buffer
    my_particle_index *mpi_buff = new my_particle_index[3 * n];
    MPI_Buffer_attach(mpi_buff, 10 * n * sizeof(my_particle_index));

    // Allocate grobal particle buffer
    my_particle_index *particles = (my_particle_index*) malloc(n * sizeof(my_particle_index));
    // Allocate local particle buffer
    my_particle_index *local_particles = (my_particle_index*) malloc(n * sizeof(my_particle_index));

    // Allocate particle simulation coeffiency
    size2 = sqrt(density * n);
    double size = sqrt(density * n);

    n_bins_side = max(1, sqrt(density * n) / (0.01 * 3));
    n_bins = n_bins_side * n_bins_side;
    n_rows_proc = ceil(n_bins_side / (float)n_proc);

    init_particles_mpi(rank, n, size, particles);

    // initialize MPI PARTICLE type
    int n_local_particles, particle_size;
    int counter, cur_displs, counter_send;
    int lens[5];
    int counter_sends[n_proc];
    int displs[n_proc];

    MPI_Aint disp[5];
    MPI_Datatype temp;
    MPI_Datatype types[5];

    particle_size = sizeof(my_particle_index);
    std::fill_n(lens, 5, 1);
    std::fill_n(types, 4, MPI_DOUBLE);
    types[4] = MPI_INT;
    disp[0] = find_particle_offset(x);
    disp[1] = find_particle_offset(y);
    disp[2] = find_particle_offset(vx);
    disp[3] = find_particle_offset(vy);
    disp[4] = (size_t)&(((my_particle_index*)0)->index);

    MPI_Type_create_struct(5, lens, disp, types, &temp);
    MPI_Type_create_resized(temp, 0, particle_size, &PARTICLE);
    MPI_Type_commit(&PARTICLE);

    //scatter the paritcles to each processors base on location
    my_particle_index *particles_by_bin = new my_particle_index[n];
    for (int pro = cur_displs = counter = 0; pro < n_proc && rank == 0; cur_displs += counter_sends[pro], ++pro) {
        counter_send = 0;
        for (int i = 0; i < n; ++i) {
            if (rank_of_bin(particle_bin(size, particles[i])) != pro)      continue;
            particles_by_bin[counter] = particles[i];
            counter_send++;
            counter++;
        }
        counter_sends[pro] = counter_send;
        displs[pro] = cur_displs;
    }

    // MPI initialize and send all other var in other processors
    MPI_Bcast(&counter_sends[0], n_proc, MPI_INT, 0, MPI_COMM_WORLD);
    n_local_particles = counter_sends[rank];
    MPI_Bcast(&displs[0], n_proc, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(particles_by_bin, &counter_sends[0], &displs[0], PARTICLE, local_particles, n_local_particles, PARTICLE, 0, MPI_COMM_WORLD);

    // Initialize local bins
    std::vector<bin_t> bins;
    std::vector<int> local_bin_idxs = bins_of_rank(rank);
    init_bins(n_local_particles, size, local_particles, bins);

    //  simulate a number of time steps
    double simulation_time = read_timer();
    for (int step = 0; step < NSTEPS; step++)
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        // exchange neighbors particles for force computation
        std::vector<int> nei_ranks;
        if (rank > 0)
            nei_ranks.push_back(rank - 1);
        if (rank + 1 < n_proc)
            nei_ranks.push_back(rank + 1);
        for(auto &nei_rank : nei_ranks){
            std::vector<my_particle_index> border_particles = get_rank_border_particles(nei_rank, bins);
            int n_b_particles = border_particles.size();
            const void *buf = n_b_particles == 0 ? 0 : &border_particles[0];
            MPI_Request request;
            MPI_Ibsend(buf, n_b_particles, PARTICLE, nei_rank, 0, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
        }

        // neighbors collect border particles and assign to bins
        my_particle_index *cur_pos = local_particles + n_local_particles;
        int n_particles_received = 0;
        for (auto &nei_rank : nei_ranks){
            MPI_Status status;
            MPI_Recv(cur_pos, n, PARTICLE, nei_rank, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, PARTICLE, &n_particles_received);
            assign_particles_to_bins(n_particles_received, size, cur_pos, bins);
            cur_pos += n_particles_received;
            n_local_particles += n_particles_received;
        }

        // Zero out the accelerations
        for (int i = 0; i < n_local_particles; ++i) {
            local_particles[i].particle.ax = local_particles[i].particle.ay = 0;
        }

        // Compute forces between each local bin and its neighbors
        for (auto &idx:local_bin_idxs){
            int b1_row = idx % n_bins_side;
            int b1_col = idx / n_bins_side;
            for_bin(b2_row, b1_row){
                for_bin(b2_col, b1_col){
                    int b2 = b2_row + b2_col * n_bins_side;
                    for (auto &it1: bins[idx].particles) {
                        for (auto &it2: bins[b2].particles) {
                             it1->apply_force(it2->particle, &dmin, &davg, &navg);
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
                // Computing statistical data
                if (rnavg) {
                    absavg += rdavg/rnavg;
                    nabsavg++;
                }
                if (rdmin < absmin) absmin = rdmin;
            }
        }

        //  move particles in each bins
        for (auto &b_it: local_bin_idxs) {
            bins[b_it].moved_particles_in_bin(bins, b_it);
        }

        // refresh the particles in bins
        for (auto &b_it: local_bin_idxs) {
            bins[b_it].binning();
        }

        // exchange particles after moveing
        std::vector<int> neighbor_ranks;
        if (rank > 0)
            neighbor_ranks.push_back(rank - 1);
        if (rank + 1 < n_proc)
            neighbor_ranks.push_back(rank + 1);

        for (auto &nei_rank : neighbor_ranks) {
            std::vector<int> cur_bins = bins_of_rank(nei_rank);
            std::vector<my_particle_index> moved_particles;
            for (auto b_idx : cur_bins)
                for(auto &p: bins[b_idx].newparticles)
                    moved_particles.push_back(*p);
            int n_moved_p = moved_particles.size();
            const void *buf = n_moved_p == 0 ? 0 : &moved_particles[0];
            MPI_Request request;
            MPI_Ibsend(buf, n_moved_p, PARTICLE, nei_rank, 0, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
        }

        my_particle_index *new_local_particles = new my_particle_index[n];
        my_particle_index *tmp_pos = new_local_particles;

        for(auto &nei_rank : neighbor_ranks){
            MPI_Status status;
            MPI_Recv(tmp_pos, n, PARTICLE, nei_rank, 0, MPI_COMM_WORLD, &status);
            int n_particles_received;
            MPI_Get_count(&status, PARTICLE, &n_particles_received);
            tmp_pos += n_particles_received;
        }

        for(auto &b_idx : local_bin_idxs){
            for(auto &p : bins[b_idx].particles){
                *tmp_pos = *p;
                tmp_pos++;
            }
        }

        // Apply new_local_particles
        local_particles = new_local_particles;
        n_local_particles = tmp_pos - new_local_particles;
        // Rebin all particles
        bins.clear();
        init_bins(n_local_particles, size, new_local_particles, bins);
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
