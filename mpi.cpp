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

#define imy_particle_t_offset(attr) ((size_t)&(((imy_particle_t*)0)->attr))
#define imy_particle_t_particle_offset(attr) ((size_t)&(((imy_particle_t*)0)->particle.attr))
#define for_bin(i, j)     for(int i = max(0, j - 1); i <= min(bins_per_side - 1, j + 1); ++i)

int bins_per_side, n_bins, n_proc, rank, n, rows_per_proc;
double size2;

MPI_Datatype PARTICLE;

class my_particle_t{
public:
      double x, y, vx, vy, ax, ay;
};

class imy_particle_t{
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


class bin_t{
public:
    std::list<imy_particle_t*> particles;
    std::list<imy_particle_t*> incoming;

    void add_particles(imy_particle_t * p){
        this->particles.push_back(p);
    }

    void splice(){
        this->particles.splice(this->particles.end(), this->incoming);
    }

    void clear_incoming(){
        this->incoming.clear();
    }

    void clear_particles(){
        this->particles.clear();
    }

    void binning(){
        splice();
        clear_incoming();
    }

    void neighbor_particles(std::vector<imy_particle_t> &res){
        for(auto &p: this->particles){
            res.push_back(*p);
        }
    }

    void moved_particles_in_bin(std::vector<bin_t> &bins, int b_it){
        auto it = this->particles.begin();
        while (it != this->particles.end()) {
            imy_particle_t *p = *it;
            p->move();
            double bin_side_len = size2 / bins_per_side;
            int row_b = floor(p->particle.x / bin_side_len), col_b = floor(p->particle.y / bin_side_len);
            int new_b_idx =  row_b + col_b * bins_per_side;
            if (new_b_idx != b_it) { //if particle is not in the same position
                p->bin_idx = new_b_idx;
                this->particles.erase(it++);
                bins[new_b_idx].incoming.push_back(p);
            } else {
                it++;
            }
        }
    }
};

bool operator<(const imy_particle_t &a, const imy_particle_t &b) {
    return a.index < b.index;
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

void init_particles_mpi(int rank, int n, double size, imy_particle_t *p) {
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

int bin_of_particle(double canvas_side_len, imy_particle_t &p) {
    double bin_side_len = canvas_side_len / bins_per_side;
    int row_b = floor(p.particle.x / bin_side_len), col_b = floor(p.particle.y / bin_side_len);
    return row_b + col_b * bins_per_side;
}

std::vector<int> get_rank_neighbors(int rank) {
    std::vector<int> rank_neis;
    if (rank > 0)
        rank_neis.push_back(rank - 1);
    if (rank + 1 < n_proc)
        rank_neis.push_back(rank + 1);
    return rank_neis;
}

void assign_particles_to_bins(int n, double canvas_side_len, imy_particle_t *particles, std::vector<bin_t> &bins) {
    for (int i = 0; i < n; ++i) {
        int b_idx = particles[i].bin_idx = bin_of_particle(canvas_side_len, particles[i]);
        // bins[b_idx].particles.push_back(&particles[i]);
        bins[b_idx].add_particles(&particles[i]);
    }
}

void init_bins(int n, double size, imy_particle_t *particles, std::vector<bin_t> &bins) {
    // Create bins (most do not belong to this task)
    for (int b_idx = 0; b_idx < n_bins; b_idx++) {
        bin_t b;
        bins.push_back(b);
    }
    assign_particles_to_bins(n, size, particles, bins);
}

int rank_of_bin(int b_idx) {
    int b_row = b_idx % bins_per_side;
    return b_row / rows_per_proc;
}


std::vector<int> bins_of_rank(int rank) {
    std::vector<int> res;
    int row_s = rank * rows_per_proc,
        row_e = min(bins_per_side, rows_per_proc * (rank + 1));
    for (int row = row_s; row < row_e; ++row)
        for (int col = 0; col < bins_per_side; ++col)
            res.push_back(row + col * bins_per_side);
    return res;
}


std::vector<imy_particle_t> get_rank_border_particles(int nei_rank, std::vector<bin_t> &bins) {
    int row;
    if (nei_rank < rank) row = rank * rows_per_proc;
    else row = rows_per_proc * (rank + 1) - 1;

    std::vector<imy_particle_t> res;
    if (row < 0 || row >= bins_per_side) return res;
    for (int col = 0; col < bins_per_side; ++col) {
        bin_t &b = bins[row + col * bins_per_side];
        b.neighbor_particles(res);
    }
    return res;
}

void exchange_neighbors(double size, imy_particle_t *local_particles,
                        int *n_local_particles, std::vector<bin_t> &bins){

    std::vector<int> neighbor_ranks = get_rank_neighbors(rank);
    for(int i = 0; i < neighbor_ranks.size(); ++i){
        std::vector<imy_particle_t> border_particles = get_rank_border_particles(neighbor_ranks[i], bins);
        MPI_Request request;
        if(border_particles.empty()){
            MPI_Ibsend(0, border_particles.size(), PARTICLE, neighbor_ranks[i], 0, MPI_COMM_WORLD, &request);
        } else {
            MPI_Ibsend(&border_particles[0], border_particles.size(), PARTICLE, neighbor_ranks[i], 0, MPI_COMM_WORLD, &request);
        }
        MPI_Request_free(&request);
    }

    imy_particle_t *cur_pos = local_particles + *n_local_particles;
    int num_particles_received;
    for (std::vector<int>::const_iterator it = neighbor_ranks.begin(); it != neighbor_ranks.end(); it++){
        MPI_Status status;
        MPI_Recv(cur_pos, n, PARTICLE, *it, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, PARTICLE, &num_particles_received);
        assign_particles_to_bins(num_particles_received, size, cur_pos, bins);
        cur_pos += num_particles_received;
        *n_local_particles += num_particles_received;
    }
}

void exchange_moved(double size, imy_particle_t **local_particles_ptr,
                    std::vector<bin_t> &bins, std::vector<int> &local_bin_idxs,
                    int *n_local_particles) {
    std::vector<int> neighbor_ranks = get_rank_neighbors(rank);
    for (int i = 0; i < neighbor_ranks.size(); i++) {
        std::vector<imy_particle_t> moved_particles;
        for (int b_idx = 0; b_idx < n_bins; b_idx++) {
            if (rank_of_bin(b_idx) == neighbor_ranks[i]) {
                for(auto &it: bins[b_idx].incoming){
                    moved_particles.push_back(*it);
                }
            }
        }
        MPI_Request request;
        if(moved_particles.empty()){
            MPI_Ibsend(0, moved_particles.size(), PARTICLE, neighbor_ranks[i], 0, MPI_COMM_WORLD, &request);
        } else {
            MPI_Ibsend(&moved_particles[0], moved_particles.size(), PARTICLE, neighbor_ranks[i], 0, MPI_COMM_WORLD, &request);
        }
        MPI_Request_free(&request);
    }

    imy_particle_t *new_local_particles = new imy_particle_t[n];
    imy_particle_t *cur_pos = new_local_particles;
    //for (std::vector<int>::const_iterator it = neighbor_ranks.begin(); it != neighbor_ranks.end(); it++) {
    for(auto &it: neighbor_ranks){
        MPI_Status status;
        MPI_Recv(cur_pos, n, PARTICLE, it, 0, MPI_COMM_WORLD, &status);
        int num_particles_received;
        MPI_Get_count(&status, PARTICLE, &num_particles_received);
        cur_pos += num_particles_received;
    }

    // for (std::vector<int>::const_iterator b_it = local_bin_idxs.begin(); b_it != local_bin_idxs.end(); b_it++) {//
    for(auto &b_it: local_bin_idxs){
        for(auto &p_it: bins[b_it].particles){
        //for (std::list<imy_particle_t*>::const_iterator p_it = bins[ *b_it].particles.begin(); p_it != bins[*b_it].particles.end(); p_it++) {
            *cur_pos = *p_it;
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

    // receieve & send buffer
    imy_particle_t *mpi_buffer = new imy_particle_t[3 * n];
    MPI_Buffer_attach(mpi_buffer, 10 * n * sizeof(imy_particle_t));

    // particle initi\]]\
    alization
    imy_particle_t *particles = (imy_particle_t*) malloc(n * sizeof(imy_particle_t));
    // Allocate local particle buffer
    imy_particle_t *local_particles = (imy_particle_t*) malloc(n * sizeof(imy_particle_t));

    size2 = sqrt(density * n);
    double size = sqrt(density * n);

    bins_per_side = max(1, sqrt(density * n) / (0.01 * 3));
    n_bins = bins_per_side * bins_per_side;
    rows_per_proc = ceil(bins_per_side / (float)n_proc);

    init_particles_mpi(rank, n, size, particles);

    // initialize MPI PARTICLE
    int n_local_particles, particle_size;
    int counter, cur_displs, counter_send;
    int lens[5];
    int counter_sends[n_proc];
    int displs[n_proc];

    MPI_Aint disp[5];
    MPI_Datatype temp;
    MPI_Datatype types[5];

    particle_size = sizeof(imy_particle_t);
    std::fill_n(lens, 5, 1);
    std::fill_n(types, 4, MPI_DOUBLE);
    types[4] = MPI_INT;
    disp[0] = imy_particle_t_particle_offset(x);
    disp[1] = imy_particle_t_particle_offset(y);
    disp[2] = imy_particle_t_particle_offset(vx);
    disp[3] = imy_particle_t_particle_offset(vy);
    disp[4] = imy_particle_t_offset(index);

    MPI_Type_create_struct(5, lens, disp, types, &temp);
    MPI_Type_create_resized(temp, 0, particle_size, &PARTICLE);
    MPI_Type_commit(&PARTICLE);

    //scatter the paritcles to each processors
    imy_particle_t *particles_by_bin = new imy_particle_t[n];
    for (int pro = cur_displs = counter = 0; pro < n_proc && rank == 0; cur_displs += counter_sends[pro], ++pro) {
        counter_send = 0;
        for (int i = 0; i < n; ++i) {
            if (rank_of_bin(bin_of_particle(size, particles[i])) != pro)      continue;
            particles_by_bin[counter] = particles[i];
            counter_send++;
            counter++;
        }
        counter_sends[pro] = counter_send;
        displs[pro] = cur_displs;
    }

    MPI_Bcast(&counter_sends[0], n_proc, MPI_INT, 0, MPI_COMM_WORLD);
    n_local_particles = counter_sends[rank];
    MPI_Bcast(&displs[0], n_proc, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(particles_by_bin, &counter_sends[0], &displs[0], PARTICLE, local_particles, n_local_particles, PARTICLE, 0, MPI_COMM_WORLD);

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
        // exchange_neighbors(size, local_particles, &n_local_particles, bins);
        std::vector<int> nei_ranks = get_rank_neighbors(rank);
        for(auto &nei_rank : nei_ranks){
            std::vector<imy_particle_t> border_particles = get_rank_border_particles(nei_rank, bins);
            int n_b_particles = border_particles.size();
            const void *buf = n_b_particles == 0 ? 0 : &border_particles[0];
            MPI_Request request;
            MPI_Ibsend(buf, n_b_particles, PARTICLE, nei_rank, 0, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
        }
        // neighbors collect border particles and assign to bins
        imy_particle_t *cur_pos = local_particles + n_local_particles;
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
            int b1_row = idx % bins_per_side;
            int b1_col = idx / bins_per_side;
            for_bin(b2_row, b1_row){
                for_bin(b2_col, b1_col){
                    int b2 = b2_row + b2_col * bins_per_side;
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

        //  move particles
        for (auto &b_it: local_bin_idxs) {
            bins[b_it].moved_particles_in_bin(bins, b_it);
        }

        for (auto &b_it: local_bin_idxs) {
            bins[b_it].binning();
        }

        // exchange_moved(size, &local_particles, bins, local_bin_idxs, &n_local_particles);
        std::vector<int> neighbor_ranks = get_rank_neighbors(rank);
        for (auto &nei_rank : neighbor_ranks) {
            std::vector<int> cur_bins = bins_of_rank(nei_rank);
            std::vector<imy_particle_t> moved_particles;
            for (auto b_idx : cur_bins)
                for(auto &p: bins[b_idx].incoming)
                    moved_particles.push_back(*p);
            int n_moved_p = moved_particles.size();
            const void *buf = n_moved_p == 0 ? 0 : &moved_particles[0];
            MPI_Request request;
            MPI_Ibsend(buf, n_moved_p, PARTICLE, nei_rank, 0, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
        }
        imy_particle_t *new_local_particles = new imy_particle_t[n];
        imy_particle_t *tmp_pos = new_local_particles;
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
