# Project_particle_simulation

parallel computation simulate the particle motion

cs267 hw2

Nuochen Lyu 
Xiaoyun Zhao

Final Result
![Result](https://github.com/nlyu/Project_particle_simulation/blob/master/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-03-30%20%E4%B8%8B%E5%8D%889.53.54.png)

# Set up and run

Login to cori or any super computer center. For CUDA part you need CMU PSC bridge computer.

For CUDA part you need `module load cuda`

Compile: `make`

Request Node from Cori: `salloc -N 1 -A mp309 -t 60:00 -q debug --qos=interactive -C haswell`

Run: See Job-block

# Overview

[Link for Complete Info and detail](https://sites.google.com/lbl.gov/cs267-spr2019/hw-2?authuser=0)

Part1: Openmp, multi-thread for particle interaction [Report Link](https://docs.google.com/document/d/1TtZm3EWSEm7xQ7od3GWZEeGk8oU9POQuhkM0uKdlBAo/edit?usp=sharing)

Part2: MPI, multi-processor for particle interaction [Report Link](https://docs.google.com/document/d/1Yrqd1o0ddPRK6ziEUl3gGZ5Oc_WnX0KTBs9h7cgJy2g/edit?usp=sharing)

Part3: Cuda, Gpu for particle interaction [Report Link](https://docs.google.com/document/d/16Y9FK3KKkxwppBHggxtwby4aGbJGUOoGvAdiPV5MwEI/edit?usp=sharing)



