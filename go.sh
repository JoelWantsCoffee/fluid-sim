#!/bin/bash -l
#
#SBATCH --job-name=COSC3500_Project
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=0-00:05:00
#SBATCH --partition=cosc

#You could add these to your bashrc if you wanted
module add intel/2018.1.163
#intel MPI was conflicting with openMP for some reason?
module unload mpi/intelmpi/2018.1.163
module add mpi/openmpi-x86_64
module add compilers/cuda/11.1
make clean
make all

./fluid.out