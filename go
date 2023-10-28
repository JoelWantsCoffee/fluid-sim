#!/bin/bash -l
#
#SBATCH --job-name=COSC3500_Project
#SBATCH --cpus-per-task=4
#SBATCH --gres gpu:1
#SBATCH --time=0-00:01:00

#You could add these to your bashrc if you wanted
module load compiler-rt/latest
module add mkl/latest
module add mpi/openmpi-x86_64
module load cuda/11.1

#I would have expected the module loads to add these, but apparently not
export PATH=/opt/local/stow/cuda-11.1/bin:$PATH
export PATH=/usr/lib64/openmpi/bin:$PATH

make clean
make all

echo starting...

# nvprof fluid.out gmon.out > analysis.txt
time ./fluid.out