# module load compiler-rt/latest
# module add mkl/latest
# module add mpi/openmpi-x86_64
# module load cuda/11.1

all: fluid

fluid: fluid.o fluid_cuda.o
	nvcc -O3 -lm -o fluid.out -pg fluid.o fluid_cuda.o

fluid.o: fluid.c
	g++ -O3 -fno-pie -fno-builtin -fopenmp -mfma -mavx2 -pg -c fluid.c

fluid_cuda.o: fluid_cuda.cu fluid_cuda.cuh
	nvcc --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -O3 -c fluid_cuda.cu

debug.o:
	g++ -g -c fluid.c -o debug.o

debug: clean debug.o
	g++ -o debug debug.o -lm

clean:
	rm -f fluid.out debug fluid.o debug.o fluid_cuda.o gmon.out

test: fluid
	./fluid
	nvprof fluid gmon.out > analysis.txt
