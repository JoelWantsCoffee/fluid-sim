# module load compiler-rt/latest
# module add mkl/latest
# module add mpi/openmpi-x86_64
# module load cuda/11.1

all: fluid

fluid: fluid.o fluid_cuda.o
	nvcc -O2 -lm -o fluid.out -pg fluid.o fluid_cuda.o

fluid.o: fluid.c
	g++ -O2 -fno-pie -fno-builtin -fopenmp -mfma -mavx2 -pg -c fluid.c

fluid_cuda.o: fluid_cuda.cu fluid_cuda.cuh
	nvcc --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -O2 -c fluid_cuda.cu

debug.o:
	g++ -g -c fluid.c -o debug.o

debug: clean debug.o
	g++ -o debug debug.o -lm

clean:
	rm -f fluid debug fluid.o debug.o fluid_cuda.o gmon.out

test: fluid
	./fluid
	nvprof fluid gmon.out > analysis.txt
