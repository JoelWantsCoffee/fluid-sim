# module load compiler-rt/latest
# module add mkl/latest
# module load cuda/11.1

all: fluid

fluid: fluid.o fluid_cuda.o
	nvcc -o fluid fluid.o fluid_cuda.o -lm -pg

fluid.o: fluid.c
	g++ -O3 -c fluid.c

fluid_cuda.o: fluid_cuda.cu fluid_cuda.cuh
	nvcc --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -O3 fluid_cuda.cu

debug.o:
	g++ -g -c fluid.c -o debug.o

debug: clean debug.o
	g++ -o debug debug.o -lm

clean:
	rm -f fluid debug fluid.o debug.o fluid_cuda.o gmon.out

test: fluid
	./fluid
	gprof fluid gmon.out > analysis.txt