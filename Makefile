all: fluid

fluid: fluid.o
	gcc -O2 -fno-pie -fno-builtin -fopenmp -lm -o fluid -pg fluid.o

fluid.o: fluid.c
	gcc -O2 -fno-pie -fno-builtin -fopenmp -mfma -mavx2 -std=gnu99 -pg -c fluid.c

debug.o:
	gcc -fopenmp -mfma -mavx2 -std=gnu99 -g -c fluid.c -o debug.o

debug: clean debug.o
	gcc -fopenmp -lm -o debug debug.o

clean:
	rm -f fluid debug fluid.o debug.o gmon.out

test: fluid
	./fluid
	gprof fluid gmon.out > analysis.txt
