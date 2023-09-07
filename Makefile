all: main

main: main.o
	gcc -fopenmp -lm -o main -pg main.o

main.o: main.c
	gcc -fopenmp -mfma -mavx2 -std=gnu99 -O3 -pg -c main.c

debug.o:
	gcc -fopenmp -mfma -mavx2 -std=gnu99 -g -c main.c -o debug.o

debug: clean debug.o
	gcc -fopenmp -lm -o debug debug.o

clean:
	rm -f main debug main.o debug.o gmon.out

test: main
	./main
	gprof main gmon.out > analysis.txt
