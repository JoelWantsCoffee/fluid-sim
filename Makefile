all: main

main: main.o
	gcc -lm -o main -pg main.o

main.o: main.c
	gcc -mavx2 -std=gnu99 -O3 -pg -c main.c

debug.o:
	gcc -std=gnu99 -g -c main.c -o debug.o

debug: debug.o
	gcc -mavx2 -lm -o debug debug.o
	gdb debug

clean:
	rm -f main debug main.o debug.o gmon.out

test: main
	./main
	gprof main gmon.out > analysis.txt
