all: main

main: main.o
	gcc -lm -pg -o main main.o

main.o: main.c
	gcc -std=gnu99 -pg -c main.c

clean:
	rm -f main main.o gmon.out

test: main
	./main
	gprof main gmon.out > analysis.txt