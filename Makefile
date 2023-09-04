all: main

main: main.o
	gcc -lm -o main main.o

main.o: main.c
	gcc -std=gnu99 -pg -c main.c

debug.o:
	gcc -std=gnu99 -g -c main.c

debug: debug.o
	gcc -lm -o main main.o

clean:
	rm -f main main.o gmon.out

test: main
	./main
	gprof main gmon.out > analysis.txt