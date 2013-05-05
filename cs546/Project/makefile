all: 
	mpicc -c 2D-Conv.c
	mpicc -o 2D-Conv 2D-Conv.o -lm
	mpicc -c 2D-Conv-P1.c
	mpicc -o 2D-Conv-P1 2D-Conv-P1.o -lm
	mpicc -c 2D-Conv-P2.c
	mpicc -o 2D-Conv-P2 2D-Conv-P2.o -lm
	mpicc -c 2D-Conv-P3.c
	mpicc -o 2D-Conv-P3 2D-Conv-P3.o -lm

2D-Conv:
	mpicc -c 2D-Conv.c
	mpicc -o 2D-Conv 2D-Conv.o -lm
	
2D-Conv-P1:
	mpicc -c 2D-Conv-P1.c
	mpicc -o 2D-Conv-P1 2D-Conv-P1.o -lm
	
2D-Conv-P2:
	mpicc -c 2D-Conv-P2.c
	mpicc -o 2D-Conv-P2 2D-Conv-P2.o -lm
	
2D-Conv-P3:
	mpicc -c 2D-Conv-P3.c
	mpicc -o 2D-Conv-P3 2D-Conv-P3.o -lm

run:
	./2D-Conv

run_v1:
	mpirun -np 2 ./2D-Conv-P1

run_v2:
	mpirun -np 2 ./2D-Conv-P2

run_v3:
	mpirun -np 2 ./2D-Conv-P3