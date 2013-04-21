all: 
	mpicc -c gauss_mpi.c
	mpicc -o gauss_mpi gauss_mpi.o

run:
	mpirun -np 2 ./gauss_mpi.exe 5 2