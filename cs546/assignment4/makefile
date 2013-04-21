all: 
	mpicc -c get_data.c
	mpicc -o get_data get_data.o
	mpicc -c gauss_mpi.c
	mpicc -o gauss_mpi gauss_mpi.o
	mpicc -c gauss_mpi2.c
	mpicc -o gauss_mpi2 gauss_mpi2.o
	mpicc -c gauss_mpi2.c
	mpicc -o gauss_mpi2 gauss_mpi2.o

get_data:
	mpicc -c get_data.c
	mpicc -o get_data get_data.o

gauss_v1:
	mpicc -c gauss_mpi.c
	mpicc -o gauss_mpi gauss_mpi.o
	
gauss_v2:
	mpicc -c gauss_mpi2.c
	mpicc -o gauss_mpi2 gauss_mpi2.o
	
gauss_v3:
	mpicc -c gauss_mpi3.c
	mpicc -o gauss_mpi3 gauss_mpi3.o
	
run_v1:
	mpirun -np 2 ./gauss_mpi 5 2

run_v2:
	mpirun -np 2 ./gauss_mpi2 5 2

run_v3:
	mpirun -np 2 ./gauss_mpi3 5 2

run_get_data:
	mpirun -np 2 ./get_data