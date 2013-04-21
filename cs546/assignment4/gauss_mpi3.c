#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

#include "mpi.h"

#define MAXN 2000
void mpi_gauss();
void print_X();

// Serial algorithm is kept around for easy comparison of timings

void gauss( double *A, double *B, double *X, int N )
{
  int norm, col, row;
  double end_t;
  double start_t = MPI_Wtime();

  for( norm = 0; norm < N; norm++ ) {		// k = current row
    for( col= norm + 1; col < N; col++ ) {		// in division step
      if( A[norm*N+norm] != 0)
	    /* The Multiplier */
        A[norm*N+col] = A[norm*N+col] / A[norm*N+norm];
      else
        A[norm*N+col] = 0;
    }

    if( A[norm*N+norm] != 0 )			// calculates new value
      X[norm] = B[norm] / A[norm*N+norm];		// for equation solution
    else
      X[norm] = 0.0;

    A[norm*N+norm] = 1.0;			// sets UTM diagonal value

    for( row = norm + 1; row < N; row++ ) {		// Guassian elimination occurs
      for( col = norm + 1; col < N; col++ )		// in all remaining rows
        A[row*N+col] -= A[row*N+norm] * A[norm*N+col];

      B[row] -= A[row*N+norm] * X[norm];
      A[row*N+norm] = 0.0;
    }
  }
  end_t = MPI_Wtime();
  print_X(X, N);
  printf("\nElapsed time = %g s\n", end_t - start_t);
}

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

void parameters(int argc, char **argv, int *N) {
  int seed = 0;  /* Random seed */
  char uid[32]; /*User name */

  /* Read command-line arguments */
  srand(time_seed());  /* Randomize */

  if (argc == 3) {
    seed = atoi(argv[2]);
    srand(seed);
    printf("Random seed = %i\n", seed);
  } 
  if (argc >= 2) {
    *N = atoi(argv[1]);
    if (*N < 1 || *N > MAXN) {
      printf("N = %i is out of range.\n", *N);
      MPI_Finalize();
    }
  }
  else {
    printf("Usage: %s <matrix_dimension> [random seed]\n",
           argv[0]);    
    MPI_Finalize();
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs( double A[],  double B[],  double X[], int N) {
  int row, col;

  printf("\nInitializing...\n");
  for(row = 0; row < N; row++) {
	for(col = 0; col < N; col++) {
		A[(row*N)+col] = (double)rand() / 32768.0;
	}
	B[row] = (double)rand() / 32768.0;
	X[row] = 0.0;
  }
}

/* Print input matrices */
void print_inputs( double A[],  double B[], int N) {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
		printf("%5.2f%s", A[(row*N) + col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

void print_X( double X[], int N) {
  int row;

  if (N < 100) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

void main( int argc, char *argv[] )
{
  double *A, *B, *X, *a, *tmp, *final_X;	// var decls
  int i, j, N, row, r;
  double start_t, end_t;		// timing decls
  int p, my_rank;

  MPI_Init (&argc,&argv);
                          
  MPI_Comm_size(MPI_COMM_WORLD,&p);
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  
  if(my_rank == 0) {
	parameters(argc, argv, &N);
  }
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (p == 1) {
	A = malloc((N*N) * sizeof(double));				// space for matricies
	B = malloc(N * sizeof(double));
	X = malloc(N * sizeof(double));
	initialize_inputs(A, B, X, N);
	print_inputs(A, B, N);
	gauss( A, B, X, N);
	free(A);
	free(B);
	free(X);
  } else {
	mpi_gauss(A, B, X, N, my_rank, p);
  }
  
  MPI_Finalize();
}

void mpi_gauss(double *A, double *B, double *X, int N, int my_rank, int p) {

	double *final_X;
	double start_t, end_t;
	int i, j, row;
	
	if(my_rank == 0) {
		A = malloc((N*N) * sizeof(double));
	}			
	B = malloc(N * sizeof(double));
	X = malloc(N * sizeof(double));
	
	if(my_rank == 0) {
		initialize_inputs(A, B, X, N);
		print_inputs(A, B, N);
	}
		
	if ( ( N % p ) != 0 ) {
		printf("Unknowns must be multiple of processors.");
		MPI_Finalize();
    }

	int workload = (int) N/p;
	double a[N*workload];			// size of each submatrix
	double tmp[N*workload];		// and temp. submatrix

	if (my_rank == 0) {
		start_t = MPI_Wtime();		// first PE starts timing
		final_X = calloc(N, sizeof(double));	// and collection var
	}

					// divys up the rows to PEs
					// in blocked mapping fashion
	MPI_Scatter(A, N*workload, MPI_DOUBLE, a, N*workload, MPI_DOUBLE, 0, MPI_COMM_WORLD);

				// This for loop is called
				// for as many communicated rows
				// the current PE is expecting
				// (e.g. how many rows are above)
	for ( i=0; i < (my_rank*workload); i++ ){
				// Get data (blocking)
		MPI_Bcast(tmp,N,MPI_DOUBLE,i/workload,MPI_COMM_WORLD);
		MPI_Bcast(&(X[i]),1,MPI_DOUBLE,i/workload,MPI_COMM_WORLD);


				// adujst current row's values based
				// on values received
				// i = k (row being calculated)
				// rank = row on this processor
		for (row=0; row<workload; row++){
			for ( j=i+1; j<N; j++ ) {
				a[row*N+j] = a[row*N+j] - a[row*N+i]*tmp[j];
			}
			B[my_rank*workload+row] = B[my_rank*workload+row] - a[row*N+i]*X[i];
			a[row*N+i] = 0;
		}
	}

				// after receiving data from all prior
				// rows, begin calculcations
	for (row=0; row<workload; row++) {
		for ( j=my_rank*workload+row+1; j < N ; j++ ) {
			a[row*N+j] = a[row*N+j] / a[row*N+workload*my_rank+row];
		}
		X[my_rank*workload+row] = B[my_rank*workload+row] / a[row*N+my_rank*workload+row];
		a[row*N+my_rank*workload+row] = 1.0;

				// send your row's calculated data
		for ( i=0; i<N ; i++ )
			tmp[i] = a[row*N+i];

		MPI_Bcast (tmp,N,MPI_DOUBLE,my_rank,MPI_COMM_WORLD);
		MPI_Bcast (&(X[my_rank*workload+row]),1,MPI_DOUBLE,my_rank,MPI_COMM_WORLD);

		// update lower rows
		for ( i=row+1; i<workload; i++) {
			for ( j=my_rank*workload+row+1; j<N; j++ ) {
				a[i*N+j] = a[i*N+j] - a[i*N+row+my_rank*workload]*tmp[j];
			}
			B[my_rank*workload+i] = B[my_rank*workload+i] - a[i*N+row+my_rank*workload]*X[my_rank*workload+row];
			a[i*N+row+my_rank*workload] = 0;
		}
	}

				// set idle and receive lower row's
				// broadcast message. VERY BAD, this
				// is why we implement cyclic mapping
				// in the other version
	for (i=(my_rank+1)*workload ; i<N ; i++) {
		MPI_Bcast (tmp,N,MPI_DOUBLE,i/workload,MPI_COMM_WORLD);
		MPI_Bcast (&(X[i]),1,MPI_DOUBLE,i/workload,MPI_COMM_WORLD);
	}

				// Synchronize and gather rows
				// and y values
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(a,N*workload,MPI_DOUBLE,A,N*workload,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Gather(&(X[my_rank*workload]),workload,MPI_DOUBLE,final_X,workload,MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	if(my_rank == 0) {
		X = final_X;
		print_X(X, N);
		free(A);
	}
	
	free(B);
	free(X);
	free(tmp);
	free(a);
}