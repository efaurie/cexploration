#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

#include <mpi.h>

#define MAXN 2000
int N;

void gauss();
//void gauss( float X[],  float B[],  float X[], int my_rank, int p);

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
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
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
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
void initialize_inputs( float A[],  float B[],  float X[]) {
  int row, col;

  printf("\nInitializing...\n");
  for(row = 0; row < N; row++) {
	for(col = 0; col < N; col++) {
		A[(row*N)+col] = (float)rand() / 32768.0;
	}
	B[row] = (float)rand() / 32768.0;
	X[row] = 0.0;
  }
}

/* Print input matrices */
void print_inputs( float A[],  float B[]) {
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

void print_X( float X[]) {
  int row;

  if (N < 100) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

int main(int argc, char **argv) {
  
  /* Timing variables */
  double start_t;
  double end_t;
  
  /* MPI Variables */
  int my_rank;
  int p;
  int dest = 0;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  
  if(my_rank == 0) {
	parameters(argc, argv);
  }
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
   float A[N*N], B[N], X[N];
  
  if(my_rank == 0) {
	/* Initialize A and B */
	initialize_inputs(A, B, X);

	/* Print input matrices */
	print_inputs(A, B);

	/* Start Clock */
	printf("\nStarting clock.\n");
	start_t = MPI_Wtime();
  }

  /* Gaussian Elimination */
  gauss(A, B, X, my_rank, p);
  
  if(my_rank == 0) {
	/* Stop Clock */
	printf("Stopped clock.\n");
	end_t = MPI_Wtime();
  
	/* Display output */
	print_X(X);

	/* Display timing results */
	printf("\nElapsed time = %g ms.\n", end_t - start_t);
	printf("--------------------------------------------\n");
  }
  
  MPI_Finalize();
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
void gauss( float A[],  float B[],  float X[], int my_rank, int p) {
  int norm, row, col, i;
  float multiplier;
  int col_workload;
  int excess_work;
  float buf[(N/p)+(N%p)];
  MPI_Status status;
  
  for(norm = 0; norm < N - 1; norm++) {
	MPI_Bcast(&A[(N*norm)+norm], (N-norm), MPI_FLOAT, 0, MPI_COMM_WORLD);
	for(row = norm + 1; row < N; row++) {
		col_workload = (N-norm) / p;
		excess_work = (N-norm) % p;
		if(my_rank == 0)
			multiplier = A[(N*row)+norm] / A[(N*norm)+norm];
		MPI_Bcast(&multiplier, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		
		if(my_rank == 0) {
			for(i = 1; i < p; i++) {
				if(i < p - 1)
					MPI_Send(&A[(N*row)+norm+(col_workload*p)], col_workload, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
				else
					MPI_Send(&A[(N*row)+norm+(col_workload*p)], col_workload+excess_work, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
			}
			/* Compute Rank 0's portion */
			for(col = norm; col < norm+col_workload; col++) {
				A[(N*row)+col] -= A[(N*norm)+col] * multiplier;
			}
			
			/* Collect other portions */
			for(i = 1; i < p; i++) {
				if(i < p - 1)
					MPI_Recv(&A[(N*row)+norm+(col_workload*p)], col_workload, MPI_FLOAT, i, 2, MPI_COMM_WORLD, &status);
				else
					MPI_Recv(&A[(N*row)+norm+(col_workload*p)], col_workload+excess_work, MPI_FLOAT, i, 2, MPI_COMM_WORLD, &status);
			}
			B[row] -= B[norm] * multiplier;
		} else {
			if(my_rank < p-1) {
				MPI_Recv(buf, col_workload, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
				for(col = 0; col < col_workload; col++) {
					buf[col] -= A[(N*norm)+col] * multiplier;
				}
				MPI_Send(buf, col_workload, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
			} else {
				MPI_Recv(buf, col_workload+excess_work, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status); 
				for(col = 0; col < col_workload+excess_work; col++) {
					buf[col] -= A[(N*norm)+col] * multiplier;
				}
				MPI_Send(buf, col_workload+excess_work, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
			}
		}
	}
  }

  if(my_rank == 0) {
	for (row = N - 1; row >= 0; row--) {
		X[row] = B[row];
		for (col = N-1; col > row; col--) {
			X[row] -= A[(N*row)+col] * X[col];
		}
		X[row] /= A[(N*row)+row];
	}
  }
}

