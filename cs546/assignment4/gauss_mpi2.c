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
void workerGauss();

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
  
  if(my_rank == 0) {
	float A[N*N], B[N], X[N];
	/* Initialize A and B */
	initialize_inputs(A, B, X);

	/* Print input matrices */
	print_inputs(A, B);

	/* Start Clock */
	printf("\nStarting clock.\n");
	start_t = MPI_Wtime();
	
    gauss(A, B, X, my_rank, p);
	
	/* Stop Clock */
	end_t = MPI_Wtime();
	printf("Stopped clock.\n");
  
	/* Display output */
	print_X(X);

	/* Display timing results */
	printf("\nElapsed time = %g s\n", end_t - start_t);
	printf("--------------------------------------------\n");
  } else {
	workerGauss(my_rank, p);
  }
  
  MPI_Finalize();
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
void gauss( float A[],  float B[],  float X[], int my_rank, int p) {
  int norm, row, col, i, j;
  float multiplier, b_norm;
  int end_row;
  MPI_Status status;
  
  /*Slice up the rows */
  int row_workload = (N-1)/p;
  int excess_work = (N-1)%p;
  end_row = row_workload + excess_work;
  for(i = 1; i < p; i++) {
	MPI_Ssend(&A[(N*row_workload*i)+(N*excess_work)+N], row_workload*N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
	MPI_Ssend(&B[row_workload*i+excess_work+1], row_workload, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
  }
  
  for(norm = 0; norm <= row_workload + excess_work; norm++) {
	/* If you have the latest norm vector */
	if(norm <= row_workload + excess_work) {
		/* Send it to all ranks after yours */
		for(i = my_rank+1; i < p; i++) {
			b_norm = B[norm];
			MPI_Ssend(&A[(N*norm)+norm], N-norm, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
			MPI_Ssend(&b_norm, 1, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
		}
		
		for (row = norm + 1; row <= end_row; row++) {
			multiplier = A[N*row+norm] / A[N*norm+norm];
			if(row <= end_row) {
				for (col = norm; col < N; col++) {
					A[N*row+col] -= A[N*norm+col] * multiplier;
				}
			}
			B[row] -= B[norm] * multiplier;
		}
	}
  }
  
  for(i = 1; i < p; i++) {
	MPI_Recv(&A[(N*row_workload*i)+(N*excess_work)+N], row_workload*N, MPI_FLOAT, i, 3, MPI_COMM_WORLD, &status);
	MPI_Recv(&B[(row_workload*i)+excess_work+1], row_workload, MPI_FLOAT, i, 3, MPI_COMM_WORLD, &status);
  }
  print_inputs(A,B);

  for (row = N - 1; row >= 0; row--) {
	X[row] = B[row];
	for (col = N-1; col > row; col--) {
		X[row] -= A[(N*row)+col] * X[col];
	}
	X[row] /= A[(N*row)+row];
  }
}

void workerGauss(int my_rank, int p) {
	int norm, row, col, i;
	int end_row;
	int start_row;
	float multiplier;
	float norm_buf[N];
	float b_norm;
	MPI_Status status;
	
	/*Get working row set */
	int row_workload = (N-1)/p;
	int excess_work = (N-1)%p;
	float work_buf[row_workload*N];
	float b_buf[row_workload];
	MPI_Recv(work_buf, row_workload*N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
	MPI_Recv(b_buf, row_workload, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
	
	end_row = (my_rank*row_workload)+(row_workload+excess_work);
	
	for(norm = 0; norm < end_row; norm++) {
	
		if(end_row - norm < row_workload) {
			/* This process now has the updated norm row */
			start_row = end_row - norm;
			for(i = my_rank+1; i < p; i++) {
				b_norm = b_buf[start_row - 1];
				MPI_Ssend(&work_buf[N*(start_row-1)], N-norm, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
				MPI_Ssend(&b_norm, 1, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
			}
			for(i = 0; i < N-norm; i++) {
				norm_buf[i] = work_buf[N*(start_row-1)+norm+i];
			}
		} else {
			start_row = 0;
			MPI_Recv(norm_buf, N-norm, MPI_FLOAT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
			MPI_Recv(&b_norm, 1, MPI_FLOAT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
		}
		if(norm == end_row - 1)
			b_norm = b_buf[row_workload - 1];
		for (row = start_row; row < row_workload; row++) {
			multiplier = work_buf[(N*row)+norm] / norm_buf[0];
			int count = 0;
			for (col = norm; col < N; col++) {
				work_buf[N*row+col] -= norm_buf[count] * multiplier;
				count++;
			}
			b_buf[row] -= b_norm * multiplier;
		}
	}
	
	for(i = my_rank+1; i < p; i++) {
		b_norm = b_buf[row_workload-1];
		MPI_Ssend(&work_buf[N*(row_workload - 1)], N-norm, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
		MPI_Ssend(&b_norm, 1, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
	}
	
	MPI_Ssend(work_buf, row_workload*N, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
	MPI_Ssend(b_buf, row_workload, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
}

