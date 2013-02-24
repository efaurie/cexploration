/* Gaussian elimination without pivoting.
 * Compile with "gcc -pthread gaussPT.c" 
 */

/* Please see line 196 for gauss function and comments */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
int N;  /* Matrix size */

#define NUM_THREADS 4

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
                * It is this routine that is timed.
                * It is called only on the parent.
                */
void *ForwardElimTask();

struct task_data {
   int thread_id;
   int norm;
   int work_load;
   int start_row;
};

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
      exit(0);
    }
  }
  else {
    printf("Usage: %s <matrix_dimension> [random seed]\n",
           argv[0]);    
    exit(0);
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
  }

}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
         printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

void print_X() {
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
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */

  /* Process program parameters */
  parameters(argc, argv);

  /* Initialize A and B */
  initialize_inputs();

  /* Print input matrices */
  print_inputs();

  /* Start Clock */
  printf("\nStarting clock.\n");
  gettimeofday(&etstart, &tzdummy);
  etstart2 = times(&cputstart);

  /* Gaussian Elimination */
  gauss();

  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  print_X();

  /* Display timing results */
  printf("\nElapsed time = %g ms.\n",
         (float)(usecstop - usecstart)/(float)1000);

  printf("(CPU times are accurate to the nearest %g ms)\n",
         1.0/(float)CLOCKS_PER_SEC * 1000.0);
  printf("My total CPU time for parent = %g ms.\n",
         (float)( (cputstop.tms_utime + cputstop.tms_stime) -
         (cputstart.tms_utime + cputstart.tms_stime) ) /
         (float)CLOCKS_PER_SEC * 1000);
  printf("My system CPU time for parent = %g ms.\n",
         (float)(cputstop.tms_stime - cputstart.tms_stime) /
         (float)CLOCKS_PER_SEC * 1000);
  printf("My total CPU time for child processes = %g ms.\n",
         (float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
         (cputstart.tms_cutime + cputstart.tms_cstime) ) /
         (float)CLOCKS_PER_SEC * 1000);
      /* Contrary to the man pages, this appears not to include the parent */
  printf("--------------------------------------------\n");

  exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */

void gauss() {
  int norm, row, col;  /* Normalization row, and zeroing
                        * element row and col
                        */
  /* A structure for passing in multiple args to the pthread function
   * an array of thread elements, the attribute flag to make the 
   * threads joinable, and a status var (currently unused).
   */
  struct task_data data_array[NUM_THREADS];
  pthread_t threads[NUM_THREADS];
  pthread_attr_t attr;
  void *status;
  
  /* Variables containing information for each thread: an id
   * the number of rows each thread computes (work_load)
   * the number of rows remaining when not perfectly divisible
   * by the number of threads (work_remainder) and the row to
   * begin computation on.
   */
  int thread;
  int work_load;
  int work_remainder;
  int start_row;

  /* The threads must be joinable */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  printf("Computing In Parallel (Pthread).\n");

  /* Pthread Parallel Gaussian Elimination */
  for(norm = 0; norm < N - 1; norm++) {
    /* Determine the work_load for each thread */

   start_row = norm + 1;
   work_load = (N - start_row) / NUM_THREADS;
   work_remainder = (N - start_row) % NUM_THREADS;

   /* For each thread */
   for(thread = 0; thread < NUM_THREADS; thread++) {
      /* Define the arguments to be passed in */
      data_array[thread].thread_id = thread;
      data_array[thread].norm = norm;
      data_array[thread].start_row = start_row + (thread * work_load);

      /* If the thread is the last one, assign all work_remainder rows to it as well */
      if(thread == NUM_THREADS - 1) {
         data_array[thread].work_load = work_load + work_remainder;
         pthread_create(&threads[thread], &attr, ForwardElimTask, (void *) &data_array[thread]);
      } else {
         data_array[thread].work_load = work_load;
         pthread_create(&threads[thread], &attr, ForwardElimTask, (void *) &data_array[thread]);
      }
   }
   
   /* Release the attribute variable */
   pthread_attr_destroy(&attr);
   /* The threads must join before moving to the next iteration because of 
    * the shared B vector.
    */
   for(thread = 0; thread < NUM_THREADS; thread++) {
      pthread_join(threads[thread], &status);
   }
  }

  /* (Diagonal elements are not normalized to 1.  This is treated in back
   * substitution.)
   */

  /* Back substitution */
  for (row = N - 1; row >= 0; row--) {
    X[row] = B[row];
    for (col = N-1; col > row; col--) {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
}


/* The logic for the forward elimination */
void *ForwardElimTask(void *args) {
   /* Parse the arguments */
   struct task_data *my_task;
   my_task = (struct task_data *) args;
   int thread_id = my_task->thread_id;
   int start_row = my_task->start_row;
   int work_load = my_task->work_load;
   int norm = my_task->norm;

   int col, row;
   float multiplier;

   /* For each row assigned to the threads work load */
   for(row = start_row; row < start_row + work_load; row++) {
      multiplier = A[row][norm] / A[norm][norm];
      for(col = norm; col < N; col++) {
         A[row][col] -= A[norm][col] * multiplier;
      }
      B[row] -= B[norm] * multiplier;
   }
   pthread_exit(NULL);
}
