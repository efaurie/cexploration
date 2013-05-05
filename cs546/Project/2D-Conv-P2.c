#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "mpi.h"


typedef struct {float r; float i;} complex;
static complex ctmp;

#define C_SWAP(a,b) {ctmp=(a);(a)=(b);(b)=ctmp;}

void initialize_data();
void output_data();
void transpose();
void execute_fft();
void execute_mm();
void c_fft1d();

char f1_name[] = "1_im1";
char f2_name[] = "1_im2";
char f_out[] = "output";
MPI_Datatype mpi_complex;

void main(int argc, char **argv) {
	
	double start_t;
	double end_t;
	
	int my_rank, p;
	complex *A;
	complex *B;
	complex *C;

	/* initialize MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	/* Create MPI Datatype for Complex */
    const float nitems=2;
    int          blocklengths[2] = {1,1};
    MPI_Datatype types[2] = {MPI_FLOAT, MPI_FLOAT};
    MPI_Aint     offsets[2];

    offsets[0] = offsetof(complex, r);
    offsets[1] = offsetof(complex, i);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_complex);
    MPI_Type_commit(&mpi_complex);
	int workload = 512 / p;
	complex a[512*workload];
	complex b[512*workload];
	complex c[512*workload];
	
	/* Initialize Data*/
	if(my_rank == 0) {
		A = malloc(512*512 * sizeof(complex));
		B = malloc(512*512 * sizeof(complex));
		C = malloc(512*512 * sizeof(complex));
		initialize_data(f1_name, A);
		initialize_data(f2_name, B);
	}
	
	start_t = MPI_Wtime();
	
	/* 2D FFT on A */
	MPI_Scatter(A, 512*workload, mpi_complex, 
			   a, 512*workload, mpi_complex,
			   0, MPI_COMM_WORLD);
	execute_fft(a, 1, p, my_rank);
	MPI_Gather(a, 512*workload, mpi_complex,
			   A, 512*workload, mpi_complex,
			   0, MPI_COMM_WORLD);
	if(my_rank == 0) {
		transpose(A);
	}
	MPI_Scatter(A, 512*workload, mpi_complex, 
			   a, 512*workload, mpi_complex,
			   0, MPI_COMM_WORLD);
	execute_fft(a, 1, p, my_rank);
	
	/* 2D FFT on B */
	MPI_Scatter(B, 512*workload, mpi_complex, 
			   b, 512*workload, mpi_complex,
			   0, MPI_COMM_WORLD);
	execute_fft(b, 1, p, my_rank);
	MPI_Gather(b, 512*workload, mpi_complex,
			   B, 512*workload, mpi_complex,
			   0, MPI_COMM_WORLD);
	if(my_rank == 0) {
		transpose(B);
	}
	MPI_Scatter(B, 512*workload, mpi_complex, 
			   b, 512*workload, mpi_complex,
			   0, MPI_COMM_WORLD);
	execute_fft(b, 1, p, my_rank);
	
	/* Multiplication Step */
	execute_mm(a, b, c, p, my_rank);
	
	/* 2D FFT on C */
	execute_fft(c, -1, p, my_rank);
	MPI_Gather(c, 512*workload, mpi_complex,
			   C, 512*workload, mpi_complex,
			   0, MPI_COMM_WORLD);
	if(my_rank == 0) {
		transpose(C);
	}
	MPI_Scatter(C, 512*workload, mpi_complex, 
			   c, 512*workload, mpi_complex,
			   0, MPI_COMM_WORLD);
	execute_fft(c, -1, p, my_rank);
	MPI_Gather(c, 512*workload, mpi_complex,
			   C, 512*workload, mpi_complex,
			   0, MPI_COMM_WORLD);
	
	
	end_t = MPI_Wtime();
	
	if(my_rank == 0) {
		output_data(f_out, C);
		printf("\nElapsed time = %g s\n", end_t - start_t);
		printf("--------------------------------------------\n");
		int i;
		for(i = 0; i < 512*512; i++) {
			free(&A[i]);
			free(&B[i]);
			free(&C[i]);
		}
	}
	
	MPI_Finalize();
}

void initialize_data(char *f_name, complex data[512*512]) {
	int i, j;
	FILE *fp;
	float temp;
	
	fp = fopen(f_name, "r");
	for(i = 0; i < 512; i++) {
		for(j = 0; j < 512; j++) {
			fscanf(fp, "%g", &temp);
			data[(i*512) + j].r = temp;
			data[(i*512) + j].i = 0;
		}
	}
	fclose(fp);
}

void execute_fft(complex data[512*512], int type, int p, int my_rank) {
	int i, j, workload;
	workload = 512 / p;
	for(i = 0; i < workload; i++) {
		c_fft1d(&data[512 * i], 512, type);
	}
}

void execute_mm(complex a[512*512], complex b[512*512], complex c[512*512], int p, int my_rank) {
	int i, j, workload;
	workload = 512 / p;
	complex temp;
	int cur_loc;
	for(i = 0; i < workload; i++) {
		for(j = 0; j < 512; j++) {
			cur_loc = (i*512) + j;
			temp.r = a[cur_loc].r * b[cur_loc].r - a[cur_loc].i * b[cur_loc].i;
			temp.i = a[cur_loc].i * b[cur_loc].r - a[cur_loc].r * b[cur_loc].i;
			c[cur_loc].r = temp.r;
			c[cur_loc].i = temp.i;
		}
	}
}

void c_fft1d(complex *r, int n, int isign) {
   int m, i, i1, j, k, i2, l, l1, l2;
   float c1, c2, z;
   complex t, u;

   if (isign == 0) return;

   /* Do the bit reversal */
   i2 = n >> 1;
   j = 0;
   for (i=0;i<n-1;i++) {
      if (i < j)
         C_SWAP(r[i], r[j]);
      k = i2;
      while (k <= j) {
         j -= k;
         k >>= 1;
      }
      j += k;
   }

   /* m = (int) log2((double)n); */
   for (i=n,m=0; i>1; m++,i/=2);

   /* Compute the FFT */
   c1 = -1.0;
   c2 =  0.0;
   l2 =  1;
   for (l=0;l<m;l++) {
      l1   = l2;
      l2 <<= 1;
      u.r = 1.0;
      u.i = 0.0;
      for (j=0;j<l1;j++) {
         for (i=j;i<n;i+=l2) {
            i1 = i + l1;

            /* t = u * r[i1] */
            t.r = u.r * r[i1].r - u.i * r[i1].i;
            t.i = u.r * r[i1].i + u.i * r[i1].r;

            /* r[i1] = r[i] - t */
            r[i1].r = r[i].r - t.r;
            r[i1].i = r[i].i - t.i;

            /* r[i] = r[i] + t */
            r[i].r += t.r;
            r[i].i += t.i;
         }
         z =  u.r * c1 - u.i * c2;

         u.i = u.r * c2 + u.i * c1;
         u.r = z;
      }
      c2 = sqrt((1.0 - c1) / 2.0);
      if (isign == -1) /* FWD FFT */
         c2 = -c2;
      c1 = sqrt((1.0 + c1) / 2.0);
   }

   /* Scaling for inverse transform */
   if (isign == 1) {       /* IFFT*/
      for (i=0;i<n;i++) {
         r[i].r /= n;
         r[i].i /= n;
      }
   }
}

void transpose(complex data[512*512]) {
	int i, j;
	complex temp[512*512];
	for(i = 0; i < 512; i++) {
		for(j = 0; j < 512; j++) {
			temp[(i*512) + j] = data[(j*512) + i];
		}
	}
	data = temp;
}

void output_data(char *f_out, complex data[512*512]) {
	int i, j;
	float temp;
	FILE *fp;
	
	fp = fopen(f_out, "w");
	for(i = 0; i < 512; i++) {
		for(j = 0; j < 512; j++) {
			temp = data[(i*512) + j].r;
			fprintf(fp, "%6.2g", &temp);
		}
		fprintf(fp, "/n");
	}
	fclose(fp);
}