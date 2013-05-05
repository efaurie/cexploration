#include <stdio.h>
#include <math.h>
#include "mpi.h"


typedef struct {float r; float i;} complex;
static complex ctmp;

#define C_SWAP(a,b) {ctmp=(a);(a)=(b);(b)=ctmp;}

void initialize_data();
void dist_data();
void recv_data();
void output_data();
void collect_data();
void transpose();
void execute_fft();
void execute_mm();
void c_fft1d();

char f1_name[] = "1_im1";
char f2_name[] = "1_im2";
char f_out[] = "output";
MPI_Datatype mpi_complex;

void main(int argc, char **argv) {
	
	int my_rank, p;
	complex A[512*512], B[512*512], C[512*512];

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
	
	/* Initialize Data*/
	if(my_rank == 0) {
		initialize_data(f1_name, A);
		initialize_data(f2_name, B);
		dist_data(A, p);
		dist_data(B, p);
	} else {
		recv_data(A, p, my_rank);
		recv_data(B, p, my_rank);
	}
	
	/* 2D FFT on A */
	execute_fft(A, 1, p, my_rank);
	collect_data(A, p, my_rank);
	if(my_rank == 0) {
		transpose(A);
		dist_data(A, p);
	} else {
		recv_data(A, p, my_rank);
	}
	execute_fft(A, 1, p, my_rank);
	
	/* 2D FFT on B */
	execute_fft(B, 1, p, my_rank);
	collect_data(B, p, my_rank);
	if(my_rank == 0) {
		transpose(B);
		dist_data(B, p);
	} else {
		recv_data(B, p, my_rank);
	}
	execute_fft(B, 1, p, my_rank);
	
	/* Multiplication Step */
	execute_mm(A, B, C, p, my_rank);
	
	/* 2D FFT on C */
	execute_fft(C, -1, p, my_rank);
	collect_data(C, p, my_rank);
	if(my_rank == 0) {
		transpose(C);
		dist_data(C, p);
	} else {
		recv_data(C, p, my_rank);
	}
	execute_fft(C, 1, p, my_rank);
	collect_data(C, p, my_rank);
	
	if(my_rank == 0) {
		output_data(f_out, C);
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

void dist_data(complex data[512*512], int p) {
	int i, workload;
	workload = 512 / p;
	for(i = 1; i < p; i++) {
		MPI_Send(&data[i * workload * 512], 512*workload, mpi_complex, i, 1, MPI_COMM_WORLD);
	}
}

void recv_data(complex data[512*512], int p, int my_rank) {
	int i, workload;
	MPI_Status status;
	workload = 512 / p;
	MPI_Recv(&data[my_rank * workload * 512], 512*workload, mpi_complex, 0, 1, MPI_COMM_WORLD, &status);
}

void collect_data(complex data[512*512], int p, int my_rank) {
	int workload = 512 / p;
	if(my_rank == 0) {
		int i;
		MPI_Status status;
		for(i = 1; i < p; i++) {
			MPI_Recv(&data[512 * i * workload], 512*workload, mpi_complex, i, 2, MPI_COMM_WORLD, &status);
		}
	} else {
		int my_row = 512 * workload * my_rank;
		MPI_Send(&data[my_row], 512*workload, mpi_complex, 0, 2, MPI_COMM_WORLD);
	}
}

void execute_fft(complex data[512*512], int type, int p, int my_rank) {
	int i, j, workload;
	workload = 512 / p;
	for(i = 0; i < workload; i++) {
		c_fft1d(&data[(my_rank * workload * 512) + (512 * i)], 512, type);
	}
}

void execute_mm(complex A[512*512], complex B[512*512], complex C[512*512], int p, int my_rank) {
	int i, j, workload;
	workload = 512 / p;
	int start_loc = my_rank * workload * 512;
	complex temp;
	int cur_loc;
	for(i = 0; i < workload; i++) {
		for(j = 0; j < 512; j++) {
			cur_loc = start_loc + (i*512) + j;
			temp.r = A[cur_loc].r * B[cur_loc].r - A[cur_loc].i * B[cur_loc].i;
			temp.i = A[cur_loc].i * B[cur_loc].r - A[cur_loc].r * B[cur_loc].i;
			C[cur_loc].r = temp.r;
			C[cur_loc].i = temp.i;
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