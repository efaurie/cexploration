#include <stdio.h>
#include <math.h>


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
	
	double start_t, end_t;

	complex A[512*512], B[512*512], C[512*512];
	
	/* Initialize Data*/
	initialize_data(f1_name, A);
	initialize_data(f2_name, B);
	
	
	/* 2D FFT on A */
	execute_fft(A, 1);
	transpose(A);
	execute_fft(A, 1);
	
	
	/* 2D FFT on B */
	execute_fft(B, 1);
	transpose(B);
	execute_fft(B, 1);
	
	/* Multiplication Step */
	execute_mm(A, B, C);
	
	/* 2D FFT on C */
	execute_fft(C, -1, p, my_rank);
	transpose(C);
	execute_fft(C, -1, p, my_rank);
	
	output_data(f_out, C);
	printf("\nElapsed time = %g s\n", end_t - start_t);
	printf("--------------------------------------------\n");
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

void execute_fft(complex data[512*512], int type) {
	int i, j;
	for(i = 0; i < 512; i++) {
		c_fft1d(&data[512 * i], 512, type);
	}
}

void execute_mm(complex A[512*512], complex B[512*512], complex C[512*512]) {
	int i, j;
	complex temp;
	int cur_loc;
	for(i = 0; i < 512; i++) {
		for(j = 0; j < 512; j++) {
			cur_loc = (i * 512) + j;
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