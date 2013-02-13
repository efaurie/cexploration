#include <stdio.h>
#define e 2.718281828459045235360287471353

void main() {
   int size_int;
   int size_double;
   int size_unknown;
   int k;

   size_double = sizeof(double);
   size_int = sizeof k;
   printf("\n sizeof int = %i \n sizeof double = %i\n", size_int, size_double);
   size_unknown = sizeof e;
   printf(" sizeof e = %i \n\n", size_unknown);
}
