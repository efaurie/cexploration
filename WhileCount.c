#include <stdio.h>

#define ITERATIONS 20

void main() {
   int counter;

   printf("\n This program will simply count down from %i\n\n", ITERATIONS);

   counter = ITERATIONS;
   while (counter > 0) {
      printf(" %i. \n", counter);
      counter--;
   }

   printf("\n Goodbye! \n\n");
}
