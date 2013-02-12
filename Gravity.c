#include <stdio.h>

#define GRAVITY 9.81

void main() {
   double time;
   double distance;
   double velocity;

   printf("\n Welcome. \n"
          " Calculate the height from which a grapefruit fell.\n"
          " given the number of seconds that it was falling. \n\n");
   printf(" Input Seconds: ");

   scanf("%lg", &time);
   distance= .5 * GRAVITY * time * time;
   velocity= GRAVITY * time;

   printf("   Time of fall = %lg seconds \n", time);
   printf("   Distance of fall = %lg meters \n", distance);
   printf("   Velocity of the object = %g m/s \n\n", velocity);

   printf(" Goodbye!\n");
}
