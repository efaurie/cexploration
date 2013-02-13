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

   if(time > 0) {
      distance= .5 * GRAVITY * time * time;
      velocity= GRAVITY * time;

      printf("   Time of fall = %g seconds \n", time);
      printf("   Distance of fall = %g meters \n", distance);
      printf("   Velocity of the object = %g m/s \n\n", velocity);
   } else {
      printf(" Sorry, you entered %g, that is not a valid time!\n\n", time);
   }

   printf(" Goodbye!\n");
}
