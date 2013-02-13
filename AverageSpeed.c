#include <stdio.h>

void main() {
   int begin_miles;
   int end_miles;
   int miles;
   double hours;
   double minutes;
   double speed;

   printf("\n Miles Per Hour Computation \n");

   printf(" Odometer reading at the beginning of the trip: ");
   scanf("%i", &begin_miles);
   while(begin_miles < 0) {
      printf(" Re-enter; odometer reading must be positive: ");
      scanf("%i", &begin_miles);
   }

   printf(" Odometer reading at the end of the trip: ");
   scanf("%i", &end_miles);
   while(end_miles < begin_miles) {
      printf(" Re-enter; odometer reading must be > starting miles: ");
      scanf("%i", &end_miles);
   }

   printf(" Duration of trip in hours and minutes: ");
   scanf("%lg%lg", &hours, &minutes);
   hours = hours + (minutes / 60);
   while(hours < 0.0) {
      printf("Re-enter; hours and minutes must be >= 0: ");
      scanf("%lg%lg", &hours, &minutes);
      hours = hours + (minutes / 60);
   }

   miles = end_miles - begin_miles;
   speed = miles / hours;
   printf(" Average speed was %g mph\n", speed);

   printf("\n Goodbye! \n\n");
}
