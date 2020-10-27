#include <stdio.h>
#include <math.h>
#include <windows.h>
#include <omp.h>

double f(double x);    /* Function we're integrating */
void Trap(double a, double b, int n, double * integral);

int main(void) {
   double  integral = 0.0;   /* Store result in integral   */
   double  a, b;       /* Left and right endpoints   */
   int     n;          /* Number of trapezoids       */
   double  h;          /* Height of trapezoids       */
   int thread_count = 1;
   double start, end;

   printf("Enter a (-double), b (-double), and n (-int): \n");
   scanf("%lf%lf%d", &a, &b, &n);

   printf("Enter the number of threads: ");
   scanf("%d", &thread_count);

   start = omp_get_wtime();

#  pragma omp parallel num_threads(thread_count) 
   Trap(a, b, n, &integral);

   end = omp_get_wtime();
   
   printf("With n = %d trapezoids, our estimate\n", n);
   printf("of the integral from %f to %f = %.10f\n", a, b, integral);
   printf("Total CPU time is %f milliseconds\n", (end - start)*1000.0);

   return 0;
}  /* main */

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Use trapezoidal rule to estimate definite integral
 * Input args:  
 *    a: left endpoint
 *    b: right endpoint
 *    n: number of trapezoids
 * Output arg:
 *    integral: estimate of integral from a to b of f(x)
 */
void Trap(double a, double b, int n, double* global_result_p) {
   double  h, x, my_result;
   double  local_a, local_b;
   int  i, local_n;
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();

   h = (b-a)/n; 
   local_n = n/thread_count;  
   local_a = a + my_rank*local_n*h; 
   local_b = local_a + local_n*h; 
   my_result = (f(local_a) + f(local_b))/2.0; 
   for (i = 1; i <= local_n-1; i++) {
     x = local_a + i*h;
     my_result += f(x);
   }
   my_result = my_result*h; 

#  pragma omp critical 
   *global_result_p += my_result; 
}  /* Trap */

/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 */
double f(double x) {
   double return_val;

   return_val = sqrt(x)*sin(x)*sin(2*x)*sin(3*x)*sin(4*x);

   return return_val;
}  /* f */
