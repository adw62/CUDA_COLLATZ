%module chaos

%{
    #define SWIG_FILE_WITH_INIT 
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (int* INPLACE_ARRAY1, int DIM1) {(int* start_ints, int n)}
%inline %{
void collatz(int *start_ints, int n);
%}

%apply (double* INPLACE_ARRAY1, int DIM1) {(double* x_points, int N)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* y_points, int M)}
%inline %{
void henon(double *x_points, int N, double *y_points, int M, double a, double b, int steps);
%}

%apply (double* INPLACE_ARRAY1, int DIM1) {(double* x_points, int N)}
%inline %{
void logistics_map(double *x_points, int N, double r, int steps);
%}

%apply (double* INPLACE_ARRAY1, int DIM1) {(double* x_points, int N)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* y_points, int M)}
%inline %{
void tinkerbell(double *x_points, int N, double *y_points, int M, double a, double b,
 double c, double d, int steps);
%}

%apply (double* INPLACE_ARRAY1, int DIM1) {(double* x_points, int N)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* y_points, int M)}
%inline %{
void bogdanov(double *x_points, int N, double *y_points, int M, double eps, double k,
 double mew, int steps);
%}
