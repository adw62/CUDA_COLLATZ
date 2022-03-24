%module cuda_collatz

%{
    #define SWIG_FILE_WITH_INIT 
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (int* INPLACE_ARRAY1, int DIM1) {(int* invec, int n)}

%inline %{
void collatz(int *invec, int n);
%}
