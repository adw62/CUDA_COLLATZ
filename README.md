# CUDA_COLLATZ
CUDA code to test Collatz conjecture with Python API

# To build

swig -python -c++ example.i
nvcc --compiler-options '-fPIC' -c collatz.cu collatz_wrap.cxx -I/home/a/miniconda3/include/python3.9/ -I/home/a/miniconda3/lib/python3.9/site-packages/numpy/core/include
nvcc -shared collatz.o collatz_wrap.o -o _cuda_collatz.so

# To run

python run.py
