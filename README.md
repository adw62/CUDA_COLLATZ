# CUDA_COLLATZ
CUDA code to test Collatz conjecture with Python API

# To build

swig -python -c++ collatz.i

nvcc --compiler-options '-fPIC' -c collatz.cu collatz_wrap.cxx -I/home/a/miniconda3/include/python3.9/ -I/home/a/miniconda3/lib/python3.9/site-packages/numpy/core/include

nvcc -shared collatz.o collatz_wrap.o -o _cuda_collatz.so

# To run

python run.py

![coll](https://user-images.githubusercontent.com/38112687/160183594-851f08b5-2f34-4254-89cf-a067a14a25ec.png)
