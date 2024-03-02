# cuda_chaos
CUDA code to iterate a range of dynamic systems which exhibit chaotic behaviour with Python API

# To build

swig -python -c++ chaos.i

nvcc --compiler-options '-fPIC' -c chaos.cu chaos_wrap.cxx -I/home/a/miniconda3/include/python3.9/ -I/home/a/miniconda3/lib/python3.9/site-packages/numpy/core/include

nvcc -shared chaos.o chaos_wrap.o -o _chaos.so

# To run

```python
from chaos import collatz, henon, tinkerbell, logistics_map, bogdanov
import numpy as np

#uses a lot of memory
nx = 1024*100000
ny = nx
steps = 300
a = 0.9
b = -0.6013
c = 2.0
d = 0.50

x = np.arange(nx) - nx/2
y = np.arange(ny) - ny/2

x = x / nx
y = y / ny

x = x.astype('d')
y = y.astype('d')
tinkerbell(x, y, a, b, c, d, steps)
```
![download1](https://github.com/adw62/cuda_chaos/assets/38112687/1765b69d-8ef6-4406-ad8d-2ff9e414b247)


See Jupyter notebook for examples of other systems


