# cuda_chaos
CUDA code to iterate a range of dynamic systems which exhibit chaotic behaviour with Python API

# To build

swig -python -c++ chaos.i

#Change python and numpy locations as required
nvcc --compiler-options '-fPIC' -c chaos.cu chaos_wrap.cxx -I/home/a/miniconda3/include/python3.9/ -I/home/a/miniconda3/lib/python3.9/site-packages/numpy/core/include

nvcc -shared chaos.o chaos_wrap.o -o _chaos.so

# To run

```python
from chaos import collatz, henon, tinkerbell, logistics_map, bogdanov
import numpy as np
import matplotlib.pyplot as plt

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

plt.figure(figsize=(16, 9), dpi=600)
plt.scatter(x, y, s=0.1, linewidths=0, alpha=0.5)
text = f"""Tinkerbell
a = {a}, b = {b}, c = {c}, d = {d}"""

plt.figtext(0.05,0.00, text, fontsize=8, va="top", ha="left")

plt.xlabel('x')
plt.ylabel('y')
plt.show()
```
![t1](https://github.com/adw62/cuda_chaos/assets/38112687/70815f39-493e-4132-953b-2c743a28831a)
![t3](https://github.com/adw62/cuda_chaos/assets/38112687/159ae8d4-586c-4307-b903-afd3c3b7eddc)



See Jupyter notebook for examples of other systems


