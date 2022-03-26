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

nx = 100
ny = nx
steps = 40
a = 0.9
b = -0.6013
c = 2.0
d = 0.50

mat = [[i-nx/2, j-ny/2] for i in range(nx) for j in range(ny)]
x = [x[0]/(nx) for x in mat]
y = [x[1]/(ny) for x in mat]
x = np.array(x, 'd')
y = np.array(y, 'd')
tinkerbell(x, y, a, b, c, d, steps)
```
![tb](https://user-images.githubusercontent.com/38112687/160251441-cdd66a76-5777-4fa9-8b7c-24d17d7e94c7.png)

See Jupyter notebook for examples of other systems


