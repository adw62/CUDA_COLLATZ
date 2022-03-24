from cuda_collatz import collatz
import numpy

a = numpy.array(range(1, 1001),'i')
collatz(a)

print(a)

