from cuda_collatz import collatz
import numpy

for i in range (1):
    a = numpy.array(range((i*1000000)+1, (i+1)*1000001),'i')
    collatz(a)
    print(a)

