# <h1 class="text-center">The Basics of   ![](https://i.imgur.com/FEiml65.png)</h1>
1) The fundamental nature of Cython can be summed up as follows: Cython is Python with C data types.
2) Cython is Python: Almost any piece of Python code is also valid Cython code. The Cython compiler will convert it into C code which makes equivalent calls to the Python/C API.
3) But Cython is much more than that, because parameters and variables can be declared to have C data types. Code which manipulates Python values and C values can be freely intermixed, with conversions occurring automatically wherever possible. 
``` python=
%%cython -a
import numpy as np

def mandelbrot_cython(int[:,::1] m,
                      int size,
                      int iterations):
    cdef int i, j, n
    cdef complex z, c
    for i in range(size):
        for j in range(size):
            c = -2 + 3./size*j + 1j*(1.5-3./size*i)
            z = 0
            for n in range(iterations):
                if z.real**2 + z.imag**2 <= 100:
                    z = z*z + c
                    m[i, j] = n
                else:
                    break
```

# <h1 class="text-center">The Basics of ![](https://i.imgur.com/9B5sRj7.png)</h1>

1) Numba is an open source JIT compiler that translates a subset of Python and NumPy code into fast machine code.
2) Numba-compiled numerical algorithms in Python can approach the speeds of C or FORTRAN.
3) Numba is designed to be used with NumPy arrays and functions. 
4) Numba also works great with Jupyter notebooks for interactive computing, and with distributed execution frameworks, like Dask and Spark.


```python=
@numba.jit(nopython=True, parallel=True)
def logistic_regression(Y, X, w, iterations):
    for i in range(iterations):
        w -= np.dot(((1.0 /
              (1.0 + np.exp(-Y * np.dot(X, w)))
              - 1.0) * Y), X)
    return w
```



