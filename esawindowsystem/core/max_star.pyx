import itertools
# from functools import lru_cache
from libc.math cimport round, fmax, exp, log, abs, INFINITY

import numpy as np
from numpy import inf
import cython

# cdef object lookup = {}



# cpdef float get_max_log_lookup(float a, float b):


#     cdef int[:, :] keys_list
#     cdef int idx1, idx2

#     keys = list(itertools.product(range(-6, 6), repeat=2))
#     keys_list = [list(key) for key in keys]
#     size_arr: int = 12
#     max_log_lookup = np.zeros((size_arr, size_arr), dtype=np.double)

#     for key in keys:
#         idx1 = key[0]
#         idx2 = key[1]

#         max_log_lookup[(idx1, idx2)] = logf(1 + expf(-abs(key[0] - key[1])))

#     return max_log_lookup


@cython.ccall
cpdef double max_star_c(double a, double b):
    """Calculate the max star of a and b, which is a modified max function.

    This function is used mostly for optimization. The formal definition of max star is max*(a, b) = log(exp(a)+exp(b)).
    Since this calculation comes up all the time in calculating LLRs and is relatively expensive, it is approximated by
    max*(a, b) ~= max(a, b) + log(1+exp(-|a-b|)), where the second term serves as a correction to the max.
    The second term is cached for optimization reasons.
    """

    cdef double idx1, idx2
    cdef double return_val, lookup_val

    # When a or b is > 5, the correction term is already so small that we can discard it.
    if abs(a) > 5 or abs(b) > 5 or abs(a - b) > 5:
        return_val = fmax(a, b)
        return return_val
    else:
        return_val = fmax(a, b) + log(1 + exp(-abs(round(a) - round(b))))
        return return_val


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double max_star_recursive_c(double[:] arr):
    cdef double a, b, result
    cdef Py_ssize_t i, len_arr

    len_arr = arr.shape[0]

    a = arr[0]
    b = arr[1]
    
    result = max_star_c(a, b)

    for i in range(2, len_arr):
        result = max_star_c(result, arr[i])

    return result