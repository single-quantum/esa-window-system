import itertools
# from functools import lru_cache

import numpy as np
import cython
import numpy.typing as npt

keys = list(itertools.product(range(-6, 6), repeat=2))

size_arr: cython.Py_ssize_t = 12

max_log_lookup: cython.double[:, :] = np.zeros((size_arr, size_arr), dtype=np.double)

for key in keys:
    idx1: cython.Py_ssize_t = key[0]+6
    idx2: cython.Py_ssize_t = key[1]+6

    max_log_lookup[idx1, idx2] = np.log(1 + np.exp(-abs(key[0] - key[1])))

@cython.wraparound(False)
def max_star(a: cython.double, b: cython.double):
    """Calculate the max star of a and b, which is a modified max function.

    This function is used mostly for optimization. The formal definition of max star is max*(a, b) = log(exp(a)+exp(b)).
    Since this calculation comes up all the time in calculating LLRs and is relatively expensive, it is approximated by
    max*(a, b) ~= max(a, b) + log(1+exp(-|a-b|)), where the second term serves as a correction to the max.
    The second term is cached for optimization reasons.
    """

    return_val: np.double

    # When a or b is > 5, the correction term is already so small that we can discard it.
    if abs(a) > 5 or abs(b) > 5 or abs(a - b) > 5:
        return_val = max(a, b)
        return return_val
    elif a == -np.inf or b == -np.inf:
        return_val = max(a, b) + np.log(2)
        return return_val
    else:
        idx1: cython.Py_ssize_t = int(round(a)+6)
        idx2: cython.Py_ssize_t = int(round(b)+6)
        
        return_val = max(a, b) + max_log_lookup[idx1, idx2]
        return return_val

@cython.wraparound(False)
def max_star_recursive(arr: list | npt.NDArray) -> float:
    """Recursive implementation of the max star operator. """

    i: cython.Py_ssize_t

    result: cython.double

    result = max_star(arr[0], arr[1])
    x_max = len(arr)
    i = 2
    while i < x_max:
        result = max_star(result, arr[i])
        i += 1

    return result
