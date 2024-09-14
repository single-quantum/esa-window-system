import itertools
# from functools import lru_cache

import numpy as np
from numpy import inf
import cython
import numpy.typing as npt

def get_max_log_lookup() -> cython.double[:, :]:
    keys = list(itertools.product(range(-6, 6), repeat=2))
    size_arr: cython.int = 12
    max_log_lookup = np.zeros((size_arr, size_arr), dtype=np.double)

    for key in keys:
        idx1: cython.int = key[0]+6
        idx2: cython.int = key[1]+6

        max_log_lookup[idx1, idx2] = np.log(1 + np.exp(-abs(key[0] - key[1])))

    return max_log_lookup


@cython.ccall
def max_star(a: cython.double, b: cython.double) -> cython.double:
    """Calculate the max star of a and b, which is a modified max function.

    This function is used mostly for optimization. The formal definition of max star is max*(a, b) = log(exp(a)+exp(b)).
    Since this calculation comes up all the time in calculating LLRs and is relatively expensive, it is approximated by
    max*(a, b) ~= max(a, b) + log(1+exp(-|a-b|)), where the second term serves as a correction to the max.
    The second term is cached for optimization reasons.
    """

    max_log_lookup = get_max_log_lookup()
    max_log_lookup_view: cython.double[:, :] = max_log_lookup

    return_val: cython.double
    
    idx1: cython.int = int(round(a)+6)
    idx2: cython.int = int(round(b)+6)
    
    tmp: cython.double

    # When a or b is > 5, the correction term is already so small that we can discard it.
    if abs(a) > 5 or abs(b) > 5 or abs(a - b) > 5:
        return_val = max(a, b)
        return return_val
    elif a == -inf or b == -inf:
        return_val = max(a, b) + 0.6931471 # + np.log(2)
        return return_val
    else:
        
        return_val = max(a, b) + max_log_lookup_view[idx1, idx2]
        return return_val
