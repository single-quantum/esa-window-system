import itertools
from functools import lru_cache
import numpy as np
import numpy.typing as npt

keys = list(itertools.product(range(-6, 6), repeat=2))
max_log_lookup = {}
for key in keys:
    max_log_lookup[key] = np.log(1 + np.exp(-abs(key[0] - key[1])))

size_arr = 12

max_log_lookup_arr = np.zeros((size_arr, size_arr), dtype=np.double)

for key in keys:
    idx1 = key[0]+6
    idx2 = key[1]+6

    max_log_lookup_arr[idx1, idx2] = np.log(1 + np.exp(-abs(key[0] - key[1])))


@lru_cache(maxsize=256)
def max_star_lru(a: float, b: float) -> float:
    """Calculate the max star of a and b, which is a modified max function.

    This function is used mostly for optimization. The formal definition of max star is max*(a, b) = log(exp(a)+exp(b)).
    Since this calculation comes up all the time in calculating LLRs and is relatively expensive, it is approximated by
    max*(a, b) ~= max(a, b) + log(1+exp(-|a-b|)), where the second term serves as a correction to the max.
    The second term is cached for optimization reasons.
    """
    # When a or b is > 5, the correction term is already so small that we can discard it.
    if abs(a) > 5 or abs(b) > 5 or abs(a - b) > 5:
        return max(a, b)
    elif a == -np.inf or b == -np.inf:
        return max(a, b) + np.log(2)
    else:
        return max(a, b) + max_log_lookup[(round(a), round(b))]


def max_star_recursive(arr: list | npt.NDArray) -> float:
    """Recursive implementation of the max star operator. """
    result = max_star_lru(arr[0], arr[1])

    for i in range(2, len(arr)):
        result = max_star_lru(result, arr[i])

    return result


@lru_cache(maxsize=256)
def max_star_lru_arr_lookup(a: float, b: float) -> float:
    """Calculate the max star of a and b, which is a modified max function.

    This function is used mostly for optimization. The formal definition of max star is max*(a, b) = log(exp(a)+exp(b)).
    Since this calculation comes up all the time in calculating LLRs and is relatively expensive, it is approximated by
    max*(a, b) ~= max(a, b) + log(1+exp(-|a-b|)), where the second term serves as a correction to the max.
    The second term is cached for optimization reasons.
    """
    # When a or b is > 5, the correction term is already so small that we can discard it.
    if abs(a) > 5 or abs(b) > 5 or abs(a - b) > 5:
        return max(a, b)
    elif a == -np.inf or b == -np.inf:
        return max(a, b) + np.log(2)
    else:
        idx1 = int(round(a)+6)
        idx2 = int(round(b)+6)
        return max(a, b) + max_log_lookup_arr[idx1, idx2]
