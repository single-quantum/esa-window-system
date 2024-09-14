import numpy as np
import pytest
from pytest import approx

from esawindowsystem.core.max_star import max_star


@pytest.fixture(scope="module", params=[
    [1, 1], [-1, 1], [1, -1], [-1, -1],      # abs(a) and abs(b) < 5
    [7, 1], [-7, 1], [7, -1], [-7, -1],      # only abs(b) < 5
    [2, 6], [-2, 6], [2, -6], [-2, -6],      # only abs(a) < 5
    [6, 10], [-6, 10], [6, -10], [-6, -10],  # abs(a) and abs(b) > 5
    [0, 0], [-np.inf, np.inf]
])
# These values are carefully chosen to make sure all cases are tested
def a_and_b_values(request):
    return request.param


def test_max_star_compare_to_analytical(a_and_b_values, benchmark):
    a, b = a_and_b_values
    analytical_value = np.log(np.exp(a) + np.exp(b))
    result = benchmark(max_star, a, b)
    assert result == approx(analytical_value, rel=2E-2)
